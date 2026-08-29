import os
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from backend.api.router import agent_router
from backend.services.memory import persistent_memory
from backend.services.memory.persistent_memory import cleanup_old_sessions
from backend.services.memory.memory_manager import save_system_vector_store
from backend.services.rag.dataset_indexer import build_system_vector_store

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Build_Dataset", "bali_tourist_clean_v3.csv"
)
DEFAULT_SESSION_ID = "__default__"

_REQUIRED_ENVS = {
    "GOOGLE_API_KEY":     "Diperlukan untuk embedding RAG (GoogleGenerativeAIEmbeddings)",
    "OPENROUTER_API_KEY": "Diperlukan untuk LLM agent dan interpretasi (OpenRouter)",
}

def _validate_env() -> list[str]:
    """Kembalikan list ENV yang belum di-set."""
    return [f"{k} — {desc}" for k, desc in _REQUIRED_ENVS.items() if not os.environ.get(k)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    missing_envs = _validate_env()
    if missing_envs:
        print("❌ [Startup] ENV tidak lengkap! Server berjalan tapi fitur berikut akan gagal:")
        for m in missing_envs:
            print(f"   ⚠️  {m}")
    else:
        print("✅ [Startup] Semua ENV wajib terdeteksi.")

    if os.path.exists(DEFAULT_DATASET_PATH):
        persistent_memory.save_dataset_path(DEFAULT_SESSION_ID, "__latest_csv", DEFAULT_DATASET_PATH)
        print(f"✅ [Startup] Dataset default Bali (v3) berhasil didaftarkan.")
        print(f"   Path: {DEFAULT_DATASET_PATH}")
    else:
        print(f"⚠️  [Startup] Dataset default tidak ditemukan di: {DEFAULT_DATASET_PATH}")

    removed = cleanup_old_sessions(max_age_days=7)
    print(f"✅ [Startup] LTM cleanup: {removed} sesi lama dihapus dari memory_db.json")

    async def _build_rag_in_background():
        """
        LIM-6: Coba muat FAISS dari disk terlebih dahulu (near-instant).
        Jika tidak ada cache disk, build dari API embedding (30-60 detik).
        """
        from backend.services.rag.dataset_indexer import load_system_vector_store_from_disk

        print("🔄 [Startup] Mencoba memuat FAISS index dari disk...")
        vector_store = load_system_vector_store_from_disk()
        if vector_store:
            save_system_vector_store(vector_store)
            print("✅ [Startup] System RAG vector store dimuat dari disk (near-instant).")
            return

        print("🔄 [Startup] Tidak ada cache disk. Memulai indexing dataset ke RAG vector store...")
        print("   (Proses berjalan di background, server sudah siap menerima request)")
        try:
            loop = asyncio.get_event_loop()
            vector_store = await loop.run_in_executor(
                None,
                build_system_vector_store,
                DEFAULT_DATASET_PATH
            )
            if vector_store:
                save_system_vector_store(vector_store)
                print("✅ [Startup] System RAG vector store siap digunakan.")
            else:
                print("⚠️  [Startup] Gagal membuat system RAG vector store.")
        except Exception as e:
            print(f"❌ [Startup] Error saat indexing RAG: {e}")

    asyncio.create_task(_build_rag_in_background())

    yield
    print("🔌 [Shutdown] Server berhenti.")

app = FastAPI(
    title="WISTA AI Agent API",
    description=(
        "API untuk sistem AI Agent WISTA — Wisata Bali Intelligent Assistant. "
        "Menggabungkan ReAct Agent, Hybrid RAG, ML (RandomForest), "
        "dan Budget Optimizer berbasis Pandas untuk rekomendasi itinerary wisata Bali."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(agent_router.router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Tangkap semua unhandled exception, kembalikan JSON bersih (bukan Python traceback)."""
    print(f"❌ [Global Handler] Unhandled error di {request.url}: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Kembalikan pesan validasi yang lebih ramah pengguna."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request Validation Error",
            "detail": exc.errors(),
        }
    )

@app.get("/", summary="Root", tags=["System"])
def read_root():
    return {
        "status": "ok",
        "message": "Selamat datang di WISTA AI Agent API — v1.0.0",
        "docs": "/docs",
    }

@app.get("/health", summary="Health Check", tags=["System"])
def health_check():
    """Cek status server, ENV, dan RAG system vector store."""
    from backend.services.memory.memory_manager import get_system_vector_store
    missing = _validate_env()
    rag_ready = get_system_vector_store() is not None
    return {
        "status": "healthy" if not missing else "degraded",
        "env_ok":  len(missing) == 0,
        "missing_envs": missing,
        "rag_system_ready": rag_ready,
        "version": "1.0.0",
    }

@app.get("/rag/status", summary="RAG Build Status", tags=["RAG"])
def rag_status():
    """Cek status build system RAG vector store (idle/building/ready/failed)."""
    from backend.services.rag.dataset_indexer import build_status
    from backend.services.memory.memory_manager import get_system_vector_store
    return {
        **build_status,
        "rag_in_memory": get_system_vector_store() is not None,
    }

@app.post("/rag/rebuild", summary="Rebuild RAG Vector Store", tags=["RAG"])
async def rag_rebuild():
    """
    Trigger rebuild system RAG vector store dari dataset CSV tanpa restart server.
    Berguna jika build gagal saat startup (misalnya karena GOOGLE_API_KEY expired).
    """
    from backend.services.rag.dataset_indexer import build_status
    if build_status.get("state") == "building":
        return {"status": "skipped", "message": "RAG sedang dalam proses build. Coba lagi nanti."}

    async def _rebuild():
        import asyncio
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None, build_system_vector_store, DEFAULT_DATASET_PATH
        )
        if vector_store:
            save_system_vector_store(vector_store)
            print("✅ [RAG Rebuild] System vector store berhasil diperbarui.")
        else:
            print("❌ [RAG Rebuild] Gagal membangun vector store.")

    asyncio.create_task(_rebuild())
    return {
        "status": "accepted",
        "message": (
            f"Rebuild RAG dimulai di background dari dataset: {DEFAULT_DATASET_PATH}. "
            "Pantau progress via GET /rag/status."
        )
    }

@app.get("/rag/evaluate", summary="Evaluasi Formal Retrieval RAG", tags=["RAG"])
def rag_evaluate(k: int = 5):
    """
    Evaluasi formal retrieval quality sistem RAG menggunakan:
    - **Precision@K** : proporsi dokumen relevan dalam top-K hasil retrieve
    - **Recall@K**    : proporsi dokumen relevan yang berhasil ditemukan
    - **MRR**         : Mean Reciprocal Rank (posisi dokumen relevan pertama)
    - **Hit Rate@K**  : apakah minimal 1 dokumen relevan ada di top-K

    Evaluasi menggunakan **8 test query standar** domain wisata Bali
    dengan Pseudo Relevance Judgement (keyword-based ground truth).

    Hasil cocok untuk dicantumkan di **Bab 4 Evaluasi Sistem** pada skripsi.
    """
    from backend.services.memory.memory_manager import get_system_vector_store
    from backend.services.rag.evaluator import evaluate_rag_retrieval

    vector_store = get_system_vector_store()
    if vector_store is None:
        return {
            "status": "error",
            "message": (
                "System RAG vector store belum tersedia. "
                "Tunggu proses build selesai atau trigger POST /rag/rebuild terlebih dahulu. "
                "Pantau progress via GET /rag/status."
            )
        }

    if k < 1 or k > 20:
        return {"status": "error", "message": "Parameter K harus antara 1 dan 20."}

    result = evaluate_rag_retrieval(vector_store=vector_store, k=k)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint — real-time streaming agent
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    WebSocket endpoint untuk streaming output agent WISTA secara real-time.

    Protokol (client → server):
        JSON: {
            "prompt":       "<teks pertanyaan pengguna>",          # wajib
            "session_id":   "<session id>",                        # opsional, auto-generate jika kosong
            "file_path":    "<path file CSV/PDF di server>",       # opsional
            "dataset_name": "<nama file, misal 'data.csv'>",       # opsional
        }

    Protokol (server → client) — setiap event adalah JSON terpisah:
        {"type": "token",      "content": "<teks token LLM>"}
        {"type": "tool_start", "tool": "<nama>",  "input": {...}}
        {"type": "tool_end",   "tool": "<nama>",  "output": "<ringkasan>"}
        {"type": "image",      "data": "<base64>","format": "png"}
        {"type": "done",       "summary": "...",  "reasoning_log": [...], "session_id": "..."}
        {"type": "error",      "detail": "<pesan kesalahan>"}
    """
    import uuid as _uuid
    from backend.services.agent.main import run_agent_flow_streaming

    await websocket.accept()
    print("🔌 [WS] Client terhubung.")

    try:
        while True:
            # Tunggu pesan dari client
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                raise  # biarkan handler luar menangkap
            except Exception as recv_err:
                print(f"⚠️  [WS] Gagal menerima pesan: {recv_err}")
                break

            # Skip frame kosong (ping/pong/close frame dari Postman)
            if not raw or not raw.strip():
                continue

            try:
                payload = __import__("json").loads(raw)
            except Exception:
                await websocket.send_json({
                    "type":   "error",
                    "detail": "Pesan harus berformat JSON valid.",
                })
                continue

            prompt = payload.get("prompt", "").strip()
            if not prompt:
                await websocket.send_json({
                    "type":   "error",
                    "detail": "Field 'prompt' wajib diisi dan tidak boleh kosong.",
                })
                continue

            session_id   = payload.get("session_id") or str(_uuid.uuid4())
            file_path    = payload.get("file_path")
            dataset_name = payload.get("dataset_name")

            print(f"🔌 [WS] SESSION={session_id} | PROMPT={prompt[:80]}...")

            # Streaming: kirim setiap event langsung ke client
            async for event in run_agent_flow_streaming(
                session_id=session_id,
                prompt=prompt,
                new_file_path=file_path,
                new_dataset_name=dataset_name,
            ):
                try:
                    await websocket.send_json(event)
                except Exception as send_err:
                    # Jika event tertentu gagal di-serialize, log dan lanjutkan
                    print(f"⚠️  [WS] Gagal kirim event '{event.get('type','?')}': {send_err}")
                    try:
                        await websocket.send_json({
                            "type":   "error",
                            "detail": f"Gagal mengirim event: {send_err}",
                        })
                    except Exception:
                        pass

    except WebSocketDisconnect:
        print("🔌 [WS] Client terputus.")
    except Exception as e:
        print(f"❌ [WS] Error tidak terduga: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
