import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import base64
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import memory_manager
from backend.services.memory import persistent_memory

from backend.services.eda.main import get_eda_tools
from backend.services.visualization.main import get_visualization_tools
from backend.services.ml.main import get_ml_tools
from backend.services.rag.main import get_rag_tools
from backend.services.optimizer.main import get_optimizer_tools

_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google").lower()
_AGENT_MODEL  = os.environ.get("AGENT_MODEL")

def _build_agent_llm():
    provider = _LLM_PROVIDER

    if provider == "google":
        from api_inject import get_gemini_llm
        return get_gemini_llm(
            model=_AGENT_MODEL,
            temperature=0.3,
            convert_system_message_to_human=False,
        )
    elif provider == "groq":
        return ChatGroq(
            model=_AGENT_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.3,
        )
    else:
        return ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model=_AGENT_MODEL,
            temperature=0.3,
            request_timeout=60.0,
        )

llm = _build_agent_llm()
print(f"🤖 [Agent LLM] Provider={_LLM_PROVIDER} | Model={_AGENT_MODEL}")def _prepare_agent_executor(
    session_id: str,
    prompt: str,
    new_file_path: Optional[str] = None,
    new_dataset_name: Optional[str] = None,
):
    """
    Internal helper: jalankan semua setup (file loading, tool building, memory,
    system prompt, AgentExecutor) secara terpadu.
    Mengembalikan (agent_executor, invoke_input, memory_stm, context).
    Dipanggil oleh run_agent_flow (sync) dan run_agent_flow_streaming (async).
    """
    column_list = []
    file_path_to_use = new_file_path
    file_type = None

    if new_file_path and new_dataset_name:
        if new_dataset_name.endswith('.csv'):
            file_type = 'csv'
            try:
                with open(new_file_path, 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                if df is not None:
                    column_list = df.columns.tolist()
                persistent_memory.save_dataset_path(session_id, "__latest_csv", new_file_path)
            except Exception as e:
                print(f"Gagal membaca file CSV baru untuk kolom: {e}")
        elif new_dataset_name.endswith('.pdf'):
            file_type = 'pdf'
            persistent_memory.save_dataset_path(session_id, "__latest_pdf", new_file_path)
    elif not new_file_path:
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                file_path_to_use = dataset_info['path']
                file_type = 'csv'
                with open(file_path_to_use, 'rb') as f:
                    csv_contents_bytes = f.read()
                df = _read_csv_with_fallback(csv_contents_bytes)
                if df is not None:
                    column_list = df.columns.tolist()
            except Exception as e:
                print(f"Gagal memuat kolom dari file CSV di LTM: {e}")
        else:
            default_info = persistent_memory.get_dataset_path("__default__", "__latest_csv")
            if default_info and os.path.exists(default_info['path']):
                try:
                    file_path_to_use = default_info['path']
                    file_type = 'csv'
                    persistent_memory.save_dataset_path(session_id, "__latest_csv", file_path_to_use)
                    with open(file_path_to_use, 'rb') as f:
                        csv_contents_bytes = f.read()
                    df = _read_csv_with_fallback(csv_contents_bytes)
                    if df is not None:
                        column_list = df.columns.tolist()
                    print(f"[Agent] Menggunakan dataset default Bali v3 untuk sesi {session_id}")
                except Exception as e:
                    print(f"Gagal memuat dataset default: {e}")

    context = {}

    eda_tools = get_eda_tools(session_id, context)
    vis_tools = get_visualization_tools(session_id, context)
    ml_tools  = get_ml_tools(session_id, context)
    rag_tools = get_rag_tools(session_id, context, llm)
    optimizer_tools = get_optimizer_tools(session_id, context)

    from backend.services.agent.verify_output import create_verify_output_tool
    verify_output_tool = create_verify_output_tool(context)

    tools = eda_tools + vis_tools + ml_tools + rag_tools + optimizer_tools + [verify_output_tool]

    memory_stm  = memory_manager.get_or_create_memory(session_id)
    chat_history = memory_stm.load_memory_variables({})['chat_history']

    columns_str = ", ".join(column_list) if column_list else "Tidak ada file CSV yang dikonfirmasi."

    system_prompt = f"""
IDENTITAS PERSONA — ROLE PLAYER
Anda adalah AI Data Scientist — seorang Pakar Perjalanan Bali berbasis AI yang dibangun di atas arsitektur Neuro-Symbolic ReAct.

Karakteristik Mutlak AI Data Scientist:
- Berpikir sistematis, terstruktur, berbasis data riil.
- Berbicara dengan hangat, profesional, berbahasa Indonesia natural.
- TIDAK PERNAH berhalusinasi angka atau lokasi. Angka dan lokasi WAJIB sama persis dengan input pengguna.
- Jujur dan memberikan "Logical Pushback" jika data/budget tidak mendukung.

Kolom dataset yang tersedia saat ini:
{columns_str}

<rules>
ATURAN EKSTRAKSI & TOOL CALLING (KRITIS):
1. HARFIAH: Jika user mengetik budget "400.000", kirim 400000. JANGAN dibulatkan. Jika ada kata "juta", kalikan 1000000.
2. LOKASI WAJIB: Jika user menyebut nama wilayah tertentu seperti "Bangli", keyword lokasi WAJIB mengandung nama wilayah tersebut. Namun jika pengguna hanya menyebutkan preferensi suasana/kategori (seperti "damai", "hijau", "sepi", "alam") tanpa nama kota, JANGAN menolak! Gunakan `rag_semantic_filter` untuk memetakan preferensi tersebut menjadi kata kunci destinasi Bali yang relevan.
3. ANTI-PLAGIAT CONTOH: Jangan pernah menggunakan angka atau lokasi yang ada di dalam blok <examples> untuk menjawab pertanyaan pengguna.
4. VERIFIKASI WAJIB: Tool `verify_output` WAJIB dipanggil sebagai langkah TERAKHIR sebelum memberikan Final Answer. DILARANG Final Answer tanpa verify_output.
5. BUDGET TRANSPARENCY: Setelah menerima output optimizer, WAJIB sebutkan:
   - Total biaya aktual vs budget yang diberikan user
   - Persentase budget yang terpakai
   - Jika utilization < 80%, jelaskan alasannya dan tawarkan alternatif
6. VARIASI HARGA: Itinerary yang baik memiliki campuran destinasi berbayar dan gratis. Jangan dominasi dengan destinasi gratis saja.
7. BATASAN DOMAIN & GUARDRAIL: Anda HANYA melayani analisis data dan konsultasi wisata di provinsi Bali. Jika user meminta kode pemrograman (seperti Python, FizzBuzz, dll) atau destinasi di luar provinsi Bali (seperti Jakarta, Bandung, dll), TOLAK DENGAN SOPAN tanpa memanggil tool apapun, dan ingatkan bahwa spesialisasi Anda adalah domain pariwisata Bali.
8. PERMINTAAN ITINERARY BARU / MULTI-TURN DALAM SESI: Jika pengguna meminta itinerary baru dalam sesi yang sama (meskipun dengan budget, durasi, atau preferensi berbeda), segera rancang itinerary baru tersebut. Jangan menolak dan jangan katakan ada kesalahan pada input pengguna.
</rules>

<sop>
SOP PEMBUATAN ITINERARY (NEURO-SYMBOLIC PIPELINE):
STEP 0 - EXTRACTION: Baca <current_task> dengan teliti. Ekstrak budget_limit, duration_days, dan preferensi lokasi/kategori/suasana.
STEP 1 - RAG & SEMANTIC FILTER:
  - Jika user mengunggah PDF, ATAU jika user mengekspresikan preferensi abstrak/emosional (seperti "damai", "sepi", "healing", "hijau-hijau", "jauh dari kota", "stres"):
    WAJIB panggil `rag_semantic_filter(query=...)` untuk menerjemahkan preferensi emosional tersebut ke kata kunci destinasi konkrit!
  - Gunakan kata kunci hasil RAG untuk parameter `location_keywords` di optimizer.
STEP 2 - ML SCORE: Panggil `predict_match_score` untuk probabilitas relevansi (opsional).
STEP 3 - OPTIMIZER: Panggil `budget_optimizer_tool` MENGGUNAKAN NILAI EKSTRAK & KATA KUNCI TERBAIK.
STEP 4 - VISUALISASI: Panggil `plot_budget_breakdown` dengan daftar destinasi terpilih dari output optimizer.
STEP 5 - VERIFIKASI: Panggil `verify_output(draft=...)`.
STEP 6 - FINAL ANSWER: Narasikan itinerary per hari (Jam, Nama, Rating, Harga, Jarak, Waktu Tempuh) HANYA menggunakan data dari output tool.
</sop>
<examples>
  <example_1>
    User: "Buatkan itinerary 2 hari untuk wisata pantai di Kuta, budget saya 500.000 rupiah."
    Thought: User meminta durasi 2 hari, lokasi Kuta, kategori pantai, budget 500000. Lokasi dan kategori sudah sangat spesifik. Saya akan langsung memanggil optimizer.
    Action: Panggil budget_optimizer_tool(budget_limit=500000, location_keywords=["Kuta", "pantai"], duration_days=2, min_rating=0.0)
  </example_1>
  <example_2>
    User: "Saya mau ke Kabupaten Bangli cari wisata alam 3 hari dengan budget 1 juta."
    Thought: User meminta durasi 3 hari, lokasi Bangli, kategori alam, budget 1000000.
    Action: Panggil budget_optimizer_tool(budget_limit=1000000, location_keywords=["Bangli", "alam"], duration_days=3, min_rating=0.0)
  </example_2>
  <example_3>
    User: "Saya sedang sangat stres dari pekerjaan dan ingin mencari suasana yang damai, jauh dari hiruk-pikuk kota, dan hijau-hijau selama 1 hari. Budget saya hanya 150.000."
    Thought: User mengekspresikan preferensi suasana abstrak ("stres", "damai", "jauh dari hiruk-pikuk", "hijau-hijau") dengan durasi 1 hari dan budget 150000. Saya akan memanggil rag_semantic_filter untuk memetakan preferensi ini menjadi kata kunci konkrit destinasi Bali, lalu memanggil optimizer.
    Action: Panggil rag_semantic_filter(query="suasana damai jauh dari hiruk pikuk kota hijau hijau healing")
  </example_3>
</examples>

<current_task>
Input Pengguna: "{prompt}"
</current_task>
"""
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=8,
    )

    invoke_input = {"input": prompt, "chat_history": chat_history}
    return agent_executor, invoke_input, memory_stm, context


def run_agent_flow(session_id: str, prompt: str, new_file_path: Optional[str] = None, new_dataset_name: Optional[str] = None):
    try:
        agent_executor, invoke_input, memory_stm, context = _prepare_agent_executor(
            session_id, prompt, new_file_path, new_dataset_name
        )
        chat_history = invoke_input.get("chat_history", [])

        # ── DEBUG: Cek apa yang benar-benar dikirim ke LLM ──
        print(f"\n{'='*60}")
        print(f"🔍 [DEBUG] SESSION: {session_id}")
        print(f"🔍 [DEBUG] PROMPT USER: {prompt}")
        print(f"🔍 [DEBUG] CHAT HISTORY LENGTH: {len(chat_history)} messages")
        if chat_history:
            for i, msg in enumerate(chat_history[-4:]):  # 4 pesan terakhir
                print(f"🔍 [DEBUG]   history[{i}]: {type(msg).__name__} = {str(msg.content)[:100]}...")
        print(f"{'='*60}\n")

        response = agent_executor.invoke(invoke_input)
        agent_response = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])

        reasoning_log = []
        for step_idx, (action, observation) in enumerate(intermediate_steps):
            log_entry = {
                "step": step_idx + 1,
                "thought": action.log.split("Action:")[0].strip() if hasattr(action, 'log') else "Memutuskan aksi...",
                "tool_called": action.tool,
                "tool_input": action.tool_input,
                "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
            }
            reasoning_log.append(log_entry)

        inputs = {"input": prompt}
        outputs = {"output": agent_response}
        memory_stm.save_context(inputs, outputs)
        persistent_memory.save_chat_history(session_id, memory_stm)
        print(f"--- [STM] Konteks disimpan ke cache memori sesi {session_id} ---")

        result = {
            "summary": agent_response,
            "reasoning_log": reasoning_log
        }

        if "last_tool_name" in context:
            result["tool_name"] = context["last_tool_name"]

        if "last_tool_output" in context:
            result["data"] = context["last_tool_output"]
        if "last_image_bytes" in context:
            image_bytes = context["last_image_bytes"]
            result["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            result["image_format"] = "png"

            from backend.services.agent.interpretation import get_interpretation
            tool_name = context.get("last_tool_name", "custom plot")

            tool_params = context.get("last_tool_params", {})
            try:
                interpretation = get_interpretation(session_id, tool_name, {"tool_name": tool_name, **tool_params}, image_bytes=image_bytes)
                result["summary"] += "\n\n" + interpretation
            except Exception as e:
                print(f"Gagal generate interpretasi gambar: {e}")

        return result

    except KeyError as e:
        return {"error": "Data tidak ditemukan", "detail": f"Kolom atau key yang diperlukan tidak tersedia: {str(e)}"}
    except ValueError as e:
        return {"error": "Nilai tidak valid", "detail": f"Input atau data memiliki format yang salah: {str(e)}"}
    except TimeoutError:
        return {"error": "Agent Timeout", "detail": "Agent membutuhkan waktu terlalu lama. Coba pertanyaan yang lebih sederhana."}
    except Exception as e:
        return {"error": "Gagal menjalankan agen", "detail": str(e)}


def _json_safe(obj):
    """
    Sanitasi rekursif agar semua nilai dalam dict/list menjadi JSON-serializable murni
    (mengubah NaN/Inf menjadi None, numpy/pandas types ke native Python).
    """
    import math
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(item) for item in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return [_json_safe(x) for x in obj.tolist()]
    except ImportError:
        pass
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except ImportError:
        pass
    return str(obj)


async def run_agent_flow_streaming(
    session_id: str,
    prompt: str,
    new_file_path: Optional[str] = None,
    new_dataset_name: Optional[str] = None,
):
    """
    Async generator — wrapper streaming untuk WebSocket.
    Menggunakan astream_events() LangChain untuk emit event per-step secara real-time.
    Tidak mengubah apapun di logika tool, RAG, atau Symbolic Optimizer.

    Event types yang di-yield (dict siap dikirim via websocket.send_json):
      {"type": "token",      "content": "<token LLM>"}        ← streaming per token
      {"type": "tool_start", "tool": "<nama>", "input": {}}   ← tool dipanggil
      {"type": "tool_end",   "tool": "<nama>", "output": ""}  ← tool selesai
      {"type": "image",      "data": "<base64>", "format": "png"}
      {"type": "done",       "summary": "", "reasoning_log": [], "session_id": ""}
      {"type": "error",      "detail": "<pesan>"}
    """
    import asyncio

    # Setup dijalankan di thread pool agar tidak blocking event loop
    try:
        agent_executor, invoke_input, memory_stm, context = await asyncio.to_thread(
            _prepare_agent_executor,
            session_id, prompt, new_file_path, new_dataset_name,
        )
    except Exception as e:
        yield {"type": "error", "detail": f"Gagal menyiapkan agent: {str(e)}"}
        return

    print(f"\n{'='*60}")
    print(f"🔌 [WS STREAM] SESSION: {session_id}")
    print(f"🔌 [WS STREAM] PROMPT: {prompt}")
    print(f"{'='*60}\n")

    reasoning_log: list = []
    final_answer   = ""
    current_thought = ""

    try:
        async for event in agent_executor.astream_events(invoke_input, version="v2"):
            # Tiap event diproses secara independen — error satu event tidak
            # menghentikan seluruh stream
            try:
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # ── LLM token stream ──
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        current_thought += chunk.content
                        yield {"type": "token", "content": chunk.content}

                # ── Tool mulai dipanggil ──
                elif kind == "on_tool_start":
                    tool_input = _json_safe(data.get("input", {}))
                    thought_for_this_step = current_thought.strip()
                    current_thought = ""

                    yield {
                        "type": "tool_start",
                        "tool": name,
                        "input": tool_input,
                        "thought": thought_for_this_step or f"Memanggil tool `{name}`..."
                    }
                    reasoning_log.append({
                        "step":        len(reasoning_log) + 1,
                        "thought":     thought_for_this_step or f"Memanggil tool `{name}`...",
                        "tool":        name,
                        "tool_called": name,
                        "tool_input":  tool_input,
                        "observation": None,
                        "status":      "running",
                    })

                # ── Tool selesai ──
                elif kind == "on_tool_end":
                    raw_output = data.get("output", "")
                    obs = str(raw_output)
                    observation_str = obs[:500] + "..." if len(obs) > 500 else obs
                    for entry in reversed(reasoning_log):
                        if entry.get("tool_called") == name and entry["observation"] is None:
                            entry["observation"] = observation_str
                            entry["status"] = "success"
                            break
                    yield {"type": "tool_end", "tool": name, "output": observation_str}

                # ── AgentExecutor selesai → ambil final answer & intermediate steps ──
                elif kind == "on_chain_end" and name == "AgentExecutor":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        final_answer = output.get("output", "")
                        intermediate = output.get("intermediate_steps", [])
                        if intermediate and isinstance(intermediate, list):
                            enriched = []
                            for idx, step_item in enumerate(intermediate):
                                if isinstance(step_item, tuple) and len(step_item) == 2:
                                    act_obj, obs_val = step_item
                                    t_tool = getattr(act_obj, "tool", "")
                                    t_input = _json_safe(getattr(act_obj, "tool_input", {}))
                                    t_log = getattr(act_obj, "log", "") or ""
                                    t_clean = t_log.strip()
                                    if t_clean.startswith("Thought:"):
                                        t_clean = t_clean[len("Thought:"):].strip()
                                    o_str = str(obs_val)
                                    o_clean = o_str[:500] + "..." if len(o_str) > 500 else o_str
                                    
                                    fallback_thought = reasoning_log[idx]["thought"] if idx < len(reasoning_log) else f"Memanggil tool `{t_tool}`..."
                                    enriched.append({
                                        "step": idx + 1,
                                        "thought": t_clean or fallback_thought,
                                        "tool": t_tool,
                                        "tool_called": t_tool,
                                        "tool_input": t_input,
                                        "observation": o_clean,
                                        "status": "success",
                                    })
                            if enriched:
                                reasoning_log = enriched
                    else:
                        final_answer = str(output)

            except Exception as ev_err:
                # Log tapi lanjutkan — satu event buruk tidak boleh matikan stream
                print(f"⚠️  [WS STREAM] Lewati event '{event.get('event','')}': {ev_err}")
                continue

    except Exception as stream_err:
        # Error fatal dari astream_events itu sendiri
        print(f"⚠️  [WS STREAM] Stream event terganggu ({stream_err}), mencoba fallback invoke...")
        try:
            res = await asyncio.to_thread(agent_executor.invoke, invoke_input)
            final_answer = res.get("output", "") if isinstance(res, dict) else str(res)
        except Exception as fallback_err:
            print(f"❌ [WS STREAM] Fallback invoke juga gagal: {fallback_err}")
            yield {"type": "error", "detail": str(stream_err)}
            return

    # ── Simpan memori ──
    try:
        def _save_memory():
            memory_stm.save_context({"input": prompt}, {"output": final_answer})
            persistent_memory.save_chat_history(session_id, memory_stm)
            print(f"--- [STM/WS] Konteks disimpan ke cache sesi {session_id} ---")
        await asyncio.to_thread(_save_memory)
    except Exception as mem_err:
        print(f"⚠️  [WS STREAM] Gagal simpan memori: {mem_err}")

    # ── Bangun paket "done" ──
    done_event: dict = {
        "type":          "done",
        "summary":       final_answer,
        "reasoning_log": reasoning_log,
        "session_id":    session_id,
    }

    if "last_tool_name" in context:
        done_event["tool_name"] = context["last_tool_name"]
    if "last_tool_output" in context:
        # Sanitasi: optimizer output mungkin mengandung numpy/NaN types
        done_event["data"] = _json_safe(context["last_tool_output"])

    if "last_image_bytes" in context:
        image_bytes = context["last_image_bytes"]
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        done_event["image_base64"] = b64
        done_event["image_format"] = "png"
        # Kirim gambar dulu sebagai event terpisah
        yield {"type": "image", "data": b64, "format": "png"}
        from backend.services.agent.interpretation import get_interpretation
        tool_name   = context.get("last_tool_name", "custom plot")
        tool_params = context.get("last_tool_params", {})
        try:
            interpretation = await asyncio.wait_for(
                asyncio.to_thread(
                    get_interpretation,
                    session_id, tool_name,
                    {"tool_name": tool_name, **tool_params},
                    image_bytes,
                ),
                timeout=5.0
            )
            if interpretation and not interpretation.startswith("[Error"):
                done_event["summary"] += "\n\n" + interpretation
        except asyncio.TimeoutError:
            print("[WS] Interpretasi gambar timeout (>5s), dilewati agar respons segera selesai.")
        except Exception as interp_err:
            print(f"[WS] Gagal generate interpretasi: {interp_err}")

    yield _json_safe(done_event)

class PlotPlan(BaseModel):
    plot_type: str = Field(description="Tipe plot, harus salah satu dari: bar, box, histogram, scatter.")
    x_col: str = Field(description="Nama kolom untuk sumbu X.")
    y_col: Optional[str] = Field(default=None, description="Nama kolom untuk sumbu Y.")
    hue_col: Optional[str] = Field(default=None, description="Nama kolom untuk pewarnaan (hue).")
    orientation: str = Field(default='v', description="Orientasi plot, 'v' untuk vertikal, 'h' untuk horizontal.")

def get_plot_plan(user_prompt: str) -> dict:
    parser = JsonOutputParser(pydantic_object=PlotPlan)
    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten yang tugasnya mengekstrak parameter untuk membuat plot dari permintaan pengguna.
        {format_instructions}
        Permintaan Pengguna: {user_input}"""
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "user_input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        return {"error": "Gagal mengekstrak parameter plot.", "detail": str(e)}