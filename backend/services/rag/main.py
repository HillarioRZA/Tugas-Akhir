from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.services.memory import persistent_memory
from backend.services.memory import memory_manager
from backend.services.rag import parser as rag_parser
from backend.services.rag import vectorizer as rag_vectorizer
from backend.services.rag import retriever as rag_retriever
import json

class SemanticFilterInput(BaseModel):
    query: str = Field(description="Preferensi abstrak pengguna, contoh: 'tempat untuk healing dan bebas keramaian'")

def get_rag_tools(session_id: str, context: dict, llm: Any) -> List[Any]:
    import os

    def _read_current_pdf():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_pdf")
        if dataset_info and os.path.exists(dataset_info['path']):
            with open(dataset_info['path'], 'rb') as f:
                return f.read()
        return None

    def _get_active_vector_store():
        """
        Resolusi vector store dengan 2-layer fallback:

        Layer 1 — User Store:
            Jika user sudah upload & index PDF → pakai vector store sesii mereka.
            Ini memberikan hasil yang lebih personal/relevan dengan konten PDF-nya.

        Layer 2 — System Store (Fallback):
            Jika tidak ada PDF user → pakai system vector store dari dataset CSV.
            Selalu tersedia sejak startup server. Tidak perlu upload apapun.

        Returns:
            (vector_store, source_label)
        """
        # Layer 1: Cek user vector store
        user_vs = memory_manager.get_vector_store(session_id)
        if user_vs is not None:
            return user_vs, "user_pdf"

        # Layer 2: Fallback ke system vector store (dataset CSV)
        system_vs = memory_manager.get_system_vector_store()
        if system_vs is not None:
            return system_vs, "system_dataset"

        return None, None

    @tool
    def index_pdf() -> dict:
        """
        Gunakan ini satu kali ketika pengguna mengunggah file PDF referensi wisata.
        Mengindeks dokumen PDF agar kontennya bisa digunakan bersama knowledge base dataset.
        Setelah diindeks, rag_semantic_filter akan memprioritaskan isi PDF ini.
        """
        file_contents = _read_current_pdf()
        if not file_contents:
            return {"error": "Tidak ada file PDF yang diunggah untuk diindeks."}

        text_content = rag_parser.parse_pdf(file_contents)
        if isinstance(text_content, str) and text_content.startswith("Error:"):
            return {"error": "Gagal mem-parsing PDF.", "detail": text_content}

        vector_store = rag_vectorizer.create_vector_store(text_content)
        if isinstance(vector_store, str) and vector_store.startswith("Error:"):
            return {"error": "Gagal membuat vector store.", "detail": vector_store}

        memory_manager.save_vector_store(session_id, vector_store)
        context["last_tool_name"] = "index_pdf"
        return {
            "status": "success",
            "summary": (
                "Dokumen PDF berhasil diindeks. "
                "rag_semantic_filter sekarang akan memprioritaskan konten PDF ini "
                "di samping knowledge base dataset bawaan."
            )
        }

    @tool(args_schema=SemanticFilterInput)
    def rag_semantic_filter(query: str) -> dict:
        """
        Terjemahkan preferensi abstrak atau deskripsi wisata pengguna menjadi
        sekumpulan kata kunci konkrit (lokasi, kategori, suasana) berdasarkan
        knowledge base wisata Bali.

        Knowledge base memiliki 2 sumber (otomatis dipilih):
        - Sumber utama: knowledge base dataset wisata Bali (selalu tersedia)
        - Sumber tambahan: PDF referensi dari pengguna (jika ada)

        KATA KUNCI KEMBALIAN WAJIB DIGUNAKAN UNTUK PARAMETER OPTIMIZER.
        """
        vector_store, source = _get_active_vector_store()

        if vector_store is None:
            # Fallback graceful: sistem belum siap (indexing masih berjalan)
            return {
                "status": "fallback",
                "summary": "Knowledge base RAG sedang diinisialisasi. Menggunakan keyword langsung dari preferensi pengguna.",
                "data": [word.strip() for word in query.split() if len(word) > 3],
                "extracted_keywords": [word.strip() for word in query.split() if len(word) > 3],
                "source": "direct_extraction"
            }

        # ── Craft extraction prompt ──────────────────────────────────────────
        if source == "user_pdf":
            source_context = "dokumen referensi wisata yang diunggah pengguna"
        else:
            source_context = "knowledge base dataset wisata Bali (1400+ destinasi)"

        extraction_prompt = f"""
        Berdasarkan {source_context}, terjemahkan preferensi abstrak ini:
        "{query}"

        Menjadi KATA KUNCI LOKASI atau KATEGORI wisata yang KONKRIT dan ADA di data wisata Bali.
        Pilah antara nama kota/kabupaten, kategori (Alam/Budaya/Rekreasi/Umum), dan suasana (sepi/ramai).

        Keluarkan HANYA dalam format array JSON dari string.
        Contoh output: ["Kintamani", "Alam", "Sepi"] atau ["Seminyak", "Pantai", "Rekreasi"]
        Jangan tambahkan teks apapun selain array JSON tersebut.
        """

        answer_raw = rag_retriever.get_rag_answer(extraction_prompt, vector_store, llm)

        if isinstance(answer_raw, str) and answer_raw.startswith("Error:"):
            return {"error": "Gagal menjalankan RAG semantic filter.", "detail": answer_raw}

        # ── Parse hasil LLM ke list keywords ────────────────────────────────
        clean_json_str = answer_raw.replace('```json', '').replace('```', '').strip()

        try:
            keywords = json.loads(clean_json_str)
            if not isinstance(keywords, list):
                keywords = [str(keywords)]
        except json.JSONDecodeError:
            keywords = [chunk.strip() for chunk in clean_json_str.split(',')]

        # Bersihkan keyword kosong
        keywords = [k for k in keywords if k and k.strip()]

        context["last_tool_name"] = "rag_semantic_filter"
        context["last_tool_output"] = keywords

        return {
            "status": "success",
            "summary": (
                f"Berhasil mengekstrak {len(keywords)} kata kunci dari preferensi abstrak "
                f"(sumber: {source})."
            ),
            "data": keywords,
            "extracted_keywords": keywords,
            "source": source,  # "user_pdf" atau "system_dataset" — berguna untuk XAI
        }

    return [
        index_pdf,
        rag_semantic_filter
    ]
