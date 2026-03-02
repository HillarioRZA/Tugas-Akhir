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

    @tool
    def index_pdf() -> dict:
        """Gunakan ini satu kali saat pengguna mengunggah file PDF. Ini akan membaca dan mengindeks dokumen PDF agar siap ditanyai."""
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
        return {"status": "success", "summary": "Dokumen PDF berhasil diindeks dan siap untuk ditanyai sekarang."}

    @tool(args_schema=SemanticFilterInput)
    def rag_semantic_filter(query: str) -> dict:
        """Gunakan alat ini untuk menerjemahkan/memfilter preferensi abstrak atau deskripsi wisata pengguna menjadi sekumpulan string kata kunci konkrit (misalnya lokasi atau kategori spesifik) berdasarkan dokumen referensi PDF wisata. KATA KUNCI KEMBALIAN WAJIB DIGUNAKAN UNTUK PARAMETER OPTIMIZER."""
        vector_store = memory_manager.get_vector_store(session_id)
        if not vector_store:
             return {"error": "PDF referensi wisata belum diindeks. Silakan minta pengguna untuk mengunggah dan indeks PDF terlebih dahulu."}
        
        # Crafting a specific prompt to force LLM to extract JSON list of keywords
        extraction_prompt = f"""
        Berdasarkan dari referensi dokumen wisata, terjemahkan kueri preferensi abstrak ini: "{query}" menjadi KATA KUNCI LOKASI atau KATEGORI wisata yang kongkrit yang ada di profil dokumen. 
        Keluarkan HANYA dalam format array JSON dari string, contohnya: ["Ubud", "Kintamani", "Alam"] atau ["Pantai Kuta", "Seminyak"].
        Jangan tambahkan teks apapun selain array JSON tersebut.
        """
        answer_raw = rag_retriever.get_rag_answer(extraction_prompt, vector_store, llm)
        
        if isinstance(answer_raw, str) and answer_raw.startswith("Error:"):
            return {"error": "Gagal menjalankan RAG semantic filter.", "detail": answer_raw}
        
        # Clean markdown formatting if present
        clean_json_str = answer_raw.replace('```json', '').replace('```', '').strip()
        
        try:
             keywords = json.loads(clean_json_str)
             if not isinstance(keywords, list):
                 keywords = [str(keywords)]
        except json.JSONDecodeError:
             # Fallback if the LLM didn't return proper JSON
             keywords = [chunk.strip() for chunk in clean_json_str.split(',')]
             
        context["last_tool_name"] = "rag_semantic_filter"
        context["last_tool_output"] = keywords
        
        return {"status": "success", "summary": f"Berhasil mengekstrak kata kunci dari preferensi abstrak.", "data": keywords, "extracted_keywords": keywords}

    return [
        index_pdf,
        rag_semantic_filter
    ]
