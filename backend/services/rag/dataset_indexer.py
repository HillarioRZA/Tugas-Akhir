"""
backend/services/rag/dataset_indexer.py

Mengonversi dataset wisata Bali (CSV) menjadi vector store FAISS
yang berfungsi sebagai knowledge base bawaan sistem (tidak memerlukan PDF upload).

Setiap baris dataset diformat menjadi dokumen teks:
    "Nama: <Place_Name> | Kota: <City> | Kecamatan: <kecamatan> |
     Kategori: <Category> | Tags: <tags> | Deskripsi: <Description>"

Dokumen ini kemudian di-embed dan disimpan sebagai FAISS vector store
yang dapat di-query oleh rag_semantic_filter.
"""

import os
import time
import pandas as pd
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Status tracking untuk endpoint /rag/status dan /rag/rebuild
build_status: dict = {
    "state":      "idle",     # idle | building | ready | failed
    "progress":   0,          # jumlah dokumen yang sudah di-embed
    "total":      0,
    "error":      None,
    "built_at":   None,
}

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────

# Kolom yang digunakan sebagai knowledge base (urutan prioritas)
KNOWLEDGE_COLS = [
    "Place_Name",
    "City",
    "kecamatan",
    "Category",
    "tags",
    "Description",
    "Crowd_Density",
    "Price_Category",
]

# Embedding model (sama dengan yang dipakai vectorizer.py)
_embeddings = None

def _get_embeddings():
    """Lazy-load embeddings untuk menghindari init cost saat import."""
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embeddings


# ─────────────────────────────────────────────────────────────
# STEP 1: Format setiap baris dataset ke dokumen teks
# ─────────────────────────────────────────────────────────────

def _format_row_to_document(row: pd.Series) -> str:
    """
    Konversi satu baris DataFrame ke string dokumen.

    Format:
        Nama: Pura Kehen | Kota: Kabupaten Bangli | Kecamatan: Bangli |
        Kategori: Budaya | Tags: budaya, sepi, sakral |
        Deskripsi: Pura Kehen adalah...
    """
    parts = []
    if pd.notna(row.get("Place_Name")):
        parts.append(f"Nama: {row['Place_Name']}")
    if pd.notna(row.get("City")):
        parts.append(f"Kota: {row['City']}")
    if pd.notna(row.get("kecamatan")):
        parts.append(f"Kecamatan: {row['kecamatan']}")
    if pd.notna(row.get("Category")):
        parts.append(f"Kategori: {row['Category']}")
    if pd.notna(row.get("tags")):
        parts.append(f"Tags: {row['tags']}")
    if pd.notna(row.get("Crowd_Density")):
        parts.append(f"Keramaian: {row['Crowd_Density']}")
    if pd.notna(row.get("Price_Category")):
        parts.append(f"Harga: {row['Price_Category']}")
    if pd.notna(row.get("Description")) and str(row.get("Description")).strip():
        parts.append(f"Deskripsi: {row['Description']}")

    return " | ".join(parts)


def build_dataset_text_corpus(csv_path: str) -> Optional[str]:
    """
    Baca dataset CSV dan konversi ke satu string corpus teks.
    Setiap baris menjadi satu paragraf terpisah ('\n\n').

    Returns:
        String corpus gabungan, atau None jika gagal.
    """
    try:
        df = pd.read_csv(csv_path)
        # Ambil hanya kolom yang tersedia
        existing_cols = [c for c in KNOWLEDGE_COLS if c in df.columns]
        df_subset = df[existing_cols].copy()

        documents = []
        for _, row in df_subset.iterrows():
            doc_text = _format_row_to_document(row)
            if doc_text.strip():
                documents.append(doc_text)

        if not documents:
            print("⚠️  [Dataset Indexer] Tidak ada dokumen yang bisa dibuat dari CSV.")
            return None

        corpus = "\n\n".join(documents)
        print(f"✅ [Dataset Indexer] {len(documents)} destinasi diformat ke corpus teks.")
        return corpus

    except Exception as e:
        print(f"❌ [Dataset Indexer] Gagal membaca CSV: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# STEP 2: Bangun FAISS vector store dari corpus
# ─────────────────────────────────────────────────────────────

def build_system_vector_store(csv_path: str, max_retries: int = 3):
    """
    Bangun system-level FAISS vector store dari dataset CSV.
    Ini menjadi knowledge base bawaan yang SELALU tersedia.

    Penambahan (Task 6 RAG fix):
    - Retry mechanism dengan exponential backoff (3x per batch)
    - build_status tracking agar endpoint /rag/status dan /rag/rebuild bisa lapor progress

    Returns:
        FAISS vector store, atau None jika gagal.
    """
    global build_status
    build_status["state"]   = "building"
    build_status["error"]   = None
    build_status["built_at"] = None

    try:
        df = pd.read_csv(csv_path)
        existing_cols = [c for c in KNOWLEDGE_COLS if c in df.columns]
        df_subset = df[existing_cols].copy()

        # Buat satu dokumen per baris (bukan chunk besar)
        documents = []
        for _, row in df_subset.iterrows():
            doc_text = _format_row_to_document(row)
            if doc_text.strip():
                documents.append(doc_text)

        if not documents:
            print("⚠️  [Dataset Indexer] Tidak ada dokumen untuk diindeks.")
            build_status["state"] = "failed"
            build_status["error"] = "Tidak ada dokumen yang berhasil dibuat dari CSV."
            return None

        build_status["total"]    = len(documents)
        build_status["progress"] = 0

        print(f"🔄 [Dataset Indexer] Membuat vector store dari {len(documents)} dokumen destinasi...")
        print("   (Proses ini mungkin 30-60 detik pada pertama kali karena embedding API call)")

        embeddings   = _get_embeddings()
        batch_size   = 50
        vector_store = None

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # ── Retry dengan exponential backoff ──
            attempt     = 0
            batch_store = None
            while attempt < max_retries:
                try:
                    if vector_store is None:
                        vector_store = FAISS.from_texts(batch, embedding=embeddings)
                        batch_store  = vector_store
                    else:
                        batch_store = FAISS.from_texts(batch, embedding=embeddings)
                        vector_store.merge_from(batch_store)
                    break  # sukses, keluar from retry loop

                except Exception as batch_err:
                    attempt += 1
                    wait_sec = 2 ** attempt  # 2s, 4s, 8s
                    print(f"   [Retry {attempt}/{max_retries}] Batch {i//batch_size+1} gagal: "
                          f"{batch_err}. Menunggu {wait_sec}s...")
                    if attempt < max_retries:
                        time.sleep(wait_sec)
                    else:
                        print(f"   ❌ Batch {i//batch_size+1} gagal setelah {max_retries} percobaan. Dilanjutkan ke batch berikutnya.")

            build_status["progress"] = min(i + batch_size, len(documents))
            print(f"   Indexed {build_status['progress']}/{len(documents)} dokumen...")

        if vector_store is None:
            build_status["state"] = "failed"
            build_status["error"] = "Semua batch gagal di-embed."
            return None

        build_status["state"]    = "ready"
        build_status["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"✅ [Dataset Indexer] System vector store berhasil dibuat.")
        return vector_store

    except Exception as e:
        build_status["state"] = "failed"
        build_status["error"] = str(e)
        print(f"❌ [Dataset Indexer] Gagal membuat vector store: {e}")
        return None
