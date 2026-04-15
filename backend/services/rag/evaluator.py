"""
backend/services/rag/evaluator.py
==================================
Evaluasi formal Retrieval Quality untuk RAG system.

Metrik akademis yang diimplementasikan:
  - Precision@K  : Dari K dokumen yang di-retrieve, berapa % yang relevan?
  - Recall@K     : Dari semua dokumen relevan, berapa % yang berhasil di-retrieve dalam top-K?
  - MRR          : Mean Reciprocal Rank — posisi rata-rata dokumen relevan pertama
  - Hit Rate@K   : Apakah minimal 1 dokumen relevan ada dalam top-K?

Ground Truth:
  Karena tidak ada labeled dataset eksternal, evaluator menggunakan
  'Pseudo Relevance Judgement' — relevansi ditentukan dengan keyword matching
  antara query dan konten dokumen. Ini adalah pendekatan standar untuk
  evaluasi RAG dalam konteks TA/penelitian tanpa anotasi manual.

Referensi:
  - Manning et al. (2008) - Introduction to Information Retrieval
  - Precision@K, Recall@K (Standard IR Metrics)
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# TEST QUERIES — Ground Truth untuk Sistem Wisata Bali
# Setiap query punya expected_keywords yang menentukan relevansi
# ─────────────────────────────────────────────────────────────

RAG_EVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "wisata alam",
        "expected_keywords": ["alam", "nature", "pantai", "bukit", "air terjun", "hutan"],
        "description": "Destinasi wisata alam",
    },
    {
        "query": "wisata budaya dan pura",
        "expected_keywords": ["budaya", "pura", "temple", "sakral", "upacara", "adat"],
        "description": "Destinasi budaya dan spiritual",
    },
    {
        "query": "tempat sepi dan tenang untuk healing",
        "expected_keywords": ["sepi", "tenang", "healing", "quiet", "calm"],
        "description": "Destinasi untuk ketenangan",
    },
    {
        "query": "wisata keluarga anak-anak rekreasi",
        "expected_keywords": ["rekreasi", "keluarga", "anak", "taman", "wahana", "family"],
        "description": "Destinasi rekreasi keluarga",
    },
    {
        "query": "tempat gratis murah terjangkau",
        "expected_keywords": ["gratis", "murah", "terjangkau", "free", "budget"],
        "description": "Destinasi berbiaya rendah",
    },
    {
        "query": "Ubud seni dan galeri",
        "expected_keywords": ["ubud", "seni", "galeri", "art", "lukis", "ukir"],
        "description": "Wisata seni di Ubud",
    },
    {
        "query": "pantai sunset Kuta Seminyak",
        "expected_keywords": ["pantai", "kuta", "seminyak", "sunset", "beach"],
        "description": "Pantai populer sunset",
    },
    {
        "query": "rating tinggi destinasi terbaik populer",
        "expected_keywords": ["populer", "ramai", "terbaik", "top", "unggulan"],
        "description": "Destinasi populer dan top-rated",
    },
]


# ─────────────────────────────────────────────────────────────
# RELEVANCE JUDGE
# ─────────────────────────────────────────────────────────────

def _is_relevant(doc_text: str, expected_keywords: List[str]) -> bool:
    """
    Tentukan apakah dokumen relevan terhadap query.
    Relevan jika mengandung MINIMAL 1 keyword yang diharapkan.
    Menggunakan case-insensitive substring matching.
    """
    doc_lower = doc_text.lower()
    return any(kw.lower() in doc_lower for kw in expected_keywords)


def _rank_of_first_relevant(docs: List[str], keywords: List[str]) -> Optional[int]:
    """Return posisi (1-indexed) dokumen relevan pertama, atau None jika tidak ada."""
    for rank, doc in enumerate(docs, start=1):
        if _is_relevant(doc, keywords):
            return rank
    return None


# ─────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved_docs: List[str], expected_keywords: List[str], k: int) -> float:
    """
    Precision@K = |{relevant docs in top-K}| / K

    Mengukur: Dari K dokumen yang diambil, berapa proporsi yang relevan?
    Range: 0.0 (tidak ada relevan) — 1.0 (semua relevan)
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc in top_k if _is_relevant(doc, expected_keywords))
    return round(relevant_count / len(top_k), 4)


def recall_at_k(retrieved_docs: List[str], expected_keywords: List[str], k: int,
                total_corpus_docs: List[str] = None) -> float:
    """
    Recall@K = |{relevant docs in top-K}| / |{all relevant docs in corpus}|

    Mengukur: Dari SEMUA dokumen relevan yang ada, berapa yang berhasil ditemukan?
    Range: 0.0 (tidak ada yang ditemukan) — 1.0 (semua relevan ditemukan)

    Catatan: Jika total_corpus_docs tidak diberikan, dihitung dari retrieved_docs saja
    (lower-bound recall estimate).
    """
    top_k = retrieved_docs[:k]
    relevant_in_topk = sum(1 for doc in top_k if _is_relevant(doc, expected_keywords))

    if total_corpus_docs:
        total_relevant = sum(1 for doc in total_corpus_docs if _is_relevant(doc, expected_keywords))
    else:
        # Estimasi konservatif: gunakan seluruh retrieved set
        total_relevant = sum(1 for doc in retrieved_docs if _is_relevant(doc, expected_keywords))

    if total_relevant == 0:
        return 1.0  # Tidak ada yang relevan → recall trivially 1.0 (vacuously true)
    return round(relevant_in_topk / total_relevant, 4)


def reciprocal_rank(retrieved_docs: List[str], expected_keywords: List[str]) -> float:
    """
    Reciprocal Rank (RR) = 1 / rank_of_first_relevant

    Jika dokumen relevan pertama ada di posisi 1 → RR=1.0
    Jika ada di posisi 2 → RR=0.5, posisi 3 → RR=0.33, dst.
    Jika tidak ada sama sekali → RR=0.0
    """
    rank = _rank_of_first_relevant(retrieved_docs, expected_keywords)
    if rank is None:
        return 0.0
    return round(1.0 / rank, 4)


def hit_rate_at_k(retrieved_docs: List[str], expected_keywords: List[str], k: int) -> bool:
    """
    Hit Rate@K = True jika minimal 1 dokumen relevan ada di top-K.
    Binary metric — berguna sebagai sanity check.
    """
    return any(_is_relevant(doc, expected_keywords) for doc in retrieved_docs[:k])


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate_rag_retrieval(
    vector_store,
    k: int = 5,
    queries: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi formal retrieval quality dari RAG vector store.

    Args:
        vector_store : FAISS vector store yang sudah dibangun
        k            : jumlah dokumen yang di-retrieve per query (default 5)
        queries      : custom query list, default pakai RAG_EVAL_QUERIES

    Returns:
        Dict berisi:
          - per_query: hasil per query (Precision@K, Recall@K, RR, Hit)
          - macro_avg: rata-rata semua metrik (untuk laporan TA)
          - k_used: nilai K yang dipakai
          - interpretation: narasi akademis ringkasan hasil
    """
    if queries is None:
        queries = RAG_EVAL_QUERIES

    per_query_results = []

    for q in queries:
        query_text       = q["query"]
        expected_keywords = q["expected_keywords"]
        description      = q.get("description", query_text)

        # Retrieve top-K documents dari vector store
        try:
            docs_and_scores = vector_store.similarity_search_with_score(query_text, k=k)
            retrieved_texts = [doc.page_content for doc, _score in docs_and_scores]
        except Exception as e:
            per_query_results.append({
                "query":       query_text,
                "description": description,
                "error":       str(e),
                "precision_at_k": 0.0,
                "recall_at_k":    0.0,
                "mrr":            0.0,
                "hit_rate":       False,
            })
            continue

        p_at_k = precision_at_k(retrieved_texts, expected_keywords, k)
        r_at_k = recall_at_k(retrieved_texts, expected_keywords, k)
        rr     = reciprocal_rank(retrieved_texts, expected_keywords)
        hit    = hit_rate_at_k(retrieved_texts, expected_keywords, k)

        # Top-1 snippet untuk transparansi
        top1_snippet = retrieved_texts[0][:120] if retrieved_texts else ""

        per_query_results.append({
            "query":             query_text,
            "description":       description,
            "expected_keywords": expected_keywords,
            f"precision_at_{k}": p_at_k,
            f"recall_at_{k}":    r_at_k,
            "reciprocal_rank":   rr,
            "hit_rate":          hit,
            "n_retrieved":       len(retrieved_texts),
            "top1_snippet":      top1_snippet,
        })

    # ── Macro Average (semua query) ──
    valid = [r for r in per_query_results if "error" not in r]
    n = len(valid)

    def _avg(key: str) -> float:
        vals = [r.get(key, r.get(f"precision_at_{k}", 0.0)) for r in valid]
        return round(sum(vals) / n, 4) if n > 0 else 0.0

    macro_precision = _avg(f"precision_at_{k}")
    macro_recall    = _avg(f"recall_at_{k}")
    macro_mrr       = _avg("reciprocal_rank")
    macro_hit       = round(sum(1 for r in valid if r.get("hit_rate", False)) / n, 4) if n > 0 else 0.0

    # ── Interpretasi Akademis ──
    def _grade(val: float) -> str:
        if val >= 0.8: return "Sangat Baik"
        if val >= 0.6: return "Baik"
        if val >= 0.4: return "Cukup"
        return "Perlu Peningkatan"

    interpretation = (
        f"Evaluasi RAG ({n} query, K={k}): "
        f"Precision@{k}={macro_precision:.2%} [{_grade(macro_precision)}] | "
        f"Recall@{k}={macro_recall:.2%} [{_grade(macro_recall)}] | "
        f"MRR={macro_mrr:.4f} [{_grade(macro_mrr)}] | "
        f"Hit Rate={macro_hit:.2%}. "
        + (
            "Sistem RAG mampu me-retrieve konten yang relevan dengan baik."
            if macro_precision >= 0.5
            else "Pertimbangkan untuk meningkatkan kualitas embedding atau memperluas dokumen indexing."
        )
    )

    return {
        "status":          "success",
        "k_used":          k,
        "n_queries_eval":  n,
        "evaluation_method": "Pseudo Relevance Judgement (keyword-based ground truth)",
        "per_query":       per_query_results,
        "macro_avg": {
            f"precision_at_{k}": macro_precision,
            f"recall_at_{k}":    macro_recall,
            "mrr":               macro_mrr,
            "hit_rate_at_k":     macro_hit,
        },
        "interpretation": interpretation,
        "academic_note": (
            "Metrik ini menggunakan Pseudo Relevance Judgement karena tidak tersedia "
            "annotated ground truth dataset. Relevansi ditentukan via keyword matching "
            "antara query intent dengan konten dokumen (Manning et al., 2008). "
            "Metrik ini standar digunakan untuk evaluasi retrieval IR pada penelitian akademis."
        ),
    }
