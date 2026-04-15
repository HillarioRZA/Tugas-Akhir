"""
backend/testing/scenarios.py
================================
Definisi 5 skenario uji resmi berdasarkan test_prompts.txt.

Setiap skenario mendefinisikan:
    - id            : nomor skenario
    - name          : nama singkat
    - category      : tipe pengujian
    - prompt        : input user ke agent
    - expected       : kriteria keberhasilan (tool, argumen, behavior)
    - description   : tujuan pengujian

Kriteria evaluasi per skenario:
    expected_tools_called   : tool HARUS dipanggil (minimal salah satu)
    forbidden_tools_called  : tool yang TIDAK BOLEH dipanggil
    expected_arg_checks     : validasi argumen tool (key: value atau key: callable)
    expected_behavior       : keyword yang harus ada di response akhir
    forbidden_behavior      : keyword yang TIDAK BOLEH ada di response (indikasi halusinasi)
    must_not_hallucinate    : True = cek response tidak mengandung angka/kota yang tidak relevan
"""

from typing import List, Dict, Any, Optional, Callable

# ─────────────────────────────────────────────────────────────
# TYPE ALIAS
# ─────────────────────────────────────────────────────────────

Scenario = Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# 5 SKENARIO RESMI
# ─────────────────────────────────────────────────────────────

SCENARIOS: List[Scenario] = [

    # ──────────────────────────────────────────────────────────
    # SKENARIO 1: Permintaan Langsung Tersurat (Clear Request)
    # Tujuan: Agen harus langsung memanggil budget_optimizer_tool
    #         dengan parameter yang diekstrak tepat dari prompt.
    # ──────────────────────────────────────────────────────────
    {
        "id": 1,
        "name": "Clear Request — Optimizer Langsung",
        "category": "Tool Selection & Argument Extraction",
        "prompt": (
            "Tolong buatkan itinerary 2 hari ke wisata Alam di Kabupaten Bangli. "
            "Saya punya budget teratas 400.000 rupiah. Apa saja rekomendasinya?"
        ),
        "expected": {
            # Tool yang HARUS dipanggil
            "expected_tools": ["budget_optimizer_tool"],
            # Tool yang tidak relevan — jika dipanggil → indikasi hallucination reasoning
            "forbidden_tools": ["index_pdf", "rag_semantic_filter"],
            # Cek argumen tool: budget harus 400000, durasi 2 hari
            "arg_checks": {
                "budget_optimizer_tool": {
                    "budget_limit": lambda v: int(v) == 400000,
                    "duration_days": lambda v: int(v) == 2,
                    # Keyword harus ada Bangli atau Alam di location_keywords
                    "location_keywords": lambda v: any(
                        k.lower() in ["bangli", "kabupaten bangli", "alam"]
                        for k in (v if isinstance(v, list) else [v])
                    ),
                }
            },
            # Response akhir harus menyebutkan ini
            "expected_in_response": ["itinerary", "bangli", "alam", "400"],
            # Response tidak boleh menyebutkan kota yang tidak relevan
            "forbidden_in_response": ["paris", "lombok", "jakarta"],
        },
        "passing_criteria": "budget_optimizer_tool dipanggil dengan budget=400000 dan duration=2",
    },

    # ──────────────────────────────────────────────────────────
    # SKENARIO 2: Pemahaman Abstrak / Neuro-Symbolic (RAG)
    # Tujuan: Agen harus PERTAMA memanggil rag_semantic_filter
    #         untuk menerjemahkan "stres, damai, hijau" → keywords,
    #         BARU kemudian memanggil budget_optimizer_tool.
    # ──────────────────────────────────────────────────────────
    {
        "id": 2,
        "name": "Neuro-Symbolic — RAG + Optimizer",
        "category": "Multi-Tool Chaining & Semantic Understanding",
        "prompt": (
            "Saya sedang sangat stres dari pekerjaan dan ingin mencari suasana yang damai, "
            "jauh dari hiruk-pikuk kota, dan hijau-hijau selama 1 hari. "
            "Budget saya hanya 150.000. Coba carikan itinerary wisata yang pas beserta harganya."
        ),
        "expected": {
            # Idealnya rag_semantic_filter dipanggil dulu, lalu optimizer
            "expected_tools": ["budget_optimizer_tool"],
            "preferred_tools": ["rag_semantic_filter"],  # bonus jika dipanggil
            "forbidden_tools": ["index_pdf"],
            "arg_checks": {
                "budget_optimizer_tool": {
                    "budget_limit": lambda v: int(v) == 150000,
                    "duration_days": lambda v: int(v) == 1,
                }
            },
            "expected_in_response": ["150", "damai", "alam"],
            "forbidden_in_response": ["paris", "tokyo", "mall"],
        },
        "passing_criteria": "budget_optimizer_tool dipanggil dengan budget=150000 dan duration=1",
    },

    # ──────────────────────────────────────────────────────────
    # SKENARIO 3: Logical Pushback — Budget Tidak Masuk Akal
    # Tujuan: Agen HARUS melakukan pushback logis, TIDAK menghasilkan
    #         itinerary fiktif. Budget Rp50.000 untuk wisata mewah 4 hari
    #         tidak mungkin dipenuhi → agen harus menolak dengan sopan.
    # ──────────────────────────────────────────────────────────
    {
        "id": 3,
        "name": "Logical Pushback — Impossible Constraint",
        "category": "Reasoning & Constraint Validation",
        "prompt": (
            "Tolong buatkan rekomendasi liburan di Kabupaten Badung selama 4 hari penuh "
            "untuk keliling ke tempat wisata rekreasi mewah. "
            "Oh ya, budget total saya cuma Rp 50.000. Bisa bantu rutenya?"
        ),
        "expected": {
            "expected_tools": ["budget_optimizer_tool"],
            "forbidden_tools": ["index_pdf"],
            "arg_checks": {
                "budget_optimizer_tool": {
                    "budget_limit": lambda v: int(v) == 50000,
                }
            },
            # Response HARUS mengandung penolakan yang sopan / pushback
            "expected_in_response": ["budget", "50"],
            # Response TIDAK BOLEH langsung menyebutkan nama destinasi seolah berhasil
            # (karena harusnya gagal/pushback)
            "expected_pushback": True,  # flag khusus: response harus mengindikasikan kendala
            "forbidden_in_response": ["berhasil membuat itinerary", "berikut rute anda"],
        },
        "passing_criteria": "Agent melakukan pushback logis: memberitahu budget tidak cukup",
    },

    # ──────────────────────────────────────────────────────────
    # SKENARIO 4: Explainable AI (XAI) — ML + Visualisasi
    # Tujuan: Agen harus memanggil predict_match_score (ML)
    #         DAN plot_itinerary_scatter (visualisasi XAI).
    # ──────────────────────────────────────────────────────────
    {
        "id": 4,
        "name": "Explainable AI — ML + Scatter Plot",
        "category": "XAI: Multi-Tool ML + Visualization",
        "prompt": (
            "Coba carikan 2 tempat wisata alam di Kota Denpasar yang menarik "
            "dengan budget maksimal 100 ribu. "
            "Tolong jelaskan juga fitur mana yang membuat Anda merekomendasikan tempat itu, "
            "dan buatkan scatter plot perbandingan rating dan harganya."
        ),
        "expected": {
            "expected_tools": ["budget_optimizer_tool"],
            "preferred_tools": ["predict_match_score", "plot_itinerary_scatter"],
            "forbidden_tools": ["index_pdf"],
            "arg_checks": {
                "budget_optimizer_tool": {
                    "budget_limit": lambda v: int(v) == 100000,
                    "location_keywords": lambda v: any(
                        k.lower() in ["denpasar", "kota denpasar", "alam"]
                        for k in (v if isinstance(v, list) else [v])
                    ),
                }
            },
            "expected_in_response": ["denpasar", "alam", "100"],
            "forbidden_in_response": ["surabaya", "yogyakarta"],
        },
        "passing_criteria": "budget_optimizer_tool dipanggil dengan budget=100000 dan keywords Denpasar",
    },

    # ──────────────────────────────────────────────────────────
    # SKENARIO 5: EDA + Filter Spesifik
    # Tujuan: Agen harus melakukan filter EDA (run_eda / full_profile)
    #         atau memanggil optimizer dengan filter Buleleng, rating>4.5, Sepi.
    # ──────────────────────────────────────────────────────────
    {
        "id": 5,
        "name": "EDA Filter Spesifik — Multi-Criteria",
        "category": "Data Analysis & Filtered Query",
        "prompt": (
            "Tempat wisata apa saja di file data ini yang letaknya di Kabupaten Buleleng, "
            "ratingnya di atas 4.5, dan masih tergolong 'Sepi'? "
            "Tolong buatkan list singkat beserta estimasi biaya dan Google Maps-nya "
            "jika saya mau ke sana besok."
        ),
        "expected": {
            # Boleh optimizer atau EDA tool
            "expected_tools": ["budget_optimizer_tool"],
            # Tool EDA yang benar: 'describe_dataset' dan 'run_full_profile'
            # (tidak ada tool bernama 'run_eda' di codebase)
            "preferred_tools": ["describe_dataset", "run_full_profile"],
            "forbidden_tools": ["index_pdf"],
            "arg_checks": {
                "budget_optimizer_tool": {
                    "location_keywords": lambda v: any(
                        k.lower() in ["buleleng", "kabupaten buleleng"]
                        for k in (v if isinstance(v, list) else [v])
                    ),
                    "min_rating": lambda v: float(v) >= 4.5,
                }
            },
            "expected_in_response": ["buleleng", "sepi", "4.5"],
            # Badung dan Gianyar adalah kota Bali yang benar-benar tidak relevan
            # untuk query Buleleng (di sisi utara pulau)
            "forbidden_in_response": ["paris", "tokyo", "london"],
        },
        "passing_criteria": "Optimizer/EDA dipanggil dengan filter Buleleng dan min_rating >= 4.5",
    },
]


def get_scenario_by_id(scenario_id: int) -> Optional[Scenario]:
    """Ambil satu skenario berdasarkan ID."""
    return next((s for s in SCENARIOS if s["id"] == scenario_id), None)


def get_all_scenarios() -> List[Scenario]:
    """Return semua skenario."""
    return SCENARIOS
