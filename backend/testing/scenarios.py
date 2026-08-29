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

Scenario = Dict[str, Any]

SCENARIOS: List[Scenario] = [

    {
        "id": 1,
        "name": "Clear Request — Optimizer Langsung",
        "category": "Tool Selection & Argument Extraction",
        "prompt": (
            "Tolong buatkan itinerary 2 hari ke wisata Alam di Kabupaten Bangli. "
            "Saya punya budget teratas 400.000 rupiah. Apa saja rekomendasinya?"
        ),
        "expected": {
            "expected_tools": ["budget_optimizer_tool"],
            "forbidden_tools": ["index_pdf", "rag_semantic_filter"],
            "arg_checks": {
                "budget_optimizer_tool": {
                    "budget_limit": lambda v: int(v) == 400000,
                    "duration_days": lambda v: int(v) == 2,
                    "location_keywords": lambda v: any(
                        k.lower() in ["bangli", "kabupaten bangli", "alam"]
                        for k in (v if isinstance(v, list) else [v])
                    ),
                }
            },
            "expected_in_response": ["itinerary", "bangli", "alam", "400"],
            "forbidden_in_response": ["paris", "lombok", "jakarta"],
        },
        "passing_criteria": "budget_optimizer_tool dipanggil dengan budget=400000 dan duration=2",
    },

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
            "expected_tools": ["budget_optimizer_tool"],
            "preferred_tools": ["rag_semantic_filter"],
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
            "expected_in_response": ["budget", "50"],
            "expected_pushback": True,
            "forbidden_in_response": ["berhasil membuat itinerary", "berikut rute anda"],
        },
        "passing_criteria": "Agent melakukan pushback logis: memberitahu budget tidak cukup",
    },

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
            "expected_tools": ["budget_optimizer_tool"],
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
