"""
backend/testing/runner.py
============================
Test Runner utama framework evaluasi AI Agent WISTA.

CARA MENJALANKAN:
    # Dari root project
    python backend/testing/runner.py

    # Mode satu skenario saja (misal skenario 3)
    python backend/testing/runner.py --scenario 3

    # Mode dengan server URL kustom
    python backend/testing/runner.py --url http://localhost:8000

    # Mode dry-run (tidak kirim ke API, test evaluator saja)
    python backend/testing/runner.py --dry-run

PRASYARAT:
    - Server FastAPI berjalan di http://localhost:8000
    - Dataset default (v3) sudah ter-seed saat startup
    - Model ML sudah ada di saved_models/

OUTPUT:
    - Hasil per skenario dicetak ke terminal
    - 3 file laporan di backend/testing/reports/:
        eval_detail_YYYYMMDD_HHMMSS.json
        eval_summary_YYYYMMDD_HHMMSS.csv
        eval_narasi_YYYYMMDD_HHMMSS.txt
"""

import sys
import time
import argparse
import json
import uuid
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.testing.scenarios       import get_all_scenarios, get_scenario_by_id
from backend.testing.evaluator       import evaluate_scenario_result, compute_success_rate
from backend.testing.report_generator import (
    print_scenario_result,
    print_summary_table,
    generate_all_reports,
)

DEFAULT_API_URL = "http://localhost:8000/api/agent/execute"
REQUEST_TIMEOUT = 120
DELAY_BETWEEN   = 3

def call_agent_api(
    prompt: str,
    api_url: str,
    session_id: str,
    timeout: int = REQUEST_TIMEOUT,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kirim satu prompt ke API Agent dan kembalikan response dict.
    Menyertakan reasoning_log jika ada di response.
    Mendukung upload file opsional (dataset_override / dataset_override_pdf).
    """
    files = None
    opened_file = None
    try:
        data = {"prompt": prompt}
        headers = {"X-Session-ID": session_id}

        if file_path:
            full_path = Path(file_path)
            if not full_path.is_absolute():
                full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                opened_file = open(full_path, "rb")
                files = {"file": (full_path.name, opened_file)}
            else:
                print(f"  [WARN] File override {full_path} tidak ditemukan!")

        response = requests.post(
            api_url,
            data=data,
            files=files,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "error": "CONNECTION_ERROR",
            "summary": "",
            "reasoning_log": [],
            "detail": f"Tidak bisa terhubung ke {api_url}. Pastikan server berjalan.",
        }
    except requests.exceptions.Timeout:
        return {
            "error": "TIMEOUT",
            "summary": "",
            "reasoning_log": [],
            "detail": f"Request timeout setelah {timeout}s.",
        }
    except requests.exceptions.HTTPError as e:
        body = {}
        try:
            body = e.response.json()
        except Exception:
            pass
        return {
            "error": f"HTTP_{e.response.status_code}",
            "summary": "",
            "reasoning_log": [],
            "detail": body.get("detail", str(e)),
        }
    except Exception as e:
        return {
            "error": "UNKNOWN_ERROR",
            "summary": "",
            "reasoning_log": [],
            "detail": str(e),
        }
    finally:
        if opened_file:
            opened_file.close()


def _build_mock_response(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock response untuk mode --dry-run.
    Menggunakan format AKTUAL dari agent/main.py (intermediate_steps dict),
    bukan format string 'Invoking: `tool` with `{...}`'.
    """
    mock_data = {
        1: {
            "summary": "Berhasil membuat itinerary 2 hari di Bangli dengan budget 400000 rupiah. Wisata alam tersedia.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User minta itinerary 2 hari Alam Bangli budget 400000",
                    "tool_called": "budget_optimizer_tool",
                    "tool_input": {"budget_limit": 400000, "duration_days": 2,
                                   "location_keywords": ["Bangli", "Alam"], "min_rating": 0.0},
                    "observation": "status: success, 6 destinasi ditemukan, total 180000"
                },
                {
                    "step": 2,
                    "thought": "Membuat visualisasi scatter plot untuk XAI",
                    "tool_called": "plot_itinerary_scatter",
                    "tool_input": {"selected_destinations": ["Pura Kehen", "Trunyan"]},
                    "observation": "status: success, plot dibuat"
                }
            ],
        },
        2: {
            "summary": "Saya menemukan wisata alam yang damai dan tenang dengan budget 150000 selama 1 hari.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User minta wisata damai, tidak disebutkan PDF — skip RAG, langsung optimizer",
                    "tool_called": "budget_optimizer_tool",
                    "tool_input": {"budget_limit": 150000, "duration_days": 1,
                                   "location_keywords": ["Alam", "Sepi"], "min_rating": 0.0},
                    "observation": "status: success, 3 destinasi sepi dan damai ditemukan"
                }
            ],
        },
        3: {
            "summary": "Maaf, budget 50000 tidak cukup untuk wisata rekreasi mewah selama 4 hari. Budget minimum yang dibutuhkan jauh lebih besar. Saya sarankan minimal budget 500.000 rupiah.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User minta 4 hari mewah dengan Rp50.000 — tidak mungkin, cek optimizer dulu",
                    "tool_called": "budget_optimizer_tool",
                    "tool_input": {"budget_limit": 50000, "duration_days": 4,
                                   "location_keywords": ["Badung", "Rekreasi"], "min_rating": 0.0},
                    "observation": "status: error, budget terlalu kecil untuk 4 hari wisata mewah"
                }
            ],
        },
        4: {
            "summary": "Berikut rekomendasi wisata alam di Denpasar dengan budget 100 ribu. Skor ML menunjukkan fitur utama adalah Rating dan Popularitas.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User minta XAI — panggil optimizer dulu untuk dapat kandidat",
                    "tool_called": "budget_optimizer_tool",
                    "tool_input": {"budget_limit": 100000, "duration_days": 1,
                                   "location_keywords": ["Denpasar", "Alam"], "min_rating": 0.0},
                    "observation": "status: success, 2 destinasi ditemukan"
                },
                {
                    "step": 2,
                    "thought": "Hitung ML match score untuk XAI",
                    "tool_called": "predict_match_score",
                    "tool_input": {"user_preferences": {"category": "Alam"},
                                   "candidate_destinations": ["Taman Budaya", "Taman Werdhi Budaya"]},
                    "observation": "status: success, match scores: 87%, 72%"
                },
                {
                    "step": 3,
                    "thought": "Buat scatter plot untuk visualisasi XAI",
                    "tool_called": "plot_itinerary_scatter",
                    "tool_input": {"selected_destinations": ["Taman Budaya"]},
                    "observation": "status: success, plot dibuat"
                }
            ],
        },
        5: {
            "summary": "Tempat wisata di Buleleng dengan rating di atas 4.5 dan kategori Sepi: berikut listnya beserta Google Maps.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User minta filter Buleleng + Sepi + rating >= 4.5",
                    "tool_called": "budget_optimizer_tool",
                    "tool_input": {"budget_limit": 2000000, "duration_days": 1,
                                   "location_keywords": ["Buleleng"], "min_rating": 4.5},
                    "observation": "status: success, 5 destinasi sepi di Buleleng ditemukan"
                }
            ],
        },
        6: {
            "summary": "Maaf, sebagai AI asisten pariwisata WISTA, domain keahlian saya adalah seputar wisata dan perjalanan di Bali. Saya tidak dapat membuatkan kode pemrograman Python seperti FizzBuzz.",
            "reasoning_log": [],
        },
        7: {
            "summary": "Maaf, saya adalah asisten khusus wisata Bali. Saya tidak melayani rekomendasi untuk wilayah Jakarta. Apakah ada destinasi di Bali yang ingin Anda tanyakan?",
            "reasoning_log": [],
        },
        8: {
            "summary": "Berikut analisis lengkap dataset wisata Bali. Dataset memiliki 1454 rows dan 15 columns, statistik deskriptif dan sebaran nilai telah dihitung secara menyeluruh.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User meminta analisis profil lengkap dataset",
                    "tool_called": "run_full_profile",
                    "tool_input": {},
                    "observation": "Shape: 1454 rows, 15 columns. Missing values: 0. Summary statistics generated."
                }
            ],
        },
        9: {
            "summary": "Hasil prediksi ML match score menunjukkan Pantai Kuta memiliki match score 68% dan Tirta Empul memiliki match score 91% untuk pencinta alam yang menyukai suasana damai. Fitur paling berpengaruh adalah Crowd_Density dan Rating.",
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "Memprediksi match score menggunakan model ML",
                    "tool_called": "predict_match_score",
                    "tool_input": {
                        "user_preferences": {"category": "Alam", "crowd": "Sepi"},
                        "candidate_destinations": ["Pantai Kuta", "Tirta Empul"]
                    },
                    "observation": "Match score: Pantai Kuta 0.68, Tirta Empul 0.91. Feature importance: Crowd_Density, Rating."
                }
            ],
        },
        10: {
            "summary": (
                "Berikut analisis lengkap untuk dataset HR Analytics yang Anda upload. "
                "Dataset memiliki 1470 rows dan 35 columns tanpa missing value. "
                "Ditemukan 12 outlier pada kolom MonthlyIncome menggunakan metode IQR. "
                "Matriks korelasi menunjukkan korelasi kuat antara TotalWorkingYears dan JobLevel (r=0.78). "
                "Analisis target Attrition menunjukkan MODERATE_IMBALANCE dengan rasio 5.2x."
            ),
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "User mengunggah dataset HR Analytics dan meminta profil EDA lengkap",
                    "tool_called": "run_full_profile",
                    "tool_input": {},
                    "observation": "status: success, rows: 1470, columns: 35, missing_values: 0"
                },
                {
                    "step": 2,
                    "thought": "Mendeteksi outlier pada variabel numerik kunci",
                    "tool_called": "detect_outliers",
                    "tool_input": {"columns": ["MonthlyIncome", "TotalWorkingYears"]},
                    "observation": "status: success, outliers found: MonthlyIncome (12 rows)"
                },
                {
                    "step": 3,
                    "thought": "Menganalisis korelasi antar fitur",
                    "tool_called": "get_correlation_matrix",
                    "tool_input": {},
                    "observation": "status: success, correlation matrix computed"
                },
                {
                    "step": 4,
                    "thought": "Mengecek ketidakseimbangan kelas pada kolom target Attrition",
                    "tool_called": "detect_class_imbalance",
                    "tool_input": {"target_column": "Attrition"},
                    "observation": "status: success, ratio: 5.2x, severity: MODERATE_IMBALANCE"
                }
            ],
        },
        11: {
            "summary": (
                "Berdasarkan dokumen PDF Laporan BPS Statistik Pariwisata Bali yang diunggah, "
                "jumlah kedatangan wisatawan mancanegara (wisman) langsung ke Bali tercatat "
                "mencapai 5.273.258 kunjungan pada periode tahun terakhir."
            ),
            "reasoning_log": [
                {
                    "step": 1,
                    "thought": "Dokumen PDF terdeteksi, melakukan indexing dokumen via vector store",
                    "tool_called": "index_pdf",
                    "tool_input": {},
                    "observation": "status: success, PDF indexed successfully into session FAISS vectorstore"
                },
                {
                    "step": 2,
                    "thought": "Mencari data spesifik jumlah wisatawan asing dari dokumen BPS",
                    "tool_called": "rag_semantic_filter",
                    "tool_input": {"query": "jumlah wisatawan asing ke Bali tahun terakhir"},
                    "observation": "status: success, retrieved chunks: 'Jumlah wisman langsung ke Bali tahun 2023 sebanyak 5.273.258 kunjungan'"
                }
            ],
        },
    }
    sid = scenario["id"]
    return mock_data.get(sid, {"summary": "", "reasoning_log": []})

def run_all_tests(
    api_url: str = DEFAULT_API_URL,
    scenario_id: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """
    Jalankan semua skenario (atau satu skenario spesifik).
    """
    if scenario_id is not None:
        scenarios = [get_scenario_by_id(scenario_id)]
        if not scenarios[0]:
            print(f"[ERROR] Skenario ID {scenario_id} tidak ditemukan.")
            sys.exit(1)
    else:
        scenarios = get_all_scenarios()

    mode = "DRY-RUN (mock API)" if dry_run else f"LIVE ({api_url})"

    print("\n" + "=" * 65)
    print("  WISTA AI AGENT - AUTOMATED EVALUATION FRAMEWORK")
    print("=" * 65)
    print(f"  Mode         : {mode}")
    print(f"  Total Uji    : {len(scenarios)} skenario")
    print(f"  Passing Score: >= 60% per skenario")
    print(f"  Target TA    : Success Rate >= 90%")

    if not dry_run:
        try:
            r = requests.get(api_url.replace("/execute", "/docs"), timeout=5)
            print(f"  Server Status: OK (HTTP {r.status_code})")
        except Exception:
            print(f"  Server Status: TIDAK BISA TERHUBUNG ke {api_url}")
            print(f"  Jalankan server dulu: uvicorn backend.api.main:app --reload")
            print(f"  Atau gunakan --dry-run untuk test tanpa server.\n")
            sys.exit(1)

    print("=" * 65)
    all_results = []
    session_id  = str(uuid.uuid4())
    print(f"  Session ID   : {session_id}\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Menjalankan: Skenario {scenario['id']} — {scenario['name']}")
        print(f"  Prompt: \"{scenario['prompt'][:70]}...\"")

        file_override = scenario.get("dataset_override") or scenario.get("dataset_override_pdf")
        if file_override:
            print(f"  Dataset Override: {file_override}")

        if dry_run:
            agent_response = _build_mock_response(scenario)
            print(f"  [DRY-RUN] Menggunakan mock response.")
        else:
            print(f"  Mengirim ke agent... (timeout={REQUEST_TIMEOUT}s)")
            start = time.time()
            agent_response = call_agent_api(
                prompt=scenario["prompt"],
                api_url=api_url,
                session_id=session_id,
                file_path=file_override,
            )
            elapsed = time.time() - start
            print(f"  Response diterima dalam {elapsed:.1f}s")

        if "error" in agent_response and not agent_response.get("summary"):
            print(f"  [ERROR] API gagal: {agent_response.get('detail', agent_response['error'])}")
            agent_response["summary"] = ""
            agent_response["reasoning_log"] = []

        result = evaluate_scenario_result(scenario, agent_response)
        all_results.append(result)

        print_scenario_result(result)

        if not dry_run and i < len(scenarios):
            print(f"\n  Jeda {DELAY_BETWEEN}s sebelum skenario berikutnya...")
            time.sleep(DELAY_BETWEEN)

    summary = compute_success_rate(all_results)

    print_summary_table(all_results, summary)

    print("  Membuat file laporan...")
    report_paths = generate_all_reports(all_results, summary)

    print(f"\n  Laporan tersedia di:")
    for fmt, path in report_paths.items():
        print(f"    [{fmt.upper():4}] {path}")

    if summary["meets_proposal_target"]:
        print(f"\n  HASIL: Target proposal TERCAPAI! ({summary['success_rate']} >= 90%)")
        sys.exit(0)
    else:
        print(f"\n  HASIL: Target proposal BELUM TERCAPAI ({summary['success_rate']} < 90%)")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="WISTA AI Agent — Automated Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python backend/testing/runner.py
  python backend/testing/runner.py --scenario 3
  python backend/testing/runner.py --dry-run
  python backend/testing/runner.py --url http://localhost:8080/api/agent/execute
        """
    )
    parser.add_argument(
        "--url", type=str, default=DEFAULT_API_URL,
        help=f"URL endpoint agent (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--scenario", type=int, default=None,
        help="Jalankan hanya skenario tertentu (1-5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Jalankan evaluasi dengan mock response (tanpa server)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_tests(
        api_url=args.url,
        scenario_id=args.scenario,
        dry_run=args.dry_run,
    )
