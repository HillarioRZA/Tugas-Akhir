"""
backend/testing/report_generator.py
======================================
Modul pembuat laporan dari hasil evaluasi skenario.

Output yang dihasilkan:
  1. Tabel ASCII di terminal (real-time)
  2. File JSON detail hasil per skenario (untuk audit)
  3. File CSV ringkasan (siap copy ke Excel/Word untuk skripsi)
  4. File TXT laporan naratif (untuk bab Evaluasi skripsi)

Semua output disimpan ke backend/testing/reports/
dengan timestamp sebagai nama file.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _score_bar(score: float, width: int = 15) -> str:
    """Visualisasi skor sebagai bar ASCII."""
    filled = int(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _status_icon(is_success: bool) -> str:
    return "LULUS" if is_success else "GAGAL"


# ─────────────────────────────────────────────────────────────
# 1. TABEL TERMINAL (Real-time saat test berjalan)
# ─────────────────────────────────────────────────────────────

def print_scenario_result(result: Dict[str, Any]) -> None:
    """Print hasil satu skenario ke terminal secara rapi."""
    icon   = _status_icon(result["is_success"])
    score  = result["final_score"]
    bar    = _score_bar(score)
    dims   = result["dimensions"]

    print(f"\n{'='*65}")
    print(f"  [{icon}]  Skenario {result['scenario_id']}: {result['scenario_name']}")
    print(f"{'='*65}")
    print(f"  Kategori      : {result['category']}")
    print(f"  Skor Akhir    : {bar} {score*100:.1f}%")
    print(f"  Tools Dipanggil: {result['tools_called'] or ['(tidak ada)']}")
    print(f"  Kriteria Lulus: {result['passing_criteria']}")
    print(f"\n  --- Breakdown per Dimensi ---")
    print(f"  D1 Tool Selection  (40%): {dims['D1_tool_selection']['score']*100:.0f}%  | {dims['D1_tool_selection']['result']}")
    print(f"  D2 Argument Quality(35%): {dims['D2_argument_quality']['score']*100:.0f}%  | {dims['D2_argument_quality']['result']}")
    print(f"  D3 Response Quality(25%): {dims['D3_response_quality']['score']*100:.0f}%  | {dims['D3_response_quality']['result']}")
    if result.get("bonus_applied"):
        print(f"  Bonus Tool Opsional     : +5% (preferred tool dipanggil)")
    print(f"\n  Preview Response  : {result['raw_response_preview'][:120]}...")


def print_summary_table(results: List[Dict], summary: Dict) -> None:
    """Print tabel ringkasan semua skenario."""
    W = 75
    print(f"\n\n{'='*W}")
    print(f"  RINGKASAN EVALUASI AI AGENT WISTA")
    print(f"{'='*W}")
    header = f"{'No':<4} {'Nama Skenario':<35} {'D1':>5} {'D2':>5} {'D3':>5} {'Total':>7} {'Status':<8}"
    print(f"  {header}")
    print(f"  {'-'*(W-2)}")

    for r in results:
        dims  = r["dimensions"]
        d1    = f"{dims['D1_tool_selection']['score']*100:.0f}%"
        d2    = f"{dims['D2_argument_quality']['score']*100:.0f}%"
        d3    = f"{dims['D3_response_quality']['score']*100:.0f}%"
        total = f"{r['final_score']*100:.1f}%"
        status = _status_icon(r["is_success"])
        name  = r["scenario_name"][:34]
        print(f"  {r['scenario_id']:<4} {name:<35} {d1:>5} {d2:>5} {d3:>5} {total:>7} {status:<8}")

    print(f"  {'-'*(W-2)}")
    print(f"\n  SUCCESS RATE : {summary['success_rate']}")
    print(f"  Rata-rata Skor: {summary['average_score']*100:.1f}%")
    print(f"  Lulus / Gagal : {summary['success_count']} / {summary['failure_count']}")
    print(f"  Target Proposal: {summary['proposal_target']}")
    met = "TERCAPAI" if summary["meets_proposal_target"] else "BELUM TERCAPAI"
    print(f"  Status Target  : {met}")
    print(f"{'='*W}\n")


# ─────────────────────────────────────────────────────────────
# 2. JSON DETAIL (Audit Log)
# ─────────────────────────────────────────────────────────────

def save_json_report(results: List[Dict], summary: Dict) -> Path:
    """Simpan laporan JSON lengkap."""
    ts       = _timestamp()
    filepath = REPORTS_DIR / f"eval_detail_{ts}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"  [JSON] Laporan detail disimpan: {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────
# 3. CSV RINGKASAN (Siap pakai di Excel/Word skripsi)
# ─────────────────────────────────────────────────────────────

def save_csv_report(results: List[Dict], summary: Dict) -> Path:
    """Simpan ringkasan sebagai CSV."""
    ts       = _timestamp()
    filepath = REPORTS_DIR / f"eval_summary_{ts}.csv"

    rows = []
    for r in results:
        dims = r["dimensions"]
        rows.append({
            "No":              r["scenario_id"],
            "Nama Skenario":   r["scenario_name"],
            "Kategori":        r["category"],
            "D1 Tool (40%)":   f"{dims['D1_tool_selection']['score']*100:.0f}%",
            "D2 Argumen (35%)": f"{dims['D2_argument_quality']['score']*100:.0f}%",
            "D3 Respons (25%)": f"{dims['D3_response_quality']['score']*100:.0f}%",
            "Skor Total":      f"{r['final_score']*100:.1f}%",
            "Bonus":           "Ya" if r.get("bonus_applied") else "Tidak",
            "Status":          "Lulus" if r["is_success"] else "Gagal",
            "Tools Dipanggil": str(r["tools_called"]),
            "Kriteria":        r["passing_criteria"],
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Summary row
        writer.writerow({
            "No": "-",
            "Nama Skenario": "TOTAL / RATA-RATA",
            "Kategori": "-",
            "D1 Tool (40%)":   f"{summary['per_dimension']['D1_tool_selection_avg']*100:.1f}%",
            "D2 Argumen (35%)": f"{summary['per_dimension']['D2_argument_quality_avg']*100:.1f}%",
            "D3 Respons (25%)": f"{summary['per_dimension']['D3_response_quality_avg']*100:.1f}%",
            "Skor Total":      f"{summary['average_score']*100:.1f}%",
            "Bonus": "-",
            "Status":         f"Success Rate: {summary['success_rate']}",
            "Tools Dipanggil": "-",
            "Kriteria": summary['proposal_target'],
        })

    print(f"  [CSV] Ringkasan disimpan: {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────
# 4. TXT LAPORAN NARATIF (Untuk Bab Evaluasi Skripsi)
# ─────────────────────────────────────────────────────────────

def save_narrative_report(results: List[Dict], summary: Dict) -> Path:
    """
    Buat laporan naratif dalam Bahasa Indonesia,
    siap digunakan sebagai dasar penulisan Bab Evaluasi skripsi.
    """
    ts       = _timestamp()
    filepath = REPORTS_DIR / f"eval_narasi_{ts}.txt"

    met      = summary["meets_proposal_target"]
    target_s = "TERCAPAI" if met else "BELUM TERCAPAI"

    lines = [
        "=" * 70,
        "  LAPORAN EVALUASI AI AGENT WISTA",
        f"  Data Whisperer v1.0 — Sistem Analisis Data Wisata Bali",
        f"  Tanggal Evaluasi: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        "=" * 70,
        "",
        "A. RINGKASAN EKSEKUTIF",
        "-" * 40,
        f"   Total Skenario Diuji : {summary['total_scenarios']}",
        f"   Skenario Lulus       : {summary['success_count']}",
        f"   Skenario Gagal       : {summary['failure_count']}",
        f"   Success Rate         : {summary['success_rate']}",
        f"   Rata-rata Skor       : {summary['average_score']*100:.1f}%",
        f"   Target Proposal      : {summary['proposal_target']}",
        f"   Status Target        : {target_s}",
        "",
        "B. RATA-RATA PER DIMENSI EVALUASI",
        "-" * 40,
        f"   D1 Tool Selection Accuracy (bobot 40%)  : {summary['per_dimension']['D1_tool_selection_avg']*100:.1f}%",
        f"   D2 Argument Quality (bobot 35%)         : {summary['per_dimension']['D2_argument_quality_avg']*100:.1f}%",
        f"   D3 Response Quality (bobot 25%)         : {summary['per_dimension']['D3_response_quality_avg']*100:.1f}%",
        "",
        "C. DETAIL PER SKENARIO",
        "-" * 40,
    ]

    category_descriptions = {
        "Tool Selection & Argument Extraction": (
            "Skenario ini menguji kemampuan agent untuk mengidentifikasi tool yang "
            "tepat secara langsung dari prompt yang tersurat dan mengekstrak parameter "
            "numerik (budget, durasi) dengan akurat."
        ),
        "Multi-Tool Chaining & Semantic Understanding": (
            "Skenario ini menguji kemampuan neuro-symbolic agent: menerjemahkan "
            "preferensi abstrak pengguna ('stres', 'damai', 'hijau') menggunakan RAG, "
            "kemudian mengintegrasikannya ke dalam optimizer."
        ),
        "Reasoning & Constraint Validation": (
            "Skenario ini menguji kemampuan agent untuk mendeteksi constraint yang "
            "mustahil (budget terlalu kecil) dan memberikan respons pushback yang logis "
            "alih-alih menghasilkan hasil fiktif."
        ),
        "XAI: Multi-Tool ML + Visualization": (
            "Skenario ini menguji kemampuan Explainable AI: agent harus memanggil "
            "model ML untuk scoring dan menghasilkan visualisasi scatter plot sebagai "
            "bukti transparansi rekomendasi."
        ),
        "Data Analysis & Filtered Query": (
            "Skenario ini menguji kemampuan agent melakukan query multi-kriteria "
            "(lokasi + rating + crowd density) terhadap dataset dan menampilkan "
            "hasil yang relevan beserta informasi tambahan (Google Maps, harga)."
        ),
    }

    for r in results:
        dims    = r["dimensions"]
        status  = "LULUS" if r["is_success"] else "GAGAL"
        cat_desc = category_descriptions.get(r["category"], "")

        lines += [
            "",
            f"   Skenario {r['scenario_id']}: {r['scenario_name']}",
            f"   Kategori   : {r['category']}",
            f"   Keterangan : {cat_desc}",
            f"   Status     : {status}  (Skor: {r['final_score']*100:.1f}%)",
            f"   Tools      : {r['tools_called'] or ['(tidak ada)']}",
            f"   D1 Tool    : {dims['D1_tool_selection']['score']*100:.0f}%  - {dims['D1_tool_selection']['result']}",
            f"   D2 Argumen : {dims['D2_argument_quality']['score']*100:.0f}%  - {dims['D2_argument_quality']['result']}",
            f"   D3 Respons : {dims['D3_response_quality']['score']*100:.0f}%  - {dims['D3_response_quality']['result']}",
        ]

    lines += [
        "",
        "D. INTERPRETASI HASIL",
        "-" * 40,
        "",
        "   Sistem dievaluasi menggunakan tiga dimensi utama:",
        "   (1) Tool Selection Accuracy — ketepatan pemilihan tool oleh LLM",
        "   (2) Argument Quality — presisi ekstraksi parameter dari prompt bahasa natural",
        "   (3) Response Quality — kualitas dan akurasi narasi akhir tanpa halusinasi",
        "",
        "   Framework evaluasi dirancang untuk mengukur keandalan sistem secara",
        "   objektif dan terukur, sesuai dengan target proposal penelitian yang",
        f"   menetapkan Success Rate minimal 90% ({summary['proposal_target']}).",
        "",
        "=" * 70,
        "  [END OF REPORT]",
        "=" * 70,
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [TXT] Laporan naratif disimpan: {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_all_reports(results: List[Dict], summary: Dict) -> Dict[str, Path]:
    """Generate semua format laporan sekaligus."""
    print(f"\n  Menyimpan laporan ke: {REPORTS_DIR}")
    return {
        "json": save_json_report(results, summary),
        "csv":  save_csv_report(results, summary),
        "txt":  save_narrative_report(results, summary),
    }
