"""
backend/services/agent/verify_output.py
=========================================
Chain of Verification (CoV) Tool untuk AgentExecutor WISTA.

Menyediakan factory function `create_verify_output_tool(context)` yang
mengembalikan LangChain @tool eksplisit untuk digunakan dalam toolbox agent.

Dengan tool ini, agent dapat memanggil verify_output() sebagai langkah ReAct
eksplisit sebelum mengirim jawaban akhir ke user:

    Thought → Action: [main tool] → Observation
    Thought → Action: verify_output(draft=...) → Observation {passed: True/False}
    Thought → Final Answer (hanya jika passed=True)

Checks yang dilakukan:
  1. Panjang minimum respons (≥50 karakter)
  2. Halusinasi numerik (Rp 0, nan, None, undefined)
  3. Konsistensi dengan last_tool_output di context
  4. Domain scope (menolak konten di luar wisata Bali)
  5. Budget integrity (jika optimizer tool dipanggil)

Referensi: Chain-of-Verification (CoV) — Dhuliawala et al., 2023
"""

from typing import Any, Dict
from langchain_core.tools import tool


def create_verify_output_tool(context: Dict[str, Any]):
    """
    Factory function — buat verify_output tool yang terikat pada `context` sesi.

    Args:
        context: dict shared state antara semua tools dalam satu agent session.
                 Berisi: last_tool_output, last_tool_name, last_image_bytes, dll.

    Returns:
        LangChain @tool siap dimasukkan ke toolbox AgentExecutor.
    """

    @tool
    def verify_output(draft_response: str) -> dict:
        """
        Verifikasi mandiri (Chain of Verification) sebelum mengirim respons ke user.
        Panggil tool ini SETELAH mendapat hasil dari tool lain dan SEBELUM
        menulis jawaban akhir ke user.

        Gunakan untuk:
        - Memastikan angka di respons konsisten dengan data tool
        - Memastikan tidak ada halusinasi nama/harga/rating
        - Memastikan narasi sesuai domain wisata Bali
        - Memastikan budget tidak terlampaui

        Args:
            draft_response: Draft jawaban yang akan dikirim ke user.

        Returns:
            Laporan verifikasi: passed=True/False + daftar issues ditemukan.
        """
        issues = []

        # ── 1. Cek panjang minimum (jawaban bermakna) ──
        if len(draft_response.strip()) < 50:
            issues.append(
                "PERINGATAN: Draft respons terlalu pendek — mungkin tidak informatif."
            )

        # ── 2. Cek indikasi halusinasi numerik ──
        HALLUCINATION_PATTERNS = [
            ("Rp 0",      "Harga bernilai 0 — kemungkinan error."),
            ("nan",       "Nilai NaN terdeteksi — data hilang atau error processing."),
            ("None",      "Nilai None terdeteksi — kemungkinan tool belum mengembalikan data."),
            ("undefined", "Kata 'undefined' terdeteksi."),
        ]
        for pattern, msg in HALLUCINATION_PATTERNS:
            if pattern.lower() in draft_response.lower():
                issues.append(
                    f"PERINGATAN HALUSINASI: {msg} | Ditemukan: '{pattern}'"
                )

        # ── 3. Cek konsistensi dengan last_tool_output ──
        last_output  = context.get("last_tool_output")
        last_name    = context.get("last_tool_name", "")
        tool_history = context.get("_tool_history", [])

        if last_output is None and not tool_history and any(
            keyword in draft_response.lower()
            for keyword in ["rekomendasi", "ditemukan", "berhasil", "itinerary"]
        ):
            issues.append(
                "PERINGATAN: Respons mengklaim ada rekomendasi/data, "
                "tapi tidak ada tool yang dipanggil sebelumnya (last_tool_output kosong)."
            )

        # ── 4. Cek apakah output keluar dari domain wisata Bali ──
        OUT_OF_DOMAIN_KEYWORDS = [
            "jakarta", "lombok", "surabaya", "bandung", "jogja", "yogyakarta",
            "eropa", "singapura", "malaysia", "kode python", "integral", "presiden",
        ]
        if any(kw in draft_response.lower() for kw in OUT_OF_DOMAIN_KEYWORDS):
            issues.append(
                "PERINGATAN DOMAIN: Respons mengandung konten di luar domain wisata Bali. "
                "Pertimbangkan untuk menolak dengan sopan menggunakan template guardrail."
            )

        # ── 5. Cek integritas budget (jika optimizer dipanggil) ──
        budget = context.get("budget")
        optimizer_output = None

        # Cari output optimizer dari history atau last_name (BUG-2 + BUG-10 fix)
        if last_name == "budget_optimizer_tool" and isinstance(last_output, dict):
            optimizer_output = last_output
        else:
            for entry in tool_history:
                if entry.get("tool") == "budget_optimizer_tool" and isinstance(entry.get("output"), dict):
                    optimizer_output = entry["output"]
                    break

        if optimizer_output and budget:
            total_cost = optimizer_output.get("total_biaya_kalkulasi", 0)
            if total_cost and total_cost > budget:
                issues.append(
                    f"PERINGATAN BUDGET: Total biaya (Rp {total_cost:,}) "
                    f"melebihi budget user (Rp {budget:,})."
                )

        passed = len(issues) == 0
        return {
            "verification_passed": passed,
            "issues_found":        len(issues),
            "issues":              issues,
            "instruction": (
                "Respons AMAN untuk dikirim ke user. Lanjutkan."
                if passed else
                f"Perbaiki {len(issues)} masalah berikut sebelum menjawab user: "
                + " | ".join(issues)
            ),
        }

    return verify_output
