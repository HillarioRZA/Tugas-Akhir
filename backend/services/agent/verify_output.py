"""
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

        if len(draft_response.strip()) < 50:
            issues.append(
                "PERINGATAN: Draft respons terlalu pendek — mungkin tidak informatif."
            )

        HALLUCINATION_PATTERNS = [
            # ("Rp 0", ...) → DIHAPUS: Harga 0 = Gratis, valid di dataset wisata Bali
            ("undefined", "Kata 'undefined' terdeteksi — kemungkinan bug di tool output."),
        ]

        # Cek NaN/None hanya jika berdiri sendiri (bukan bagian dari kata lain)
        import re
        if re.search(r'\bnan\b', draft_response, re.IGNORECASE):
            # Cek apakah NaN dari konteks yang valid (seperti "distance_to_next: null")
            # Jika NaN muncul dalam narasi harga/rating, itu memang error
            nan_context = re.findall(r'.{0,30}\bnan\b.{0,30}', draft_response, re.IGNORECASE)
            price_related_nan = any(
                any(kw in ctx.lower() for kw in ['harga', 'price', 'rp', 'rating', 'biaya'])
                for ctx in nan_context
            )
            if price_related_nan:
                issues.append(
                    "PERINGATAN: Nilai NaN terdeteksi di konteks harga/rating — data mungkin error."
                )

        if re.search(r'\bNone\b', draft_response):
            issues.append(
                "PERINGATAN: Nilai None terdeteksi — kemungkinan tool belum mengembalikan data."
            )



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

        OUT_OF_DOMAIN_KEYWORDS = [
            "jakarta", "lombok", "surabaya", "bandung", "jogja", "yogyakarta",
            "eropa", "singapura", "malaysia", "kode python", "integral", "presiden",
        ]
        if any(kw in draft_response.lower() for kw in OUT_OF_DOMAIN_KEYWORDS):
            issues.append(
                "PERINGATAN DOMAIN: Respons mengandung konten di luar domain wisata Bali. "
                "Pertimbangkan untuk menolak dengan sopan menggunakan template guardrail."
            )

        budget = context.get("budget")
        optimizer_output = None

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
