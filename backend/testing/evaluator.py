"""
backend/testing/evaluator.py
================================
Logic evaluasi hasil percobaan skenario AI Agent.

Setiap skenario dievaluasi berdasarkan 4 dimensi:

  [D1] Tool Selection Accuracy
       Apakah tool yang benar dipanggil? (dari reasoning_log)

  [D2] Argument Quality
       Apakah argumen tool sesuai dengan yang diekstrak dari prompt?

  [D3] Response Quality
       Apakah response akhir mengandung kata kunci yang diharapkan?
       Tidak mengandung kata terlarang (indikasi halusinasi)?

  [D4] Logical Behavior
       Untuk Skenario 3: apakah agent melakukan pushback?
       Untuk Skenario 2/4: apakah tool opsional juga dipanggil (bonus)?

Skor per dimensi: 0.0 (gagal) / 0.5 (parsial) / 1.0 (lulus)
Skor total: rata-rata tertimbang → SUCCESS jika >= 0.6
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────

# Keyword pushback — response mengandung salah satu ini → pushback detected
PUSHBACK_KEYWORDS = [
    "budget", "terlalu kecil", "tidak cukup", "tidak memungkinkan",
    "tidak bisa", "tidak dapat", "maaf", "mohon maaf",
    "anggaran", "kurang", "minimum", "minimal"
]

# Kata yang menandakan agent berhasil (buruk untuk skenario pushback)
SUCCESS_KEYWORDS_BAD_FOR_PUSHBACK = [
    "berikut itinerary", "berikut rute", "berhasil membuat", "rekomendasi anda",
    "inilah itinerary", "hari pertama", "hari kedua"
]


# ─────────────────────────────────────────────────────────────
# HELPER PARSER
# ─────────────────────────────────────────────────────────────

def _extract_tool_calls_from_log(reasoning_log: List) -> Dict[str, Any]:
    """
    Parse reasoning_log dari agent response untuk mengekstrak:
    - Daftar tool yang dipanggil
    - Argumen tiap tool

    Format AKTUAL dari agent/main.py (LangChain intermediate_steps):
      List of dict: {
        "step": int,
        "thought": str,
        "tool_called": str,      <-- nama tool
        "tool_input": dict,      <-- argumen tool
        "observation": str
      }

    Fallback: jika entry adalah string dengan format:
      "Invoking: `tool_name` with `{...args...}`"
    (digunakan oleh dry-run mock)
    """
    import json
    tool_calls = {}  # {tool_name: [args_dict, ...]}

    for entry in reasoning_log:
        # ── Format utama: dict dari intermediate_steps ──
        if isinstance(entry, dict):
            tool_name = entry.get("tool_called", "")
            tool_input = entry.get("tool_input", {})
            if tool_name:
                if tool_name not in tool_calls:
                    tool_calls[tool_name] = []
                # tool_input bisa berupa dict atau string JSON
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except Exception:
                        tool_input = {}
                tool_calls[tool_name].append(tool_input if isinstance(tool_input, dict) else {})

        # ── Fallback: string "Invoking: `tool` with `{...}`" (dry-run mock) ──
        elif isinstance(entry, str):
            match = re.search(
                r"Invoking:\s*`([^`]+)`\s+with\s+`(\{.*?\})`",
                entry, re.DOTALL
            )
            if match:
                tool_name = match.group(1).strip()
                args_str  = match.group(2).strip()
                try:
                    args_dict = json.loads(args_str.replace("'", '"'))
                except Exception:
                    args_dict = {}
                if tool_name not in tool_calls:
                    tool_calls[tool_name] = []
                tool_calls[tool_name].append(args_dict)

    return tool_calls


def _normalize_text(text: str) -> str:
    """Lowercase dan strip tanda baca untuk perbandingan."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# ─────────────────────────────────────────────────────────────
# DIMENSI 1: Tool Selection Accuracy
# ─────────────────────────────────────────────────────────────

def evaluate_tool_selection(
    tool_calls: Dict[str, Any],
    expected_tools: List[str],
    forbidden_tools: List[str],
    preferred_tools: Optional[List[str]] = None,
) -> Tuple[float, str]:
    """
    Returns: (score 0.0-1.0, detail_message)

    Scoring:
      1.0  = semua expected_tools dipanggil, tidak ada forbidden_tools
      0.75 = expected_tools dipanggil, tidak ada forbidden (preferred tidak dipanggil)
      0.5  = sebagian expected dipanggil
      0.0  = tidak ada expected yang dipanggil ATAU ada forbidden yang dipanggil
    """
    called = set(tool_calls.keys())
    expected_set  = set(expected_tools)
    forbidden_set = set(forbidden_tools)
    preferred_set = set(preferred_tools or [])

    # Cek forbidden
    forbidden_called = called & forbidden_set
    if forbidden_called:
        return 0.0, f"GAGAL: Tool terlarang dipanggil: {forbidden_called}"

    # Cek expected
    missing_expected = expected_set - called
    if not missing_expected:
        # Semua expected terpenuhi
        preferred_called = called & preferred_set
        if preferred_set and preferred_called:
            return 1.0, f"LULUS: Semua expected + preferred tools dipanggil: {called & (expected_set | preferred_set)}"
        return 1.0, f"LULUS: Semua expected tools dipanggil: {called & expected_set}"
    elif len(missing_expected) < len(expected_set):
        return 0.5, f"PARSIAL: Beberapa expected tools tidak dipanggil: {missing_expected}"
    else:
        return 0.0, f"GAGAL: Tidak ada expected tools yang dipanggil. Tools dipanggil: {called}"


# ─────────────────────────────────────────────────────────────
# DIMENSI 2: Argument Quality
# ─────────────────────────────────────────────────────────────

def evaluate_arguments(
    tool_calls: Dict[str, Any],
    arg_checks: Dict[str, Dict[str, Any]],
) -> Tuple[float, str, List[Dict]]:
    """
    Validasi argumen yang dikirim ke setiap tool.

    arg_checks format:
      { "tool_name": { "arg_key": expected_value_or_callable } }

    Returns: (score, summary_message, detail_list)
    """
    if not arg_checks:
        return 1.0, "Tidak ada arg checks yang didefinisikan.", []

    total_checks = 0
    passed_checks = 0
    details = []

    for tool_name, checks in arg_checks.items():
        if tool_name not in tool_calls:
            # Tool tidak dipanggil → arg check tidak bisa divalidasi
            details.append({
                "tool": tool_name,
                "status": "SKIP",
                "reason": f"Tool '{tool_name}' tidak dipanggil, arg tidak bisa dievaluasi."
            })
            continue

        # Ambil argumen dari call terakhir tool ini
        args_list = tool_calls[tool_name]
        args = args_list[-1] if args_list else {}

        for arg_key, expected in checks.items():
            total_checks += 1
            actual_value = args.get(arg_key)

            if actual_value is None:
                details.append({
                    "tool": tool_name, "arg": arg_key,
                    "status": "GAGAL",
                    "actual": None,
                    "expected": str(expected),
                    "reason": f"Argumen '{arg_key}' tidak ada di call."
                })
                continue

            try:
                if callable(expected):
                    result = expected(actual_value)
                else:
                    result = (str(actual_value) == str(expected))

                if result:
                    passed_checks += 1
                    details.append({
                        "tool": tool_name, "arg": arg_key,
                        "status": "LULUS",
                        "actual": actual_value,
                        "reason": "Argumen sesuai ekspektasi."
                    })
                else:
                    details.append({
                        "tool": tool_name, "arg": arg_key,
                        "status": "GAGAL",
                        "actual": actual_value,
                        "expected": str(expected),
                        "reason": f"Nilai '{actual_value}' tidak memenuhi kondisi."
                    })
            except Exception as e:
                details.append({
                    "tool": tool_name, "arg": arg_key,
                    "status": "ERROR",
                    "actual": actual_value,
                    "reason": f"Exception saat validasi: {e}"
                })

    if total_checks == 0:
        return 1.0, "Tool tidak dipanggil, tidak ada argumen untuk dievaluasi.", details

    score = passed_checks / total_checks
    summary = f"{passed_checks}/{total_checks} argumen lulus validasi."
    return round(score, 2), summary, details


# ─────────────────────────────────────────────────────────────
# DIMENSI 3: Response Quality
# ─────────────────────────────────────────────────────────────

def evaluate_response_quality(
    summary: str,
    expected_in_response: List[str],
    forbidden_in_response: List[str],
    expected_pushback: bool = False,
) -> Tuple[float, str, List[Dict]]:
    """
    Evaluasi kualitas response akhir dari agent.
    """
    normalized = _normalize_text(summary)
    details = []
    total   = 0
    passed  = 0

    # Cek expected keywords
    for kw in expected_in_response:
        total += 1
        found = kw.lower() in normalized
        status = "LULUS" if found else "GAGAL"
        passed += 1 if found else 0
        details.append({
            "type": "expected_keyword",
            "keyword": kw,
            "status": status,
            "found": found
        })

    # Cek forbidden keywords (indikasi halusinasi)
    for kw in forbidden_in_response:
        total += 1
        found = kw.lower() in normalized
        # Forbidden ditemukan → GAGAL
        status = "GAGAL" if found else "LULUS"
        passed += 0 if found else 1
        details.append({
            "type": "forbidden_keyword",
            "keyword": kw,
            "status": status,
            "found": found,
            "note": "Indikasi halusinasi kota/tempat yang tidak relevan" if found else ""
        })

    # Cek pushback behavior (khusus Skenario 3)
    if expected_pushback:
        total += 1
        pushback_found = any(kw in normalized for kw in PUSHBACK_KEYWORDS)
        bad_found = any(kw in normalized for kw in
                       [_normalize_text(bk) for bk in SUCCESS_KEYWORDS_BAD_FOR_PUSHBACK])

        if pushback_found and not bad_found:
            passed += 1
            details.append({"type": "pushback_check", "status": "LULUS",
                            "reason": "Agent melakukan pushback yang tepat."})
        elif not pushback_found:
            details.append({"type": "pushback_check", "status": "GAGAL",
                            "reason": "Agent tidak melakukan pushback padahal constraint tidak memungkinkan."})
        else:
            details.append({"type": "pushback_check", "status": "GAGAL",
                            "reason": "Agent seolah berhasil meski constraint mustahil."})

    score = round(passed / total, 2) if total > 0 else 0.0
    summary_msg = f"{passed}/{total} response quality checks lulus."
    return score, summary_msg, details


# ─────────────────────────────────────────────────────────────
# EVALUATOR UTAMA
# ─────────────────────────────────────────────────────────────

def evaluate_scenario_result(
    scenario: Dict[str, Any],
    agent_response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluasi hasil satu skenario secara menyeluruh.

    Args:
        scenario      : Definisi skenario dari scenarios.py
        agent_response: JSON response dari API /api/agent/execute

    Returns:
        Dict hasil evaluasi lengkap dengan score dan detail per dimensi.
    """
    expected    = scenario.get("expected", {})
    summary_txt = agent_response.get("summary", "")
    reasoning   = agent_response.get("reasoning_log", [])

    # Parse tool calls dari reasoning log
    tool_calls = _extract_tool_calls_from_log(reasoning)

    # ── D1: Tool Selection ──
    d1_score, d1_msg = evaluate_tool_selection(
        tool_calls=tool_calls,
        expected_tools=expected.get("expected_tools", []),
        forbidden_tools=expected.get("forbidden_tools", []),
        preferred_tools=expected.get("preferred_tools", []),
    )

    # ── D2: Argument Quality ──
    d2_score, d2_msg, d2_details = evaluate_arguments(
        tool_calls=tool_calls,
        arg_checks=expected.get("arg_checks", {}),
    )

    # ── D3: Response Quality ──
    d3_score, d3_msg, d3_details = evaluate_response_quality(
        summary=summary_txt,
        expected_in_response=expected.get("expected_in_response", []),
        forbidden_in_response=expected.get("forbidden_in_response", []),
        expected_pushback=expected.get("expected_pushback", False),
    )

    # ── Weighted Final Score ──
    # D1 Tool Selection: 40% (paling kritis untuk TA)
    # D2 Argument Quality: 35% (anti-halusinasi argumen)
    # D3 Response Quality: 25%
    weights = {"d1": 0.40, "d2": 0.35, "d3": 0.25}
    final_score = round(
        d1_score * weights["d1"] +
        d2_score * weights["d2"] +
        d3_score * weights["d3"],
        3
    )

    is_success = final_score >= 0.6  # 60% = threshold lulus

    # Bonus: preferred tools dipanggil → +5% tapi tidak melampaui 1.0
    preferred = set(expected.get("preferred_tools", []))
    called    = set(tool_calls.keys())
    bonus     = 0.05 if preferred and (preferred & called) else 0.0
    final_score_with_bonus = min(1.0, round(final_score + bonus, 3))

    return {
        "scenario_id":    scenario["id"],
        "scenario_name":  scenario["name"],
        "category":       scenario["category"],
        "prompt":         scenario["prompt"][:80] + "...",
        "is_success":     is_success,
        "final_score":    final_score_with_bonus,
        "passing_score":  0.6,
        "bonus_applied":  bonus > 0,
        "tools_called":   list(tool_calls.keys()),
        "dimensions": {
            "D1_tool_selection": {
                "score": d1_score, "weight": "40%",
                "result": d1_msg
            },
            "D2_argument_quality": {
                "score": d2_score, "weight": "35%",
                "result": d2_msg,
                "details": d2_details
            },
            "D3_response_quality": {
                "score": d3_score, "weight": "25%",
                "result": d3_msg,
                "details": d3_details
            },
        },
        "passing_criteria": scenario.get("passing_criteria", "-"),
        "raw_response_preview": summary_txt[:200] + "..." if len(summary_txt) > 200 else summary_txt,
    }


def compute_success_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hitung Success Rate keseluruhan dari semua hasil evaluasi.
    """
    total    = len(results)
    success  = sum(1 for r in results if r["is_success"])
    avg_score = round(sum(r["final_score"] for r in results) / total, 3) if total > 0 else 0

    # Per dimensi
    avg_d1 = round(sum(r["dimensions"]["D1_tool_selection"]["score"]  for r in results) / total, 3)
    avg_d2 = round(sum(r["dimensions"]["D2_argument_quality"]["score"] for r in results) / total, 3)
    avg_d3 = round(sum(r["dimensions"]["D3_response_quality"]["score"] for r in results) / total, 3)

    return {
        "total_scenarios":    total,
        "success_count":      success,
        "failure_count":      total - success,
        "success_rate":       f"{round(success/total*100, 1)}%" if total > 0 else "0%",
        "average_score":      avg_score,
        "per_dimension": {
            "D1_tool_selection_avg":  avg_d1,
            "D2_argument_quality_avg": avg_d2,
            "D3_response_quality_avg": avg_d3,
        },
        "meets_proposal_target": success / total >= 0.90 if total > 0 else False,
        "proposal_target":        "Success Rate >= 90%",
    }
