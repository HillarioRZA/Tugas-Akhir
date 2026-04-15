from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.services.memory import persistent_memory
from backend.services.optimizer.travel_logic import calculate_optimized_itinerary
from backend.utils.read_csv import _read_csv_with_fallback
import pandas as pd
import os

from pydantic import BaseModel, Field, field_validator
import json

class OptimizerInput(BaseModel):
    budget_limit: int = Field(description="Maksimal budget dalam angka yang tersedia")
    location_keywords: list[str] = Field(description="Daftar kata kunci yang didapat dari RAG filter")
    duration_days: int = Field(default=1, description="Durasi perjalanan dalam hari. Gunakan 1 jika pengguna tidak menyebutkan spesifik waktu.")
    min_rating: float = Field(default=0.0, description="Rating minimum yang diinginkan, default 0.0")

    @field_validator('location_keywords', mode='before')
    def parse_location_keywords(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                return [item.strip() for item in v.split(',') if item.strip()]
        return v

    @field_validator('budget_limit', 'duration_days', mode='before')
    def parse_int(cls, v):
        if isinstance(v, str):
            try:
                return int(float(v))
            except Exception:
                pass
        return v
        
    @field_validator('min_rating', mode='before')
    def parse_float(cls, v):
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                pass
        return v

def get_optimizer_tools(session_id: str, context: dict) -> List[Any]:
    def _get_current_df():
        if "_cached_df" in context and context["_cached_df"] is not None:
            return context["_cached_df"]
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                with open(dataset_info['path'], 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                if df is not None:
                    context["_cached_df"] = df
                return df
            except Exception as e:
                print(f"Error memuat CSV untuk Optimizer: {e}")
                return None
        return None

    @tool(args_schema=OptimizerInput)
    def budget_optimizer_tool(budget_limit: int, location_keywords: list[str], duration_days: int = 1, min_rating: float = 0.0) -> dict:
        """
        Tool wajib untuk merancang itinerary perjalanan!
        SELALU gunakan alat ini SETELAH Anda mendapatkan daftar kata kunci.
        PENTING: JANGAN MENGARANG/BERHALUSINASI ANGKA! Ekstrak angka BUDGET (contoh: '400.000' -> 400000) dan DURASI hari secara eksak dari prompt pengguna terbaru.
        Alat ini menggunakan perhitungan matematis deterministik untuk mencocokkan budget dan destinasi.
        Jika hasil alat ini "budget terlalu kecil", Anda WAJIB menyampaikan Logical Pushback ke pengguna.
        """
        # BUG-8 fix: validasi input minimum
        duration_days = max(1, duration_days)
        # BUG-2 fix: simpan budget ke context untuk verify_output CoV
        context["budget"] = budget_limit

        df = _get_current_df()
        if df is None:
            return {"error": "Dataset destinasi belum tersedia. Harap upload data CSV."}
            
        success, message, recommendations, total_cost, daily_structure = calculate_optimized_itinerary(
            df=df,
            budget_limit=budget_limit,
            location_keywords=location_keywords,
            duration_days=duration_days,
            min_rating=min_rating
        )
        
        context["last_tool_name"] = "budget_optimizer_tool"
        
        if not success:
            context["last_tool_output"] = message
            context.setdefault("_tool_history", []).append({"tool": "budget_optimizer_tool", "output": message})
            return {"status": "error", "error_message": message, "instruction": "Lakukan Logical Pushback (Tolak permintaan karena tidak masuk akal secara logis/budgets). Berikan alasan."}
            
        result_data = {
            "pesan": message,
            "total_biaya_kalkulasi": total_cost,
            "rekomendasi_itinerary": recommendations,          # flat list (backward compat)
            "itinerary_per_hari": daily_structure,             # struktur per hari + geografi
        }
        
        context["last_tool_output"] = result_data
        context.setdefault("_tool_history", []).append({"tool": "budget_optimizer_tool", "output": result_data})
        
        # Buat summary singkat antar hari untuk agent
        daily_summary_lines = []
        if daily_structure:
            for day_key in sorted(k for k in daily_structure if k.startswith("hari_")):
                day_data = daily_structure[day_key]
                daily_summary_lines.append(day_data.get("day_summary", ""))
        
        day_summary_str = " | ".join(daily_summary_lines) if daily_summary_lines else ""
        
        return {
            "status": "success",
            "summary": (
                f"Berhasil menghitung itinerary {duration_days} hari dengan total Rp {total_cost:,}. "
                f"{len(recommendations)} destinasi. {day_summary_str}"
            ),
            "data": result_data
        }

    return [budget_optimizer_tool]
