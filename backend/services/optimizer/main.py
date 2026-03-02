from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.services.memory import persistent_memory
from backend.services.optimizer.travel_logic import calculate_optimized_itinerary
from backend.utils.read_csv import _read_csv_with_fallback
import pandas as pd
import os

class OptimizerInput(BaseModel):
    budget_limit: int = Field(description="Maksimal budget dalam angka yang tersedia")
    location_keywords: list[str] = Field(description="Daftar kata kunci yang didapat dari RAG filter")
    duration_days: int = Field(default=1, description="Durasi perjalanan dalam hari. Gunakan 1 jika pengguna tidak menyebutkan spesifik waktu.")
    min_rating: float = Field(default=0.0, description="Rating minimum yang diinginkan, default 0.0")

def get_optimizer_tools(session_id: str, context: dict) -> List[Any]:
    def _get_current_df():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                with open(dataset_info['path'], 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                return df
            except Exception as e:
                print(f"Error memuat CSV untuk Optimizer: {e}")
                return None
        return None

    @tool(args_schema=OptimizerInput)
    def budget_optimizer_tool(budget_limit: int, location_keywords: list[str], duration_days: int = 1, min_rating: float = 0.0) -> dict:
        """
        Tool wajib untuk merancang itinerary perjalanan!
        SELALU gunakan alat ini SETELAH Anda mendapatkan daftar kata kunci dari RAG Semantic Filter.
        Alat ini menggunakan perhitungan matematis deterministik untuk mencocokkan budget dan destinasi.
        Jika hasil alat ini "budget terlalu kecil", Anda WAJIB menyampaikan Logical Pushback ke pengguna.
        """
        df = _get_current_df()
        if df is None:
            return {"error": "Dataset destinasi belum tersedia. Harap upload data CSV."}
            
        success, message, recommendations, total_cost = calculate_optimized_itinerary(
            df=df,
            budget_limit=budget_limit,
            location_keywords=location_keywords,
            duration_days=duration_days,
            min_rating=min_rating
        )
        
        context["last_tool_name"] = "budget_optimizer_tool"
        
        if not success:
            context["last_tool_output"] = message
            return {"status": "error", "error_message": message, "instruction": "Lakukan Logical Pushback (Tolak permintaan karena tidak masuk akal secara logis/budgets). Berikan alasan."}
            
        result_data = {
            "pesan": message,
            "total_biaya_kalkulasi": total_cost,
            "rekomendasi_itinerary": recommendations
        }
        
        context["last_tool_output"] = result_data
        
        return {
            "status": "success",
            "summary": f"Berhasil menghitung itinerary dengan total {total_cost}. Pilihan: {len(recommendations)} tempat.",
            "data": result_data
        }

    return [budget_optimizer_tool]
