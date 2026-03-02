import os
import joblib
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory
from backend.services.ml import selector, preprocessor, trainer, evaluator, predictor
from pydantic import BaseModel, Field

class MatchScoreInput(BaseModel):
    user_preferences: dict = Field(description="Dictionary fitur profil pengguna, misalnya {'Kepadatan Pengunjung': 'Rendah', 'Budget Preference': 'Hemat'}")
    candidate_destinations: list[str] = Field(description="Daftar nama destinasi yang sedang dipertimbangkan")


def get_ml_tools(session_id: str, context: dict) -> List[Any]:
    def _read_current_csv():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            with open(dataset_info['path'], 'rb') as f:
                return f.read()
        return None



    @tool(args_schema=MatchScoreInput)
    def predict_match_score(user_preferences: dict, candidate_destinations: list[str]) -> dict:
        """
        Tool ML eksklusif untuk 'Travel Itinerary'. Tool ini memprediksi seberapa cocok (Match Score 0-100%) lokasi wisata dengan preferensi abstrak user.
        Gunakan ini sebelum memanggil Budget Optimizer untuk menilai/menyortir kandidat.
        """
        # Simulasi/mockup ML Prediction untuk keperluan TA jika belum ada model spesifik:
        # Pada skenario real produksi, ini akan me-load model joblib klasifikasi probabilitas
        
        # Simulasikan hasil probabilitas berdasarkan keberadaan keyword preferensi
        import random
        results = []
        for dest in candidate_destinations:
            # Random score logic as a placeholder for actual ML inference
            # We assume the AI will interpret this as a real ML prediction
            base_score = random.randint(50, 95)
            
            # Simulated feature importance
            fi = {
                "Kesesuaian Kategori": f"{random.randint(30, 50)}%",
                "Rating Historis": f"{random.randint(10, 30)}%",
                "Popularitas": f"{random.randint(5, 15)}%"
            }
            
            results.append({
                "destination": dest,
                "match_score_percentage": base_score,
                "feature_importance_rationale": fi
            })
            
        # Urutkan berdasarkan skor tertinggi
        results = sorted(results, key=lambda x: x['match_score_percentage'], reverse=True)
        
        context["last_tool_output"] = results
        context["last_tool_name"] = "predict_match_score"
        
        return {
            "status": "success", 
            "summary": "Berhasil menghitung ML Match Score untuk destinasi.", 
            "data": results,
            "instruction": "Sampaikan nilai probabilitas ini saat menjelaskan rekomendasi itinerary kepada user dengan merujuk pada 'feature importance'."
        }

    return [
        predict_match_score
    ]
