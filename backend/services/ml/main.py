"""
backend/services/ml/main.py
============================
ML Tools untuk AI Agent WISTA.

predict_match_score:
  Menggunakan RandomForestClassifier yang sudah dilatih pada dataset bali_tourist_clean_v3.csv
  untuk memprediksi seberapa cocok (Match Score 0-100%) suatu destinasi dengan preferensi user.

  Alur:
  1. Load model dari saved_models/ (lazy-loaded, di-cache setelah pertama kali)
  2. Cari data destinasi kandidat dari dataset CSV
  3. Terapkan penyesuaian preferensi user (category/crowd/price matching)
  4. Return sorted scores + feature importance dari model
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory

# ─────────────────────────────────────────────────────────────
# PATH MODEL
# Model disimpan di folder saved_models/ di root project.
# Gunakan travel_trainer.py (di folder ini) untuk melatih ulang model.
# ─────────────────────────────────────────────────────────────

# backend/services/ml/main.py → naik 4 level → project root
_ML_BASE = Path(__file__).parent.parent.parent.parent / "saved_models"
_MODEL_PATH        = _ML_BASE / "travel_rf_model.joblib"
_PREPROCESSOR_PATH = _ML_BASE / "travel_preprocessor.joblib"
_METADATA_PATH     = _ML_BASE / "travel_model_metadata.joblib"

# ─────────────────────────────────────────────────────────────
# LAZY LOADER — Model di-load sekali, di-cache di memory modul
# ─────────────────────────────────────────────────────────────

_model_cache = {}

def _load_ml_assets():
    """Load model, preprocessor, metadata (lazy, cached)."""
    if "model" not in _model_cache:
        if not _MODEL_PATH.exists():
            return None, None, None
        try:
            _model_cache["model"]        = joblib.load(_MODEL_PATH)
            _model_cache["preprocessor"] = joblib.load(_PREPROCESSOR_PATH)
            _model_cache["metadata"]     = joblib.load(_METADATA_PATH)
            print(f"[ML] Model RandomForest berhasil dimuat dari: {_MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Gagal memuat model: {e}")
            return None, None, None
    return _model_cache["model"], _model_cache["preprocessor"], _model_cache["metadata"]


# ─────────────────────────────────────────────────────────────
# HELPER: Preference Adjustment Score
# Menyesuaikan base ML score dengan preferensi eksplisit user
# ─────────────────────────────────────────────────────────────

def _apply_preference_adjustment(base_score: float, row: pd.Series, user_prefs: dict) -> float:
    """
    Mendapatkan adjusted score dengan menambahkan/mengurangi bonus berdasarkan
    kecocokan antara preferensi user dengan atribut destinasi.

    base_score: probabilitas dari RandomForest (0.0 - 1.0)
    Adjustment range: ±0.15 (tidak mengubah dominasi model ML)
    """
    adjustment = 0.0

    # ── Kecocokan Kategori ──
    pref_category = user_prefs.get("category", user_prefs.get("kategori", ""))
    if pref_category and isinstance(pref_category, str):
        dest_category = str(row.get("Category", "")).lower()
        if pref_category.lower() in dest_category or dest_category in pref_category.lower():
            adjustment += 0.08

    # ── Kecocokan Crowd / Keramaian ──
    pref_crowd = user_prefs.get("crowd_preference", user_prefs.get("keramaian", ""))
    if pref_crowd and isinstance(pref_crowd, str):
        crowd_map = {
            "sepi": "Sepi", "quiet": "Sepi", "tenang": "Sepi", "healing": "Sepi",
            "sedang": "Sedang", "moderate": "Sedang",
            "ramai": "Ramai", "crowded": "Ramai",
            "sangat ramai": "Sangat Ramai", "populer": "Sangat Ramai",
        }
        target_crowd = crowd_map.get(pref_crowd.lower(), "")
        dest_crowd   = str(row.get("Crowd_Density", ""))
        if target_crowd and dest_crowd == target_crowd:
            adjustment += 0.07

    # ── Kecocokan Budget / Harga ──
    pref_budget = user_prefs.get("budget_preference", user_prefs.get("budget", ""))
    dest_price  = float(row.get("Price", 0))
    if pref_budget:
        budget_map = {
            "gratis": 0, "free": 0,
            "murah": 20000,  "budget": 20000, "hemat": 20000, "terjangkau": 20000,
            "sedang": 75000,
            "mahal": 200000, "premium": 200000, "mewah": 200000,
        }
        budget_threshold = budget_map.get(str(pref_budget).lower(), None)
        if budget_threshold is not None:
            if dest_price <= budget_threshold:
                adjustment += 0.05
            elif dest_price > budget_threshold * 2:
                adjustment -= 0.05

    # Clamp final score ke range [0.0, 1.0]
    return float(np.clip(base_score + adjustment, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA
# ─────────────────────────────────────────────────────────────

class MatchScoreInput(BaseModel):
    user_preferences: dict = Field(
        description=(
            "Preferensi user dalam bentuk dictionary. Keys yang dikenali: "
            "'category' (Alam/Budaya/Rekreasi/Umum), "
            "'crowd_preference' (sepi/sedang/ramai/sangat ramai), "
            "'budget_preference' (gratis/murah/sedang/mahal/premium). "
            "Contoh: {'category': 'Alam', 'crowd_preference': 'sepi', 'budget_preference': 'murah'}"
        )
    )
    candidate_destinations: list[str] = Field(
        description="Daftar nama destinasi wisata yang ingin dinilai relevansinya."
    )


# ─────────────────────────────────────────────────────────────
# TOOL FACTORY
# ─────────────────────────────────────────────────────────────

def get_ml_tools(session_id: str, context: dict) -> List[Any]:

    def _read_current_df() -> Optional[pd.DataFrame]:
        if "_cached_df" in context and context["_cached_df"] is not None:
            return context["_cached_df"]
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info["path"]):
            with open(dataset_info["path"], "rb") as f:
                contents = f.read()
            df = _read_csv_with_fallback(contents)
            if df is not None:
                context["_cached_df"] = df
            return df
        return None

    @tool(args_schema=MatchScoreInput)
    def predict_match_score(user_preferences: dict, candidate_destinations: list[str]) -> dict:
        """
        Tool ML untuk 'Travel Itinerary': memprediksi seberapa cocok (Match Score 0-100%)
        setiap destinasi wisata dengan preferensi abstrak user menggunakan model RandomForest
        yang sudah dilatih pada dataset wisata Bali.
        Gunakan ini sebelum memanggil Budget Optimizer untuk menilai relevansi kandidat.
        Output berisi match_score, label prediksi, feature_importance, dan preference_adjustment.
        """

        # ── 1. Load model assets ──
        model, preprocessor, metadata = _load_ml_assets()

        if model is None:
            # Fallback informatif jika model belum dilatih
            return {
                "status": "warning",
                "message": (
                    "Model ML belum tersedia. Jalankan 'python Build_Dataset/train_travel_ml.py' "
                    "untuk melatih model terlebih dahulu."
                ),
                "data": []
            }

        # ── 2. Load dataset & cari kandidat ──
        df = _read_current_df()
        if df is None:
            return {"status": "error", "message": "Dataset tidak ditemukan.", "data": []}

        name_col = next(
            (c for c in df.columns if "name" in c.lower() or "place" in c.lower() or "nama" in c.lower()),
            None
        )
        if name_col is None:
            return {"status": "error", "message": "Kolom nama destinasi tidak ditemukan di dataset.", "data": []}

        # Filter baris yang nama destinasinya ada di candidate list
        candidate_mask = df[name_col].isin(candidate_destinations)
        candidate_df   = df[candidate_mask].copy()

        if candidate_df.empty:
            return {
                "status": "warning",
                "message": f"Tidak ada dari {len(candidate_destinations)} destinasi kandidat yang ditemukan di dataset.",
                "data": []
            }

        # ── 3. Siapkan feature matrix ──
        feature_cols = metadata["all_features"]
        available_cols = [c for c in feature_cols if c in candidate_df.columns]
        missing_cols   = [c for c in feature_cols if c not in candidate_df.columns]

        if missing_cols:
            print(f"[ML] Warning: Kolom fitur tidak ditemukan: {missing_cols}. Mengisi dengan default.")
        for mc in missing_cols:
            candidate_df[mc] = 0 if mc in ["Price", "Rating", "jumlah_rating"] else "Umum"

        X_candidates = candidate_df[feature_cols]

        try:
            X_proc = preprocessor.transform(X_candidates)
        except Exception as e:
            return {"status": "error", "message": f"Preprocessing gagal: {e}", "data": []}

        # ── 4. Prediksi probabilitas Match Score ──
        try:
            proba = model.predict_proba(X_proc)  # shape: (n, 2)
            base_scores = proba[:, 1]            # prob of class 1 (top destination)
        except Exception as e:
            return {"status": "error", "message": f"Prediksi model gagal: {e}", "data": []}

        # ── 5. Preference Adjustment + Build Results ──
        grouped_fi  = metadata.get("grouped_feature_importance", {})
        top_fi_list = [
            {"fitur": feat, "kontribusi": f"{round(imp * 100, 1)}%"}
            for feat, imp in list(grouped_fi.items())[:4]
        ]

        results = []
        for i, (_, row) in enumerate(candidate_df.iterrows()):
            dest_name   = row[name_col]
            base_score  = float(base_scores[i])
            adj_score   = _apply_preference_adjustment(base_score, row, user_preferences)
            label       = "Top Destination" if adj_score >= 0.5 else "Biasa"
            score_pct   = round(adj_score * 100, 1)

            results.append({
                "destination":           dest_name,
                "match_score_percentage": score_pct,
                "prediction_label":      label,
                "base_ml_score":         round(base_score * 100, 1),
                "preference_adjustment": round((adj_score - base_score) * 100, 1),
                "category":              row.get("Category", "-"),
                "crowd_density":         row.get("Crowd_Density", "-"),
                "price":                 int(row.get("Price", 0)),
                "rating":                float(row.get("Rating", 0.0)),
            })

        # Sort descending by match score
        results = sorted(results, key=lambda x: x["match_score_percentage"], reverse=True)

        # ── 6. Simpan output ke context ──
        context["last_tool_output"] = results
        context["last_tool_name"]   = "predict_match_score"
        context.setdefault("_tool_history", []).append({"tool": "predict_match_score", "output": results})

        model_metrics = metadata.get("metrics", {})

        return {
            "status": "success",
            "summary": (
                f"Berhasil menghitung ML Match Score untuk {len(results)} dari "
                f"{len(candidate_destinations)} destinasi kandidat menggunakan "
                f"RandomForestClassifier (Accuracy={model_metrics.get('accuracy', 0)*100:.1f}%, "
                f"ROC-AUC={model_metrics.get('roc_auc', 0)*100:.1f}%)."
            ),
            "xai_feature_importance": top_fi_list,
            "model_info": {
                "type":      metadata.get("model_type", "RandomForestClassifier"),
                "accuracy":  f"{model_metrics.get('accuracy', 0)*100:.1f}%",
                "roc_auc":   f"{model_metrics.get('roc_auc', 0)*100:.1f}%",
                "f1_score":  f"{model_metrics.get('f1_score', 0)*100:.1f}%",
                "label_logic": metadata.get("label_description", "-"),
            },
            "data": results,
            "instruction": (
                "Sampaikan Match Score ini kepada user lengkap dengan feature importance (XAI) "
                "sebagai bukti transparansi model. Untuk destinasi dengan preference_adjustment != 0, "
                "jelaskan bahwa skor disesuaikan karena cocok/tidak cocok dengan preferensi user."
            )
        }

    @tool
    def retrain_model() -> dict:
        """
        Latih ulang model RandomForest dari dataset CSV terkini dan simpan ke saved_models/.
        Gunakan saat pengguna meminta: 'Update model ML', 'Latih ulang model',
        atau 'Model sudah outdated, tolong perbarui'.
        Proses training membutuhkan 10-30 detik.
        Setelah selesai, model cache di-reset sehingga predict_match_score
        otomatis menggunakan model yang baru.
        """
        import threading

        train_result: dict = {}
        error_holder: list = []

        def _run_training():
            try:
                from backend.services.ml.travel_trainer import train_and_save
                result = train_and_save()
                train_result.update(result)
                # Invalidate cache agar model baru langsung dipakai
                _model_cache.clear()
                print("[ML] Model cache di-reset. Model baru siap digunakan.")
            except Exception as e:
                error_holder.append(str(e))

        thread = threading.Thread(target=_run_training, daemon=True)
        thread.start()
        thread.join(timeout=120)  # maks 2 menit

        if error_holder:
            return {
                "status": "error",
                "message": f"Training gagal: {error_holder[0]}",
            }

        if not train_result:
            return {
                "status": "timeout",
                "message": "Training melebihi batas waktu 120 detik. Coba jalankan manual: python -m backend.services.ml.travel_trainer",
            }

        metrics = train_result.get("metrics", {})
        context["last_tool_output"] = train_result
        context["last_tool_name"]   = "retrain_model"

        return {
            "status": "success",
            "summary": (
                f"Model RandomForest berhasil dilatih ulang. "
                f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}% | "
                f"ROC-AUC: {metrics.get('roc_auc', 0)*100:.1f}% | "
                f"F1: {metrics.get('f1_score', 0)*100:.1f}%"
            ),
            "new_metrics": metrics,
            "training_samples":   train_result.get("training_samples", "?"),
            "grouped_importance": train_result.get("grouped_feature_importance", {}),
            "instruction": (
                "Sampaikan ke user bahwa model sudah diperbarui dan berikan ringkasan "
                "performa model baru (Accuracy, ROC-AUC, F1)."
            )
        }

    @tool
    def check_model_drift() -> dict:
        """
        Cek apakah model ML masih relevan terhadap dataset terkini (Model Drift Monitoring).
        Bandingkan statistik dataset saat training vs dataset saat ini.
        Gunakan saat pengguna bertanya: 'Apakah model masih valid?',
        'Dataset sudah diupdate, apakah model perlu dilatih ulang?',
        atau 'Cek drift model'.
        Mengembalikan laporan drift dengan severity: OK / WARNING / CRITICAL.
        """
        # ── 1. Load metadata model ──
        model, preprocessor, metadata = _load_ml_assets()
        if model is None:
            return {
                "status": "error",
                "message": "Model ML belum dilatih. Jalankan retrain_model terlebih dahulu.",
            }

        train_snapshot = metadata.get("dataset_snapshot")
        trained_at     = metadata.get("trained_at", "tidak diketahui")
        if not train_snapshot:
            return {
                "status": "warning",
                "message": (
                    "Model belum menyimpan dataset snapshot (model lama sebelum ML-3 fix). "
                    "Jalankan retrain_model untuk memperbarui metadata dengan snapshot."
                )
            }

        # ── 2. Load dataset terkini ──
        df = _read_current_df()
        if df is None:
            return {"status": "error", "message": "Dataset tidak ditemukan untuk dibandingkan."}

        # ── 3. Hitung statistik saat ini ──
        current_stats = {
            "n_rows":          int(len(df)),
            "price_mean":      round(float(df["Price"].mean()),  2) if "Price"        in df.columns else None,
            "price_std":       round(float(df["Price"].std()),   2) if "Price"        in df.columns else None,
            "rating_mean":     round(float(df["Rating"].mean()), 2) if "Rating"       in df.columns else None,
            "rating_std":      round(float(df["Rating"].std()),  2) if "Rating"       in df.columns else None,
            "jumlah_rat_mean": round(float(df["jumlah_rating"].mean()), 2) if "jumlah_rating" in df.columns else None,
            "categories":      sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else [],
        }

        # ── 4. Deteksi drift per metrik ──
        THRESHOLD_MEAN = 0.15   # ±15% perubahan mean = WARNING
        THRESHOLD_STD  = 0.25   # ±25% perubahan std  = WARNING
        issues = []

        def _pct_change(old, new) -> float:
            if old and old != 0:
                return abs(new - old) / abs(old)
            return 0.0

        # Row count drift
        row_change = _pct_change(train_snapshot.get("n_rows", 0), current_stats["n_rows"])
        if row_change > 0.20:
            issues.append({
                "metric":    "n_rows",
                "training":  train_snapshot.get("n_rows"),
                "current":   current_stats["n_rows"],
                "pct_change": round(row_change * 100, 1),
                "severity":  "WARNING"
            })

        # Price mean drift
        for key, thr in [("price_mean", THRESHOLD_MEAN), ("price_std", THRESHOLD_STD),
                          ("rating_mean", THRESHOLD_MEAN), ("rating_std", THRESHOLD_STD)]:
            old = train_snapshot.get(key)
            new = current_stats.get(key)
            if old is not None and new is not None:
                pct = _pct_change(old, new)
                if pct > thr:
                    issues.append({
                        "metric":     key,
                        "training":   old,
                        "current":    new,
                        "pct_change": round(pct * 100, 1),
                        "severity":   "CRITICAL" if pct > thr * 2 else "WARNING",
                    })

        # Category drift (new categories added)
        old_cats = set(train_snapshot.get("categories", []))
        new_cats = set(current_stats["categories"])
        added_cats   = list(new_cats - old_cats)
        removed_cats = list(old_cats - new_cats)
        if added_cats or removed_cats:
            issues.append({
                "metric":        "categories",
                "added":         added_cats,
                "removed":       removed_cats,
                "severity":      "WARNING",
            })

        # ── 5. Hitung severity keseluruhan ──
        critical_count = sum(1 for i in issues if i.get("severity") == "CRITICAL")
        warning_count  = sum(1 for i in issues if i.get("severity") == "WARNING")

        if critical_count > 0:
            overall = "CRITICAL"
            recommendation = "Model WAJIB dilatih ulang. Data telah berubah signifikan dari waktu training."
        elif warning_count > 0:
            overall = "WARNING"
            recommendation = "Model sebaiknya dilatih ulang. Ada perubahan distribusi data yang cukup signifikan."
        else:
            overall = "OK"
            recommendation = "Model masih relevan. Tidak ada drift signifikan yang terdeteksi."

        context["last_tool_output"] = {
            "drift_severity": overall,
            "issues": issues,
        }
        context["last_tool_name"] = "check_model_drift"

        return {
            "status":          "success",
            "drift_severity":  overall,
            "trained_at":      trained_at,
            "issues_found":    len(issues),
            "issues":          issues,
            "training_snapshot": train_snapshot,
            "current_stats":   current_stats,
            "recommendation":  recommendation,
            "instruction": (
                f"Laporkan status drift ({overall}) kepada user dengan tabel perbandingan "
                f"training vs saat ini. Jika severity WARNING/CRITICAL, sarankan memanggil "
                f"retrain_model untuk memperbarui model."
            )
        }

    return [predict_match_score, retrain_model, check_model_drift]

