"""
backend/services/ml/travel_trainer.py
========================================
Script training model ML untuk fitur Travel Recommendation WISTA.

Jalankan SEKALI (atau saat dataset berubah) dari root project:
    python -m backend.services.ml.travel_trainer

atau langsung:
    python backend/services/ml/travel_trainer.py

OUTPUT (backend/services/ml/saved_models/):
    travel_rf_model.joblib        -> Trained RandomForestClassifier
    travel_preprocessor.joblib    -> ColumnTransformer preprocessing pipeline
    travel_model_metadata.joblib  -> Metrics, feature importance, config

DESAIN MODEL:
-------------
Algoritma   : RandomForestClassifier (n_estimators=200, max_depth=15)
Target Label: 'is_top_destination' (binary classification)
  1 = destinasi menarik (attractive)
  0 = biasa

Composite Attractiveness Score (domain-weighted):
  40% -> Rating          (kualitas layanan dan pengalaman)
  30% -> Popularity      = log1p(jumlah_rating), normalized (kepercayaan)
  30% -> Value for Money = 1 - normalized(Price) (aksesibilitas budget)

Threshold: destinasi >= median attractiveness score => label = 1

FITUR (Features):
  Numerik    : Price, Rating, jumlah_rating
  Kategorikal: Category, Crowd_Density
"""

import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ── Root project = 4 level atas dari file ini ──
# backend/services/ml/travel_trainer.py
# └── backend/services/ml/
# └── backend/services/
# └── backend/
# └── <PROJECT_ROOT>
_THIS_FILE   = Path(__file__).resolve()
_ML_DIR      = _THIS_FILE.parent                     # backend/services/ml/
PROJECT_ROOT = _ML_DIR.parent.parent.parent           # <project root>

# Tambahkan root ke sys.path agar import backend.* bisa bekerja
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ─────────────────────────────────────────────────────────────
# PATH CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_PATH      = PROJECT_ROOT / "Build_Dataset" / "bali_tourist_clean_v3.csv"
MODELS_DIR        = PROJECT_ROOT / "saved_models"    # D:\...\Data_Whisperer_v1.0\saved_models\
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH        = MODELS_DIR / "travel_rf_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "travel_preprocessor.joblib"
METADATA_PATH     = MODELS_DIR / "travel_model_metadata.joblib"

# Fitur yang digunakan model
NUMERIC_FEATURES     = ["Price", "Rating", "jumlah_rating"]
CATEGORICAL_FEATURES = ["Category", "Crowd_Density"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Tags feature engineering — top-N tag paling sering muncul
TAGS_COL      = "tags"          # kolom tag di CSV (contoh: "alam, sepi, bukit")
TAGS_TOP_N    = 20              # ambil 20 tag terpopuler sebagai binary fitur
TAGS_SEP      = ","             # separator antar tag


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def extract_top_tags(df: pd.DataFrame, col: str = TAGS_COL, top_n: int = TAGS_TOP_N) -> list:
    """
    Ambil top-N tag paling sering muncul dari kolom tags.
    Contoh input: "alam, sepi, bukit" → ["alam", "sepi", "bukit"]
    Returns: list tag vocab (sorted by frequency).
    """
    if col not in df.columns:
        return []
    all_tags = []
    for val in df[col].dropna():
        tags = [t.strip().lower() for t in str(val).split(TAGS_SEP) if t.strip()]
        all_tags.extend(tags)
    from collections import Counter
    top_tags = [tag for tag, _ in Counter(all_tags).most_common(top_n)]
    return top_tags


def build_tag_features(df: pd.DataFrame, tag_vocab: list, col: str = TAGS_COL) -> pd.DataFrame:
    """
    Buat binary feature matrix dari kolom tags.
    Setiap tag dalam vocab jadi kolom binary (1 jika ada, 0 jika tidak).
    Return: DataFrame dengan kolom 'tag_<tagname>' untuk setiap tag di vocab.
    """
    rows = []
    for val in df[col] if col in df.columns else pd.Series([None] * len(df)):
        tag_set = set()
        if pd.notna(val):
            tag_set = {t.strip().lower() for t in str(val).split(TAGS_SEP) if t.strip()}
        rows.append({f"tag_{t}": int(t in tag_set) for t in tag_vocab})
    return pd.DataFrame(rows, index=df.index)


def compute_grouped_importance(feature_names, importances, known_features):
    """
    Menggabungkan importance kolom one-hot (mis. cat__Category_Alam)
    kembali ke nama fitur aslinya (mis. Category).
    """
    grouped = {}
    for feat_name, imp in zip(feature_names, importances):
        raw     = feat_name.split("__", 1)[-1]   # hapus prefix num__ / cat__
        matched = "Other"
        for known in known_features:
            if raw == known or raw.startswith(known + "_") or raw.startswith(known + " "):
                matched = known
                break
        grouped[matched] = grouped.get(matched, 0.0) + imp
    return {k: round(v, 4) for k, v in sorted(grouped.items(), key=lambda x: x[1], reverse=True)}


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def train_and_save():
    """
    Full training pipeline: load data -> feature engineering -> train -> evaluate -> save.
    Returns metadata dict untuk keperluan testing.
    """
    print("\n" + "=" * 60)
    print("  WISTA - TRAVEL ML MODEL TRAINING PIPELINE")
    print("=" * 60)

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── 1. Load Dataset ──
    print(f"\n[1/8] Memuat dataset: {DATASET_PATH.name}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}\n"
            "Jalankan dulu: python Build_Dataset/clean_dataset_v3.py"
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"      -> {len(df)} baris, {len(df.columns)} kolom")

    # ── 1b. Dataset Snapshot untuk Drift Monitoring ──
    dataset_snapshot = {
        "n_rows":          int(len(df)),
        "n_cols":          int(len(df.columns)),
        "price_mean":      round(float(df["Price"].mean()),  2) if "Price"       in df.columns else None,
        "price_std":       round(float(df["Price"].std()),   2) if "Price"       in df.columns else None,
        "rating_mean":     round(float(df["Rating"].mean()), 2) if "Rating"      in df.columns else None,
        "rating_std":      round(float(df["Rating"].std()),  2) if "Rating"      in df.columns else None,
        "jumlah_rat_mean": round(float(df["jumlah_rating"].mean()), 2) if "jumlah_rating" in df.columns else None,
        "categories":      sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else [],
        "snapshot_at":     trained_at,
    }

    # ── 2. Feature Engineering: Buat Label ──
    print(f"\n[2/8] Membuat label 'is_top_destination' (composite attractiveness score)...")

    rating_norm = (df["Rating"] - df["Rating"].min()) / (df["Rating"].max() - df["Rating"].min() + 1e-9)
    log_pop     = np.log1p(df["jumlah_rating"])
    pop_norm    = (log_pop - log_pop.min()) / (log_pop.max() - log_pop.min() + 1e-9)
    p_max       = df["Price"].max()
    value_score = 1.0 - (df["Price"] / p_max) if p_max > 0 else pd.Series(1.0, index=df.index)

    attractiveness           = rating_norm * 0.40 + pop_norm * 0.30 + value_score * 0.30
    threshold                = attractiveness.median()
    df["is_top_destination"] = (attractiveness >= threshold).astype(int)

    n_top = int(df["is_top_destination"].sum())
    print(f"      Bobot: Rating=40%, Popularitas=30%, Value for Money=30%")
    print(f"      Threshold (median): {threshold:.4f}")
    print(f"      Label 1 (Top Destination): {n_top} ({n_top/len(df)*100:.1f}%)")
    print(f"      Label 0 (Biasa)          : {len(df)-n_top} ({(len(df)-n_top)/len(df)*100:.1f}%)")

    # ── 3. Tags Feature Engineering (ML-3 Fix) ──
    print(f"\n[3/8] Membangun tag features dari kolom '{TAGS_COL}'...")
    tag_vocab = extract_top_tags(df, col=TAGS_COL, top_n=TAGS_TOP_N)
    has_tags  = len(tag_vocab) > 0 and TAGS_COL in df.columns

    if has_tags:
        tags_df = build_tag_features(df, tag_vocab, col=TAGS_COL)
        print(f"      -> {len(tag_vocab)} top tags diekstrak: {tag_vocab[:10]}... (top 10)")
        print(f"      -> Tag feature matrix shape: {tags_df.shape}")
    else:
        tags_df  = pd.DataFrame(index=df.index)
        tag_vocab = []
        print("      -> Kolom tags tidak ditemukan. Dilewati.")

    # ── 4. Split ──
    print(f"\n[4/8] Split data (80% train / 20% test, stratified)...")
    X_base = df[ALL_FEATURES].copy()
    y      = df["is_top_destination"]

    # Concatenate tag binary features ke base features
    if has_tags:
        X = pd.concat([X_base.reset_index(drop=True), tags_df.reset_index(drop=True)], axis=1)
        effective_features = ALL_FEATURES + [f"tag_{t}" for t in tag_vocab]
    else:
        X = X_base
        effective_features = ALL_FEATURES

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"      Total features: {len(effective_features)} ({len(NUMERIC_FEATURES)} numerik + {len(CATEGORICAL_FEATURES)} kategorik + {len(tag_vocab)} tag binary)")

    # ── 5. Preprocessing Pipeline ──
    print(f"\n[5/8] Membangun preprocessing pipeline (StandardScaler + OneHotEncoder + Tag passthrough)...")
    # Tag features sudah biner (0/1), tidak perlu encoding tambahan — pakai passthrough
    tag_col_names = [f"tag_{t}" for t in tag_vocab]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = [
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
    if tag_col_names:
        # "passthrough" = sklearn built-in, mendukung get_feature_names_out()
        # FunctionTransformer TIDAK mendukung get_feature_names_out() → error
        transformers.append(
            ("tag", "passthrough", tag_col_names)
        )

    preprocessor = ColumnTransformer(transformers)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print(f"      -> Feature matrix shape (train): {X_train_proc.shape}")

    # ── 6. Training ──
    print(f"\n[6/8] Melatih RandomForestClassifier (n_estimators=200, max_depth=15)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_proc, y_train)
    print(f"      -> Selesai!")

    # ── Feature Importance (base model, SEBELUM kalibrasi) ──
    feature_names_out   = preprocessor.get_feature_names_out()
    base_importances    = model.feature_importances_.copy()
    grouped_fi          = compute_grouped_importance(feature_names_out, base_importances, ALL_FEATURES)

    # ── 6b. Kalibrasi Probabilitas — Platt Scaling (LIM-5 fix) ──
    print(f"\n[6b/8] Kalibrasi probabilitas (Platt Scaling / Sigmoid)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',
        cv=3,
        n_jobs=-1
    )
    calibrated_model.fit(X_train_proc, y_train)
    print(f"      -> Kalibrasi selesai (3-fold Sigmoid).")

    # ── 7. Evaluasi ──
    print(f"\n[7/8] Evaluasi pada test set:")
    y_pred            = model.predict(X_test_proc)
    y_pred_prob_base  = model.predict_proba(X_test_proc)[:, 1]
    y_pred_prob_cal   = calibrated_model.predict_proba(X_test_proc)[:, 1]

    # Brier Score (mengukur kualitas kalibrasi — lower is better)
    brier_base       = round(float(brier_score_loss(y_test, y_pred_prob_base)), 4)
    brier_calibrated = round(float(brier_score_loss(y_test, y_pred_prob_cal)), 4)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test,  y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test,    y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test,        y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test,   y_pred_prob_base)), 4),
        "brier_score_base":       brier_base,
        "brier_score_calibrated": brier_calibrated,
        "calibration_method":     "Platt Scaling (Sigmoid, 3-fold CV)",
    }

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"      {metric_name:<25}: {value:.4f}")
        else:
            print(f"      {metric_name:<25}: {value}")

    print(f"\n      Classification Report:")
    report_lines = classification_report(
        y_test, y_pred, target_names=["Biasa (0)", "Top Dest (1)"]
    ).split("\n")
    for line in report_lines:
        print(f"      {line}")

    cv_scores = cross_val_score(model, X_train_proc, y_train, cv=5, scoring="f1")
    print(f"\n      5-Fold CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ── Feature Importance (dari base_importances) ──
    print(f"\n      Feature Importance (grouped):")
    for feat, imp in grouped_fi.items():
        print(f"      {feat:<25}: {imp:.4f} ({imp*100:.1f}%)")

    top_10_raw = sorted(
        [{"feature": f, "importance": round(float(i), 4)}
         for f, i in zip(feature_names_out, base_importances)],
        key=lambda x: x["importance"], reverse=True
    )[:10]

    # ── 8. Simpan ──
    print(f"\n[8/8] Menyimpan model dan metadata ke: {MODELS_DIR}")
    metadata = {
        "model_type":               "RandomForestClassifier",
        "n_estimators":             200,
        "max_depth":                15,
        "target_column":            "is_top_destination",
        "label_description":        "1=top destination (attractive), 0=biasa",
        "attractiveness_weights":   {"rating": 0.40, "popularity": 0.30, "value_for_money": 0.30},
        "label_threshold":          float(threshold),
        "features_numeric":         NUMERIC_FEATURES,
        "features_categorical":     CATEGORICAL_FEATURES,
        "features_tag_vocab":       tag_vocab,          # ML-3: tag vocab untuk drift check
        "all_features":             effective_features,  # includes tag_* columns
        "feature_names_out":        list(feature_names_out),
        "grouped_feature_importance": grouped_fi,
        "top_10_raw_features":      top_10_raw,
        "metrics":                  metrics,
        "cv_f1_mean":               round(float(cv_scores.mean()), 4),
        "cv_f1_std":                round(float(cv_scores.std()), 4),
        "training_samples":         int(len(X_train)),
        "test_samples":             int(len(X_test)),
        "dataset_path":             str(DATASET_PATH),
        "models_dir":               str(MODELS_DIR),
        # ML-2: Dataset snapshot untuk drift monitoring
        "dataset_snapshot":         dataset_snapshot,
        "trained_at":               trained_at,
    }

    joblib.dump(calibrated_model, MODEL_PATH)   # LIM-5: simpan model terkalibrasi
    joblib.dump(preprocessor,      PREPROCESSOR_PATH)
    joblib.dump(metadata,          METADATA_PATH)

    print(f"      -> {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"      -> {PREPROCESSOR_PATH.name} ({PREPROCESSOR_PATH.stat().st_size:.0f} bytes)")
    print(f"      -> {METADATA_PATH.name} ({METADATA_PATH.stat().st_size:.0f} bytes)")

    print(f"\n{'='*60}")
    print(f"  SELESAI - Model siap digunakan oleh AI Agent WISTA!")
    print(f"{'='*60}\n")

    return metadata


if __name__ == "__main__":
    train_and_save()
