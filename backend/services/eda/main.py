import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory

# generate_custom_plot dihapus — digantikan oleh 5 chart tools
# di backend/services/visualization/main.py yang lebih lengkap dan
# terintegrasi langsung dengan dark theme sistem.

def get_eda_tools(session_id: str, context: dict) -> List[Any]:
    def _read_current_csv():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            with open(dataset_info['path'], 'rb') as f:
                return f.read()
        return None

    @tool
    def describe_dataset() -> dict:
        """Untuk ringkasan statistik (mean, std, etc.)."""
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            res = df.describe().to_dict()
            context["last_tool_output"] = res
            context["last_tool_name"] = "describe_dataset"
            return {"status": "success", "summary": "Berhasil mendapatkan ringkasan statistik dataset.", "data": res}
        except Exception as e:
            return {"error": str(e)}

    @tool
    def run_full_profile() -> dict:
        """Untuk analisis umum atau profil lengkap dataset (shape, info, nulls, skewness)."""
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "file content is None"}

            description = df.describe().to_dict()

            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info_str = info_buffer.getvalue()

            missing_values = df.isnull().sum().to_dict()
            missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()

            df_numeric = df.select_dtypes(include=['number'])
            skewness = df_numeric.skew().to_dict()

            res = {
                "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "statistical_summary": description,
                "data_info": info_str,
                "missing_values_summary": {
                    "count": missing_values,
                    "percentage": missing_percentage
                },
                "skewness": skewness
            }
            context["last_tool_output"] = res
            context["last_tool_name"] = "run_full_profile"
            return {"status": "success", "summary": "Berhasil menjalankan full data profile.", "data": res}
        except Exception as e:
             return {"error": str(e)}

    @tool
    def detect_outliers() -> dict:
        """
        Deteksi outlier pada semua kolom numerik dataset menggunakan dua metode:
        1. IQR (Interquartile Range): outlier di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        2. Z-Score: nilai dengan |z| > 3 (lebih dari 3 standar deviasi dari rata-rata)
        Juga menghasilkan box plot visual untuk kolom numerik yang punya outlier.
        Gunakan untuk analisis kualitas data atau ketika pengguna bertanya tentang anomali/outlier.
        """
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            num_df = df.select_dtypes(include="number")
            if num_df.empty:
                return {"error": "Tidak ada kolom numerik di dataset."}

            results = {}
            total_outlier_rows = set()
            cols_with_outliers = []

            for col in num_df.columns:
                series = num_df[col].dropna()

                # Metode IQR
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                iqr_mask  = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
                iqr_count = int(iqr_mask.sum())

                # Metode Z-Score
                z_scores  = np.abs((series - series.mean()) / series.std())
                z_mask    = z_scores > 3
                z_count   = int(z_mask.sum())

                outlier_indices = list(series[iqr_mask | z_mask].index[:5])
                total_outlier_rows.update(series[iqr_mask].index.tolist())

                results[col] = {
                    "iqr_outliers":    iqr_count,
                    "zscore_outliers": z_count,
                    "pct_outlier":     round(iqr_count / len(series) * 100, 2),
                    "range_valid":     [round(float(Q1 - 1.5 * IQR), 2), round(float(Q3 + 1.5 * IQR), 2)],
                    "sample_outlier_rows": outlier_indices,
                }
                if iqr_count > 0:
                    cols_with_outliers.append(col)

            summary = {
                "total_numeric_cols":     len(num_df.columns),
                "total_outlier_rows_iqr": len(total_outlier_rows),
                "pct_rows_with_outlier":  round(len(total_outlier_rows) / len(df) * 100, 2),
                "cols_with_outliers":     cols_with_outliers,
                "per_column": results,
            }

            # ── Visualisasi Box Plot untuk kolom yang punya outlier ──
            if cols_with_outliers:
                n_cols = min(len(cols_with_outliers), 4)   # maks 4 kolom per baris
                fig, axes = plt.subplots(
                    1, n_cols,
                    figsize=(4 * n_cols, 5),
                    squeeze=False
                )
                fig.patch.set_facecolor("#1e1e2e")

                for idx, col in enumerate(cols_with_outliers[:n_cols]):
                    ax = axes[0][idx]
                    ax.set_facecolor("#2a2a3e")

                    series = num_df[col].dropna()
                    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                    IQR    = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                    # Box plot
                    bp = ax.boxplot(
                        series,
                        patch_artist=True,
                        notch=False,
                        vert=True,
                        widths=0.6,
                    )
                    bp["boxes"][0].set_facecolor("#89b4fa")
                    bp["boxes"][0].set_alpha(0.7)
                    bp["medians"][0].set_color("#f38ba8")
                    bp["medians"][0].set_linewidth(2)
                    for flier in bp["fliers"]:
                        flier.set(marker="o", color="#fab387", markersize=4, alpha=0.6)

                    # Garis batas valid
                    ax.axhline(lower, color="#a6e3a1", linestyle="--", lw=1, label=f"IQR lower: {lower:.0f}")
                    ax.axhline(upper, color="#cba6f7", linestyle="--", lw=1, label=f"IQR upper: {upper:.0f}")

                    n_out = results[col]["iqr_outliers"]
                    ax.set_title(f"{col}\n({n_out} outlier IQR)", color="#cdd6f4", fontsize=9)
                    ax.tick_params(colors="#cdd6f4")
                    ax.spines["bottom"].set_color("#45475a")
                    ax.spines["left"].set_color("#45475a")
                    ax.legend(fontsize=7, facecolor="#313244", labelcolor="#cdd6f4")

                plt.suptitle(
                    f"Box Plot Outlier Detection — {len(cols_with_outliers)} Kolom Bermasalah",
                    color="#cba6f7", fontsize=12
                )
                plt.tight_layout()

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=120)
                plt.close()
                img_buffer.seek(0)
                context["last_image_bytes"]  = img_buffer.getvalue()
                context["last_tool_name"]    = "outlier-boxplot"
                context["last_tool_params"]  = {"cols_with_outliers": cols_with_outliers}

            context["last_tool_output"] = summary
            context["last_tool_name"]   = "detect_outliers"
            return {
                "status": "success",
                "summary": (
                    f"Deteksi outlier selesai. {len(total_outlier_rows)} baris berpotensi outlier "
                    f"dari {len(df)} total baris ({summary['pct_rows_with_outlier']}%). "
                    f"Box plot divisualisasikan untuk {len(cols_with_outliers)} kolom bermasalah."
                ),
                "data": summary,
                "instruction": (
                    "Jelaskan kolom mana yang punya outlier tertinggi dan apa implikasinya terhadap "
                    "analisis atau model ML. Box plot sudah dikirimkan sebagai visualisasi."
                )
            }
        except Exception as e:
            return {"error": str(e)}

    @tool
    def get_correlation_matrix() -> dict:
        """
        Hitung matriks korelasi Pearson antar semua kolom numerik di dataset.
        Return data korelasi dalam format JSON yang bisa dianalisis agent.
        Gunakan ini saat pengguna bertanya: 'Apakah rating berkorelasi dengan harga?',
        'Fitur mana yang paling berkorelasi satu sama lain?',
        atau 'Tampilkan matriks korelasi sebagai data'.
        Berbeda dengan plot_correlation_heatmap yang hanya menghasilkan gambar —
        tool ini memberikan NILAI NUMERIK korelasi untuk analisis naratif.
        """
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                return {"error": "Minimal 2 kolom numerik diperlukan untuk matriks korelasi."}

            corr = num_df.corr(method="pearson").round(4)
            corr_dict = corr.to_dict()

            # Pasangan dengan korelasi kuat (|r| >= 0.5)
            strong_pairs = []
            moderate_pairs = []
            for i, col_a in enumerate(corr.columns):
                for j, col_b in enumerate(corr.columns):
                    if j <= i:
                        continue
                    r = float(corr.loc[col_a, col_b])
                    entry = {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "pearson_r": round(r, 4),
                        "interpretation": (
                            "Sangat kuat positif" if r >= 0.8 else
                            "Kuat positif"        if r >= 0.5 else
                            "Sangat kuat negatif" if r <= -0.8 else
                            "Kuat negatif"        if r <= -0.5 else
                            "Lemah/tidak ada"
                        )
                    }
                    if abs(r) >= 0.5:
                        strong_pairs.append(entry)
                    elif abs(r) >= 0.3:
                        moderate_pairs.append(entry)

            strong_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

            result = {
                "n_numeric_cols":   num_df.shape[1],
                "columns_analyzed": list(num_df.columns),
                "correlation_matrix": corr_dict,
                "strong_correlations":    strong_pairs,    # |r| >= 0.5
                "moderate_correlations":  moderate_pairs,  # 0.3 <= |r| < 0.5
                "insight_summary": (
                    f"Ditemukan {len(strong_pairs)} pasangan fitur dengan korelasi kuat (|r| >= 0.5). "
                    f"{len(moderate_pairs)} pasangan dengan korelasi moderat (|r| 0.3-0.5). "
                    + (f"Pasangan terkorelasi kuat: "
                       + ", ".join(f"{p['feature_a']} ↔ {p['feature_b']} (r={p['pearson_r']})"
                                   for p in strong_pairs[:3])
                       if strong_pairs else "Tidak ada korelasi kuat antar fitur.")
                ),
            }

            context["last_tool_output"] = result
            context["last_tool_name"]   = "correlation_matrix"
            return {
                "status": "success",
                "summary": result["insight_summary"],
                "data": result,
                "instruction": (
                    "Narasikan temuan korelasi kuat (jika ada), jelaskan implikasi terhadap "
                    "pemilihan fitur model ML dan potensi masalah multikolinearitas."
                )
            }
        except Exception as e:
            return {"error": str(e)}

    @tool
    def analyze_vif() -> dict:
        """
        Hitung Variance Inflation Factor (VIF) untuk semua kolom numerik.
        VIF mengukur multikolinearitas: VIF > 10 berarti kolom sangat berkorelasi dengan kolom lain,
        yang bisa merusak performa model ML. Gunakan sebelum training model untuk seleksi fitur.
        """
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            num_df = df.select_dtypes(include="number").dropna()
            if num_df.shape[1] < 2:
                return {"error": "Minimal 2 kolom numerik diperlukan untuk menghitung VIF."}

            vif_data = {}
            for i, col in enumerate(num_df.columns):
                try:
                    vif_val = variance_inflation_factor(num_df.values, i)
                    vif_data[col] = {
                        "vif":        round(float(vif_val), 3),
                        "risk_level": "Tinggi (>10, multikolinear)" if vif_val > 10
                                      else "Sedang (5-10)" if vif_val > 5
                                      else "Rendah (<5, aman)"
                    }
                except Exception:
                    vif_data[col] = {"vif": None, "risk_level": "Tidak dapat dihitung"}

            high_vif = [k for k, v in vif_data.items() if v["vif"] and v["vif"] > 10]

            result = {
                "vif_per_column": vif_data,
                "high_multicollinearity_cols": high_vif,
                "recommendation": (
                    f"Perhatikan kolom: {high_vif}. Pertimbangkan untuk menghapus atau menggabungkan."
                    if high_vif else "Tidak ada multikolinearitas kritis. Dataset siap untuk modeling."
                )
            }

            context["last_tool_output"] = result
            context["last_tool_name"]   = "vif"
            return {"status": "success", "summary": f"Analisis VIF selesai. {len(high_vif)} kolom berpotensi multikolinear.", "data": result}
        except Exception as e:
            return {"error": str(e)}

    return [
        describe_dataset,
        run_full_profile,
        detect_outliers,
        get_correlation_matrix,
        analyze_vif,
    ]