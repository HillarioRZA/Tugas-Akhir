import os
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory

# ── Style global untuk semua chart ───────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.facecolor"] = "#1e1e2e"
plt.rcParams["axes.facecolor"]   = "#2a2a3e"
plt.rcParams["axes.labelcolor"]  = "#cdd6f4"
plt.rcParams["xtick.color"]      = "#cdd6f4"
plt.rcParams["ytick.color"]      = "#cdd6f4"
plt.rcParams["text.color"]       = "#cdd6f4"
plt.rcParams["grid.color"]       = "#45475a"

_ACCENT_COLORS = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8", "#94e2d5"]

class PlotItineraryInput(BaseModel):
    selected_destinations: list[str] = Field(
        description="Daftar nama destinasi wisata yang terpilih/direkomendasikan dari Optimizer."
    )

class PlotDistributionInput(BaseModel):
    column: str = Field(description="Nama kolom numerik yang ingin dilihat distribusinya (contoh: 'Rating', 'Price').")

class PlotCategoryInput(BaseModel):
    category_column: str = Field(
        default="Category",
        description="Nama kolom kategori untuk dianalisis (contoh: 'Category', 'City')."
    )
    top_n: int = Field(
        default=10,
        description="Jumlah kategori teratas yang ditampilkan (default 10)."
    )

class PlotCorrelationInput(BaseModel):
    pass  # Tidak perlu parameter, auto-detect kolom numerik

class PlotBudgetBreakdownInput(BaseModel):
    selected_destinations: list[str] = Field(
        description="Daftar nama destinasi wisata yang terpilih dari Optimizer untuk pie chart budget."
    )


def get_visualization_tools(session_id: str, context: dict) -> List[Any]:
    def _read_current_csv():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            with open(dataset_info['path'], 'rb') as f:
                return f.read()
        return None

    def _save_figure_to_context(tool_name: str, params: dict) -> dict:
        """Helper: simpan BytesIO buffer ke context dan return success response."""
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        img_buffer.seek(0)
        context["last_image_bytes"]  = img_buffer.getvalue()
        context["last_tool_name"]    = tool_name
        context["last_tool_params"]  = params
        return {"status": "success", "summary": f"Chart '{tool_name}' berhasil dibuat.", "instruction": "Kirimkan narasi insight dari chart yang baru saja digambar."}

    def _find_col(df: pd.DataFrame, *keywords) -> Optional[str]:
        """Cari nama kolom yang mengandung salah satu keyword (case insensitive)."""
        for kw in keywords:
            for col in df.columns:
                if kw.lower() in col.lower():
                    return col
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 1: Scatter Plot XAI — Harga vs Rating (existing, direfaktor)
    # ──────────────────────────────────────────────────────────────────────────
    @tool(args_schema=PlotItineraryInput)
    def plot_itinerary_scatter(selected_destinations: list[str]) -> dict:
        """
        [XAI Chart] Scatter plot Harga vs Rating seluruh destinasi.
        Destinasi terpilih dari Optimizer dilingkari warna accent, sisanya abu-abu transparan.
        WAJIB dipanggil setelah Budget Optimizer menghasilkan itinerary.
        """
        file_contents = _read_current_csv()
        if not file_contents:
            return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            price_col  = _find_col(df, 'price', 'harga', 'cost')
            rating_col = _find_col(df, 'rating')
            name_col   = _find_col(df, 'place_name', 'name', 'nama', 'place', 'destination')

            if not all([price_col, rating_col, name_col]):
                return {"error": "Kolom harga/rating/nama tidak ditemukan."}

            fig, ax = plt.subplots(figsize=(13, 8))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#2a2a3e")

            unselected = df[~df[name_col].isin(selected_destinations)]
            ax.scatter(unselected[price_col], unselected[rating_col],
                       color="#6c7086", alpha=0.35, s=40, label="Tidak terpilih")

            selected = df[df[name_col].isin(selected_destinations)]
            if not selected.empty:
                ax.scatter(selected[price_col], selected[rating_col],
                           color="#cba6f7", s=160, edgecolors="#f5c2e7",
                           linewidths=2, zorder=5, label="✅ Rekomendasi Itinerary")
                for _, row in selected.iterrows():
                    ax.annotate(
                        row[name_col],
                        (row[price_col], row[rating_col]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8.5, color="#cdd6f4",
                        arrowprops=dict(arrowstyle="->", color="#89b4fa", lw=0.8)
                    )

            ax.set_title("🔍 XAI: Mengapa Destinasi Ini Dipilih? (Rating vs Harga)", fontsize=14, color="#cba6f7", pad=15)
            ax.set_xlabel(f"Harga Tiket ({price_col})", fontsize=11, color="#89b4fa")
            ax.set_ylabel(f"Rating ({rating_col})", fontsize=11, color="#89b4fa")
            ax.legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3, color="#45475a")

            return _save_figure_to_context("itinerary-visualization", {"selected": selected_destinations})
        except Exception as e:
            plt.close()
            return {"error": f"Error membuat scatter plot: {e}"}

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 2: Histogram Distribusi — distribusi kolom numerik + KDE curve
    # ──────────────────────────────────────────────────────────────────────────
    @tool(args_schema=PlotDistributionInput)
    def plot_distribution_histogram(column: str) -> dict:
        """
        Buat histogram distribusi dengan KDE (Kernel Density Estimation) untuk kolom numerik.
        Gunakan untuk melihat distribusi Rating, Harga, atau kolom numerik lainnya.
        Contoh: 'Tampilkan distribusi Rating' atau 'Distribusi harga tiket wisata Bali'.
        """
        file_contents = _read_current_csv()
        if not file_contents:
            return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            # Cari kolom yang cocok (fuzzy)
            target_col = None
            for col in df.columns:
                if column.lower() in col.lower() or col.lower() in column.lower():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        target_col = col
                        break
            if not target_col:
                return {"error": f"Kolom numerik '{column}' tidak ditemukan. Kolom tersedia: {list(df.select_dtypes(include='number').columns)}"}

            data = df[target_col].dropna()
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#2a2a3e")

            ax.hist(data, bins=40, color="#89b4fa", alpha=0.7, edgecolor="#45475a", label="Frekuensi")

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 300)
                ax.plot(x_range, kde(x_range) * len(data) * (data.max() - data.min()) / 40,
                        color="#f38ba8", lw=2.5, label="KDE (Densitas)")
            except ImportError:
                pass

            # Garis mean & median
            ax.axvline(data.mean(),   color="#a6e3a1", lw=2, linestyle="--", label=f"Mean: {data.mean():.2f}")
            ax.axvline(data.median(), color="#fab387", lw=2, linestyle=":",  label=f"Median: {data.median():.2f}")

            ax.set_title(f"📊 Distribusi {target_col} — {len(data):,} Destinasi Wisata Bali", fontsize=14, color="#cba6f7", pad=15)
            ax.set_xlabel(target_col, fontsize=11, color="#89b4fa")
            ax.set_ylabel("Frekuensi", fontsize=11, color="#89b4fa")
            ax.legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3, color="#45475a")

            stats_text = f"Min: {data.min():.0f}  |  Max: {data.max():.0f}  |  Std: {data.std():.2f}  |  Skew: {data.skew():.2f}"
            fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, color="#9399b2")

            return _save_figure_to_context("distribution-histogram", {"column": column})
        except Exception as e:
            plt.close()
            return {"error": f"Error membuat histogram: {e}"}

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 3: Bar Chart Top Kategori/Kota — frekuensi per kategori atau kota
    # ──────────────────────────────────────────────────────────────────────────
    @tool(args_schema=PlotCategoryInput)
    def plot_category_bar(category_column: str = "Category", top_n: int = 10) -> dict:
        """
        Buat bar chart jumlah destinasi per kategori, kota, atau kolom kategorikal lainnya.
        Gunakan untuk: 'Berapa destinasi per kategori?', 'Kota mana yang punya wisata terbanyak?',
        'Tampilkan bar chart kategori wisata Bali'.
        Parameter category_column: nama kolom (default 'Category'), bisa juga 'City', 'Kecamatan', dll.
        """
        file_contents = _read_current_csv()
        if not file_contents:
            return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            # Fuzzy match kolom
            target_col = None
            for col in df.columns:
                if category_column.lower() in col.lower():
                    target_col = col
                    break
            if not target_col:
                return {"error": f"Kolom '{category_column}' tidak ditemukan. Tersedia: {list(df.columns)}"}

            counts = df[target_col].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#2a2a3e")

            bars = ax.barh(counts.index[::-1], counts.values[::-1],
                           color=_ACCENT_COLORS * (top_n // len(_ACCENT_COLORS) + 1),
                           edgecolor="#1e1e2e", linewidth=0.8)

            # Label nilai di dalam bar
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 5, bar.get_y() + bar.get_height() / 2,
                        f"{int(w):,}", va="center", fontsize=9, color="#cdd6f4")

            ax.set_title(f"🏖️ Top {top_n} — Distribusi Destinasi per {target_col}", fontsize=14, color="#cba6f7", pad=15)
            ax.set_xlabel("Jumlah Destinasi", fontsize=11, color="#89b4fa")
            ax.set_ylabel(target_col, fontsize=11, color="#89b4fa")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3, color="#45475a")
            ax.set_xlim(0, counts.max() * 1.15)

            fig.text(0.5, 0.01, f"Total destinasi: {len(df):,} | Unique {target_col}: {df[target_col].nunique()}",
                     ha="center", fontsize=9, color="#9399b2")

            return _save_figure_to_context("category-bar", {"column": category_column, "top_n": top_n})
        except Exception as e:
            plt.close()
            return {"error": f"Error membuat bar chart: {e}"}

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 4: Correlation Heatmap — korelasi antar kolom numerik
    # ──────────────────────────────────────────────────────────────────────────
    @tool(args_schema=PlotCorrelationInput)
    def plot_correlation_heatmap() -> dict:
        """
        Buat heatmap korelasi antar semua kolom numerik di dataset.
        Berguna untuk XAI: menunjukkan hubungan antara Rating, Harga, jumlah_rating, dll.
        Gunakan saat pengguna bertanya: 'Apakah ada korelasi antara harga dan rating?'
        atau 'Tampilkan heatmap korelasi dataset'.
        """
        file_contents = _read_current_csv()
        if not file_contents:
            return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                return {"error": "Tidak cukup kolom numerik untuk menghitung korelasi (minimal 2)."}

            corr = num_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))  # Tampilkan hanya segitiga bawah

            fig, ax = plt.subplots(figsize=(max(8, len(corr) * 1.2), max(6, len(corr) * 1.0)))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#2a2a3e")

            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, linecolor="#313244",
                ax=ax, annot_kws={"size": 9, "color": "#cdd6f4"},
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("🔥 Correlation Heatmap — Kolom Numerik Dataset Wisata Bali",
                         fontsize=14, color="#cba6f7", pad=15)
            ax.tick_params(colors="#cdd6f4")

            strong_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i):
                    val = corr.iloc[i, j]
                    if abs(val) >= 0.5:
                        strong_pairs.append(f"{corr.columns[i]} ↔ {corr.columns[j]}: {val:.2f}")

            summary = f"Korelasi kuat (|r|≥0.5): {', '.join(strong_pairs) if strong_pairs else 'tidak ada'}"
            fig.text(0.5, 0.0, summary, ha="center", fontsize=8.5, color="#9399b2", wrap=True)

            plt.tight_layout()
            return _save_figure_to_context("correlation-heatmap", {})
        except Exception as e:
            plt.close()
            return {"error": f"Error membuat heatmap: {e}"}

    # ──────────────────────────────────────────────────────────────────────────
    # TOOL 5: Pie Chart Budget Breakdown — alokasi budget per destinasi terpilih
    # ──────────────────────────────────────────────────────────────────────────
    @tool(args_schema=PlotBudgetBreakdownInput)
    def plot_budget_breakdown(selected_destinations: list[str]) -> dict:
        """
        [XAI Chart] Pie chart alokasi budget (harga tiket) per destinasi yang terpilih dari Optimizer.
        Tampilkan setelah Budget Optimizer selesai sebagai visualisasi komposisi pengeluaran wisata.
        Cocok untuk pertanyaan: 'Tampilkan breakdown budget itinerary saya'.
        """
        file_contents = _read_current_csv()
        if not file_contents:
            return {"error": "Tidak ada file CSV yang ditemukan."}
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}

            price_col = _find_col(df, 'price', 'harga', 'cost')
            name_col  = _find_col(df, 'place_name', 'name', 'nama', 'place', 'destination')

            if not price_col or not name_col:
                return {"error": "Kolom harga atau nama destinasi tidak ditemukan."}

            selected = df[df[name_col].isin(selected_destinations)][[name_col, price_col]].copy()
            selected[price_col] = pd.to_numeric(selected[price_col], errors="coerce").fillna(0)

            if selected.empty or selected[price_col].sum() == 0:
                return {"error": "Tidak ada data harga valid untuk destinasi yang dipilih."}

            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#1e1e2e")

            wedges, texts, autotexts = ax.pie(
                selected[price_col],
                labels=selected[name_col],
                autopct="%1.1f%%",
                colors=_ACCENT_COLORS * (len(selected) // len(_ACCENT_COLORS) + 1),
                startangle=140,
                wedgeprops={"edgecolor": "#1e1e2e", "linewidth": 2},
                pctdistance=0.82,
            )
            for text in texts:
                text.set_color("#cdd6f4")
                text.set_fontsize(9.5)
            for autotext in autotexts:
                autotext.set_color("#1e1e2e")
                autotext.set_fontsize(9)
                autotext.set_fontweight("bold")

            total = int(selected[price_col].sum())
            ax.set_title(
                f"💰 Budget Breakdown Itinerary\nTotal: Rp {total:,}",
                fontsize=14, color="#cba6f7", pad=20
            )

            # Legend dengan nilai rupiah
            legend_labels = [f"{row[name_col]}: Rp {int(row[price_col]):,}" for _, row in selected.iterrows()]
            ax.legend(legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                      ncol=2, fontsize=9, facecolor="#313244", labelcolor="#cdd6f4")

            return _save_figure_to_context("budget-breakdown", {"selected": selected_destinations})
        except Exception as e:
            plt.close()
            return {"error": f"Error membuat pie chart budget: {e}"}

    return [
        plot_itinerary_scatter,        # Tool 1 — scatter XAI (existing, refactored)
        plot_distribution_histogram,   # Tool 2 — histogram + KDE (NEW)
        plot_category_bar,             # Tool 3 — bar chart top kategori/kota (NEW)
        plot_correlation_heatmap,      # Tool 4 — heatmap korelasi (NEW)
        plot_budget_breakdown,         # Tool 5 — pie chart budget breakdown (NEW)
    ]