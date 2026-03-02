import os
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory

def generate_custom_plot(
    file_contents: bytes,
    plot_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    hue_col: Optional[str] = None,
    orientation: str = 'v'
):
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None:
            return "error: Gagal membaca file CSV."

        required_cols = [c for c in [x_col, y_col, hue_col] if c is not None]
        for col in required_cols:
            if col not in df.columns:
                return f"error: Kolom '{col}' tidak ditemukan di dalam dataset."

        is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        is_y_numeric = y_col and pd.api.types.is_numeric_dtype(df[y_col])

        if plot_type in ['scatter', 'box'] and not y_col:
            return f"error: Plot tipe '{plot_type}' membutuhkan kolom sumbu Y (y_col)."
        if plot_type == 'scatter' and not (is_x_numeric and is_y_numeric):
            return "error: Scatter plot membutuhkan kolom X dan Y yang keduanya numerik."
        if plot_type == 'box' and not is_y_numeric:
             return "error: Box plot membutuhkan kolom Y yang numerik."

        plt.figure(figsize=(12, 8))
        plot_kwargs = {'data': df, 'x': x_col, 'y': y_col, 'hue': hue_col}

        if orientation == 'h' and plot_type in ['bar', 'box']:
            plot_kwargs['x'], plot_kwargs['y'] = y_col, x_col
            plot_kwargs['orient'] = 'h'

        if plot_type == 'histogram':
            sns.histplot(data=df, x=x_col, hue=hue_col, kde=True)
        elif plot_type == 'bar':
            if y_col is None:
                 sns.countplot(**{k: v for k, v in plot_kwargs.items() if k != 'y'})
            else:
                 sns.barplot(**plot_kwargs)
        elif plot_type == 'box':
            sns.boxplot(**plot_kwargs)
        elif plot_type == 'scatter':
            sns.scatterplot(**plot_kwargs)
        else:
            return f"error: Tipe plot '{plot_type}' tidak didukung."

        title = f'{plot_type.capitalize()} of {x_col}'
        if y_col: title += f' by {y_col}'
        if hue_col: title += f' (colored by {hue_col})'

        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()

    except Exception as e:
        return f"error: Terjadi kesalahan saat membuat plot - {str(e)}"

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



    return [
        describe_dataset,
        run_full_profile
    ]