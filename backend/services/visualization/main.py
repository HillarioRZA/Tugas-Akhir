import os
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import persistent_memory

class PlotItineraryInput(BaseModel):
    selected_destinations: list[str] = Field(description="Daftar nama destinasi wisata yang terpilih/direkomendasikan dari Optimizer.")


def get_visualization_tools(session_id: str, context: dict) -> List[Any]:
    def _read_current_csv():
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            with open(dataset_info['path'], 'rb') as f:
                return f.read()
        return None



    @tool(args_schema=PlotItineraryInput)
    def plot_itinerary_scatter(selected_destinations: list[str]) -> dict:
        """
        Tool Explainable AI eksklusif untuk 'Travel Itinerary'.
        Alat ini menggambar scatter plot (Harga vs Rating) dari seluruh destinasi. 
        Destinasi yang terpilih dilingkari warna cerah, yang tidak terpilih diberi warna abu-abu.
        WAJIB memanggil ini setelah mendapatkan finalisasi Itinerary dari Budget Optimizer.
        """
        file_contents = _read_current_csv()
        if not file_contents: return {"error": "Tidak ada file CSV yang ditemukan."}
        
        try:
            df = _read_csv_with_fallback(file_contents)
            if df is None: return {"error": "Gagal membaca file CSV."}
            
            # Cari kolom harga dan rating
            price_cols = [col for col in df.columns if 'price' in col.lower() or 'harga' in col.lower() or 'cost' in col.lower()]
            rating_cols = [col for col in df.columns if 'rating' in col.lower()]
            name_cols = [col for col in df.columns if 'name' in col.lower() or 'nama' in col.lower() or 'place' in col.lower() or 'destination' in col.lower()]

            if not price_cols or not rating_cols or not name_cols:
                 return {"error": "Gagal menemukan kolom harga, rating, atau nama destinasi pada dataset untuk plotting."}
                 
            price_col = price_cols[0]
            rating_col = rating_cols[0]
            name_col = name_cols[0]

            plt.figure(figsize=(12, 8))
            
            # Plot data yang tidak terpilih (abu-abu, transparan)
            unselected_df = df[~df[name_col].isin(selected_destinations)]
            sns.scatterplot(data=unselected_df, x=price_col, y=rating_col, color='grey', alpha=0.3, label='Other Destinations')
            
            # Plot data yang terpilih (warna mencolok, solid)
            selected_df = df[df[name_col].isin(selected_destinations)]
            if not selected_df.empty:
                sns.scatterplot(data=selected_df, x=price_col, y=rating_col, color='red', s=100, edgecolor='black', zorder=5, label='Recommended Itinerary')
                
                # Tambahkan label nama hanya untuk yang terpilih agar tidak penuh
                for _, row in selected_df.iterrows():
                    plt.text(row[price_col], row[rating_col] + 0.05, row[name_col], fontsize=9, ha='center')

            plt.title('Explainable AI: Kenapa destinasi ini dipilih? (Rating vs Harga)', fontsize=14)
            plt.xlabel(f'Price ({price_col})', fontsize=12)
            plt.ylabel(f'Rating ({rating_col})', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            image_bytes = img_buffer.getvalue()

            context["last_image_bytes"] = image_bytes
            context["last_tool_name"] = "itinerary-visualization"
            context["last_tool_params"] = {"selected_destinations": selected_destinations}
            
            return {
                "status": "success", 
                "summary": "Berhasil memplot Scatter Plot Itinerary sebagai XAI.", 
                "instruction": "Kirimkan narasi bahwa gambar plot telah berhasil digambar sebagai justifikasi pilihan itinerary."
            }
        except Exception as e:
             return {"error": f"Error membuat plot itinerary: {e}"}

    return [
        plot_itinerary_scatter
    ]