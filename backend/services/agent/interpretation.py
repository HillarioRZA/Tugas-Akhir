from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from backend.services.memory import persistent_memory
import json
import base64
from typing import Optional,List,Dict
from dotenv import load_dotenv
load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(model="openai/gpt-oss-120b")

def get_interpretation(session_id: str,tool_name: str, tool_output, image_bytes: Optional[bytes] = None,  baseline_metrics: Optional[dict] = None) -> str:
    """Fungsi interpretasi universal untuk data dan gambar."""
    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Fallback to an older vision LLM temporarily if using Groq
        # Groq's qwen model requires a specifically formatted base64 url
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Anda adalah AI data analyst. Jelaskan insight utama dari gambar {tool_name} ini. Jika ada anomali atau pola menarik, sebutkan. Berikan juga rekomendasi langkah selanjutnya berdasarkan visualisasi ini."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
        
        try:
            if isinstance(llm, ChatGroq):
                # Khusus untuk Groq, model teks biasa tidak mendukung vision
                # Kita harus override sementara menggunakan model vision Groq (Llama-3.2)
                vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview")
                response = vision_llm.invoke([message])
            else:
                # Jika menggunakan Gemini (ChatGoogleGenerativeAI), model utamanya sudah natively support vision
                response = llm.invoke([message])
                
            return response.content
        except Exception as e:
            return f"Gagal membaca gambar secara spesifik. Sistem Vision saat ini tidak mendukung pembacaan grafik. Error teknis: {str(e)}"
    else:
        output_str = json.dumps(tool_output, indent=2)

        baseline_str = json.dumps(baseline_metrics, indent=2) if baseline_metrics else "Tidak ada data baseline."

        if baseline_metrics:
            baseline_str = json.dumps(baseline_metrics, indent=2, default=str)

        prompt_templates = {
            "full-profile": """Anda adalah AI data analyst yang sedang menyajikan temuan utama kepada klien.
            Berdasarkan data profil berikut: {data}.
            Buatlah sebuah ringkasan naratif ('Cerita dari Data'). Sorot 3-4 poin paling krusial seperti masalah kualitas data (missing values), fitur yang distribusinya paling aneh (skewed), dan temuan menarik lainnya.
            Akhiri dengan memberikan 2 rekomendasi konkret untuk langkah analisis selanjutnya.""",
            "skewness": """Berikut adalah data skewness: {data}.
            Identifikasi kolom dengan skewness paling tinggi (positif atau negatif). Jelaskan artinya secara sederhana.
            Jika ada kolom dengan skewness di luar rentang -1 dan 1, berikan rekomendasi spesifik (misal: 'pertimbangkan transformasi logaritma pada kolom X').""",
            "vif": """Berikut adalah hasil perhitungan VIF: {data}.
            Jelaskan apa itu VIF secara singkat. Identifikasi fitur dengan VIF di atas 10 (jika ada) dan jelaskan mengapa ini bisa menjadi masalah (multikolinearitas).
            Berikan rekomendasi yang jelas, seperti 'pertimbangkan untuk menghapus fitur X atau menggabungkannya dengan fitur lain'.""",
            "run_ml_pipeline": """Anda adalah seorang AI data scientist yang sedang melaporkan hasil pemodelan kepada manajer.
            Berikut adalah laporan metrik evaluasi dari model yang baru dilatih: {data}.
            Jelaskan arti dari hasil ini dalam beberapa poin:
            1.  Sebutkan tipe masalah dan model yang digunakan.
            2.  Jelaskan metrik utamanya (Akurasi untuk klasifikasi, R2 Score untuk regresi) dengan bahasa yang mudah dimengerti.
            3.  Berikan kesimpulan akhir tentang seberapa baik performa model tersebut.""",
           "run_tuned_ml_pipeline": """Anda adalah AI data scientist yang melaporkan hasil tuning.
            
            Ini adalah metrik dari model BARU yang sudah di-tuning:
            {data}
            
            Ini adalah metrik dari model SEBELUMNYA (baseline):
            {baseline_data}
            
            Jelaskan hasilnya:
            1. Sebutkan model dan parameter terbaik yang ditemukan.
            2. Bandingkan metrik utamanya (Akurasi/R2 Score) dari model BARU dengan model BASELINE.
            3. Jelaskan dengan angka spesifik (misal: "ada peningkatan akurasi sebesar 1.5%")
            4. Berikan kesimpulan apakah tuning ini berhasil.""",
            
            "get_feature_importance": """Anda adalah seorang analis yang menjelaskan faktor-faktor penting kepada klien. Berikut adalah daftar fitur paling berpengaruh dalam model: {data}.
            
            Sebutkan 3 fitur teratas dan jelaskan kemungkinan artinya dalam konteks bisnis secara sederhana. Contoh: 'Fitur 'masa_berlangganan' paling penting, artinya semakin lama seseorang berlangganan, semakin besar pengaruhnya terhadap prediksi.'"""
        }
        
        template = prompt_templates.get(tool_name, "Berikut adalah hasil analisis: {data}. Ringkas hasilnya.")
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        return chain.invoke({"data": output_str, "baseline_data": baseline_str}).content
