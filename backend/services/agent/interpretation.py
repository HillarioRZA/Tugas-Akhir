from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from backend.services.memory import persistent_memory
import json
import base64
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
load_dotenv()

# Fix C6: Model interpretasi dibaca dari ENV — ubah di .env tanpa menyentuh kode
# Default: openrouter/hunter-alpha via OpenRouter
_INTERP_MODEL = os.environ.get("INTERPRETATION_MODEL", "openrouter/hunter-alpha")
_BASE_URL     = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

llm = ChatOpenAI(
    base_url=_BASE_URL,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model=_INTERP_MODEL
)

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
                vision_llm = ChatGroq(model="llama-4-maverick-17b-128e-instruct")
                response = vision_llm.invoke([message])
            else:
                response = llm.invoke([message])

            return response.content

        # C4 Fix: bedakan tipe error agar lebih mudah di-debug
        except Exception as e:
            err_str = str(e).lower()
            if "api key" in err_str or "unauthorized" in err_str or "401" in err_str:
                return "[Error C4-AUTH] API key tidak valid atau tidak memiliki akses ke model vision ini."
            elif "timeout" in err_str or "timed out" in err_str or "connectionerror" in err_str:
                return "[Error C4-TIMEOUT] Koneksi ke model vision timeout. Coba lagi sebentar."
            elif "quota" in err_str or "rate limit" in err_str or "429" in err_str:
                return "[Error C4-RATELIMIT] Rate limit API tercapai. Tunggu beberapa detik dan coba lagi."
            elif "model" in err_str and ("not found" in err_str or "does not exist" in err_str):
                return f"[Error C4-MODEL] Model vision tidak ditemukan: {_INTERP_MODEL}. Periksa INTERPRETATION_MODEL di .env."
            else:
                return f"[Error C4-UNKNOWN] Gagal membaca gambar. Detail: {str(e)}"
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
