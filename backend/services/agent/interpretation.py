from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import base64
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

_INTERP_PROVIDER = "google".lower()
_INTERP_MODEL    = os.environ.get("INTERPRETATION_MODEL")

def _build_interp_llm():
    provider = _INTERP_PROVIDER

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=_INTERP_MODEL,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0,
        )
    elif provider == "groq":
        return ChatGroq(
            model=_INTERP_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0,
        )
    else:
        return ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model=_INTERP_MODEL,
            temperature=0,
        )

llm = _build_interp_llm()
print(f"🖼️  [Interp LLM] Provider={_INTERP_PROVIDER} | Model={_INTERP_MODEL}")

_VISION_PROMPTS = {
    "itinerary-visualization": (
        "Anda adalah AI travel advisor yang menganalisis scatter plot XAI (Explainable AI) wisata Bali.\n\n"
        "Gambar ini menunjukkan scatter plot Harga vs Rating seluruh destinasi wisata Bali. "
        "Titik berwarna ungu dengan lingkaran adalah destinasi yang TERPILIH oleh optimizer, "
        "sedangkan titik abu-abu adalah destinasi lainnya.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Dimana posisi destinasi terpilih dibanding keseluruhan data? (kuadran mana: murah+bagus, mahal+bagus, dll)\n"
        "2. Apakah optimizer sudah memilih destinasi dengan value terbaik (rating tinggi, harga wajar)?\n"
        "3. Adakah destinasi alternatif yang terlihat menarik tapi tidak terpilih?\n"
        "4. Berikan 1 insight menarik dari pola sebaran data."
    ),

    "distribution-histogram": (
        "Anda adalah AI data analyst yang menganalisis histogram distribusi data wisata Bali.\n\n"
        "Gambar ini menunjukkan histogram dengan KDE (Kernel Density Estimation), "
        "garis mean (hijau), dan garis median (oranye). "
        "Statistik ringkas (Min, Max, Std, Skew) ditampilkan di bawah chart.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Bagaimana bentuk distribusi? (normal, right-skewed, left-skewed, bimodal)\n"
        "2. Apakah ada gap yang besar antara mean dan median? Apa artinya?\n"
        "3. Apakah ada konsentrasi data di rentang tertentu?\n"
        "4. Jika distribusi skewed, apa implikasinya untuk analisis wisata?"
    ),

    "category-bar": (
        "Anda adalah AI data analyst yang menganalisis bar chart kategori wisata Bali.\n\n"
        "Gambar ini menunjukkan jumlah destinasi per kategori/kota, diurutkan dari terbanyak.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Kategori/kota mana yang mendominasi? Berapa persentasenya dari total?\n"
        "2. Apakah ada ketimpangan signifikan antar kategori?\n"
        "3. Kategori mana yang paling sedikit? Apakah ini peluang atau keterbatasan data?\n"
        "4. Berikan 1 rekomendasi berdasarkan distribusi ini."
    ),

    "correlation-heatmap": (
        "Anda adalah AI data analyst yang menganalisis heatmap korelasi data wisata Bali.\n\n"
        "Gambar ini menunjukkan matriks korelasi Pearson antar kolom numerik. "
        "Warna merah = korelasi positif kuat, biru = korelasi negatif kuat, putih = tidak berkorelasi. "
        "Korelasi kuat (|r|≥0.5) disebutkan di bawah chart.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Pasangan fitur mana yang paling berkorelasi? Jelaskan arti korelasinya.\n"
        "2. Apakah ada multikolinearitas yang perlu diperhatikan untuk model ML?\n"
        "3. Apakah ada temuan kontra-intuitif? (misal: harga TIDAK berkorelasi dengan rating)\n"
        "4. Berikan rekomendasi untuk feature engineering atau seleksi fitur."
    ),

    "budget-breakdown": (
        "Anda adalah AI travel planner yang menganalisis pie chart budget wisata Bali.\n\n"
        "Gambar ini menunjukkan proporsi alokasi biaya tiket masuk per destinasi yang terpilih. "
        "Total biaya ditampilkan di judul chart, dan nilai Rupiah per destinasi di legenda.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Destinasi mana yang menghabiskan proporsi budget terbesar? Berapa persen?\n"
        "2. Apakah alokasi budget sudah seimbang atau terlalu berat di satu destinasi?\n"
        "3. Apakah ada destinasi gratis/sangat murah yang memberikan value bagus?\n"
        "4. Berikan tips penghematan atau optimasi budget."
    ),

    "outlier-boxplot": (
        "Anda adalah AI data analyst yang menganalisis box plot outlier data wisata Bali.\n\n"
        "Gambar ini menunjukkan box plot untuk kolom-kolom numerik yang terdeteksi memiliki outlier. "
        "Garis hijau = batas bawah IQR, garis ungu = batas atas IQR, titik oranye = outlier.\n\n"
        "Jelaskan dalam Bahasa Indonesia:\n"
        "1. Kolom mana yang paling banyak outlier? Apa penyebabnya?\n"
        "2. Apakah outlier ini masuk akal dalam konteks wisata? (misal: destinasi premium sangat mahal)\n"
        "3. Apakah perlu dilakukan penanganan outlier (capping, removal, atau biarkan)?\n"
        "4. Berikan rekomendasi spesifik."
    ),
}

_DEFAULT_VISION_PROMPT = (
    "Anda adalah AI data analyst yang menganalisis visualisasi data wisata Bali.\n\n"
    "Jelaskan dalam Bahasa Indonesia:\n"
    "1. Apa yang ditunjukkan oleh chart/grafik ini?\n"
    "2. Apa insight atau pola utama yang terlihat?\n"
    "3. Apakah ada anomali atau temuan menarik?\n"
    "4. Berikan 1 rekomendasi berdasarkan visualisasi ini."
)

def get_interpretation(
    session_id: str,
    tool_name: str,
    tool_output,
    image_bytes: Optional[bytes] = None,
    baseline_metrics: Optional[dict] = None,
) -> str:
    """
    Interpretasi visual chart menggunakan Vision LLM.
    Dipanggil oleh main.py HANYA ketika tool menghasilkan gambar
    (context["last_image_bytes"] ada). Mengirim gambar ke Vision LLM
    dan mengembalikan narasi insight dalam Bahasa Indonesia.
    Args:
        session_id: ID sesi pengguna
        tool_name: Nama chart/tool (key di _VISION_PROMPTS)
        tool_output: Output data tool (untuk konteks tambahan)
        image_bytes: Bytes gambar PNG dari matplotlib
        baseline_metrics: Tidak digunakan (legacy parameter)
    Returns:
        String narasi interpretasi dari Vision LLM
    """
    if not image_bytes:
        return "[Skipped] Tidak ada gambar untuk diinterpretasi."

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    vision_prompt = _VISION_PROMPTS.get(tool_name, _DEFAULT_VISION_PROMPT)

    if _INTERP_PROVIDER == "google":
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
        )
    elif _INTERP_PROVIDER == "groq":
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
        )
    else:
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]
        )

    try:
        if _INTERP_PROVIDER == "groq":
            _groq_vision = os.environ.get("GROQ_VISION_MODEL", "llama-4-scout-17b-16e-instruct")
            vision_llm = ChatGroq(model=_groq_vision, temperature=0)
            print(f"🖼️  [Groq Vision] Switching to {_groq_vision} for image interpretation")
            response = vision_llm.invoke([message])
        else:
            response = llm.invoke([message])

        return response.content

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
