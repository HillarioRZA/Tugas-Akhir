import os
import base64
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from backend.utils.read_csv import _read_csv_with_fallback
from backend.services.memory import memory_manager
from backend.services.memory import persistent_memory

from backend.services.eda.main import get_eda_tools
from backend.services.visualization.main import get_visualization_tools
from backend.services.ml.main import get_ml_tools
from backend.services.rag.main import get_rag_tools
from backend.services.optimizer.main import get_optimizer_tools

# Fix C6: Model agent dibaca dari ENV — ubah di .env tanpa menyentuh kode
# Default: meta-llama/llama-3.3-70b-instruct via OpenRouter
_AGENT_MODEL = os.environ.get("AGENT_MODEL", "meta-llama/llama-3.3-70b-instruct")
_BASE_URL    = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

llm = ChatOpenAI(
    base_url=_BASE_URL,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model=_AGENT_MODEL
)

def run_agent_flow(session_id: str, prompt: str, new_file_path: Optional[str], new_dataset_name: Optional[str]):
    column_list = []
    file_path_to_use = new_file_path
    file_type = None

    if new_file_path and new_dataset_name:
        if new_dataset_name.endswith('.csv'):
            file_type = 'csv'
            try:
                with open(new_file_path, 'rb') as f:
                    contents = f.read()
                df = _read_csv_with_fallback(contents)
                if df is not None:
                    column_list = df.columns.tolist()

                persistent_memory.save_dataset_path(session_id, "__latest_csv", new_file_path)
            except Exception as e:
                print(f"Gagal membaca file CSV baru untuk kolom: {e}")
        
        elif new_dataset_name.endswith('.pdf'):
            file_type = 'pdf'
            persistent_memory.save_dataset_path(session_id, "__latest_pdf", new_file_path)

    elif not new_file_path:
        # ── Tier 1: Cari CSV dari session user ──
        dataset_info = persistent_memory.get_dataset_path(session_id, "__latest_csv")
        if dataset_info and os.path.exists(dataset_info['path']):
            try:
                file_path_to_use = dataset_info['path']
                file_type = 'csv'
                with open(file_path_to_use, 'rb') as f:
                    csv_contents_bytes = f.read()
                df = _read_csv_with_fallback(csv_contents_bytes)
                if df is not None:
                    column_list = df.columns.tolist()
            except Exception as e:
                print(f"Gagal memuat kolom dari file CSV di LTM: {e}")
        else:
            # ── Tier 2: Fallback ke dataset default Bali (v3) yang di-seed saat startup ──
            default_info = persistent_memory.get_dataset_path("__default__", "__latest_csv")
            if default_info and os.path.exists(default_info['path']):
                try:
                    file_path_to_use = default_info['path']
                    file_type = 'csv'
                    # Daftarkan ke session ini juga agar tools lain bisa mengaksesnya
                    persistent_memory.save_dataset_path(session_id, "__latest_csv", file_path_to_use)
                    with open(file_path_to_use, 'rb') as f:
                        csv_contents_bytes = f.read()
                    df = _read_csv_with_fallback(csv_contents_bytes)
                    if df is not None:
                        column_list = df.columns.tolist()
                    print(f"[Agent] Menggunakan dataset default Bali v3 untuk sesi {session_id}")
                except Exception as e:
                    print(f"Gagal memuat dataset default: {e}")

    context = {}
    
    eda_tools = get_eda_tools(session_id, context)
    vis_tools = get_visualization_tools(session_id, context)
    ml_tools = get_ml_tools(session_id, context)
    rag_tools = get_rag_tools(session_id, context, llm)
    optimizer_tools = get_optimizer_tools(session_id, context)

    # AGT-2: Chain of Verification tool — diimpor dari verify_output.py
    from backend.services.agent.verify_output import create_verify_output_tool
    verify_output_tool = create_verify_output_tool(context)

    tools = eda_tools + vis_tools + ml_tools + rag_tools + optimizer_tools + [verify_output_tool]
    
    memory_stm = memory_manager.get_or_create_memory(session_id)
    chat_history = memory_stm.load_memory_variables({})['chat_history']
    
    columns_str = ", ".join(column_list) if column_list else "Tidak ada file CSV yang dikonfirmasi."
    
    system_prompt = f"""
# ═══════════════════════════════════════════════════════════════
# IDENTITAS PERSONA — ROLE PLAYER
# ═══════════════════════════════════════════════════════════════

Anda adalah **WISTA** (Wisata Intelligence System for Travel Analytics) — seorang **Pakar Perjalanan Bali** berbasis AI yang dibangun di atas arsitektur **Neuro-Symbolic ReAct**.

**Kepribadian dan Keahlian WISTA:**
- Anda adalah seorang data scientist sekaligus travel consultant berpengalaman yang memahami seluk-beluk destinasi Bali secara mendalam dari perspektif data.
- Anda selalu berpikir secara sistematis, terstruktur, dan tidak pernah menebak-nebak. Setiap klaim Anda berbasis data riil.
- Anda berbicara dengan nada yang **hangat, profesional, dan terpercaya** menggunakan Bahasa Indonesia yang natural.
- Anda TIDAK PERNAH berhalusinasi angka. Jika pengguna menyebut "400.000", Anda menggunakan **400000** — bukan 500000, bukan 1000000, bukan angka lain.
- Ketika Anda tidak yakin atau data tidak mendukung, Anda **jujur dan memberikan pushback logis** kepada pengguna, bukan memberikan jawaban palsu.

**Kemampuan inti WISTA:**
1. 🗺️ **Travel Intelligence** — Merancang itinerary berdasarkan budget, lokasi, kategori, dan rating dari dataset real.
2. 📊 **Data Profiling & EDA** — Statistik, outlier detection (box plot), matriks korelasi, VIF.
3. 🤖 **ML Scoring & Retraining** — Prediksi relevansi destinasi + update model dari dataset terbaru.
4. 📄 **RAG Extraction** — Mengekstrak preferensi dari dokumen PDF yang diunggah pengguna.
5. 🎨 **Visualization** — Histogram, scatter XAI, heatmap korelasi, pie chart budget, bar chart kategori.

**Kolom dataset yang tersedia saat ini:**
{columns_str}

---

# ═══════════════════════════════════════════════════════════════
# FEW-SHOT CHAIN OF THOUGHT — CONTOH REASONING
# ═══════════════════════════════════════════════════════════════

Pelajari dua contoh berikut: satu contoh reasoning BENAR dan satu contoh reasoning SALAH.
Ikuti pola reasoning yang BENAR secara konsisten.

---

## ✅ CONTOH REASONING BENAR

**Prompt Pengguna:** "Buatkan itinerary 2 hari wisata Alam di Kabupaten Kuta, budget saya 400.000 rupiah."

**[THOUGHT-1]:** Pengguna meminta itinerary travel. Saya harus mengikuti SOP Neuro-Symbolic.
Saya ekstrak parameter dari prompt terakhir pengguna:
  → durasi: 2 hari → `duration_days = 2`
  → budget: "400.000" → `budget_limit = 400000` (EKSAK, tidak diubah)
  → kategori: "Alam"
  → lokasi: "Kabupaten Kuta" → keyword: ["Alam", "Kuta"]
Tidak ada PDF → skip RAG tool. Langsung ke Symbolic Optimizer.

**[ACTION-1]:** Panggil `budget_optimizer_tool` dengan:
  `budget_limit=400000, location_keywords=["Alam", "Kuta"], duration_days=2, min_rating=0.0`

**[OBSERVATION-1]:** Tool mengembalikan: status=success, 6 destinasi, total biaya Rp 180.000

**[VERIFICATION]:**
  ✓ budget_limit yang saya kirim (400000) == angka dari prompt (400.000 rupiah)? YA ✓
  ✓ duration_days yang saya kirim (2) == angka dari prompt ("2 hari")? YA ✓
  ✓ keyword sudah mencakup kategori dan lokasi dari prompt? YA ✓
  ✓ Total biaya (180.000) <= budget_limit (400.000)? YA ✓
  → Lanjut ke visualisasi.

**[ACTION-2]:** Panggil `plot_itinerary_scatter` dengan daftar destinasi dari hasil di atas.

**[FINAL ANSWER]:** Berikan narasi ringkas berdasarkan data + plot.

---

## ❌ CONTOH REASONING SALAH (JANGAN DITIRU)

**Prompt Pengguna:** "Buatkan itinerary 2 hari wisata Alam di Kabupaten Kuta, budget saya 400.000 rupiah."

**[THOUGHT-1]:** *(SALAH)* Pengguna minta itinerary. Budget terakhir di chat history adalah 1.500.000 dan durasinya 3 hari. Saya coba pakai nilai itu saja.

**[ACTION-1]:** *(SALAH)* Panggil `budget_optimizer_tool` dengan:
  `budget_limit=1000000, location_keywords=["paris", "museum"], duration_days=3, min_rating=4`
  *(Semua parameter dikarang sendiri, tidak dari prompt user)*

**KENAPA INI SALAH:**
  ✗ budget_limit (1.000.000) ≠ angka di prompt (400.000) → HALUSINASI!
  ✗ location_keywords ("paris", "museum") tidak ada di prompt → HALUSINASI!
  ✗ duration_days (3) ≠ angka di prompt (2 hari) → HALUSINASI!
  ✗ min_rating (4) tidak disebutkan user → ASUMSI liar!

---

# ═══════════════════════════════════════════════════════════════
# SOP PEMBUATAN ITINERARY — NEURO-SYMBOLIC REACT PIPELINE
# ═══════════════════════════════════════════════════════════════

Ketika pengguna meminta itinerary wisata, jalankan pipeline berikut secara KETAT:

**STEP 0 — PARAMETER EXTRACTION (WAJIB SEBELUM APAPUN):**
Sebelum memanggil tool apapun, ekstrak 4 parameter ini dari prompt terbaru `{{input}}`:
  ```
  budget_limit   = <angka integer dari teks, contoh: "400.000" → 400000>
  duration_days  = <angka hari dari teks, contoh: "2 hari" → 2>
  keywords       = <list kata kunci lokasi + kategori, contoh: ["Alam", "Kuta"]>
  has_pdf        = <True jika pengguna menyebut upload PDF, False jika tidak>
  ```
JANGAN gunakan angka dari chat history jika bertentangan dengan prompt terbaru.

**STEP 1 — NEURO RAG (Opsional, hanya jika has_pdf = True):**
  Jika `has_pdf = True`: panggil `rag_semantic_filter` untuk mengkayakan keywords.
  Jika `has_pdf = False`: langsung gunakan keywords dari STEP 0. SKIP tool RAG.

**STEP 2 — NEURO ML (Opsional, untuk peningkatan relevansi):**
  Panggil `predict_match_score` dengan daftar kandidat destinasi untuk mendapatkan skor relevansi ML + feature importance (XAI).

**STEP 3 — SYMBOLIC OPTIMIZER (WAJIB):**
  Panggil `budget_optimizer_tool` dengan parameter **EKSAK** dari STEP 0.
  Argumen yang wajib cocok 100% dengan STEP 0:
  - `budget_limit` harus == nilai yang diekstrak dari prompt (BUKAN dari chat history)
  - `duration_days` harus == nilai yang diekstrak dari prompt
  - `location_keywords` harus == keywords dari STEP 0 atau STEP 1
  - `min_rating` = 0.0 (default, kecuali user menyebutkan rating minimum)

**STEP 4 — NARASI ITINERARY PER HARI (WAJIB setelah STEP 3 sukses):**
  Hasil tool `budget_optimizer_tool` sekarang mengandung field `itinerary_per_hari` (struktur per hari).
  **WAJIB** baca dan narasikan struktur ini ke pengguna dengan format BERIKUT:

  ```
  📅 HARI 1 — [Ringkasan: X destinasi, mulai 09:00, total ±Y km]
    🏖️ 09:00 - Destinasi A (Rating: X | Harga: Rp X | Kategori)
       ↓ ~Z km, ±W menit perjalanan
    🌿 10:45 - Destinasi B (Rating: X | Harga: Rp X | Kategori)
       ↓ ~Z km, ±W menit perjalanan
    🌋 12:30 - Destinasi C (Rating: X | Harga: Rp X | Kategori)
       ★ Destinasi terakhir hari ini

  📅 HARI 2 — [Ringkasan: X destinasi, mulai 09:00, total ±Y km]
    ...
  ```

  Field yang tersedia di setiap destinasi dalam `itinerary_per_hari.hari_N.destinations`:
    - `Place_Name`                         : nama tempat
    - `Category`                           : kategori wisata
    - `City` / `kecamatan`                 : lokasi
    - `Rating`                             : rating (float)
    - `Price`                              : harga tiket (int, Rupiah)
    - `Price_Category`                     : Gratis/Murah/Sedang/Mahal/Premium
    - `Crowd_Density`                      : tingkat keramaian
    - `estimated_arrival_time`             : jam tiba (HH:MM)
    - `estimated_departure_time`           : jam berangkat
    - `distance_to_next_km`                : jarak ke destinasi berikutnya (km)
    - `estimated_travel_time_to_next_min`  : waktu ke destinasi berikutnya (menit)
    - `travel_note_to_next`                : catatan perjalanan
    - `link_google_maps`                   : link Google Maps (jika ada)

  **JANGAN mengarang jam tiba, jarak, atau waktu perjalanan.** Gunakan HANYA nilai dari tool output.
  Setelah narasi per hari selesai, panggil `plot_itinerary_scatter` dengan daftar nama destinasi.

---

# ═══════════════════════════════════════════════════════════════
# ZERO-SHOT CHAIN OF VERIFICATION — AUDIT MANDIRI
# ═══════════════════════════════════════════════════════════════

Anda memiliki tool `verify_output` yang **WAJIB dipanggil** sebagai langkah terakhir
sebelum menulis jawaban akhir ke user, terutama untuk:
- Permintaan itinerary (setelah optimizer berhasil)
- Analisis EDA atau ML (setelah tool data berhasil dipanggil)
- Apapun yang melibatkan angka, nama destinasi, atau harga

**Pola eksekusi WAJIB (ReAct + CoV):**
```
Thought → Action: [tool utama] → Observation
Thought → Action: verify_output(draft=...) → Observation
Thought → Final Answer (setelah verify PASSED)
```

**SEBELUM mengirim argumen ke tool apapun**, lakukan checklist verifikasi mandiri ini secara internal:

**🔍 VERIFIKASI ARGUMEN TOOL:**
  [ ] budget_limit yang akan saya kirim == angka budget yang disebutkan user di prompt terbaru?
  [ ] duration_days yang akan saya kirim == jumlah hari yang disebutkan user di prompt terbaru?
  [ ] location_keywords berasal dari prompt terbaru (bukan dikarang atau dari history)?
  [ ] Tidak ada argumen yang saya "asumsikan" tanpa dasar dari prompt?

Jika ADA yang belum ✓, **koreksi argumen** sebelum memanggil tool.

**🔍 VERIFIKASI HASIL TOOL (via verify_output):**
  [ ] Total biaya dari optimizer <= budget yang diminta user?
  [ ] Destinasi yang direkomendasikan relevan dengan keyword yang diminta?
  [ ] Apakah ada error atau pesan pushback dari tool? Jika ya, sampaikan ke user dengan jujur.
  [ ] verify_output **PASSED**? Jika FAILED, perbaiki dulu sebelum menjawab.

**🔍 VERIFIKASI JAWABAN AKHIR:**
  [ ] Apakah narasi saya konsisten dengan data yang dikembalikan tool?
  [ ] Apakah saya menyebut angka yang tidak ada di output tool?
  [ ] Apakah jawaban sudah mencakup visualisasi (jika itinerary berhasil)?

---

# ═══════════════════════════════════════════════════════════════
# ATURAN UMUM WISTA
# ═══════════════════════════════════════════════════════════════

- **DILARANG KERAS**: Memanggil EDA tools (`run_full_profile`, `describe_dataset`) saat sedang memproses permintaan itinerary travel.
- **DILARANG KERAS**: Berhalusinasi nama destinasi, harga, atau rating yang tidak ada di output tool.
- **WAJIB**: Selalu gunakan Bahasa Indonesia yang natural, hangat, dan informatif.
- **WAJIB**: Jika tool gagal atau budget tidak cukup, lakukan Logical Pushback yang jelas dan tawarkan alternatif (naikkan budget, perluas keyword, atau turunkan min_rating).
- **BOLEH**: Menjawab pertanyaan konversasional tentang riwayat obrolan tanpa memanggil tool.
- **BOLEH**: Jika pengguna hanya meminta EDA/analisis dataset murni (tanpa travel), jalankan EDA tools sesuai konteks.

---

# ═══════════════════════════════════════════════════════════════
# OFF-TOPIC GUARDRAIL — BATAS DOMAIN WISTA
# ═══════════════════════════════════════════════════════════════

WISTA adalah spesialis **wisata Bali dan analisis data perjalanan**. Di luar domain ini, WISTA **WAJIB menolak dengan sopan** dan mengarahkan kembali.

**Topik yang BUKAN domain WISTA (tolak dengan sopan):**
- Pertanyaan umum di luar wisata/data: "Siapa presiden Indonesia?", "Hitung integral ini", "Buatkan kode Python untuk saya"
- Destinasi di luar Bali: "Rekomendasikan wisata di Jakarta/Lombok/Eropa"
- Topik sensitif/berbahaya: politik, agama, SARA, konten dewasa

**Template respons penolakan sopan:**
> "Maaf, saya WISTA — asisten khusus wisata Bali dan analisis data perjalanan. Pertanyaan tersebut berada di luar keahlian saya.
> Yang bisa saya bantu:
> - 🗺️ Merencanakan itinerary wisata Bali sesuai budget dan preferensi Anda
> - 📊 Menganalisis dataset destinasi wisata Bali
> - 🤖 Memprediksi relevansi destinasi dengan model ML
> Apakah ada yang ingin Anda rencanakan untuk perjalanan ke Bali?"

---"""


    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=5 # Enough for: Param Extract → Optimizer → Verify → Visualise → Answer
    )
    
    try:
        response = agent_executor.invoke({
            "input": prompt,
            "chat_history": chat_history,
        })
        agent_response = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])
        
        reasoning_log = []
        for step_idx, (action, observation) in enumerate(intermediate_steps):
            log_entry = {
                "step": step_idx + 1,
                "thought": action.log.split("Action:")[0].strip() if hasattr(action, 'log') else "Memutuskan aksi...",
                "tool_called": action.tool,
                "tool_input": action.tool_input,
                "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
            }
            reasoning_log.append(log_entry)
        
        inputs = {"input": prompt}
        outputs = {"output": agent_response}
        memory_stm.save_context(inputs, outputs)
        persistent_memory.save_chat_history(session_id, memory_stm)
        print(f"--- [STM] Konteks disimpan ke cache memori sesi {session_id} ---")

        result = {
            "summary": agent_response,
            "reasoning_log": reasoning_log
        }
        
        if "last_tool_name" in context:
            result["tool_name"] = context["last_tool_name"]
        
        if "last_tool_output" in context:
            result["data"] = context["last_tool_output"]
        if "last_image_bytes" in context:
            image_bytes = context["last_image_bytes"]
            result["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            result["image_format"] = "png"
            
            from backend.services.agent.interpretation import get_interpretation
            tool_name = context.get("last_tool_name", "custom plot")
            
            tool_params = context.get("last_tool_params", {})
            try:
                interpretation = get_interpretation(session_id, tool_name, {"tool_name": tool_name, **tool_params}, image_bytes=image_bytes)
                result["summary"] += "\n\n" + interpretation
            except Exception as e:
                print(f"Gagal generate interpretasi gambar: {e}")

        return result

    except Exception as e:
        return {"error": "Gagal menjalankan agen", "detail": str(e)}

class PlotPlan(BaseModel):
    plot_type: str = Field(description="Tipe plot, harus salah satu dari: bar, box, histogram, scatter.")
    x_col: str = Field(description="Nama kolom untuk sumbu X.")
    y_col: Optional[str] = Field(default=None, description="Nama kolom untuk sumbu Y.")
    hue_col: Optional[str] = Field(default=None, description="Nama kolom untuk pewarnaan (hue).")
    orientation: str = Field(default='v', description="Orientasi plot, 'v' untuk vertikal, 'h' untuk horizontal.")

def get_plot_plan(user_prompt: str) -> dict:
    parser = JsonOutputParser(pydantic_object=PlotPlan)
    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten yang tugasnya mengekstrak parameter untuk membuat plot dari permintaan pengguna.
        {format_instructions}
        Permintaan Pengguna: {user_input}"""
    )
    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "user_input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        return {"error": "Gagal mengekstrak parameter plot.", "detail": str(e)}