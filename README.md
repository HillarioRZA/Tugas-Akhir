<h1 align="center">🧠 PROJECT TA</h1>
<h3 align="center">Dashboard AI Data Scientist Menggunakan Reasoning dan Multi-Tools untuk Analisis Data</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LLM-Multi--LLM_Agnostic-4285F4?logo=google&logoColor=white" alt="Multi-LLM"/>
  <img src="https://img.shields.io/badge/LangChain-ReAct_Agent-1C3C3C?logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/License-Academic-orange" alt="License"/>
  <img src="https://img.shields.io/badge/Tests-16%2F16_Passed_(100%25)-brightgreen" alt="Tests"/>
</p>

---

## 📋 Daftar Isi

- [Ringkasan Proyek](#-ringkasan-proyek)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Fitur Utama](#-fitur-utama)
- [Tech Stack](#-tech-stack)
- [Struktur Direktori](#-struktur-direktori)
- [Prasyarat](#-prasyarat)
- [Instalasi & Setup](#-instalasi--setup)
- [Menjalankan Aplikasi](#-menjalankan-aplikasi)
- [Multi-Tools Registry](#-multi-tools-registry)
- [Modul Backend Detail](#-modul-backend-detail)
- [Komponen Frontend](#-komponen-frontend)
- [Protokol WebSocket](#-protokol-websocket)
- [Arsitektur Memori](#-arsitektur-memori)
- [Formulasi Matematika](#-formulasi-matematika)
- [Testing Framework](#-testing-framework)
- [Diagram Arsitektur](#-diagram-arsitektur)
- [Dataset](#-dataset)
- [Environment Variables](#-environment-variables)
- [Kontribusi Akademis](#-kontribusi-akademis)
- [Penulis](#-penulis)

---

## 🎯 Ringkasan Proyek

**PROJECT TA (AI Data Scientist)** adalah sistem *Dashboard AI Data Scientist* berorientasi **Neuro-Symbolic** yang memadukan pemahaman bahasa alami *Large Language Model* (Gemini 2.5 Flash) dengan ketelitian eksekusi alat komputasi deterministik berbasis Python, Pandas, dan Scikit-Learn.

### Masalah yang Diselesaikan

Model bahasa besar (LLM) modern sangat fasih berbahasa, namun memiliki kelemahan fundamental:
- **Halusinasi Numerik** — LLM kerap mengarang angka saat diminta menghitung total anggaran
- **Inventori Fiktif** — Mencampurkan data dari internet dengan data pengguna
- **Amnesia Multi-Turn** — Mengulang rekomendasi yang sama pada percakapan lanjutan
- **Zero Constraint Enforcement** — Tidak mampu menjamin kepatuhan terhadap batasan angka ketat

### Solusi: Arsitektur Neuro-Symbolic

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM 1 (NEURO)                         │
│  LLM Gemini 2.5 Flash — Pemahaman bahasa, perencanaan,     │
│  ekstraksi intent & entitas, narasi penjelasan              │
└─────────────────────┬───────────────────────────────────────┘
                      │ JSON Function Calling (ReAct)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM 2 (SYMBOLIC)                      │
│  Python Deterministic Engine — Pandas, Scikit-Learn,        │
│  AHP Saaty, Knapsack, TSP Haversine, FAISS                 │
└─────────────────────────────────────────────────────────────┘
```

LLM bertindak sebagai **koordinator cerdas** yang memahami maksud pengguna dan mendelegasikan komputasi ke modul deterministik. Hasil komputasi kemudian dinarasikan kembali dalam bahasa alami yang mudah dipahami.

---

## 🏗 Arsitektur Sistem

<p align="center">
  <img src="image_visualisasi_arsitektur/Arsitektur Sistem Menyeluruh.jpeg" alt="System Architecture" width="750"/>
</p>

Sistem terbagi menjadi **tiga lapisan utama**:

| Lapisan | Teknologi | Fungsi |
|:---|:---|:---|
| **Frontend** | React 18 + Vite + Tailwind CSS | Antarmuka chat interaktif, upload file, reasoning panel |
| **API Gateway** | FastAPI + Uvicorn (ASGI) | HTTP REST + WebSocket Streaming (6 event types) |
| **Core Engine** | LangChain ReAct + Python Multi-Tools | Orchestrator agen, komputasi deterministik, memori persisten |

---

## ✨ Fitur Utama

### 🔬 Empat Spektrum Analisis Data

| Tingkat | Analisis | Kapabilitas |
|:---:|:---|:---|
| 1 | **Deskriptif** | Statistik ringkasan, profil data, missing values, skewness |
| 2 | **Diagnostik** | Deteksi outlier (IQR + Z-Score), VIF multikolinearitas, korelasi Pearson, class imbalance |
| 3 | **Prediktif** | Random Forest + HistGradientBoosting, kalibrasi probabilitas, XAI Feature Importance |
| 4 | **Preskriptif** | Knapsack Greedy, AHP Saaty, TSP Haversine, Adaptive Second-Pass Swap |

### 🛡️ Mekanisme Keamanan & Integritas

- **Static Function Mapping Whitelist** — LLM hanya boleh memanggil fungsi terdaftar
- **Chain-of-Verification (CoV)** — 5 lapis pemeriksaan sebelum jawaban akhir
- **Three-Tier Guardrails** — Domain geografis, intent non-analitik, kelayakan logis
- **Logical Pushback** — Penolakan sopan terhadap input yang tidak masuk akal

### 🧠 Kecerdasan Kontekstual

- **Dual-Layer Memory** — STM (WindowMemory k=10) + LTM (TinyDB JSON)
- **Pareto Soft De-prioritization** — Penalti -0.15 untuk destinasi yang pernah direkomendasikan
- **7-Day TTL Auto-Cleanup** — Pembersihan otomatis sesi yang tidak aktif
- **Data-Aware Priming** — Agen membaca skema kolom sebelum merencanakan tindakan

---

## 🛠 Tech Stack

### Backend

| Komponen | Teknologi | Versi |
|:---|:---|:---|
| Server Framework | FastAPI + Uvicorn | 0.104+ |
| Orkestrasi Agen | LangChain Core + LangChain Community | 0.1+ |
| Model Bahasa (LLM) | Google Gemini 2.5 Flash | Latest |
| Komputasi Data | Pandas, NumPy, Scikit-Learn | 1.3+ |
| Analisis Statistik | Statsmodels, SciPy | - |
| Vector Database | FAISS-CPU | 1.7+ |
| Dokumen PDF | PyMuPDF (fitz) | - |
| Basis Data Memori | TinyDB (NoSQL JSON) | 4.8+ |
| Visualisasi | Matplotlib, Seaborn | - |
| Validasi Data | Pydantic | 2.0+ |
| Testing | Pytest, Pytest-Asyncio | 7.4+ |

### Frontend

| Komponen | Teknologi | Versi |
|:---|:---|:---|
| UI Framework | React | 19.2+ |
| Bundler | Vite | 8.2+ |
| Styling | Tailwind CSS | 4.3+ |
| Icons | Lucide React | - |
| Komunikasi | Native WebSocket API | - |

---

## 📁 Struktur Direktori

```
Data_Whisperer_v1.0/
│
├── 📂 backend/                          # Server-side application
│   ├── 📂 api/
│   │   ├── main.py                      # FastAPI app, lifespan, CORS, startup
│   │   └── 📂 router/
│   │       └── agent_router.py          # HTTP upload + WebSocket chat endpoint
│   │
│   ├── 📂 services/
│   │   ├── 📂 agent/                    # 🧠 ReAct Agent Core
│   │   │   ├── main.py                  # Agent orchestrator (630 lines)
│   │   │   │                            #   - LLM provider factory (Google/Groq/OpenRouter)
│   │   │   │                            #   - System prompt engineering
│   │   │   │                            #   - Tool registration & whitelist
│   │   │   │                            #   - WebSocket streaming generator
│   │   │   ├── verify_output.py         # Chain-of-Verification (CoV) gateway
│   │   │   └── interpretation.py        # Vision-LLM chart interpretation (7 templates)
│   │   │
│   │   ├── 📂 eda/                      # 📊 Exploratory Data Analysis
│   │   │   └── main.py                  # 7 tools: describe, outlier, correlation,
│   │   │                                #   VIF, skewness, class_imbalance, full_profile
│   │   │
│   │   ├── 📂 ml/                       # 🤖 Machine Learning Pipeline
│   │   │   ├── main.py                  # Tool registry: train, predict, feature_imp
│   │   │   ├── travel_trainer.py        # RandomForest + CalibratedClassifierCV
│   │   │   └── 📂 generic/             # Blueprint untuk generic ML (future work)
│   │   │
│   │   ├── 📂 optimizer/               # 🎯 Prescriptive Optimizer
│   │   │   ├── main.py                  # Pydantic schema + tool wrapper
│   │   │   └── travel_logic.py          # 634 lines:
│   │   │                                #   - AHP Saaty (3×3 pairwise matrix)
│   │   │                                #   - Haversine distance formula
│   │   │                                #   - Greedy 0-1 Knapsack selection
│   │   │                                #   - Adaptive Second-Pass Swap
│   │   │                                #   - Nearest Neighbor TSP routing
│   │   │                                #   - Daily itinerary partitioning
│   │   │
│   │   ├── 📂 rag/                      # 📄 Retrieval-Augmented Generation
│   │   │   ├── main.py                  # RAG tool registry + hybrid search
│   │   │   ├── parser.py                # PDF text extraction (PyMuPDF)
│   │   │   ├── vectorizer.py            # Google Generative AI Embeddings
│   │   │   ├── retriever.py             # FAISS similarity search (top-k=10)
│   │   │   ├── dataset_indexer.py       # System corpus vector store builder
│   │   │   └── evaluator.py             # RAG metrics: Precision@K, MRR, Hit Rate
│   │   │
│   │   ├── 📂 memory/                  # 💾 Dual-Layer Memory
│   │   │   ├── memory_manager.py        # STM: WindowMemory k=10, runtime cache
│   │   │   ├── persistent_memory.py     # LTM: TinyDB JSON, 7-day TTL cleanup
│   │   │   └── memory_db.json           # NoSQL persistent store
│   │   │
│   │   ├── 📂 visualization/           # 📈 Chart Generation
│   │   │   └── main.py                  # Scatter, histogram, heatmap, bar, pie
│   │   │                                #   → PNG Base64 via WebSocket
│   │   │
│   │   ├── 📂 feature_engineering/     # 🔮 Future: Feature extraction module
│   │   └── 📂 preprocessing/           # 🔮 Future: User-controlled preprocessing
│   │
│   ├── 📂 testing/                      # 🧪 Automated Testing Framework
│   │   ├── scenarios.py                 # 11 benchmark scenarios definition
│   │   ├── evaluator.py                 # 3-dimension scoring engine (D1/D2/D3)
│   │   ├── runner.py                    # Test runner + mock/live mode
│   │   ├── report_generator.py          # JSON/CSV/TXT report generation
│   │   ├── ws_test.py                   # WebSocket streaming automated test
│   │   └── 📂 reports/                 # Generated test reports
│   │
│   └── 📂 utils/
│       └── read_csv.py                  # CSV reader with encoding fallback
│
├── 📂 frontend/                         # Client-side React application
│   ├── index.html                       # Entry point
│   ├── package.json                     # npm dependencies
│   ├── vite.config.js                   # Vite bundler configuration
│   └── 📂 src/
│       ├── main.jsx                     # React DOM root
│       ├── App.jsx                      # Main app layout (3-panel design)
│       ├── App.css                      # Global styles
│       ├── index.css                    # Base styles
│       ├── 📂 components/
│       │   ├── ChatArea.jsx             # Message bubbles, auto-scroll
│       │   ├── ChatInput.jsx            # Text input, file upload (CSV/PDF)
│       │   ├── ItineraryCard.jsx        # Travel itinerary daily cards
│       │   ├── ReasoningPanel.jsx       # Thought-Action-Observation viewer
│       │   ├── RightPanel.jsx           # Developer metrics panel
│       │   └── Sidebar.jsx              # Session list & navigation
│       └── 📂 hooks/
│           └── useAgentWebSocket.js     # WebSocket client (6 event handler)
│
├── 📂 data/
│   └── 📂 dataset/                      # Multi-domain test datasets
│       ├── hr_analytics.csv             # IBM HR Attrition (1472×35)
│       ├── student_performance.csv      # Student scores (1000×8)
│       └── bps_pariwisata_bali.pdf      # BPS Bali Tourism Statistics
│
├── 📂 Build_Dataset/
│   └── bali_tourist_clean_v3.csv        # Primary Bali destinations dataset
│
├── 📂 image_visualisasi_arsitektur/     # 13 architecture diagrams
│
├── .env                                 # API keys configuration
├── requirements.txt                     # Python dependencies
├── memory_db.json                       # TinyDB persistent storage
└── README.md                            # ← Anda sedang di sini
```

---

## 📋 Prasyarat

| Komponen | Versi Minimal |
|:---|:---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |
| RAM | 8 GB (direkomendasikan 16 GB) |
| Storage | 2 GB ruang kosong |

**API Key yang Diperlukan:**
- Google API Key (untuk Gemini 2.5 Flash + Embeddings)
- Opsional: Groq API Key, OpenRouter API Key (provider LLM alternatif)

---

## 🚀 Instalasi & Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd project_ta
```

### 2. Setup Backend (Python)

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi (Windows)
.\venv\Scripts\activate

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Frontend (Node.js)

```bash
cd frontend
npm install
cd ..
```

### 4. Konfigurasi Environment Variables

Buat file `.env` di root project:

```env
# Wajib
GOOGLE_API_KEY="your_google_api_key_here"

# Opsional (provider LLM alternatif)
GROQ_API_KEY="your_groq_key_here"
OPENROUTER_API_KEY="your_openrouter_key_here"
OPENAI_API_KEY="your_openai_key_here"

# Model Configuration
AGENT_MODEL="gemini-2.5-flash"
INTERPRETATION_MODEL="gemini-2.5-flash"
```

---

## ▶️ Menjalankan Aplikasi

### Terminal 1 — Backend Server

```bash
uvicorn backend.api.main:app --reload
```

Server akan berjalan di `http://localhost:8000`

**Startup Log yang Diharapkan:**
```
✅ [Startup] Semua ENV wajib terdeteksi.
✅ [Startup] Dataset default Bali (v3) berhasil didaftarkan.
✅ [Startup] LTM cleanup: 0 sesi lama dihapus dari memory_db.json
🤖 [Agent LLM] Provider=google | Model=gemini-2.5-flash
🔄 [Startup] Mencoba memuat FAISS index dari disk...
✅ [Startup] System RAG vector store dimuat dari disk (near-instant).
```

### Terminal 2 — Frontend Dev Server

```bash
cd frontend
npm run dev
```

Aplikasi akan terbuka di `http://localhost:5173`

### Production Build

```bash
cd frontend
npm run build
npm run preview
```

---

## 🔧 Multi-Tools Registry

Seluruh fungsi komputasi yang dapat dipanggil oleh agen terdaftar dalam **Static Function Mapping Whitelist**:

```python
# backend/services/agent/main.py L123
tools = eda_tools + vis_tools + ml_tools + rag_tools + optimizer_tools + [verify_output_tool]
```

### Daftar Lengkap Tools

| Kategori | Tool Name | Input Parameters | Output |
|:---|:---|:---|:---|
| **EDA** | `describe_dataset` | `file_path` | Statistik 5-angka, missing values, skewness |
| | `detect_outliers` | `file_path`, `method` | Indeks pencilan, persentase anomali, boxplot |
| | `get_correlation_matrix` | `file_path`, `threshold` | Matriks korelasi Pearson + heatmap |
| | `analyze_vif` | `file_path` | Skor VIF per fitur + indikator risiko |
| | `detect_class_imbalance` | `target_column` | Rasio ketimpangan + severity level |
| | `run_full_profile` | `file_path` | Profil data komprehensif |
| **ML** | `train_model` | `file_path`, `target_col` | Accuracy, Precision, Recall, F1, ROC-AUC |
| | `predict_match_score` | `features`, `preferences` | Probabilitas terkalibrasi (0–100%) |
| | `extract_feature_imp` | `model_id` | Permutation Importance + MDI |
| **RAG** | `index_pdf_document` | `pdf_path`, `chunk_size` | Indeks vektor FAISS + metadata |
| | `rag_semantic_filter` | `query`, `top_k` | Potongan teks relevan + sitasi halaman |
| **Optimizer** | `budget_optimizer_tool` | `budget`, `duration`, `location` | Itinerary harian, total biaya, utilisasi |
| **Verification** | `verify_output` | `draft_response` | Hasil validasi CoV 5 tahap |
| **Visualization** | `plot_itinerary_scatter` | `selected_destinations` | Scatter plot harga vs rating (Base64) |
| | `plot_distribution_histogram` | `column` | Histogram + KDE (Base64) |
| | `plot_correlation_heatmap` | — | Heatmap korelasi (Base64) |
| | `plot_category_bar` | `category_column`, `top_n` | Bar chart distribusi (Base64) |
| | `plot_budget_breakdown` | `selected_destinations` | Pie chart (≤8 destinasi) atau horizontal bar chart (>8 destinasi) alokasi budget (Base64) |

---

## 📦 Modul Backend Detail

### 1. Agent Orchestrator (`backend/services/agent/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Alur Kerja Siklus Penalaran ReAct & Chain-of-Verification.jpeg" alt="ReAct Cycle" width="700"/>
</p>

**File: `main.py` (630 lines)**

Inti dari sistem — mengorkestrasikan seluruh siklus ReAct:

- **LLM Provider Factory** — Mendukung 3 provider: Google Gemini, Groq, OpenRouter
- **System Prompt Engineering** — Data-aware prompt dengan kolom dataset aktif
- **Tool Registration** — Whitelist registry seluruh fungsi deterministik
- **WebSocket Streaming** — Generator async untuk 6 tipe event
- **Agent Executor** — LangChain `AgentExecutor` dengan `max_iterations=8`

**File: `verify_output.py` (130 lines)**

Implementasi Chain-of-Verification (CoV):
- Pemeriksaan panjang minimum respons
- Deteksi pola halusinasi (`undefined`, `NaN` dalam konteks harga)
- Validasi konsistensi data numerik terhadap output tool
- Pemeriksaan budget constraint ($\sum c_i \le B$)

**File: `interpretation.py` (9347 bytes)**

Vision-LLM untuk interpretasi grafik otomatis:
- 7 template prompt spesifik per tipe chart
- Multimodal input (Base64 image → LLM → narasi bahasa alami)

---

### 2. EDA Module (`backend/services/eda/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Alur Kerja Automated Exploratory Data Analysis (EDA) Diagnostik.jpeg" alt="EDA Pipeline" width="700"/>
</p>

**File: `main.py` (20643 bytes)**

Modul diagnostik data tabular otomatis:

| Fitur | Metode | Formula |
|:---|:---|:---|
| Deteksi Outlier | IQR Tukey | $x < Q_1 - 1.5 \times IQR$ atau $x > Q_3 + 1.5 \times IQR$ |
| Deteksi Outlier | Z-Score | $\|Z\| = \left\| \frac{x - \mu}{\sigma} \right\| \ge 3.0$ |
| Multikolinearitas | VIF | $VIF_i = \frac{1}{1 - R_i^2}$ |
| Class Imbalance | Imbalance Ratio | $IR = \frac{\max(C)}{\min(C)}$ |

---

### 3. ML Module (`backend/services/ml/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Alur Kerja Supervised Machine Learning & Explainable AI (XAI).jpeg" alt="ML Pipeline" width="700"/>
</p>

**File: `travel_trainer.py` (15108 bytes)**

Pipeline pembelajaran mesin end-to-end:

```
Data → ColumnTransformer → Stratified Split (80:20)
    → RandomForestClassifier (n=200, depth=15)
    → CalibratedClassifierCV (isotonic)
    → Permutation Importance + MDI
    → Match Score (0–100%)
```

---

### 4. Optimizer Module (`backend/services/optimizer/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Alur Kerja Prescriptive Optimizer Pipeline.jpeg" alt="Optimizer Pipeline" width="700"/>
</p>

**File: `travel_logic.py` (634 lines)**

Konstanta konfigurasi:
- `MAX_PLACES_PER_DAY = 6` — Maksimal 6 destinasi per hari
- `VISIT_DURATION_MIN = 60` — Estimasi 60 menit kunjungan per destinasi
- `AVERAGE_SPEED_KMH = 30` — Kecepatan rata-rata kendaraan di Bali
- `START_HOUR = 9` — Jam mulai kunjungan (09:00)

Pipeline optimasi multi-tahap:

1. **AHP Saaty Weighting** — Matriks perbandingan 3×3 (Rating, Value, Crowd)
   - Bobot: $w_r ≈ 0.637$, $w_v ≈ 0.258$, $w_c ≈ 0.105$
   - Consistency Ratio: $CR = 0.0334 < 0.10$ ✅

2. **Composite Scoring** — Skor AHP + Random Jitter ±5% + Pareto Penalty -0.15

3. **Greedy 0-1 Knapsack** — Seleksi destinasi dengan constraint $\sum c_i \le B$
   - Limitasi: max 1 destinasi gratis per hari
   - Normalisasi: `FREE_VALUE_SCORE = 0.5`

4. **Adaptive Second-Pass Swap** — Jika utilisasi < 80%, tukar destinasi murah ke premium

5. **Nearest Neighbor TSP** — Rute harian optimal berbasis Haversine distance (maks 6 lokasi/hari, 60 menit/kunjungan):
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)$$

---

### 5. RAG Module (`backend/services/rag/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Alur Kerja Retrieval-Augmented Generation (RAG) Hibrida Dua Tingkat.jpeg" alt="RAG Architecture" width="700"/>
</p>

Arsitektur RAG dua tingkat:

| Layer | Sumber | Trigger |
|:---|:---|:---|
| **Layer 1** | Dokumen PDF yang diunggah pengguna | Jika pengguna upload PDF |
| **Layer 2** | Korpus deskripsi wisata internal (disk) | Fallback jika tidak ada PDF |

- **Text Chunking:** 1000 karakter + overlap
- **Embedding:** Google Generative AI Embeddings
- **Vector Store:** FAISS-CPU (cosine similarity)
- **Retrieval:** Top-K=10 most similar chunks

---

### 6. Memory Module (`backend/services/memory/`)

<p align="center">
  <img src="image_visualisasi_arsitektur/Arsitektur Dual-Layer Memory & State Management.jpeg" alt="Memory Architecture" width="700"/>
</p>

| Layer | Storage | Lifetime | Data |
|:---|:---|:---|:---|
| **STM** | RAM (Python dict) | Per-session | DataFrame, FAISS index, WindowMemory (k=10) |
| **LTM** | TinyDB JSON file | Persistent | Chat history, preferences, visited destinations |

**TinyDB Schema (3 tabel):**

| Tabel | Fields | Deskripsi |
|:---|:---|:---|
| `sessions` | session_id, created_at, last_active, paths | Metadata sesi kerja |
| `messages` | session_id, role, content, reasoning_steps | Riwayat percakapan + reasoning |
| `user_preferences` | session_id, budget, duration, previously_visited | Profil pengguna |

---

## 🎨 Komponen Frontend

<p align="center">
  <img src="image_visualisasi_arsitektur/Arsitektur Komponen Frontend Dashboard.jpeg" alt="Frontend Components" width="700"/>
</p>

| Komponen | File | Fungsi |
|:---|:---|:---|
| `App.jsx` | 7207 bytes | Layout utama 3-panel, session management |
| `Sidebar.jsx` | 4507 bytes | Daftar sesi, tombol "New Analysis" |
| `ChatArea.jsx` | 6696 bytes | Message bubbles, auto-scroll, typing indicator |
| `ChatInput.jsx` | 9421 bytes | Text input, file upload (CSV/PDF), chip preview |
| `ItineraryCard.jsx` | 9678 bytes | Kartu jadwal harian, tab per hari, metadata |
| `ReasoningPanel.jsx` | 3544 bytes | Thought-Action-Observation viewer |
| `RightPanel.jsx` | 3895 bytes | Developer metrics, model performance |
| `useAgentWebSocket.js` | 15490 bytes | WebSocket client hook, 6 event handler |

---

## 🔌 Protokol WebSocket

<p align="center">
  <img src="image_visualisasi_arsitektur/Arsitektur Protokol Komunikasi Real-Time WebSocket.jpeg" alt="WebSocket Protocol" width="700"/>
</p>

Koneksi: `ws://localhost:8000/ws/chat/{session_id}`

### 6 Tipe Event

| Event | Direction | Payload | Deskripsi |
|:---|:---:|:---|:---|
| `token` | Server → Client | `{type: "token", content: "..."}` | Streaming teks kata-per-kata |
| `tool_start` | Server → Client | `{type: "tool_start", tool: "..."}` | Notifikasi awal eksekusi tool |
| `tool_end` | Server → Client | `{type: "tool_end", output: {...}}` | Hasil komputasi tool |
| `image` | Server → Client | `{type: "image", image_base64: "..."}` | Grafik visualisasi (PNG Base64) |
| `done` | Server → Client | `{type: "done", session_id: "..."}` | Penanda selesai |
| `error` | Server → Client | `{type: "error", message: "..."}` | Notifikasi kesalahan |

---

## 🧠 Arsitektur Memori

```
┌────────────────────────────────────────────────────────────┐
│                  SHORT-TERM MEMORY (STM)                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ WindowMemory    │  │ Pandas       │  │ FAISS Index  │  │
│  │ k=10 messages   │  │ DataFrame    │  │ (Vector DB)  │  │
│  └─────────────────┘  └──────────────┘  └──────────────┘  │
│  Lifetime: Session only | Storage: RAM                     │
├────────────────────────────────────────────────────────────┤
│                   LONG-TERM MEMORY (LTM)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TinyDB (memory_db.json)                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │ sessions │  │ messages │  │ user_preferences │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│  Lifetime: Persistent | TTL: 7 days auto-cleanup           │
└────────────────────────────────────────────────────────────┘
```

---

## 📐 Formulasi Matematika

### Deteksi Outlier
$$IQR = Q_3 - Q_1 \qquad BB = Q_1 - 1.5 \times IQR \qquad BA = Q_3 + 1.5 \times IQR$$

### Z-Score
$$Z = \frac{x - \mu}{\sigma}$$

### Variance Inflation Factor
$$VIF_i = \frac{1}{1 - R_i^2}$$

### AHP Consistency
$$CI = \frac{\lambda_{\max} - n}{n - 1} \qquad CR = \frac{CI}{RI}$$

### Haversine Distance
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)$$

### Travel Time Estimation
$$T = \left(\frac{d}{30} \times 60\right) + 10 \text{ menit}$$

### Knapsack Objective
$$\text{Maximize} \quad \sum_{i=1}^{N} s_i x_i \quad \text{s.t.} \quad \sum_{i=1}^{N} c_i x_i \le B$$

### Pareto Penalty
$$\text{Score}_{new} = \text{Score}_{AHP} - 0.15$$

---

## 🧪 Testing Framework

### Desain 16 Skenario Benchmark × 3 Dimensi Evaluasi

Evaluasi empiris sistem dievaluasi secara kuantitatif melalui pembobotan multi-dimensi:
$$\text{Final Score} = 0.40 \times D_1 + 0.35 \times D_2 + 0.25 \times D_3$$

| ID | Skenario | Kategori | Sequence Tool Utama |
|:---:|:---|:---|:---|
| **S01** | Prescriptive Core Travel Optimization | Optimizer | `budget_optimizer_tool` $\rightarrow$ `plot_budget_breakdown` $\rightarrow$ `verify_output` |
| **S02** | Neuro-Symbolic Multi-Tool Chaining | Semantic Chaining | `rag_semantic_filter` $\rightarrow$ `budget_optimizer_tool` $\rightarrow$ `plot_budget_breakdown` $\rightarrow$ `verify_output` |
| **S03** | Reasoning & Boundary Validation | Logical Pushback | `budget_optimizer_tool` $\rightarrow$ `verify_output` (Pagu Mustahil) |
| **S04** | Explainable AI (XAI) & Scatter Plot | XAI & Visual Analytics | `budget_optimizer_tool` $\rightarrow$ `plot_itinerary_scatter` $\rightarrow$ `verify_output` |
| **S05** | Multi-Criteria Filtered Data Query | Multi-Filter Query | `run_full_profile` $\rightarrow$ `rag_semantic_filter` $\rightarrow$ `budget_optimizer_tool` $\rightarrow$ `verify_output` |
| **S06** | Guardrail Keamanan Domain | Domain Safety | Nol Tool Call (Penolakan Sintaks Coding Langsung) |
| **S07** | Guardrail Geografis Wilayah | Boundary Defense | Nol Tool Call (Penolakan Wilayah Luar Bali / Jakarta) |
| **S08** | Automated EDA Full Profile | Autonomous Profiling | `run_full_profile` $\rightarrow$ `verify_output` |
| **S09** | Machine Learning Match Score | ML Inference & XAI | `predict_match_score` $\rightarrow$ `verify_output` |
| **S10** | Cross-Domain Generalization | Generalizability Test | `run_full_profile` $\rightarrow$ `detect_outliers` $\rightarrow$ `get_correlation_matrix` $\rightarrow$ `verify_output` |
| **S11** | Document Intelligence Q&A | Unstructured RAG | `index_pdf` $\rightarrow$ `rag_semantic_filter` $\rightarrow$ `verify_output` |
| **S12** | Multi-Turn Dynamic Re-Planning | Stateful Memory & Pareto | `rag_semantic_filter` $\rightarrow$ `budget_optimizer_tool` $\rightarrow$ `verify_output` |
| **S13** | Adversarial Jailbreak Defense | Cybersecurity | Nol Tool Call (Penolakan Rekayasa Prompt DAN) |
| **S14** | Extreme Luxury Budget Upper-Bound | Upper-Bound Stress Test | `rag_semantic_filter` $\rightarrow$ `budget_optimizer_tool` $\rightarrow$ `plot_budget_breakdown` $\rightarrow$ `verify_output` |
| **S16** | Incomplete & Ambiguous Query Dialogue | Mixed-Initiative Dialogue | Nol Tool Call (3 Pertanyaan Klarifikasi Proaktif) |
| **S17** | Cross-Regency Transit Fatigue | Spatial Intelligence | `budget_optimizer_tool` (Gabungan) $\rightarrow$ `budget_optimizer_tool` (3 Zona Mandiri) $\rightarrow$ `verify_output` |

*Catatan:* Skenario S15 (*Malformed / Dirty CSV Handling*) dialokasikan sebagai agenda riset lanjutan.

### 3 Dimensi Evaluasi

| Dimensi | Bobot | Fokus Penilaian |
|:---|:---:|:---|
| **D1: Tool Selection Accuracy** | 40% | Ketepatan memilih fungsi yang diharapkan & menghindari tool terlarang |
| **D2: Argument & Constraint Extraction** | 35% | Akurasi translasi bahasa alami ke parameter terstruktur |
| **D3: Response Integrity & Anti-Hallucination** | 25% | Kepatuhan matematis anggaran $\sum c_i \le B$, audit CoV, dan ketiadaan inventori fiktif |

### Menjalankan Tests

```bash
# Dry-run mode (tanpa server live)
python -m backend.testing.runner --dry-run

# Live mode (memerlukan server berjalan)
python -m backend.testing.runner

# WebSocket test
python -m backend.testing.ws_test
```

### Hasil Evaluasi

```
===========================================================================
  RINGKASAN EVALUASI BENCHMARK SISTEM (16 SKENARIO)
===========================================================================
  SUCCESS RATE : 100.0%
  Rata-rata 3D : 100.0 / 100.0 (Grade A)
  Lulus / Gagal: 16 / 0
  Target       : >= 90% → ✅ TERCAPAI SEMPURNA
===========================================================================
```

---

## 🖼️ Diagram Arsitektur

Seluruh 13 diagram arsitektur tersedia di folder `image_visualisasi_arsitektur/`:

| No | Diagram | Digunakan di |
|:---:|:---|:---|
| 1 | Arsitektur Sistem Menyeluruh | Bab III |
| 2 | Siklus Penalaran ReAct & Chain-of-Verification | Bab II, III |
| 3 | Paradigma Neuro-Symbolic AI | Bab II |
| 4 | Alur Kerja RAG Hibrida Dua Tingkat | Bab II, III |
| 5 | Alur Kerja AHP (Analytic Hierarchy Process) | Bab II |
| 6 | Alur Kerja EDA Diagnostik | Bab III |
| 7 | Alur Kerja ML & Explainable AI (XAI) | Bab III |
| 8 | Pipeline Prescriptive Optimizer | Bab III |
| 9 | Mekanisme Guardrails & Logical Pushback | Bab III |
| 10 | Arsitektur Dual-Layer Memory | Bab III |
| 11 | Komponen Frontend Dashboard | Bab III |
| 12 | Protokol WebSocket (6 Event Types) | Bab III |
| 13 | Kerangka Pengujian Otomatis Suite 11 | Bab IV |

---

## 📊 Dataset

### Dataset Primer

| Dataset | Format | Rows × Cols | Deskripsi |
|:---|:---:|:---:|:---|
| `bali_tourist_clean_v3.csv` | CSV | ~200 × 15+ | Destinasi wisata Bali (rating, harga, koordinat, kategori) |

### Dataset Sekunder (Multi-Domain Testing)

| Dataset | Format | Rows × Cols | Domain |
|:---|:---:|:---:|:---|
| `hr_analytics.csv` | CSV | 1472 × 35 | IBM HR Employee Attrition |
| `student_performance.csv` | CSV | 1000 × 8 | Performa akademik siswa |
| `bps_pariwisata_bali.pdf` | PDF | 4.16 MB | Statistik BPS Pariwisata Bali |

---

## 🔐 Environment Variables

| Variable | Required | Deskripsi |
|:---|:---:|:---|
| `GOOGLE_API_KEY` | ✅ | API key untuk Gemini 2.5 Flash + Google Embeddings |
| `GROQ_API_KEY` | ❌ | API key untuk Groq (provider alternatif) |
| `OPENROUTER_API_KEY` | ❌ | API key untuk OpenRouter (provider alternatif) |
| `OPENAI_API_KEY` | ❌ | API key untuk OpenAI (provider alternatif) |
| `AGENT_MODEL` | ✅ | Nama model LLM (default: `gemini-2.5-flash`) |
| `INTERPRETATION_MODEL` | ✅ | Model untuk interpretasi chart (default: `gemini-2.5-flash`) |

---

## 🎓 Kontribusi Akademis

### 4 Pilar Novelty

1. **Arsitektur Neuro-Symbolic Hibrida** — Memisahkan penalaran kualitatif (LLM) dari komputasi kuantitatif (Python Engine)
2. **Zero-Hallucination Budget Guarantee** — Knapsack deterministik menjamin $\sum c_i \le B$ 100%
3. **Pareto Multi-Turn De-prioritization** — Penalti -0.15 menyeimbangkan eksplorasi vs eksploitasi
4. **Autonomous Tool Orchestration + Domain Guardrails** — Agen memilih tools secara otonom dengan batasan domain

### Referensi Kunci

- Yao et al. (2022) — ReAct: Synergizing Reasoning and Acting in Language Models
- Kautz (2022) — Neuro-Symbolic AI (AAAI Engelmore Lecture)
- Saaty (1980) — The Analytic Hierarchy Process
- Lewis et al. (2020) — Retrieval-Augmented Generation
- Schick et al. (2023) — Toolformer: Language Models Can Teach Themselves to Use Tools

---

## 📄 API Endpoints

### HTTP REST

| Method | Endpoint | Deskripsi |
|:---|:---|:---|
| `POST` | `/api/agent/upload` | Upload file CSV/PDF (multipart/form-data) |
| `POST` | `/api/agent/execute` | Eksekusi agent action |
| `GET` | `/health` | Health check |

### WebSocket

| Endpoint | Deskripsi |
|:---|:---|
| `ws://localhost:8000/ws/chat/{session_id}` | Streaming chat + reasoning |

---

## 👤 Penulis

**Reza Pratama Tandjung**  
NRP: 221116984  
Program Studi: S1 Teknik Informatika — *Intelligent System*  
Institut Sains dan Teknologi Terpadu Surabaya (ISTTS)

**Pembimbing:**
- Dr. Ir. Herman Budianto, M.M. (Pembimbing Utama)
- Prof. Dr. Ir. Esther Irawati Setiawan, S.Kom, M.Kom (Co-Pembimbing)

---
