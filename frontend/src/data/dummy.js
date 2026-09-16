// Dummy data untuk sesi chat
export const DUMMY_SESSIONS = [
  { id: "1", label: "Itinerary Bangli 2 Hari", time: "2 menit lalu", icon: "🗺️" },
  { id: "2", label: "Wisata Gianyar Budget 300K", time: "Kemarin", icon: "🌿" },
  { id: "3", label: "Pantai Badung 1 Hari", time: "2 hari lalu", icon: "🏖️" },
];

export const DUMMY_MODELS = [
  { id: 1, name: "RandomForest_v2", sub: "Bangli Trip" },
  { id: 2, name: "XGBoost_v1", sub: "Badung Trip" },
];

// Dummy itinerary card data
export const DUMMY_ITINERARY = {
  budget: 400000,
  totalCost: 55000,
  days: [
    {
      day: 1,
      date: "Hari 1",
      summary: "3 destinasi | 09:00 – 14:41 | ±26.0 km",
      destinations: [
        {
          time: "09:00",
          name: "Mount Batur ATV Adventure",
          category: "Alam",
          price: 0,
          rating: 5.0,
          city: "Kintamani, Bangli",
          travel: "↓ ~22.71 km · ±55 menit",
        },
        {
          time: "11:25",
          name: "Yangapi Waterfall",
          category: "Alam",
          price: 0,
          rating: 5.0,
          city: "Tembuku, Bangli",
          travel: "↓ ~3.28 km · ±16 menit",
        },
        {
          time: "13:11",
          name: "Wisata Alam Gredeg",
          category: "Umum",
          price: 15000,
          rating: 5.0,
          city: "Tembuku, Bangli",
          travel: "★ Destinasi terakhir hari ini",
        },
      ],
    },
    {
      day: 2,
      date: "Hari 2",
      summary: "3 destinasi | 09:00 – 15:09 | ±40.1 km",
      destinations: [
        {
          time: "09:00",
          name: "ATV Kintamani Batur",
          category: "Alam",
          price: 10000,
          rating: 4.9,
          city: "Kintamani, Bangli",
          travel: "↓ ~9.67 km · ±29 menit",
        },
        {
          time: "10:59",
          name: "Hutan Folklore",
          category: "Alam",
          price: 5000,
          rating: 4.8,
          city: "Kintamani, Bangli",
          travel: "↓ ~30.45 km · ±70 menit",
        },
        {
          time: "13:39",
          name: "Puri Jati Delod Utu",
          category: "Budaya",
          price: 25000,
          rating: 4.8,
          city: "Susut, Bangli",
          travel: "★ Destinasi terakhir hari ini",
        },
      ],
    },
  ],
};

export const DUMMY_MESSAGES = [
  {
    id: 1,
    role: "user",
    text: "Buatkan itinerary 2 hari wisata Alam di Kabupaten Bangli dengan budget 400.000 rupiah.",
    time: "09:14",
  },
  {
    id: 2,
    role: "agent",
    text: "Tentu! Saya sudah menyiapkan rancangan itinerary 2 hari untuk wisata Alam di Kabupaten Bangli dengan budget Rp 400.000. Total biaya yang dibutuhkan hanya Rp 55.000, jadi masih banyak sisa budget!",
    time: "09:14",
    hasItinerary: true,
    reasoningSteps: [
      { step: 1, tool: "budget_optimizer_tool", status: "success", detail: "budget=400000, keywords=['Alam','Bangli'], days=2" },
      { step: 2, tool: "plot_itinerary_scatter", status: "success", detail: "6 destinasi diplot" },
      { step: 3, tool: "verify_output", status: "success", detail: "Verifikasi passed — 0 isu ditemukan" },
    ],
  },
];

export const DUMMY_STATS = {
  model_name: "RandomForest (Calibrated)",
  accuracy: 94.5,
  precision: 95.1,
  recall: 93.8,
  f1_score: 94.4,
  roc_auc: 98.3,
  features: [
    { name: "Jumlah Ulasan", value: 20.0, color: "#34d399" },
    { name: "Kategori Wisata", value: 12.1, color: "#fbbf24" },
    { name: "Harga Tiket", value: 11.6, color: "#4f6ef7" },
    { name: "Kepadatan", value: 8.0, color: "#f87171" },
    { name: "Rating Google", value: 7.3, color: "#c084fc" },
  ],
  dataset: {
    file: "bali_tourist_clean_v3.csv",
    rows: 1452,
    columns: 15,
  },
  drift_status: "Optimal (No Drift)",
};

export const INITIAL_SESSIONS = [
  {
    id: "1",
    label: "Itinerary Bangli 2 Hari",
    time: "2 menit lalu",
    icon: "🗺️",
    messages: DUMMY_MESSAGES,
  },
  {
    id: "2",
    label: "Wisata Gianyar Budget 300K",
    time: "Kemarin",
    icon: "🌿",
    messages: [
      {
        id: 201,
        role: "user",
        text: "Rekomendasi wisata Gianyar dengan budget 300 ribu",
        time: "14:20",
      },
      {
        id: 202,
        role: "agent",
        text: "Berikut rekomendasi destinasi wisata di Gianyar yang hemat dan populer:\n\n1. **Air Terjun Kanto Lampo** — Rp 20.000 (Air terjun berundak yang estetik)\n2. **Goa Gajah** — Rp 50.000 (Situs bersejarah dan budaya)\n3. **Pasar Seni Sukawati** — Gratis (Pusat kerajinan dan oleh-oleh Bali)\n4. **Pura Tirta Empul** — Rp 50.000 (Pura pemandian suci bersejarah)\n\n💰 **Total tiket masuk:** Rp 120.000 dari budget Rp 300.000. Sisa **Rp 180.000** bisa dialokasikan untuk kuliner khas Gianyar seperti Nasi Ayam Kedewatan dan transportasi!",
        time: "14:20",
      },
    ],
  },
  {
    id: "3",
    label: "Pantai Badung 1 Hari",
    time: "2 hari lalu",
    icon: "🏖️",
    messages: [
      {
        id: 301,
        role: "user",
        text: "Pantai terbaik di Badung untuk 1 hari",
        time: "11:05",
      },
      {
        id: 302,
        role: "agent",
        text: "Berikut rute pantai 1 hari terbaik di Kabupaten Badung Selatan:\n\n- **Pagi (09:00 - 11:30)**: **Pantai Pandawa** (Rp 15.000) — Tebing kapur megah dengan patung Panca Pandawa dan pasir putih bersih.\n- **Siang (12:00 - 14:30)**: **Pantai Melasti** (Rp 10.000) — Jalan berliku di antara tebing eksotis dan air laut jernih toska.\n- **Sore (16:30 - 18:30)**: **Pantai Padang Padang** (Rp 15.000) — Pantai ikonik untuk menikmati matahari terbenam (*sunset*).\n\n💡 *Semua pantai memiliki akses jalan beraspal yang mulus dan fasilitas lengkap.*",
        time: "11:05",
      },
    ],
  },
];

