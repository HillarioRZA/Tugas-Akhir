// Dummy data untuk sesi chat
export const DUMMY_SESSIONS = [
  { id: 1, label: "Itinerary Bangli 2 Hari", time: "2 menit lalu", icon: "🗺️" },
  { id: 2, label: "Wisata Gianyar Budget 300K", time: "Kemarin", icon: "🌿" },
  { id: 3, label: "Pantai Badung 1 Hari", time: "2 hari lalu", icon: "🏖️" },
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
  accuracy: 85,
  precision: 92,
  recall: 78,
  features: [
    { name: "Rating", value: 88, color: "#34d399" },
    { name: "Harga", value: 72, color: "#4f6ef7" },
    { name: "Kategori", value: 60, color: "#fbbf24" },
    { name: "Lokasi", value: 45, color: "#f87171" },
  ],
  dataset: {
    file: "bali_tourist_clean_v3.csv",
    rows: 1247,
    columns: 18,
  },
};
