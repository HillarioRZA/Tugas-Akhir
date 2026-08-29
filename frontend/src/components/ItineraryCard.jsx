import { useState } from 'react';

const CATEGORY_STYLE = {
  "Alam":   "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  "Budaya": "bg-purple-500/15 text-purple-400 border-purple-500/25",
  "Pantai": "bg-sky-500/15 text-sky-400 border-sky-500/25",
  "Umum":   "bg-slate-500/15 text-slate-400 border-slate-500/25",
};

function StarRating({ value }) {
  return (
    <span className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map(i => (
        <svg key={i}
          className={`w-2.5 h-2.5 ${i <= Math.round(value) ? 'text-amber-400' : 'text-slate-600'}`}
          viewBox="0 0 20 20" fill="currentColor">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      ))}
      <span className="ml-1 text-xs" style={{ color: 'var(--text-secondary)' }}>
        {value.toFixed(1)}
      </span>
    </span>
  );
}

function DestinationRow({ dest, isLast }) {
  return (
    <div className="flex gap-3 relative">
      {/* Timeline line */}
      {!isLast && (
        <div className="absolute left-[18px] top-7 bottom-0 w-px"
          style={{ background: 'var(--border)' }} />
      )}

      {/* Time + dot */}
      <div className="flex flex-col items-center shrink-0 w-9">
        <span className="text-xs font-mono font-semibold mb-1"
          style={{ color: 'var(--accent-light)' }}>
          {dest.time}
        </span>
        <div className="w-2 h-2 rounded-full shrink-0 mt-0.5"
          style={{ background: 'var(--accent)', boxShadow: '0 0 6px var(--accent)' }} />
      </div>

      {/* Card content */}
      <div className="flex-1 mb-4 rounded-xl p-3.5 transition-all hover:brightness-105"
        style={{ background: 'var(--card-bg)', border: '1px solid var(--border)' }}>
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold leading-tight truncate"
              style={{ color: 'var(--text-primary)' }}>
              {dest.name}
            </p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              📍 {dest.city}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${CATEGORY_STYLE[dest.category] ?? CATEGORY_STYLE["Umum"]}`}>
              {dest.category}
            </span>
            <span className="text-sm font-semibold"
              style={{ color: dest.price === 0 ? 'var(--success)' : 'var(--text-primary)' }}>
              {dest.price === 0 ? 'Gratis' : `Rp ${dest.price.toLocaleString('id-ID')}`}
            </span>
          </div>
        </div>
        <div className="mt-2 flex items-center justify-between">
          <StarRating value={dest.rating} />
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{dest.travel}</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalisasi output JSON dari budget_optimizer_tool ke format internal
//
// Struktur payload.data yang dikirim backend (context["last_tool_output"]):
// {
//   pesan: "...",
//   total_biaya_kalkulasi: 15000,
//   budget_limit: 100000,
//   rekomendasi_itinerary: [...],
//   itinerary_per_hari: {
//     hari_1: {
//       destinations: [
//         {
//           Place_Name, Category, City, kecamatan, Price, Rating,
//           estimated_arrival_time, travel_note_to_next, ...
//         }
//       ],
//       day_summary: "Hari 1: 3 destinasi | Mulai 09:00 ...",
//       total_destinations: 3,
//     },
//     hari_2: { ... },      // hanya ada jika duration_days >= 2
//     total_hari: 1,
//   }
// }
// ─────────────────────────────────────────────────────────────────────────────
function normalizeData(raw) {
  if (!raw) return null;

  // Format A — sudah ternormalisasi (misal dari dummy / format lama)
  if (Array.isArray(raw.days)) return raw;

  // Format B — output backend optimizer: ambil dari itinerary_per_hari
  const hariSource = raw.itinerary_per_hari;
  if (!hariSource || typeof hariSource !== 'object') {
    console.warn('[ItineraryCard] itinerary_per_hari tidak ditemukan di data:', raw);
    return null;
  }

  // Ambil key "hari_N" dan urutkan
  const hariKeys = Object.keys(hariSource)
    .filter(k => /^hari_\d+$/.test(k))
    .sort((a, b) =>
      parseInt(a.split('_')[1], 10) - parseInt(b.split('_')[1], 10)
    );

  if (hariKeys.length === 0) {
    console.warn('[ItineraryCard] Tidak ada key hari_N di itinerary_per_hari:', hariSource);
    return null;
  }

  const days = hariKeys.map((key, idx) => {
    const h = hariSource[key];
    const dayNum = idx + 1;

    const destinations = (h.destinations ?? []).map(d => ({
      name:     d.Place_Name    ?? d.name     ?? '(tanpa nama)',
      category: d.Category      ?? d.category ?? 'Umum',
      city:     d.kecamatan
        ? `${d.kecamatan}, ${d.City ?? ''}`.replace(/,\s*$/, '').trim()
        : (d.City ?? d.city ?? ''),
      price:    d.Price  ?? d.price  ?? 0,
      rating:   d.Rating ?? d.rating ?? 0,
      time:     d.estimated_arrival_time ?? d.time   ?? '--:--',
      travel:   d.travel_note_to_next    ?? d.travel ?? '',
    }));

    return {
      day:          dayNum,
      date:         `Hari ${dayNum}`,
      summary:      h.day_summary ?? `${destinations.length} destinasi`,
      destinations,
    };
  });

  return {
    days,
    budget:    raw.budget_limit           ?? 0,
    totalCost: raw.total_biaya_kalkulasi  ?? 0,
  };
}

export default function ItineraryCard({ data }) {
  const [openDay, setOpenDay] = useState(1);

  const normalized = normalizeData(data);

  // Guard: jika data tidak bisa dinormalisasi, jangan render apapun
  if (!normalized || normalized.days.length === 0) return null;

  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ background: 'var(--panel-bg)', border: '1px solid var(--border)' }}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--card-bg)' }}>
        <div className="flex items-center gap-2">
          <span className="text-base">🗺️</span>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Itinerary Wisata
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{
              background: 'var(--accent-glow)',
              color: 'var(--accent-light)',
              border: '1px solid rgba(79,110,247,0.3)',
            }}>
            {normalized.days.length} Hari
          </span>
        </div>
        <div className="text-right">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Total Biaya</p>
          <p className="text-sm font-bold" style={{ color: 'var(--success)' }}>
            Rp {normalized.totalCost.toLocaleString('id-ID')}
          </p>
          {normalized.budget > 0 && (
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              dari Rp {normalized.budget.toLocaleString('id-ID')} budget
            </p>
          )}
        </div>
      </div>

      {/* Day Tabs — hanya tampil jika lebih dari 1 hari */}
      {normalized.days.length > 1 && (
        <div className="flex border-b" style={{ borderColor: 'var(--border)' }}>
          {normalized.days.map(d => (
            <button key={d.day}
              onClick={() => setOpenDay(d.day)}
              className="flex-1 px-3 py-2.5 text-xs font-semibold transition-all duration-150"
              style={{
                background:   openDay === d.day ? 'var(--panel-bg)' : 'var(--card-bg)',
                color:        openDay === d.day ? 'var(--accent-light)' : 'var(--text-muted)',
                borderBottom: openDay === d.day ? '2px solid var(--accent)' : '2px solid transparent',
              }}>
              {d.date}
              <span className="block text-xs font-normal mt-0.5"
                style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
                {d.summary}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* 1 hari: tampilkan summary inline di bawah header */}
      {normalized.days.length === 1 && normalized.days[0].summary && (
        <div className="px-4 py-2 text-xs"
          style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', background: 'var(--card-bg)' }}>
          {normalized.days[0].summary}
        </div>
      )}

      {/* Destinations */}
      <div className="px-4 pt-4 pb-2 max-h-80 overflow-y-auto">
        {normalized.days
          .filter(d => normalized.days.length === 1 || d.day === openDay)
          .flatMap(d => d.destinations)
          .map((dest, i, arr) => (
            <DestinationRow key={i} dest={dest} isLast={i === arr.length - 1} />
          ))}
      </div>
    </div>
  );
}
