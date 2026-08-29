import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

MAX_PLACES_PER_DAY   = 3    # Maks destinasi per hari
AVERAGE_SPEED_KMH    = 30   # Kecepatan rata-rata kendaraan di Bali (km/h), kondisi jalan lokal
VISIT_DURATION_MIN   = 90   # Estimasi durasi kunjungan per tempat (menit)
START_HOUR           = 9    # Jam mulai kunjungan (09:00)


def calculate_ahp_weights() -> Dict[str, Any]:
    """
    Hitung bobot komposit menggunakan Analytic Hierarchy Process (AHP).

    Pairwise comparison matrix (skala Saaty 1-9):
    Kriteria: Rating, Value for Money, Crowd Density

    Judgments (berdasarkan domain knowledge pariwisata Bali):
    - Rating vs Value    : 3 (Rating moderately more important — kualitas pengalaman
                              adalah indikator utama kepuasan wisatawan)
    - Rating vs Crowd    : 5 (Rating strongly more important — wisatawan memprioritaskan
                              kualitas destinasi di atas level keramaian)
    - Value vs Crowd     : 2 (Value slightly more important — efisiensi budget lebih
                              diprioritaskan daripada kenyamanan kerumunan)

    Returns:
        Dict berisi weights, lambda_max, CI, CR, dan metadata AHP.
    """
    A = np.array([
        [1,     3,     5   ],
        [1/3,   1,     2   ],
        [1/5,   1/2,   1   ],
    ])
    criteria = ["Rating", "Value_for_Money", "Crowd_Density"]

    col_sums = A.sum(axis=0)
    A_norm   = A / col_sums

    weights = A_norm.mean(axis=1)

    n          = len(criteria)
    Aw         = A @ weights
    lambda_max = float(np.mean(Aw / weights))
    CI         = (lambda_max - n) / (n - 1)
    RI_table   = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    CR         = CI / RI_table[n]

    return {
        "criteria":    criteria,
        "weights":     dict(zip(criteria, np.round(weights, 4).tolist())),
        "W_RATING":    round(float(weights[0]), 4),
        "W_VALUE":     round(float(weights[1]), 4),
        "W_CROWD":     round(float(weights[2]), 4),
        "lambda_max":  round(float(lambda_max), 4),
        "CI":          round(float(CI), 6),
        "CR":          round(float(CR), 6),
        "consistent":  CR < 0.10,
        "method":      "Analytic Hierarchy Process (Saaty, 1980)",
        "pairwise_judgments": {
            "Rating_vs_Value": "3 (Rating moderately more important)",
            "Rating_vs_Crowd": "5 (Rating strongly more important)",
            "Value_vs_Crowd":  "2 (Value slightly more important)",
        },
    }

_AHP_RESULT = calculate_ahp_weights()

W_RATING = _AHP_RESULT["W_RATING"]
W_VALUE  = _AHP_RESULT["W_VALUE"]
W_CROWD  = _AHP_RESULT["W_CROWD"]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Hitung jarak (km) antara dua titik koordinat menggunakan rumus Haversine.
    Rumus: d = 2R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))
    R = 6371 km (jari-jari bumi).

    Args:
        lat1, lon1 : koordinat titik awal (derajat desimal)
        lat2, lon2 : koordinat titik tujuan (derajat desimal)

    Returns:
        Jarak dalam kilometer (float).
    """
    R = 6371.0

    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(R * c, 2)


def estimate_travel_minutes(distance_km: float) -> int:
    """
    Estimasi waktu perjalanan dalam menit berdasarkan jarak.
    Menggunakan kecepatan rata-rata kendaraan di Bali (30 km/h).
    Tambah 10 menit overhead (parkir, dll).
    """
    travel_hours  = distance_km / AVERAGE_SPEED_KMH
    travel_minutes = int(travel_hours * 60) + 10
    return travel_minutes

def sort_by_nearest_neighbor(
    destinations: List[Dict[str, Any]],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> List[Dict[str, Any]]:
    """
    Urutkan destinasi menggunakan algoritma Nearest Neighbor Greedy
    untuk meminimalkan total jarak perjalanan dalam satu hari.

    Algoritma:
        1. Mulai dari destinasi pertama (titik awal)
        2. Dari posisi saat ini, cari destinasi TERDEKAT yang belum dikunjungi
        3. Pindah ke sana, ulangi sampai semua destinasi dikunjungi

    Ini adalah aproksimasi Travelling Salesman Problem (TSP) yang
    deterministik dan efisien untuk jumlah destinasi kecil (≤ 3/hari).

    Args:
        destinations : list of dicts, masing-masing punya lat_col & lon_col
        lat_col      : nama kolom latitude di dict
        lon_col      : nama kolom longitude di dict

    Returns:
        List destinasi yang sudah diurutkan secara geografis.
    """
    if len(destinations) <= 1:
        return destinations

    valid   = [d for d in destinations if d.get(lat_col) and d.get(lon_col)]
    invalid = [d for d in destinations if not (d.get(lat_col) and d.get(lon_col))]

    if len(valid) <= 1:
        return destinations

    visited   = [False] * len(valid)
    ordered   = []
    current_i = 0
    visited[current_i] = True
    ordered.append(valid[current_i])

    for _ in range(len(valid) - 1):
        current = ordered[-1]
        current_lat = current[lat_col]
        current_lon = current[lon_col]

        min_dist = float("inf")
        next_i   = -1

        for j, dest in enumerate(valid):
            if visited[j]:
                continue
            dist = haversine_distance(current_lat, current_lon, dest[lat_col], dest[lon_col])
            if dist < min_dist:
                min_dist = dist
                next_i   = j

        if next_i != -1:
            visited[next_i] = True
            ordered.append(valid[next_i])

    return ordered + invalid

def enrich_with_travel_metadata(
    destinations: List[Dict[str, Any]],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    start_hour: int = START_HOUR,
) -> List[Dict[str, Any]]:
    """
    Tambahkan field metadata perjalanan ke setiap destinasi:
        - distance_to_next_km        : jarak ke destinasi berikutnya (km)
        - estimated_travel_time_to_next_min : estimasi waktu ke destinasi berikutnya (menit)
        - estimated_arrival_time     : jam tiba (format HH:MM)
        - estimated_departure_time   : jam berangkat (setelah kunjungan selesai)

    Args:
        destinations : list of dicts yang sudah diurutkan secara geografis
        lat_col      : kolom latitude
        lon_col      : kolom longitude
        start_hour   : jam mulai kunjungan pertama (default 09:00)

    Returns:
        List destinasi dengan field tambahan.
    """
    enriched = []
    current_time_min = start_hour * 60

    for i, dest in enumerate(destinations):
        dest = dest.copy()

        arrival_h = current_time_min // 60
        arrival_m = current_time_min % 60
        dest["estimated_arrival_time"] = f"{arrival_h:02d}:{arrival_m:02d}"

        departure_time_min = current_time_min + VISIT_DURATION_MIN
        depart_h = departure_time_min // 60
        depart_m = departure_time_min % 60
        dest["estimated_departure_time"] = f"{depart_h:02d}:{depart_m:02d}"

        if i < len(destinations) - 1:
            next_dest = destinations[i + 1]
            cur_lat  = dest.get(lat_col)
            cur_lon  = dest.get(lon_col)
            nxt_lat  = next_dest.get(lat_col)
            nxt_lon  = next_dest.get(lon_col)

            if all(v is not None for v in [cur_lat, cur_lon, nxt_lat, nxt_lon]):
                dist_km      = haversine_distance(cur_lat, cur_lon, nxt_lat, nxt_lon)
                travel_min   = estimate_travel_minutes(dist_km)
                dest["distance_to_next_km"]                 = dist_km
                dest["estimated_travel_time_to_next_min"]   = travel_min
                dest["travel_note_to_next"]                 = (
                    f"~{dist_km} km, ±{travel_min} menit perjalanan ke destinasi berikutnya"
                )
            else:
                dest["distance_to_next_km"]                 = None
                dest["estimated_travel_time_to_next_min"]   = None
                dest["travel_note_to_next"]                 = "Data koordinat tidak tersedia"

            current_time_min = departure_time_min + (dest.get("estimated_travel_time_to_next_min") or 15)
        else:
            dest["distance_to_next_km"]                 = None
            dest["estimated_travel_time_to_next_min"]   = None
            dest["travel_note_to_next"]                 = "Destinasi terakhir hari ini"

        enriched.append(dest)

    return enriched

def build_daily_itinerary(
    selected_items: List[Dict[str, Any]],
    duration_days: int,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    places_per_day: int = MAX_PLACES_PER_DAY,
) -> Dict[str, Any]:
    """
    Bagi daftar destinasi terpilih ke dalam struktur per hari.

    Proses per hari:
        1. Ambil slice maks `places_per_day` destinasi
        2. Urutkan secara geografis (Nearest Neighbor)
        3. Enrich dengan metadata waktu perjalanan
        4. Hitung total jarak hari itu

    Returns:
        {
          "hari_1": {
              "destinations": [...],      # List destinasi + metadata
              "total_destinations": int,
              "total_distance_km": float,
              "day_summary": str,
          },
          "hari_2": { ... },
          ...
          "total_hari": int,
          "total_destinasi": int,
        }
    """
    daily_result = {}
    total_destinasi = 0

    for day_num in range(1, duration_days + 1):
        start_idx = (day_num - 1) * places_per_day
        end_idx   = start_idx + places_per_day
        day_items = selected_items[start_idx:end_idx]

        if not day_items:
            break

        sorted_items = sort_by_nearest_neighbor(day_items, lat_col, lon_col)
        enriched = enrich_with_travel_metadata(sorted_items, lat_col, lon_col)

        total_distance = sum(
            d.get("distance_to_next_km") or 0
            for d in enriched
            if d.get("distance_to_next_km") is not None
        )
    
        names = [d.get("Place_Name", d.get("Nama", "Tempat")) for d in enriched]
        first_arr = enriched[0].get("estimated_arrival_time", "09:00") if enriched else "09:00"
        last_dep  = enriched[-1].get("estimated_departure_time", "17:00") if enriched else "17:00"

        day_summary = (
            f"Hari {day_num}: {len(enriched)} destinasi | "
            f"Mulai {first_arr} — Selesai {last_dep} | "
            f"Total Jarak ±{round(total_distance, 1)} km"
        )

        daily_result[f"hari_{day_num}"] = {
            "destinations":      enriched,
            "total_destinations": len(enriched),
            "total_distance_km": round(total_distance, 2),
            "day_summary":       day_summary,
            "destination_names": names,
        }
        total_destinasi += len(enriched)

    daily_result["total_hari"]      = len([k for k in daily_result if k.startswith("hari_")])
    daily_result["total_destinasi"] = total_destinasi

    return daily_result

def calculate_optimized_itinerary(
    df: pd.DataFrame,
    budget_limit: int,
    location_keywords: List[str],
    duration_days: int = 1,
    min_rating: float = 0.0,
) -> Tuple[bool, str, List[Dict[str, Any]], int, Optional[Dict[str, Any]]]:
    """
    Fungsi utama optimizer itinerary berbasis Pandas.

    Returns:
        (is_success, message, flat_list, total_cost, daily_structure)
        - is_success     : True jika berhasil
        - message        : pesan sukses atau error
        - flat_list      : list flat semua rekomendasi (kompatibilitas backward)
        - total_cost     : total biaya yang dikalkulasi
        - daily_structure: dict berstruktur per hari dengan metadata geografis
                           (None jika gagal)
    """

    duration_days = max(1, int(duration_days))
    budget_limit  = max(0, int(budget_limit))

    if "Rating" in df.columns:
        filtered_df = df[df["Rating"] >= min_rating].copy()
    else:
        filtered_df = df.copy()

    if (
        location_keywords
        and len(location_keywords) > 0
        and location_keywords[0].lower() not in ("semua", "")
    ):
        all_text_cols = list(filtered_df.select_dtypes(include=["object", "string"]).columns)
        tags_col      = "tags" if "tags" in filtered_df.columns else None
        fallback_cols = [c for c in all_text_cols if c != tags_col]

        # ─── Smart Filter: pisahkan KATEGORI (OR) vs LOKASI (AND) ───
        # Deteksi keyword lokasi dari data aktual di kolom City & kecamatan
        _geo_names = set()
        if "City" in filtered_df.columns:
            for v in filtered_df["City"].dropna().unique():
                for part in str(v).lower().replace("kabupaten", "").replace("kota", "").split():
                    if len(part) > 2:
                        _geo_names.add(part.strip())
        if "kecamatan" in filtered_df.columns:
            for v in filtered_df["kecamatan"].dropna().unique():
                _geo_names.add(str(v).strip().lower())

        geo_kws     = [kw for kw in location_keywords if kw.lower().strip() in _geo_names]
        content_kws = [kw for kw in location_keywords if kw.lower().strip() not in _geo_names]

        def _build_kw_mask(kw):
            """Bangun mask untuk satu keyword di semua kolom teks."""
            m = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            if tags_col:
                m = m | filtered_df[tags_col].astype(str).str.contains(kw, case=False, na=False)
            for col in fallback_cols:
                m = m | filtered_df[col].astype(str).str.contains(kw, case=False, na=False)
            return m

        # Kategori/content keywords → OR (user mau Budaya ATAU Pantai)
        if content_kws:
            cat_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for kw in content_kws:
                cat_mask = cat_mask | _build_kw_mask(kw)
        else:
            cat_mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)

        # Lokasi/geographic keywords → AND (user mau destinasi DI Bangli)
        if geo_kws:
            loc_mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)
            for kw in geo_kws:
                loc_mask = loc_mask & _build_kw_mask(kw)
        else:
            loc_mask = pd.Series([True] * len(filtered_df), index=filtered_df.index)

        # Final: (Budaya OR Pantai) AND (Bangli)
        mask = cat_mask & loc_mask
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        return (
            False,
            f"Tidak ada destinasi wisata yang cocok dengan rating >= {min_rating} "
            f"dan kata kunci: {location_keywords}.",
            [], 0, None
        )

    price_cols = [
        col for col in filtered_df.columns
        if ("price" in col.lower() or "harga" in col.lower() or "cost" in col.lower())
        and pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    rating_cols = [col for col in filtered_df.columns if "rating" in col.lower()]

    if not price_cols:
        return (
            False,
            "Tidak dapat menemukan kolom Harga (Price) numerik di dalam dataset.",
            [], 0, None
        )

    price_col  = price_cols[0]
    rating_col = rating_cols[0] if rating_cols else None

    if rating_col and filtered_df[rating_col].max() > filtered_df[rating_col].min():
        r_min = filtered_df[rating_col].min()
        r_max = filtered_df[rating_col].max()
        rating_norm = (filtered_df[rating_col] - r_min) / (r_max - r_min + 1e-9)
    elif rating_col:
        rating_norm = pd.Series(1.0, index=filtered_df.index)
    else:
        rating_norm = pd.Series(0.5, index=filtered_df.index)

    p_min = filtered_df[price_col].min()
    p_max = filtered_df[price_col].max()
    if p_max > p_min:
        price_norm  = (filtered_df[price_col] - p_min) / (p_max - p_min + 1e-9)
    else:
        price_norm  = pd.Series(0.5, index=filtered_df.index)
    value_norm = 1.0 - price_norm

    CROWD_SCORE = {"sepi": 1.0, "sedang": 0.6, "ramai": 0.3, "sangat ramai": 0.0}
    crowd_col   = next(
        (c for c in filtered_df.columns if "crowd" in c.lower() or "keramaian" in c.lower()),
        None
    )
    if crowd_col:
        crowd_norm = filtered_df[crowd_col].apply(
            lambda v: CROWD_SCORE.get(str(v).strip().lower(), 0.5)
        )
    else:
        crowd_norm = pd.Series(0.5, index=filtered_df.index)

    filtered_df = filtered_df.copy()
    filtered_df["composite_score"] = (
        W_RATING * rating_norm +
        W_VALUE  * value_norm  +
        W_CROWD  * crowd_norm
    ).round(4)

    filtered_df = filtered_df.sort_values(by="composite_score", ascending=False)

    selected_items: List[Dict[str, Any]] = []
    current_cost  = 0
    max_places    = duration_days * MAX_PLACES_PER_DAY
    MAX_SAME_CATEGORY = 2
    categories_per_day: Dict[int, Dict[str, int]] = {}

    cat_col = next(
        (c for c in filtered_df.columns if "category" in c.lower() or "kategori" in c.lower()),
        None
    )

    for _, row in filtered_df.iterrows():
        if len(selected_items) >= max_places:
            break

        try:
            item_price = float(row[price_col])
        except (ValueError, TypeError):
            continue

        if current_cost + item_price > budget_limit:
            continue

        current_day = len(selected_items) // MAX_PLACES_PER_DAY

        if cat_col:
            category = str(row.get(cat_col, "Umum")).strip()
            day_cats = categories_per_day.setdefault(current_day, {})
            if day_cats.get(category, 0) >= MAX_SAME_CATEGORY:
                continue

            day_cats[category] = day_cats.get(category, 0) + 1

        selected_items.append(row.to_dict())
        current_cost += int(item_price)
    if not selected_items and not filtered_df.empty:
        min_price = filtered_df[price_col].min()
        return (
            False,
            f"Budget Rp {budget_limit:,} terlalu kecil. "
            f"Tempat termurah dari kriteria Anda berharga Rp {int(min_price):,}. "
            f"Silakan naikkan budget Anda.",
            [], 0, None
        )

    lat_col = "latitude"  if "latitude"  in filtered_df.columns else None
    lon_col = "longitude" if "longitude" in filtered_df.columns else None

    daily_structure = None
    if lat_col and lon_col:
        daily_structure = build_daily_itinerary(
            selected_items=selected_items,
            duration_days=duration_days,
            lat_col=lat_col,
            lon_col=lon_col,
        )

    message = (
        f"Berhasil membuat rancangan itinerary {duration_days} hari "
        f"dengan {len(selected_items)} destinasi wisata."
    )

    return True, message, selected_items, current_cost, daily_structure
