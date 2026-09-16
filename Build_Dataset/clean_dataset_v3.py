"""
Build_Dataset/clean_dataset_v3.py
==================================
Script pembangun dataset wisata Bali yang diperbaiki secara signifikan.

PERBAIKAN DARI v2:
1. Price: Deterministik berdasarkan (Kategori + Popularitas), bukan random.
   - Destinasi "Sangat Ramai" → harga cenderung lebih tinggi
   - Seeded dengan ID tempat agar reproducible
   - Harga 0 (gratis) dipertahankan untuk wisata alam tertentu

2. Description: Unik per destinasi, mempertimbangkan:
   - Nama tempat
   - Kategori sub-type (pantai, gunung, museum, pura, dll) yang dideteksi dari nama
   - Kecamatan & Kota
   - Level popularitas (berdasarkan Crowd_Density)
   - Karakteristik rating

3. Tambah kolom `price_category` (Gratis / Murah / Sedang / Mahal / Premium)
   untuk mempermudah filter agent dan RAG semantic search.

4. Tambah kolom `tags` berisi kata kunci semantik unik per destinasi
   (sangat berguna untuk RAG vector search agar bisa membedakan antar destinasi)

OUTPUT: Build_Dataset/bali_tourist_clean_v3.csv
"""

import pandas as pd
import numpy as np
import hashlib
import re

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HARGA DETERMINISTIK
# Menggunakan hash ID tempat sebagai seed agar reproducible
# ─────────────────────────────────────────────────────────────

PRICE_CONFIG = {
    # (Kategori, Crowd_Density) → list harga kandidat
    ("Alam",     "Sepi"):        [0, 0, 5000, 10000],
    ("Alam",     "Sedang"):      [0, 10000, 15000, 20000],
    ("Alam",     "Ramai"):       [10000, 20000, 25000, 30000],
    ("Alam",     "Sangat Ramai"): [15000, 20000, 30000, 50000],

    ("Budaya",   "Sepi"):        [10000, 15000, 20000, 25000],
    ("Budaya",   "Sedang"):      [20000, 25000, 30000, 50000],
    ("Budaya",   "Ramai"):       [25000, 30000, 50000, 75000],
    ("Budaya",   "Sangat Ramai"): [30000, 50000, 75000, 100000],

    ("Rekreasi", "Sepi"):        [25000, 50000, 75000, 100000],
    ("Rekreasi", "Sedang"):      [50000, 75000, 100000, 125000],
    ("Rekreasi", "Ramai"):       [75000, 100000, 125000, 150000],
    ("Rekreasi", "Sangat Ramai"): [100000, 125000, 150000, 200000],

    ("Umum",     "Sepi"):        [0, 0, 10000, 15000],
    ("Umum",     "Sedang"):      [0, 10000, 15000, 20000],
    ("Umum",     "Ramai"):       [10000, 20000, 25000, 30000],
    ("Umum",     "Sangat Ramai"): [15000, 25000, 30000, 50000],
}

def get_deterministic_price(id_tempat: str, kategori: str, crowd: str) -> int:
    """
    Memilih harga dari daftar kandidat secara deterministik berdasarkan hash ID tempat.
    Seeded → harga sama setiap kali script dijalankan.
    """
    candidates = PRICE_CONFIG.get((kategori, crowd), [0, 10000, 20000, 50000])
    # Hash ID tempat → integer → indeks ke dalam list kandidat
    hash_int = int(hashlib.md5(id_tempat.encode()).hexdigest(), 16)
    idx = hash_int % len(candidates)
    return candidates[idx]


# ─────────────────────────────────────────────────────────────
# DETEKSI SUB-TYPE DARI NAMA TEMPAT
# ─────────────────────────────────────────────────────────────

SUBTYPE_KEYWORDS = {
    # Alam
    "pantai":     "pantai",
    "beach":      "pantai",
    "bukit":      "bukit",
    "gunung":     "gunung",
    "air terjun": "air terjun",
    "danau":      "danau",
    "sawah":      "sawah",
    "hutan":      "hutan",
    "kebun":      "kebun",
    "taman":      "taman",
    "teluk":      "teluk",
    "laut":       "pantai",
    "sumber":     "mata air",
    "laguna":     "laguna",
    "mangrove":   "hutan mangrove",

    # Budaya
    "pura":       "pura",
    "temple":     "pura",
    "museum":     "museum",
    "candi":      "candi",
    "istana":     "istana",
    "puri":       "puri",
    "monumen":    "monumen",

    # Rekreasi
    "waterboom":  "wahana air",
    "waterpark":  "wahana air",
    "safari":     "safari",
    "zoo":        "kebun binatang",
    "kebun binatang": "kebun binatang",
    "adventure":  "wahana petualangan",
    "snorkeling": "snorkeling",
    "diving":     "diving",
    "surfing":    "surfing",
    "spa":        "spa",
    "resort":     "resort",
    "villa":      "villa",
}

def detect_subtype(nama: str) -> str:
    nama_lower = nama.lower()
    for kw, subtype in SUBTYPE_KEYWORDS.items():
        if kw in nama_lower:
            return subtype
    return None


# ─────────────────────────────────────────────────────────────
# GENERATOR DESKRIPSI UNIK PER DESTINASI
# ─────────────────────────────────────────────────────────────

def generate_rich_description(row: pd.Series) -> str:
    nama      = row["Place_Name"]
    kategori  = row["Category"]
    kota      = row["City"]
    kecamatan = row["kecamatan"]
    rating    = row["Rating"]
    crowd     = row["Crowd_Density"]
    price     = row["Price"]
    subtype   = detect_subtype(nama)

    # ── Rating narrative ──
    if rating >= 4.8:
        rating_desc = "dengan reputasi luar biasa dan ulasan sangat memuaskan"
    elif rating >= 4.5:
        rating_desc = "yang sangat direkomendasikan oleh para pengunjung"
    elif rating >= 4.0:
        rating_desc = "dengan penilaian baik dari wisatawan"
    else:
        rating_desc = "yang menawarkan pengalaman tersendiri bagi pengunjung"

    # ── Crowd narrative ──
    crowd_map = {
        "Sepi":        "suasananya tenang dan jarang ramai, sangat cocok untuk yang ingin menyendiri atau healing",
        "Sedang":      "tingkat kunjungannya sedang, tidak terlalu ramai sehingga nyaman dinikmati",
        "Ramai":       "cukup ramai dikunjungi, terutama di akhir pekan dan musim liburan",
        "Sangat Ramai":"sangat populer dan ramai dikunjungi, pastikan datang lebih awal untuk menghindari antrean",
    }
    crowd_desc = crowd_map.get(crowd, "memiliki tingkat kunjungan yang bervariasi")

    # ── Price narrative ──
    if price == 0:
        price_desc = "gratis tanpa biaya tiket masuk"
    elif price <= 15000:
        price_desc = f"dengan harga tiket terjangkau sekitar Rp {price:,}".replace(",", ".")
    elif price <= 50000:
        price_desc = f"dengan harga tiket masuk Rp {price:,}".replace(",", ".")
    else:
        price_desc = f"dengan harga tiket Rp {price:,} yang sebanding dengan fasilitas yang ditawarkan".replace(",", ".")

    # ── Subtype-specific intro ──
    intro_map = {
        "pantai": f"{nama} adalah destinasi pantai memukau di {kecamatan}, {kota}",
        "bukit": f"{nama} merupakan destinasi bukit yang menawarkan panorama alam menakjubkan di {kecamatan}, {kota}",
        "gunung": f"{nama} adalah gunung berapi yang menjadi ikon alam Bali di wilayah {kecamatan}, {kota}",
        "air terjun": f"{nama} adalah air terjun yang tersembunyi di antara pepohonan tropis {kecamatan}, {kota}",
        "danau": f"{nama} merupakan danau alami yang indah di kawasan {kecamatan}, {kota}",
        "sawah": f"{nama} adalah hamparan sawah terasering yang ikonik di {kecamatan}, {kota}",
        "hutan": f"{nama} merupakan kawasan hutan alami di {kecamatan}, {kota}",
        "pura": f"{nama} adalah pura suci yang menjadi pusat kegiatan spiritual di {kecamatan}, {kota}",
        "museum": f"{nama} adalah museum yang menyimpan koleksi berharga tentang budaya dan sejarah Bali di {kecamatan}, {kota}",
        "candi": f"{nama} merupakan situs candi bersejarah yang terletak di {kecamatan}, {kota}",
        "puri": f"{nama} adalah puri atau istana kerajaan Bali yang bersejarah di {kecamatan}, {kota}",
        "monumen": f"{nama} adalah monumen bersejarah yang berdiri kokoh di kawasan {kecamatan}, {kota}",
        "wahana air": f"{nama} adalah destinasi wahana air yang seru untuk keluarga di {kecamatan}, {kota}",
        "wahana petualangan": f"{nama} menawarkan berbagai wahana petualangan yang memacu adrenalin di {kecamatan}, {kota}",
        "snorkeling": f"{nama} adalah spot snorkeling terbaik dengan keindahan bawah laut di {kecamatan}, {kota}",
        "diving": f"{nama} adalah spot diving andalan dengan terumbu karang yang indah di {kecamatan}, {kota}",
        "surfing": f"{nama} merupakan destinasi surfing populer dengan ombak khas Bali di {kecamatan}, {kota}",
        "laguna": f"{nama} adalah laguna berair jernih yang menawan di {kecamatan}, {kota}",
        "hutan mangrove": f"{nama} adalah kawasan hutan mangrove yang penting secara ekologi di {kecamatan}, {kota}",
        "taman": f"{nama} adalah taman yang asri dan nyaman untuk bersantai di {kecamatan}, {kota}",
    }

    # ── Fallback intro by category ──
    fallback_intro = {
        "Alam":     f"{nama} adalah destinasi alam tersembunyi di {kecamatan}, {kota}",
        "Budaya":   f"{nama} adalah tempat wisata budaya yang kaya akan nilai sejarah di {kecamatan}, {kota}",
        "Rekreasi": f"{nama} adalah destinasi rekreasi seru di {kecamatan}, {kota}",
        "Umum":     f"{nama} adalah tempat wisata yang menarik di {kecamatan}, {kota}",
    }

    intro = intro_map.get(subtype) if subtype else fallback_intro.get(kategori, f"{nama} terletak di {kecamatan}, {kota}")

    # ── Assemble description ──
    desc = (
        f"{intro}, {rating_desc}. "
        f"Tempat ini {crowd_desc}, dan dapat dikunjungi {price_desc}. "
    )

    # ── Category-specific closing sentence ──
    if kategori == "Alam":
        desc += f"Nikmati keindahan alam Bali yang autentik, udara segar, dan pemandangan yang menenangkan jiwa di sini."
    elif kategori == "Budaya":
        desc += f"Tempat ini sangat edukatif dan memberikan wawasan mendalam tentang seni, tradisi, dan warisan budaya Bali."
    elif kategori == "Rekreasi":
        desc += f"Destinasi ini menyediakan berbagai aktivitas seru yang cocok untuk keluarga, pasangan, maupun grup wisata."
    else:  # Umum
        desc += f"Destinasi yang fleksibel untuk berbagai jenis wisatawan, cocok untuk berfoto, bersantai, dan menikmati suasana Bali."

    return desc


# ─────────────────────────────────────────────────────────────
# GENERATOR TAGS SEMANTIK PER DESTINASI
# Tags ini sangat berguna untuk RAG vector search
# ─────────────────────────────────────────────────────────────

def generate_tags(row: pd.Series) -> str:
    tags = []

    # Kategori
    kategori = row["Category"]
    tags.append(kategori.lower())

    # Kota dan kecamatan
    kota = row["City"].replace("Kabupaten ", "").replace("Kota ", "")
    tags.append(kota.lower())
    tags.append(row["kecamatan"].lower())

    # Crowd
    crowd_tags = {
        "Sepi":        ["sepi", "tenang", "quiet", "uncrowded", "healing", "damai"],
        "Sedang":      ["sedang", "tidak terlalu ramai", "nyaman"],
        "Ramai":       ["ramai", "populer", "popular"],
        "Sangat Ramai":["sangat ramai", "sangat populer", "terkenal", "famous", "must visit"],
    }
    tags.extend(crowd_tags.get(row["Crowd_Density"], []))

    # Subtype
    subtype = detect_subtype(row["Place_Name"])
    if subtype:
        tags.append(subtype)

    # Price range tags
    price = row["Price"]
    if price == 0:
        tags.extend(["gratis", "free", "tanpa biaya"])
    elif price <= 20000:
        tags.extend(["murah", "hemat", "budget friendly", "terjangkau"])
    elif price <= 75000:
        tags.extend(["harga sedang"])
    else:
        tags.extend(["harga mahal", "premium", "mewah"])

    # Rating tags
    if row["Rating"] >= 4.7:
        tags.extend(["highly rated", "top rated", "rekomendasi utama"])
    elif row["Rating"] >= 4.4:
        tags.extend(["direkomendasikan", "bagus"])

    # Category-specific
    if kategori == "Alam":
        tags.extend(["wisata alam", "nature", "outdoor", "hijau", "pemandangan"])
    elif kategori == "Budaya":
        tags.extend(["wisata budaya", "sejarah", "edukasi", "culture", "heritage"])
    elif kategori == "Rekreasi":
        tags.extend(["wisata rekreasi", "liburan keluarga", "fun", "hiburan"])

    return ", ".join(sorted(set(tags)))


# ─────────────────────────────────────────────────────────────
# KATEGORISASI HARGA
# ─────────────────────────────────────────────────────────────

def get_price_category(price: int) -> str:
    if price == 0:
        return "Gratis"
    elif price <= 20000:
        return "Murah"
    elif price <= 50000:
        return "Sedang"
    elif price <= 100000:
        return "Mahal"
    else:
        return "Premium"


# ─────────────────────────────────────────────────────────────
# FUNGSI CROWD DENSITY (ditingkatkan)
# ─────────────────────────────────────────────────────────────

def get_crowd_density(jumlah_rating) -> str:
    try:
        jml = float(jumlah_rating)
        if pd.isna(jml): return "Sedang"
        if jml < 50:    return "Sepi"
        elif jml < 300:  return "Sedang"
        elif jml < 1000: return "Ramai"
        else:            return "Sangat Ramai"
    except:
        return "Sedang"


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def build_dataset():
    input_file  = r"d:\Semester 10\Data_Whisperer_v1.0\data\dataset\Bali dest\bali_tourist_destination.csv"
    output_file = r"d:\Semester 10\Data_Whisperer_v1.0\Build_Dataset\bali_tourist_clean_v3.csv"

    print("📥  Membaca dataset Kaggle asli...")
    df = pd.read_csv(input_file, encoding="utf-8")
    print(f"    → {len(df)} baris, {len(df.columns)} kolom")

    # ── Rename ──
    df.rename(columns={
        "nama_tempat_wisata": "Place_Name",
        "kategori":           "Category",
        "kabupaten_kota":     "City",
        "rating":             "Rating",
    }, inplace=True)

    # ── Missing values ──
    df["Category"] = df["Category"].fillna("Umum")
    df["Rating"]   = df["Rating"].fillna(4.0)

    # ── Crowd Density (lebih sensitif threshold) ──
    print("🔢  Menghitung Crowd_Density...")
    df["Crowd_Density"] = df["jumlah_rating"].apply(get_crowd_density)

    # ── Price: DETERMINISTIK via hash ID ──
    print("💰  Menghitung Price secara deterministik...")
    df["Price"] = df.apply(
        lambda row: get_deterministic_price(row["id_tempat"], row["Category"], row["Crowd_Density"]),
        axis=1
    )

    # ── Price Category ──
    df["Price_Category"] = df["Price"].apply(get_price_category)

    # ── Tags (sebelum description karena description bisa pakai tags) ──
    print("🏷️   Membuat Tags semantik...")
    df["tags"] = df.apply(generate_tags, axis=1)

    # ── Description: UNIK per destinasi ──
    print("📝  Membuat Deskripsi unik per destinasi...")
    df["Description"] = df.apply(generate_rich_description, axis=1)

    # ── Pilih & Urutkan kolom final ──
    final_cols = [
        "Place_Name", "Description", "Category", "City", "kecamatan",
        "Price", "Price_Category", "Rating", "jumlah_rating",
        "Crowd_Density", "tags",
        "latitude", "longitude", "link_google_maps", "link_gambar"
    ]
    df = df[final_cols]

    # ── Simpan ──
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n✅  Dataset berhasil disimpan ke: {output_file}")
    print(f"    Jumlah baris  : {len(df)}")
    print(f"    Jumlah kolom  : {len(df.columns)}")
    print(f"\n📊  Distribusi Category:")
    print(df["Category"].value_counts().to_string())
    print(f"\n📊  Distribusi Crowd_Density:")
    print(df["Crowd_Density"].value_counts().to_string())
    print(f"\n📊  Distribusi Price_Category:")
    print(df["Price_Category"].value_counts().to_string())
    print(f"\n📊  Statistik Price:")
    print(df["Price"].describe().to_string())
    print(f"\n📄  Sample 2 Deskripsi:")
    print(f"\n[1] {df['Place_Name'].iloc[0]}")
    print(f"    {df['Description'].iloc[0]}")
    print(f"    Tags: {df['tags'].iloc[0]}")
    print(f"\n[2] {df['Place_Name'].iloc[100]}")
    print(f"    {df['Description'].iloc[100]}")
    print(f"    Tags: {df['tags'].iloc[100]}")

    return df


if __name__ == "__main__":
    build_dataset()
