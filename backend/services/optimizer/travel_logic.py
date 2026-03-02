import pandas as pd
from typing import List, Dict, Any, Tuple

def calculate_optimized_itinerary(
    df: pd.DataFrame, 
    budget_limit: int, 
    location_keywords: List[str], 
    duration_days: int = 1, 
    min_rating: float = 0.0
) -> Tuple[bool, str, List[Dict[str, Any]], int]:
    """
    Fungsi murni berbasis Pandas untuk mencari kombinasi tempat wisata berdasarkan constraints.
    Returns: (is_success, message_or_error, list_of_recommendations, total_cost)
    """
    
    # 1. Filter Rating
    if 'Rating' in df.columns:
        filtered_df = df[df['Rating'] >= min_rating]
    else:
        filtered_df = df.copy()

    # 2. Filter Keyword Semantic
    if location_keywords and len(location_keywords) > 0 and location_keywords[0].lower() != "semua" and location_keywords[0] != "":
        # Buat mask pencarian string regex (case-insensitive)
        # Asumsikan kolom pencarian ada di 'City', 'Category', atau nama kolom relevan lainnya. Kita cari di seluruh kolom teks.
        text_cols = filtered_df.select_dtypes(include=['object', 'string']).columns
        
        mask = pd.Series([False]*len(filtered_df), index=filtered_df.index)
        for keyword in location_keywords:
            keyword_mask = pd.Series([False]*len(filtered_df), index=filtered_df.index)
            for col in text_cols:
                keyword_mask = keyword_mask | filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
            mask = mask | keyword_mask
        
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        return False, f"Tidak ada destinasi wisata yang cocok dengan rating >= {min_rating} dan kata kunci: {location_keywords}.", [], 0

    # 3. Urutkan berdasarkan Rating tertinggi lalu Harga terendah (Greedy approach simple)
    # Penamaan kolom harga standar, coba cari kolom yang mengandung kata 'price', 'harga'
    price_cols = [col for col in filtered_df.columns if 'price' in col.lower() or 'harga' in col.lower() or 'cost' in col.lower()]
    rating_cols = [col for col in filtered_df.columns if 'rating' in col.lower()]

    if not price_cols:
        return False, "Tidak dapat menemukan kolom Harga (Price) di dalam dataset untuk melakukan kalkulasi budget.", [], 0
    
    price_col = price_cols[0]
    rating_col = rating_cols[0] if rating_cols else None

    if rating_col:
        filtered_df = filtered_df.sort_values(by=[rating_col, price_col], ascending=[False, True])
    else:
        filtered_df = filtered_df.sort_values(by=price_col, ascending=True)

    # 4. Iterasikan Pengambilan
    selected_items = []
    current_cost = 0
    
    for _, row in filtered_df.iterrows():
        try:
            item_price = float(row[price_col])
        except (ValueError, TypeError):
             continue # Skip barang rusak harganya
             
        # Perkiraan kasar: tempat penginapan mungkin dikali duration_days.
        # Karena kita tidak tahu persis skemanya, kita asumsikan harga wisata dibayar harian/sekali
        # Untuk generalisasi: Kita tambahkan destination terus sampai budget habis
        
        if current_cost + item_price <= budget_limit:
            selected_items.append(row.to_dict())
            current_cost += int(item_price)
            
            # Berhenti jika kita sudah merasa cukup (misal 3 tempat per hari)
            if len(selected_items) >= duration_days * 3:
                break
                
    # 5. Logical Pushback jika tidak dapat tempat sama sekali padahal dataset terfilter > 0
    if not selected_items and not filtered_df.empty:
        min_price_available = filtered_df[price_col].min()
        return False, f"Budget Rp {budget_limit} terlalu kecil. Tempat termurah dari hasil filter kriteria Anda berharga Rp {min_price_available}. Silakan naikkan budget Anda.", [], 0

    return True, "Berhasil membuat rancangan itinerary.", selected_items, current_cost
