import pandas as pd
import random
import numpy as np

def generate_description(nama, kategori, kota):
    # Deskripsi sederhana berdasarkan kategori
    desc_map = {
        "Alam": f"{nama} adalah salah satu destinasi alam terbaik di {kota}. Tempat ini menawarkan pemandangan natural yang asri, udara segar, dan jauh dari hiruk pikuk kota. Sangat cocok untuk bersantai, healing, dan menikmati keindahan alam Bali.",
        "Budaya": f"Kunjungi {nama} untuk merasakan kekayaan budaya dan sejarah di {kota}. Tempat ini penuh dengan nilai seni, tradisi lokal, dan arsitektur khas Bali. Edukatif dan menenangkan, cocok untuk wisatawan yang ingin mengenal Bali lebih dalam.",
        "Rekreasi": f"{nama} merupakan tempat rekreasi populer di {kota}, ideal untuk liburan keluarga atau teman. Fasilitas lengkap, suasana menyenangkan, dan banyak aktivitas seru. Tempat yang pas untuk melepas penat dan bersenang-senang.",
        "Umum": f"Tempat wisata {nama} yang terletak di {kota} ini menawarkan daya tarik unik untuk segala usia. Tempat yang bagus untuk berfoto, berjalan-jalan santai, dan menikmati suasana lokal Bali."
    }
    return desc_map.get(kategori, f"{nama} adalah tempat menarik di {kota} dengan pemandangan indah dan suasana khas. Destinasi yang wajib dikunjungi saat berada di Bali.")

def get_random_price(kategori):
    # Harga acak masuk akal
    if kategori == "Alam":
         return random.choice([0, 10000, 20000, 50000])
    elif kategori == "Budaya":
         return random.choice([20000, 30000, 50000, 100000])
    elif kategori == "Rekreasi":
         return random.choice([50000, 100000, 150000, 200000])
    else:
         return random.choice([0, 10000, 25000, 50000])

def get_crowd_density(jumlah_rating):
    try:
        jml = float(jumlah_rating)
        if pd.isna(jml): return "Sedang"
        if jml < 100: return "Sepi"
        elif jml < 500: return "Sedang"
        elif jml < 1500: return "Ramai"
        else: return "Sangat Ramai"
    except:
        return "Sedang"

def clean_and_transform_dataset():
    input_file = r"d:\Semester 10\Data_Whisperer_v1.0\data\dataset\Bali dest\bali_tourist_destination.csv"
    output_file = r"d:\Semester 10\Data_Whisperer_v1.0\Build_Dataset\bali_tourist_clean_v2.csv"
    
    try:
        # Load dataset
        df = pd.read_csv(input_file)
        
        # Rename columns
        df.rename(columns={
            'nama_tempat_wisata': 'Place_Name',
            'kategori': 'Category',
            'kabupaten_kota': 'City',
            'rating': 'Rating'
        }, inplace=True)
        
        # Filter only needed core cols for now (bisa tetap simpan long/lat jika perlu)
        core_cols = [
            'Place_Name', 'Category', 'City', 'kecamatan', 
            'Rating', 'jumlah_rating', 'latitude', 'longitude', 
            'link_google_maps', 'link_gambar'
        ]
        df = df[core_cols].copy()
        
        # Handle NA in Category and Rating
        df['Category'] = df['Category'].fillna("Umum")
        df['Rating'] = df['Rating'].fillna(4.0)
        
        # Add new mock columns
        print("Generating mock Price, Crowd_Density, and Description...")
        df['Price'] = df['Category'].apply(get_random_price)
        
        df['Crowd_Density'] = df['jumlah_rating'].apply(get_crowd_density)
        
        df['Description'] = df.apply(lambda row: generate_description(row['Place_Name'], row['Category'], row['City']), axis=1)
        
        # Reorder columns
        final_cols = [
            'Place_Name', 'Description', 'Category', 'City', 'kecamatan', 
            'Price', 'Rating', 'jumlah_rating', 'Crowd_Density', 
            'latitude', 'longitude', 'link_google_maps', 'link_gambar'
        ]
        df = df[final_cols]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully cleaned and saved to {output_file}")
        print(f"Total rows: {len(df)}")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_and_transform_dataset()
