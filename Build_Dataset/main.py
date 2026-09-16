import pandas as pd
import random

def generate_mock_travel_dataset():
    """
    Men-generate dataset dummy wisata Indonesia yang sempurna sesuai dengan
    skema Neuro-Symbolic Agent kita (Price, Rating, Description, dst.)
    """
    
    # Destinasi Wisata Mock (Campuran Bali & Yogyakarta)
    destinations = [
        {"Place_Name": "Pantai Kuta", "City": "Bali", "Category": "Pantai", "Price": 0, "Rating": 4.2, "Crowd_Density": "Ramai",
         "Description": "Pantai pasir putih ikonik di Bali yang sangat populer untuk berselancar dan menikmati pemandangan matahari terbenam. Selalu dipadati turis mancanegara."},
        {"Place_Name": "Desa Wisata Penglipuran", "City": "Bali", "Category": "Budaya", "Price": 25000, "Rating": 4.8, "Crowd_Density": "Sedang",
         "Description": "Desa adat tertradisional di Bali dengan tata ruang yang sangat bersih dan asri. Cocok untuk healing mencari ketenangan budaya lokal."},
        {"Place_Name": "Monkey Forest Ubud", "City": "Bali", "Category": "Alam", "Price": 80000, "Rating": 4.5, "Crowd_Density": "Sedang",
         "Description": "Hutan sakral yang dikelilingi pura kuno dan dihuni oleh ratusan kera liar. Teduh, sejuk, dan memberikan vibes spiritual."},
        {"Place_Name": "Mount Batur Sunrise Trekking", "City": "Bali", "Category": "Petualangan", "Price": 350000, "Rating": 4.7, "Crowd_Density": "Sepi",
         "Description": "Pendakian gunung api aktif untuk melihat matahari terbit. Membutuhkan fisik yang kuat. Sangat tenang dan menawarkan pemandangan spektakuler."},
        {"Place_Name": "Candi Borobudur", "City": "Yogyakarta", "Category": "Sejarah", "Price": 50000, "Rating": 4.9, "Crowd_Density": "Ramai",
         "Description": "Candi Buddha terbesar di dunia. Kaya akan nilai sejarah dan arsitektur megah peninggalan wangsa Syailendra."},
        {"Place_Name": "Jalan Malioboro", "City": "Yogyakarta", "Category": "Belanja", "Price": 0, "Rating": 4.6, "Crowd_Density": "Sangat Ramai",
         "Description": "Pusat perbelanjaan jalanan legendaris di Jogja. Buka hingga malam, penuh dengan pedagang kaki lima, musisi jalanan, dan sangat hidup."},
        {"Place_Name": "Pantai Parangtritis", "City": "Yogyakarta", "Category": "Pantai", "Price": 10000, "Rating": 4.3, "Crowd_Density": "Ramai",
         "Description": "Pantai dengan ombak besar di selatan Jogja. Terkenal dengan mitos pesisir selatan dan pemandangan matahari terbenam yang eksotis."},
        {"Place_Name": "Goa Jomblang", "City": "Yogyakarta", "Category": "Petualangan", "Price": 500000, "Rating": 4.8, "Crowd_Density": "Sepi",
         "Description": "Gua vertikal dengan cahaya surga (cahaya matahari yang menembus lubang gua). Eksklusif, sepi, tiket mahal, namun menawarkan pengalaman tak terlupakan untuk kaum introvert."},
        {"Place_Name": "Campuhan Ridge Walk", "City": "Bali", "Category": "Alam", "Price": 0, "Rating": 4.6, "Crowd_Density": "Sepi",
         "Description": "Jalur trekking ikonik di perbukitan Ubud dengan pemandangan ilalang hijau yang menyejukkan mata. Sangat sunyi dan cocok untuk olahraga pagi atau sore."},
        {"Place_Name": "Museum Ullen Sentalu", "City": "Yogyakarta", "Category": "Sejarah", "Price": 100000, "Rating": 4.9, "Crowd_Density": "Sedang",
         "Description": "Museum seni dan budaya Jawa yang letaknya tersembunyi di lereng Merapi. Udaranya sejuk, suasananya sangat tenang dan eksklusif dengan aturan dilarang memotret besar-besaran."}
    ]

    df = pd.DataFrame(destinations)
    
    # Generate CSV file
    output_filename = "data_wisata_mock_TA.csv"
    df.to_csv(output_filename, index=False)
    print(f"✅ Berhasil membuat mock dataset: {output_filename}")
    print("Dataset ini sudah disesuaikan persis dengan skema yang dibutuhkan oleh AI Agent Anda!")
    
if __name__ == "__main__":
    generate_mock_travel_dataset()
