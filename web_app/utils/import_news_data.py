import os
import csv
import psycopg2
from datetime import datetime
from dateutil import parser  # 💡 Parser yang lebih fleksibel

# ==========================
# KONEKSI KE DATABASE
# ==========================
conn = psycopg2.connect("dbname=newsportal user=postgres password=24Dappe08")
conn.autocommit = True
cur = conn.cursor()

# ==========================
# LOKASI FOLDER FILE CSV
# ==========================
folder_path = r"C:\Program Files\PostgreSQL\18\data\import"

# Ambil semua file CSV di folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print(f"📁 Ditemukan {len(csv_files)} file CSV untuk diproses.")

# ==========================
# PEMETAAN BULAN INDONESIA -> INGGRIS
# ==========================
bulan_map = {
    # Singkatan
    "Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr",
    "Mei": "May", "Jun": "Jun", "Jul": "Jul", "Agu": "Aug",
    "Sep": "Sep", "Okt": "Oct", "Nov": "Nov", "Des": "Dec",
    # Panjang
    "Januari": "January", "Februari": "February", "Maret": "March", "April": "April",
    "Mei": "May", "Juni": "June", "Juli": "July", "Agustus": "August",
    "September": "September", "Oktober": "October", "November": "November", "Desember": "December"
}

# ==========================
# FUNGSI PARSING WAKTU FLEXIBLE
# ==========================
def parse_waktu(waktu_raw):
    if not waktu_raw:
        return None

    waktu_raw = waktu_raw.strip()

    # Hilangkan teks tambahan umum
    waktu_raw = waktu_raw.replace("Tayang:", "").replace("WIB", "").strip()

    # Ganti bulan Indonesia ke Inggris (baik singkatan maupun panjang)
    for indo, eng in bulan_map.items():
        waktu_raw = waktu_raw.replace(indo, eng)

    # Hilangkan nama hari kalau ada koma
    if ',' in waktu_raw:
        parts = waktu_raw.split(',', 1)
        if len(parts) > 1:
            waktu_raw = parts[1].strip()

    # Coba parse otomatis
    try:
        return parser.parse(waktu_raw, dayfirst=True)  # dayfirst=True untuk format Indonesia
    except Exception:
        print(f"⚠️ Gagal parse waktu: '{waktu_raw}'")
        return None


# ==========================
# LOOPING SEMUA FILE CSV
# ==========================
for filename in csv_files:
    path = os.path.join(folder_path, filename)
    print(f"\n=== 📄 Memproses file: {filename} ===")

    try:
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Lewati header
            batch = []

            for i, row in enumerate(reader):
                if i >= 100:  # Limit 100 baris per file
                    break

                if len(row) < 5:
                    print(f"⚠️ Baris {i} dilewati (kolom kurang): {row}")
                    continue

                title, link, category, time_str, content = row[:5]

                # Parsing waktu
                time_obj = parse_waktu(time_str)

                batch.append((title, link, category, time_obj, content))

            print(f"✅ Siap insert {len(batch)} data dari {filename}")

            if batch:
                query = """
                    INSERT INTO public.news_articles (title, link, category, time, content)
                    VALUES (%s,%s,%s,%s,%s)
                """
                cur.executemany(query, batch)
                print(f"✅ Insert selesai untuk {filename}")
            else:
                print(f"⚠️ Tidak ada data valid untuk diinsert dari {filename}")

    except Exception as e:
        print(f"❌ Error saat membaca {filename}: {e}")

# ==========================
# TUTUP KONEKSI
# ==========================
cur.close()
conn.close()
print("\n=== 🎉 Semua file selesai diproses ===")