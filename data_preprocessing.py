import pandas as pd
import numpy as np

print("=== TAHAP 1: PREPROCESSING DATASET 1 (ISPU) ===")
ispu_files = [
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2010.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2011.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2012.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2013.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2014.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2015.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2016.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2017.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2018.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2019.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2020.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2021.csv",
    "Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2022.csv"
]

ispu_dfs = []
for f in ispu_files:
    try:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns={
            'pm_10': 'pm10', 'pm_duakomalima': 'pm25', 
            'lokasi_spku': 'stasiun', '\ufeffcritical': 'critical'
        })
        ispu_dfs.append(df)
    except Exception as e:
        pass 

df_ispu = pd.concat(ispu_dfs, ignore_index=True)

# ---> Menampilkan Tipe Data dan Missing Value RAW DATA ISPU <---
print("\n[INFO RAW DATA ISPU (SEBELUM DIBERSIHKAN)]")
print("Total Baris Raw Data ISPU:", len(df_ispu))
print("\n1. Tipe Data Asli:")
print(df_ispu.dtypes)
print("\n2. Missing Value Asli:")
print(df_ispu.isnull().sum())
print("-" * 50)

# Proses Cleaning
df_ispu['tanggal'] = pd.to_datetime(df_ispu['tanggal'], errors='coerce')
df_ispu['tahun'] = df_ispu['tanggal'].dt.year
df_ispu['bulan'] = df_ispu['tanggal'].dt.month

kolom_polutan = ['pm10', 'so2', 'co', 'o3', 'no2']
for col in kolom_polutan:
    df_ispu[col] = pd.to_numeric(df_ispu[col], errors='coerce')
    df_ispu[col] = df_ispu[col].fillna(df_ispu[col].median()) # Imputasi Median

df_ispu = df_ispu.drop_duplicates()
df_ispu_bulanan = df_ispu.groupby(['tahun', 'bulan'])[kolom_polutan].mean().reset_index()
df_ispu_bulanan.to_csv('ispu_bersih.csv', index=False)


print("\n=== TAHAP 2: PREPROCESSING DATASET 2 (KENDARAAN) ===")
veh_files = [
    ("kendaraan_2016.csv", 2016),
    ("kendaraan_2020.csv", 2020),
    ("kendaraan_2021.csv", 2021),
    ("kendaraan_2022.csv", 2022)
]

veh_dfs = []
for file_name, thn in veh_files:
    try:
        # skiprows=3 digunakan untuk melewati header ganda dari format BPS
        df_v = pd.read_csv(file_name, skiprows=3) 
        df_bersih = pd.DataFrame()
        df_bersih['tahun'] = [thn] * 12
        df_bersih['bulan'] = range(1, 13)
        
        # Ekstrak angka dan konversi ke integer
        kolom_jumlah = df_v.iloc[:12, -1].astype(str).str.replace(r'\D', '', regex=True)
        df_bersih['jumlah_kendaraan'] = pd.to_numeric(kolom_jumlah, errors='coerce')
        veh_dfs.append(df_bersih)
    except Exception as e:
        pass

df_kendaraan = pd.concat(veh_dfs, ignore_index=True)

# ---> Menampilkan Informasi Kendaraan <---
print("\n[INFO RAW DATA KENDARAAN (SEBELUM DIBERSIHKAN)]")
print("Total Baris Raw Data Kendaraan:", len(df_kendaraan))
print("Missing Value Kendaraan (Setelah Pivot):")
print(df_kendaraan.isnull().sum())
print("-" * 50)

df_kendaraan['jumlah_kendaraan'] = df_kendaraan['jumlah_kendaraan'].ffill()
df_kendaraan.to_csv('kendaraan_bersih.csv', index=False)


print("\n=== TAHAP 3: MERGE KEDUA DATASET (FINAL DATA) ===")
# Inner Join
df_final = pd.merge(df_ispu_bulanan, df_kendaraan, on=['tahun', 'bulan'], how='inner')
df_final.to_csv('gabungan_ispu_kendaraan.csv', index=False)

print("\n[INFO DATA GABUNGAN (SESUDAH DIBERSIHKAN)]")
print("Total Baris Data Final:", len(df_final))
print("\n1. Tipe Data Final:")
print(df_final.dtypes)
print("\n2. Cek Missing Value Final (Harus 0 semua):")
print(df_final.isnull().sum())
print("-" * 50)
print("-> File 'gabungan_ispu_kendaraan.csv' BERHASIL DIBUAT!")