import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# 1. LOAD DATA & PRA-PEMROSESAN
# =============================================================================
df = pd.read_csv("gabungan_ispu_kendaraan.csv")

# Normalisasi Fitur (Min-Max Scaling) agar skala seragam [0, 1]
scaler = MinMaxScaler()
kolom_fitur = ["jumlah_kendaraan", "pm10", "so2", "co", "o3", "no2"]
df_norm = df.copy()
df_norm[kolom_fitur] = scaler.fit_transform(df[kolom_fitur])

# Konfigurasi estetika grafik selaras dengan aplikasi (AeroMetrics Jakarta)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d6d3d1'
plt.rcParams['axes.linewidth'] = 0.8

def style_axis(ax):
    """
    Menerapkan styling sumbu grafik yang serasi dengan aplikasi:
    latar belakang krem lembut (#fffdfa), grid putus-putus, dan tanpa bingkai atas/kanan.
    """
    ax.set_facecolor('#fffdfa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d6d3d1')
    ax.spines['bottom'].set_color('#d6d3d1')
    ax.grid(True, linestyle='--', linewidth=0.6, color='#e7e5e4', alpha=0.85)
    ax.tick_params(colors='#57534e', labelsize=9)

print("=== MEMPROSES MODEL MACHINE LEARNING & VISUALISASI ===")

# =============================================================================
# VISUALISASI 6: LINEAR REGRESSION (NO2 VS VOLUME KENDARAAN)
# =============================================================================
X = df_norm[["jumlah_kendaraan"]]
y = df_norm["no2"]

model_lr = LinearRegression()
model_lr.fit(X, y)
y_pred = model_lr.predict(X)

nilai_r2 = r2_score(y, y_pred)
nilai_mae = mean_absolute_error(y, y_pred)
nilai_rmse = np.sqrt(mean_squared_error(y, y_pred))

fig_lr, ax_lr = plt.subplots(figsize=(10, 4.2), dpi=300)
fig_lr.patch.set_facecolor('#fffdfa')

sns.scatterplot(
    x=X['jumlah_kendaraan'], y=y, ax=ax_lr,
    color='#0d3b2e', s=85, alpha=0.88, edgecolor='white', linewidth=0.8,
    label=f'Observasi Aktual (n = {len(df)})'
)
sns.lineplot(
    x=X['jumlah_kendaraan'], y=y_pred, ax=ax_lr,
    color='#d16a5a', linewidth=2.8,
    label=f'Garis Regresi OLS (R² = {nilai_r2:.2f} / {nilai_r2*100:.1f}%)'
)

style_axis(ax_lr)
ax_lr.set_xlabel("Jumlah Kendaraan Terdaftar (Skala Normalisasi 0–1)", fontsize=10, labelpad=8, color='#44403c')
ax_lr.set_ylabel("Konsentrasi NO2 (Skala Normalisasi 0–1)", fontsize=10, labelpad=8, color='#44403c')
ax_lr.set_title("Model Regresi Linear: Volume Kendaraan vs Konsentrasi NO2 DKI Jakarta", fontsize=11.5, fontweight='bold', pad=12, loc='left', color='#0d3b2e')
ax_lr.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=9)

plt.tight_layout()
plt.savefig("grafik_6_regresi.png", bbox_inches="tight", dpi=300)
plt.close()
print("-> [1/3] Grafik 6 (Linear Regression) berhasil disimpan: 'grafik_6_regresi.png'")


# =============================================================================
# VISUALISASI 7: K-MEANS CLUSTERING (3 DIAGRAM SESUAI APLIKASI)
# =============================================================================
# Diagram pada modul K-Means terbagi menjadi 3 visualisasi:
#   1. Evaluasi Elbow Method (Inertia / WCSS) untuk mencari titik siku klaster optimal
#   2. Evaluasi Silhouette Score (K = 2 s.d. 6) untuk menguji kualitas separasi klaster
#   3. Pemetaan Distribusi Klaster K-Means (K = 3) pada data nyata (Kendaraan vs PM10)
# =============================================================================

X_cluster = df_norm[['jumlah_kendaraan', 'pm10']]
k_range = range(2, 7)
inertia_list = []
silhouette_list = []

# Menghitung metrik evaluasi untuk K = 2 s.d. 6
for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = km_temp.fit_predict(X_cluster)
    inertia_list.append(km_temp.inertia_)
    silhouette_list.append(silhouette_score(X_cluster, labels_temp))

# --- DIAGRAM 1 & 2: EVALUASI K-MEANS (ELBOW METHOD & SILHOUETTE SCORE) ---
fig_eval, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)
fig_eval.patch.set_facecolor('#fffdfa')

# Subplot 1: Elbow Method
ax_elbow.plot(list(k_range), inertia_list, marker='o', markersize=6, color='#0d3b2e', linewidth=2.2)
ax_elbow.axvline(x=3, color='#d16a5a', linestyle='--', linewidth=1.5, label='Titik Siku Optimal (K = 3)')
style_axis(ax_elbow)
ax_elbow.set_title("Evaluasi Elbow Method (Inertia / WCSS)", fontsize=11, fontweight='bold', loc='left', pad=10, color='#0d3b2e')
ax_elbow.set_xlabel("Jumlah Klaster (K)", fontsize=9.5)
ax_elbow.set_ylabel("Inertia (WCSS)", fontsize=9.5)
ax_elbow.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5)

# Subplot 2: Silhouette Score
ax_sil.plot(list(k_range), silhouette_list, marker='s', markersize=6, color='#d16a5a', linewidth=2.2)
ax_sil.axvline(x=3, color='#0d3b2e', linestyle='--', linewidth=1.5, label='Skor Maksimum (K = 3)')
style_axis(ax_sil)
ax_sil.set_title("Evaluasi Silhouette Score (K = 2 s.d. 6)", fontsize=11, fontweight='bold', loc='left', pad=10, color='#0d3b2e')
ax_sil.set_xlabel("Jumlah Klaster (K)", fontsize=9.5)
ax_sil.set_ylabel("Silhouette Score", fontsize=9.5)
ax_sil.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5)

plt.tight_layout()
plt.savefig("grafik_7a_evaluasi_kmeans.png", bbox_inches="tight", dpi=300)
plt.close()
print("-> [2/3] Grafik 7A (Evaluasi Elbow & Silhouette berdampingan) berhasil disimpan: 'grafik_7a_evaluasi_kmeans.png'")

# Opsi simpan individual untuk Elbow Method & Silhouette Score
fig_elbow_single, ax_e_s = plt.subplots(figsize=(6, 4), dpi=300)
fig_elbow_single.patch.set_facecolor('#fffdfa')
ax_e_s.plot(list(k_range), inertia_list, marker='o', markersize=6, color='#0d3b2e', linewidth=2.2)
ax_e_s.axvline(x=3, color='#d16a5a', linestyle='--', linewidth=1.5, label='Titik Siku (K = 3)')
style_axis(ax_e_s)
ax_e_s.set_title("Grafik Elbow Method (Inertia / WCSS)", fontsize=11, fontweight='bold', pad=10, color='#0d3b2e')
ax_e_s.set_xlabel("Jumlah Klaster (K)", fontsize=9.5)
ax_e_s.set_ylabel("Inertia (WCSS)", fontsize=9.5)
ax_e_s.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5)
plt.tight_layout()
plt.savefig("grafik_7_elbow.png", bbox_inches="tight", dpi=300)
plt.close()

fig_sil_single, ax_s_s = plt.subplots(figsize=(6, 4), dpi=300)
fig_sil_single.patch.set_facecolor('#fffdfa')
ax_s_s.plot(list(k_range), silhouette_list, marker='s', markersize=6, color='#d16a5a', linewidth=2.2)
ax_s_s.axvline(x=3, color='#0d3b2e', linestyle='--', linewidth=1.5, label='Skor Maksimum (K = 3)')
style_axis(ax_s_s)
ax_s_s.set_title("Grafik Silhouette Score (K = 2 s.d. 6)", fontsize=11, fontweight='bold', pad=10, color='#0d3b2e')
ax_s_s.set_xlabel("Jumlah Klaster (K)", fontsize=9.5)
ax_s_s.set_ylabel("Silhouette Score", fontsize=9.5)
ax_s_s.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5)
plt.tight_layout()
plt.savefig("grafik_7_silhouette.png", bbox_inches="tight", dpi=300)
plt.close()


# --- DIAGRAM 3: PEMETAAN DISTRIBUSI KLASTER (K-MEANS, K = 3) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Klaster'] = kmeans.fit_predict(X_cluster)
sil_k3 = silhouette_score(X_cluster, df['Klaster'])

df['Label Klaster'] = df['Klaster'].map({
    0: 'Klaster A — Terkendali (Kendaraan & Polusi Rendah-Sedang)',
    1: 'Klaster B — Zona Kritis (Kendaraan & Polusi Tinggi)',
    2: 'Klaster C — Transisi Musiman (Kendaraan Sedang, Polusi Variatif)'
})

fig_cluster, ax_c = plt.subplots(figsize=(10, 5), dpi=300)
fig_cluster.patch.set_facecolor('#fffdfa')

sns.scatterplot(
    data=df, x='jumlah_kendaraan', y='pm10', hue='Label Klaster',
    palette=['#7dc8e8', '#d16a5a', '#e3a869'], s=110, alpha=0.92, edgecolor='white', linewidth=0.9, ax=ax_c
)

style_axis(ax_c)
ax_c.set_xlabel("Jumlah Kendaraan Terdaftar (Puluhan Juta Unit)", fontsize=10, labelpad=8)
ax_c.set_ylabel("Konsentrasi Partikulat PM10 (ISPU)", fontsize=10, labelpad=8)
ax_c.set_title("Pemetaan Distribusi Klaster Kualitas Udara DKI Jakarta (K-Means, K = 3)", fontsize=11.5, fontweight='bold', loc='left', pad=12, color='#0d3b2e')
ax_c.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig("grafik_7_kmeans.png", bbox_inches="tight", dpi=300)
plt.close()
print("-> [3/3] Grafik 7B (Scatter Plot Klaster K-Means) berhasil disimpan: 'grafik_7_kmeans.png'")

print("\n=== SEMUA GRAFIK MACHINE LEARNING ===")
print("File tersimpan:")
print(" 1. grafik_6_regresi.png          (Linear Regression NO2 vs Kendaraan)")
print(" 2. grafik_7a_evaluasi_kmeans.png  (Evaluasi Elbow Method & Silhouette Score 2-in-1)")
print(" 3. grafik_7_elbow.png             (Elbow Method WCSS terpisah)")
print(" 4. grafik_7_silhouette.png        (Silhouette Score terpisah)")
print(" 5. grafik_7_kmeans.png            (Scatter Plot 3 Klaster K-Means)")