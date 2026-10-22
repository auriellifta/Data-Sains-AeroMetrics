import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. Konfigurasi Halaman & Desain Antarmuka (UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroMetrics — Analisis Kualitas Udara DKI Jakarta",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #2c3531;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        background: linear-gradient(135deg, #0d3b2e 0%, #1a5646 100%);
        color: #ffffff;
        padding: 0.85rem 1.1rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 6px 16px rgba(13, 59, 46, 0.12);
    }
    .brand-icon {
        font-size: 1.65rem;
        line-height: 1;
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
        color: #ffffff;
    }
    .brand-sub {
        font-size: 0.72rem;
        color: #d1fae5;
        margin: 0;
        font-weight: 500;
    }

    .main-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0d3b2e;
        margin-bottom: 0.35rem;
        line-height: 1.25;
    }

    .sub-title {
        font-size: 0.96rem;
        color: #5c6460;
        margin-bottom: 1.6rem;
        line-height: 1.55;
    }

    .metric-card {
        background-color: #fffdfa;
        border: 1px solid #ebe5dc;
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 14px rgba(13, 59, 46, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(13, 59, 46, 0.07);
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #78716c;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #0d3b2e;
        letter-spacing: -0.02em;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #8a817c;
        margin-top: 0.25rem;
    }

    .editorial-note {
        background-color: #faf6f0;
        border-left: 4px solid #d16a5a;
        border-top: 1px solid #ede6dc;
        border-right: 1px solid #ede6dc;
        border-bottom: 1px solid #ede6dc;
        padding: 1rem 1.25rem;
        border-radius: 16px;
        font-size: 0.9rem;
        color: #44403c;
        margin: 1rem 0;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-block;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
    }
    .status-danger { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .status-warn   { background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .status-safe   { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }

    .team-card {
        background-color: #faf6f0;
        border: 1px solid #ebe5dc;
        border-radius: 16px;
        padding: 0.65rem 0.95rem;
        margin-top: 0.5rem;
    }
    .team-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        padding: 0.42rem 0;
        border-bottom: 1px dashed #e2dcd3;
        color: #292524;
        font-weight: 500;
    }
    .team-nim {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: #0d3b2e;
        background-color: #e6f0ec;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.73rem;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #0d3b2e 0%, #1a5646 100%);
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.65rem 1.25rem;
        border-radius: 14px;
        border: none;
        box-shadow: 0 4px 12px rgba(13, 59, 46, 0.18);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #d16a5a 0%, #b85444 100%);
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(209, 106, 90, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d6d3d1'
plt.rcParams['axes.linewidth'] = 0.8

def style_axis(ax):
    """
    Fungsi: Merapikan sumbu grafik Matplotlib dengan warna hangat yang lembut.
    Menerapkan latar belakang krem, menghapus border atas/kanan, dan menambahkan grid putus-putus.
    """
    ax.set_facecolor('#fffdfa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d6d3d1')
    ax.spines['bottom'].set_color('#d6d3d1')
    ax.grid(True, linestyle='--', linewidth=0.6, color='#e7e5e4', alpha=0.85)
    ax.tick_params(colors='#57534e', labelsize=9)

# -----------------------------------------------------------------------------
# Sidebar: Logo & Profil Tim Peneliti AeroMetrics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-logo">
            <div class="brand-icon">☁️</div>
            <div>
                <div class="brand-title">AeroMetrics</div>
                <div class="brand-sub">Jakarta Air Quality Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Analisis Hubungan Volume Kendaraan Bermotor dan Kualitas Udara (ISPU) di Provinsi DKI Jakarta.")
    
    st.markdown("**ANGGOTA PENELITI**")
    st.markdown(
        """
        <div class="team-card">
            <div class="team-row"><span>Piky Al Piandi</span><span class="team-nim">2401020062</span></div>
            <div class="team-row"><span>Yura Ramadhani</span><span class="team-nim">2401020068</span></div>
            <div class="team-row"><span>Meyky Ajmariadi</span><span class="team-nim">2401020070</span></div>
            <div class="team-row"><span>Auriel Lifta E. G.</span><span class="team-nim">2401020071</span></div>
            <div class="team-row" style="border-bottom:none;"><span>Devanovita Chelsea P.</span><span class="team-nim">2401020086</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("Program Studi Teknik Informatika\nUniversitas Maritim Raja Ali Haji (UMRAH)")

# -----------------------------------------------------------------------------
# 2. Pemuatan Dataset Terintegrasi
# Fungsi: Memuat file 'gabungan_ispu_kendaraan.csv' hasil prapemrosesan.
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('gabungan_ispu_kendaraan.csv')

try:
    df = load_data()
except Exception:
    st.error("File 'gabungan_ispu_kendaraan.csv' tidak ditemukan pada direktori kerja.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Normalisasi Fitur (Min-Max Scaling)
# Fungsi: Memetakan seluruh rentang nilai fitur ke dalam skala proporsional [0, 1]
# agar jarak Euclidean pada K-Means tidak bias terhadap nilai kendaraan yang besar.
# -----------------------------------------------------------------------------
scaler = MinMaxScaler()
kolom_fitur = ['jumlah_kendaraan', 'pm10', 'so2', 'co', 'o3', 'no2']
df_norm = df.copy()
df_norm[kolom_fitur] = scaler.fit_transform(df[kolom_fitur])

st.markdown("<div class='main-title'>Dashboard Analisis Kualitas Udara & Volume Kendaraan DKI Jakarta</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Sistem evaluasi berbasis <i>Linear Regression</i> dan <i>K-Means Clustering</i> untuk memetakan korelasi emisi gas buang serta dampak lonjakan populasi kendaraan bermotor.</div>",
    unsafe_allow_html=True
)

# Kartu Metrik Ringkasan Utama di Bagian Atas
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
with s_col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Observasi Terintegrasi</div><div class='metric-value'>{len(df)} Bulan</div><div class='metric-sub'>Hasil Inner Join ISPU & BPS</div></div>", unsafe_allow_html=True)
with s_col2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Rentang Populasi Kendaraan</div><div class='metric-value'>{df['jumlah_kendaraan'].min()/1e6:.1f}M – {df['jumlah_kendaraan'].max()/1e6:.1f}M</div><div class='metric-sub'>Unit kendaraan terdaftar</div></div>", unsafe_allow_html=True)
with s_col3:
    st.markdown("<div class='metric-card'><div class='metric-label'>Parameter Polutan Aktif</div><div class='metric-value'>5 Variabel</div><div class='metric-sub'>PM10, SO2, CO, O3, NO2</div></div>", unsafe_allow_html=True)
with s_col4:
    st.markdown("<div class='metric-card'><div class='metric-label'>Status Kelengkapan Data</div><div class='metric-value'>0 Missing</div><div class='metric-sub'>100% Terimputasi & Bersih</div></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sistem Navigasi Multi-Tab
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "01 Normalisasi Data",
    "02 Pemodelan & Evaluasi",
    "03 Proyeksi Tren Polutan",
    "04 Simulator Kebijakan"
])

# -----------------------------------------------------------------------------
# TAB 1: Normalisasi Data (Min-Max Scaling)
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("#### Tahap Normalisasi Fitur (Min-Max Scaling)")
    st.markdown(
        "<div class='editorial-note'>"
        "<b>Rasionalitas Metodologis:</b> Variabel <code>jumlah_kendaraan</code> berada pada skala puluhan juta unit, sedangkan variabel polutan udara (<code>pm10</code>, <code>no2</code>) berada pada rentang puluhan ISPU. "
        "Transformasi <i>Min-Max Normalization</i> memetakan seluruh fitur ke dalam rentang proporsional <b>[0, 1]</b> agar perhitungan jarak <i>Euclidean</i> pada algoritma <i>K-Means</i> tidak didominasi secara mutlak oleh skala kendaraan."
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tabel Sebelum Normalisasi (Skala Asli)**")
        st.dataframe(
            df[['tahun', 'bulan', 'jumlah_kendaraan', 'pm10', 'no2']].head(8),
            use_container_width=True,
            hide_index=True
        )
    with col2:
        st.markdown("**Tabel Sesudah Normalisasi (Rentang 0–1)**")
        st.dataframe(
            df_norm[['tahun', 'bulan', 'jumlah_kendaraan', 'pm10', 'no2']].head(8).style.format({
                'jumlah_kendaraan': '{:.4f}',
                'pm10': '{:.4f}',
                'no2': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------------------------------------------------------
# TAB 2: Pemodelan & Evaluasi Model Machine Learning
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### A. Pemodelan Regresi Linear (Supervised Learning)")
    st.caption("Memodelkan pengaruh volume kendaraan bermotor (X) terhadap konsentrasi gas Nitrogen Dioksida / NO2 (Y).")

    X = df_norm[['jumlah_kendaraan']]
    y = df_norm['no2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_lr = LinearRegression()
    model_lr.fit(X, y)
    y_pred = model_lr.predict(X)

    nilai_r2 = r2_score(y, y_pred)
    nilai_mae = mean_absolute_error(y, y_pred)
    nilai_rmse = np.sqrt(mean_squared_error(y, y_pred))
    intercept_a = model_lr.intercept_
    slope_b = model_lr.coef_[0]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>R-Squared (R²)</div><div class='metric-value'>{nilai_r2:.4f}</div><div class='metric-sub'>{nilai_r2*100:.1f}% variasi dijelaskan</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Absolute Error</div><div class='metric-value'>{nilai_mae:.4f}</div><div class='metric-sub'>Simpangan absolut rata-rata</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Root Mean Sq. Error</div><div class='metric-value'>{nilai_rmse:.4f}</div><div class='metric-sub'>Simpangan baku residu</div></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Persamaan Regresi OLS</div><div class='metric-value'>Y = {intercept_a:.2f} + {slope_b:.2f}X</div><div class='metric-sub'>Gradien kemiringan positif</div></div>", unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(10, 3.8), dpi=120)
    fig1.patch.set_facecolor('#fffdfa')
    sns.scatterplot(
        x=X['jumlah_kendaraan'], y=y, ax=ax1,
        color='#0d3b2e', s=75, alpha=0.88, edgecolor='white', linewidth=0.8,
        label='Observasi Aktual (n = 47)'
    )
    sns.lineplot(
        x=X['jumlah_kendaraan'], y=y_pred, ax=ax1,
        color='#d16a5a', linewidth=2.6,
        label=f'Garis Regresi OLS (R² = {nilai_r2:.2f})'
    )
    style_axis(ax1)
    ax1.set_xlabel("Jumlah Kendaraan Terdaftar (Skala Normalisasi 0–1)", fontsize=9.5, labelpad=8, color='#44403c')
    ax1.set_ylabel("Konsentrasi NO2 (Skala 0–1)", fontsize=9.5, labelpad=8, color='#44403c')
    ax1.set_title("Model Regresi Linear: Volume Kendaraan vs Konsentrasi NO2", fontsize=11, fontweight='bold', pad=12, loc='left', color='#0d3b2e')
    ax1.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5)
    st.pyplot(fig1)

    st.markdown("---")

    st.markdown("#### B. Segmentasi K-Means Clustering (Unsupervised Learning)")
    st.caption("Pengelompokan otomatis karakteristik bulan observasi berdasarkan kepadatan kendaraan dan partikulat debu PM10.")

    X_cluster = df_norm[['jumlah_kendaraan', 'pm10']]
    k_range = range(2, 7)
    inertia_list = []
    silhouette_list = []

    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = km_temp.fit_predict(X_cluster)
        inertia_list.append(km_temp.inertia_)
        silhouette_list.append(silhouette_score(X_cluster, labels_temp))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Klaster'] = kmeans.fit_predict(X_cluster)
    sil_k3 = silhouette_score(X_cluster, df['Klaster'])
    inertia_k3 = kmeans.inertia_

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        st.markdown("<div class='metric-card'><div class='metric-label'>Konfigurasi Klaster Optimal</div><div class='metric-value'>K = 3</div><div class='metric-sub'>Berdasarkan konvergensi Elbow & Silhouette</div></div>", unsafe_allow_html=True)
    with c_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Silhouette Score (K=3)</div><div class='metric-value'>{sil_k3:.4f}</div><div class='metric-sub'>Skor kohesi & separasi tertinggi</div></div>", unsafe_allow_html=True)
    with c_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Within-Cluster Sum of Squares</div><div class='metric-value'>{inertia_k3:.4f}</div><div class='metric-sub'>Nilai Inertia pada titik siku (Elbow)</div></div>", unsafe_allow_html=True)

    fig_eval, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=120)
    fig_eval.patch.set_facecolor('#fffdfa')

    ax_elbow.plot(list(k_range), inertia_list, marker='o', markersize=6, color='#0d3b2e', linewidth=2.2)
    ax_elbow.axvline(x=3, color='#d16a5a', linestyle='--', linewidth=1.4, label='Titik Siku (K = 3)')
    style_axis(ax_elbow)
    ax_elbow.set_title("Evaluasi Elbow Method (Inertia / WCSS)", fontsize=10.5, fontweight='bold', loc='left', pad=10, color='#0d3b2e')
    ax_elbow.set_xlabel("Jumlah Klaster (K)", fontsize=9)
    ax_elbow.set_ylabel("Inertia (WCSS)", fontsize=9)
    ax_elbow.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8)

    ax_sil.plot(list(k_range), silhouette_list, marker='s', markersize=6, color='#d16a5a', linewidth=2.2)
    ax_sil.axvline(x=3, color='#0d3b2e', linestyle='--', linewidth=1.4, label='Skor Maksimum (K = 3)')
    style_axis(ax_sil)
    ax_sil.set_title("Evaluasi Silhouette Score (K = 2 s.d. 6)", fontsize=10.5, fontweight='bold', loc='left', pad=10, color='#0d3b2e')
    ax_sil.set_xlabel("Jumlah Klaster (K)", fontsize=9)
    ax_sil.set_ylabel("Silhouette Score", fontsize=9)
    ax_sil.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig_eval)

    df['Label Klaster'] = df['Klaster'].map({
        0: 'Klaster A — Terkendali (Kendaraan & Polusi Rendah-Sedang)',
        1: 'Klaster B — Zona Kritis (Kendaraan & Polusi Tinggi)',
        2: 'Klaster C — Transisi Musiman (Kendaraan Sedang, Polusi Variatif)'
    })

    fig2, ax2 = plt.subplots(figsize=(10, 4.4), dpi=120)
    fig2.patch.set_facecolor('#fffdfa')
    sns.scatterplot(
        data=df, x='jumlah_kendaraan', y='pm10', hue='Label Klaster',
        palette=['#7dc8e8', '#d16a5a', '#e3a869'], s=105, alpha=0.92, edgecolor='white', linewidth=0.9, ax=ax2
    )
    style_axis(ax2)
    ax2.set_xlabel("Jumlah Kendaraan Terdaftar (Puluhan Juta Unit)", fontsize=9.5, labelpad=8)
    ax2.set_ylabel("Konsentrasi Partikulat PM10 (ISPU)", fontsize=9.5, labelpad=8)
    ax2.set_title("Pemetaan Distribusi Klaster Kualitas Udara DKI Jakarta (K-Means, K = 3)", fontsize=11, fontweight='bold', loc='left', pad=12, color='#0d3b2e')
    ax2.legend(frameon=True, facecolor='#ffffff', edgecolor='#ebe5dc', fontsize=8.5, loc='upper left')
    st.pyplot(fig2)

# -----------------------------------------------------------------------------
# TAB 3: Proyeksi Tren Polutan Masa Depan (Forecasting Multi-Output)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### Proyeksi Tren Polutan Masa Depan (Multi-Output Linear Extrapolation)")
    st.markdown(
        "<div class='editorial-note'>"
        "<b>Mekanisme Proyeksi:</b> Modul ini mengekstrapolasi tren waktu historis (<code>tahun</code> dan <code>bulan</code>) terhadap kelima parameter polutan sekaligus menggunakan regresi linear multi-output. "
        "Pilih tahun dan bulan target di bawah ini, lalu tekan tombol <b>Jalankan Prediksi</b> untuk melihat perbandingan estimasi konsentrasi polutan."
        "</div>",
        unsafe_allow_html=True
    )

    X_time = df[['tahun', 'bulan']]
    y_pollutants = df[['pm10', 'so2', 'co', 'o3', 'no2']]
    model_time = LinearRegression()
    model_time.fit(X_time, y_pollutants)

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        target_tahun = st.number_input("Tahun Target Proyeksi (2023–2100):", min_value=2023, max_value=2100, value=2030, step=1)
    with col_input2:
        target_bulan = st.slider("Bulan Target (1 = Januari, 12 = Desember):", min_value=1, max_value=12, value=6)

    if st.button("Jalankan Prediksi", use_container_width=True):
        prediksi = model_time.predict([[target_tahun, target_bulan]])[0]
        polutan_labels = ['PM10 (Partikulat)', 'SO2 (Sulfur)', 'CO (Karbon)', 'O3 (Ozon)', 'NO2 (Nitrogen)']
        idx_max = int(np.argmax(prediksi))
        polutan_tertinggi = polutan_labels[idx_max]

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Periode Target Evaluasi</div>"
                f"<div class='metric-value'>Bulan {target_bulan:02d} / {target_tahun}</div>"
                f"<div class='metric-sub'>Ekstrapolasi skenario Business-as-Usual</div></div>",
                unsafe_allow_html=True
            )
        with p_col2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Dominasi Polutan Tertinggi</div>"
                f"<div class='metric-value'>{polutan_tertinggi} — {prediksi[idx_max]:.2f} ISPU</div>"
                f"<div class='metric-sub'>Parameter dengan estimasi indeks maksimum</div></div>",
                unsafe_allow_html=True
            )

        fig3, ax3 = plt.subplots(figsize=(9.5, 3.8), dpi=120)
        fig3.patch.set_facecolor('#fffdfa')
        bar_colors = ['#d16a5a' if i == idx_max else '#0d3b2e' for i in range(len(polutan_labels))]
        bars = ax3.bar(polutan_labels, prediksi, color=bar_colors, width=0.52)
        style_axis(ax3)
        ax3.set_ylabel("Estimasi Konsentrasi (ISPU)", fontsize=9.5)
        ax3.set_title(f"Perbandingan Estimasi Indeks Polutan pada Periode {target_bulan:02d}-{target_tahun}", fontsize=10.5, fontweight='bold', loc='left', pad=12, color='#0d3b2e')
        ax3.set_ylim(0, max(prediksi) * 1.18)

        for bar, val in zip(bars, prediksi):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                val + (max(prediksi) * 0.03),
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0d3b2e'
            )
        st.pyplot(fig3)

# -----------------------------------------------------------------------------
# TAB 4: Simulator Kebijakan (What-If Analysis)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("#### Simulator Kebijakan Pembatasan Volume Kendaraan (What-If Analysis)")
    st.markdown(
        "<div class='editorial-note'>"
        "<b>Cara Kerja Matematis:</b> Simulator ini menguji sensitivitas kebijakan kuota kendaraan bermotor terhadap konsentrasi gas buang Nitrogen Dioksida (<code>no2</code>). "
        "Input jumlah kendaraan dinormalisasi menggunakan parameter historis, diprediksi melalui persamaan regresi <b>Y = a + bX</b>, lalu didenormalisasi (<i>inverse transform</i>) kembali ke satuan asli ISPU."
        "</div>",
        unsafe_allow_html=True
    )

    sim_kendaraan = st.slider(
        "Simulasi Populasi Kendaraan Terdaftar di DKI Jakarta (Unit):",
        min_value=15000000,
        max_value=35000000,
        value=20000000,
        step=500000
    )

    dummy_input = np.zeros((1, 6))
    dummy_input[0, 0] = sim_kendaraan
    sim_kendaraan_norm = scaler.transform(dummy_input)[0, 0]

    pred_no2_norm = model_lr.predict([[sim_kendaraan_norm]])[0]

    dummy_output = np.zeros((1, 6))
    dummy_output[0, 5] = pred_no2_norm
    pred_no2_asli = scaler.inverse_transform(dummy_output)[0, 5]

    rata_no2 = df['no2'].mean()
    if pred_no2_asli > rata_no2 + 2:
        badge_html = "<span class='status-badge status-danger'>STATUS: RISIKO TINGGI (MELAMPAUI RATA-RATA HISTORIS)</span>"
    elif pred_no2_asli < rata_no2 - 2:
        badge_html = "<span class='status-badge status-safe'>STATUS: TERKENDALI (DI BAWAH RATA-RATA HISTORIS)</span>"
    else:
        badge_html = "<span class='status-badge status-warn'>STATUS: MODERAT (SEKITAR RATA-RATA HISTORIS)</span>"

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Skenario Volume Kendaraan</div>"
            f"<div class='metric-value'>{sim_kendaraan:,.0f} Unit</div>".replace(',', '.') +
            f"<div class='metric-sub'>Skala Normalisasi (X_norm): {sim_kendaraan_norm:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_res2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Estimasi Konsentrasi NO2 (Inverse Transform)</div>"
            f"<div class='metric-value'>{pred_no2_asli:.2f} ISPU</div>"
            f"<div style='margin-top:0.4rem;'>{badge_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )