import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="OutfitSight — Prediksi Harga & Penjualan",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Noto+Serif+JP:wght@400;600&display=swap');

:root {
    --bg:        #F5F5F5;
    --white:     #FFFFFF;
    --black:     #000000;
    --border:    #E8E8E8;
    --border2:   #D0D0D0;
    --text:      #1A1A1A;
    --text2:     #5A5A5A;
    --text3:     #9A9A9A;
    --red:       #E60012;
    --red2:      #FF1A2E;
    --red-bg:    #FFF5F5;
    --red-lt:    #FFFAFA;
    --green:     #2E7D32;
    --green-bg:  #F1F8F1;
    --radius:    0px;
    --radius-s:  2px;
    --shadow:    0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    --shadow-md: 0 8px 32px rgba(0,0,0,0.12);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans JP', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text);
    letter-spacing: 0.01em;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 0 1.25rem 4rem 1.25rem !important;
    max-width: 1200px !important;
     width: 100% !important
}

/* ── NAVBAR ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    height: 56px;
    background: var(--black);
    margin: 0 -1.25rem 0 -1.25rem;
    padding: 0 2rem;
    position: sticky; top: 0; z-index: 999;
}
.nav-brand { display: flex; align-items: center; gap: 1rem; }
.nav-logo {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: var(--white); letter-spacing: 0.08em;
    text-transform: uppercase;
}
.nav-sep { width: 1px; height: 20px; background: rgba(255,255,255,0.25); }
.nav-sub { font-size: 0.68rem; color: rgba(255,255,255,0.55); letter-spacing: 0.12em; text-transform: uppercase; font-weight: 400; }
.nav-badge {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.3rem 0.9rem;
    border: 1px solid rgba(255,255,255,0.3); border-radius: 0;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: rgba(255,255,255,0.75); background: rgba(255,255,255,0.05);
}
.nav-dot { width: 5px; height: 5px; border-radius: 50%; background: #4ADE80; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0.3} }

/* ── HERO ── */
.hero {
    position: relative;
    margin: 0 -1.25rem 2rem -1.25rem;
    padding: 3.5rem 2.5rem 3rem 2.5rem;
    background: var(--white);
    border-bottom: 3px solid var(--black);
    overflow: hidden;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--red); margin-bottom: 1.1rem;
    border-left: 3px solid var(--red); padding-left: 0.6rem;
}
.hero-title {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: clamp(2.8rem, 5.5vw, 4.8rem);
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: -0.03em;
    color: var(--black);
    margin-bottom: 0.2rem;
}
.hero-title-sub {
    font-family: 'Noto Serif JP', serif;
    font-size: clamp(2.2rem, 4.5vw, 3.8rem);
    font-weight: 400; font-style: italic;
    color: var(--red); line-height: 1.1;
    letter-spacing: -0.02em; margin-bottom: 1.2rem;
}
.hero-desc {
    font-size: 0.88rem; color: var(--text2);
    line-height: 1.75; max-width: 520px; font-weight: 400;
    border-left: 2px solid var(--border2); padding-left: 1rem;
}
.hero-bottom {
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}
.hero-stats { display: flex; gap: 0; }
.hstat {
    display: flex; flex-direction: column; gap: 0.15rem;
    padding: 0 2.5rem 0 0;
}
.hstat + .hstat {
    padding: 0 2.5rem;
    border-left: 1px solid var(--border2);
}
.hstat-val {
    font-family: 'Noto Sans JP', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    color: var(--black); letter-spacing: -0.04em;
}
.hstat-label {
    font-size: 0.55rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--text3);
}
.hero-deco {
    font-size: 5rem; opacity: 0.06; font-weight: 700; letter-spacing: -0.05em;
    color: var(--black); user-select: none;
}

/* ── STEPPER ── */
.stepper {
    display: flex; align-items: center; gap: 0;
    margin-bottom: 1.75rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.step {
    display: flex; align-items: center; gap: 0.6rem;
}
.step-num {
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    border-radius: 50%;
    flex-shrink: 0;
}
.step-num.done { background: var(--black); color: white; }
.step-num.active { background: var(--red); color: white; }
.step-num.idle { background: var(--border); color: var(--text3); }
.step-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; }
.step-label.done { color: var(--text3); }
.step-label.active { color: var(--text); }
.step-label.idle { color: var(--text3); }
.step-sep { flex: 1; max-width: 40px; height: 1px; background: var(--border2); margin: 0 0.5rem; }

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.55rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── CHIP GROUP ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
.chip {
    padding: 0.4rem 0.9rem;
    border: 1.5px solid var(--border2); border-radius: 0;
    font-size: 0.75rem; font-weight: 600;
    color: var(--text2); background: white;
    cursor: pointer; transition: all 0.15s;
    letter-spacing: 0.03em;
}
.chip:hover { border-color: var(--black); color: var(--black); }
.chip.active { border-color: var(--black); background: var(--black); color: white; }

/* ── COLUMN PANELS ── */
[data-testid="column"] > div:first-child {
    background: var(--white) !important;
    border: 1.5px solid var(--black) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 rgba(0,0,0,0.06) !important;
    padding: 1.75rem !important;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:first-child {
    background: var(--white) !important;
    border: 1.5px solid var(--black) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 rgba(0,0,0,0.06) !important;
    padding: 1.75rem !important;
    margin-bottom: 1.5rem;
}

/* ── PANEL HEADER ── */
.panel-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 1.75rem; padding-bottom: 1.1rem;
    border-bottom: 2px solid var(--black);
}
.panel-label {
    font-size: 0.55rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text3); display: block; margin-bottom: 0.3rem;
}
.panel-title { font-size: 1rem; font-weight: 700; color: var(--black); display: block; }
.panel-icon {
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
    border: 1.5px solid var(--border); background: var(--bg);
}

/* ── SELECTBOX ── */
div[data-baseweb="select"] > div {
    background: var(--bg) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-size: 0.85rem !important;
    min-height: 46px !important;
    transition: border-color 0.15s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--black) !important;
}
div[data-baseweb="select"] svg { fill: var(--text2) !important; }
[data-baseweb="popover"] {
    background: var(--white) !important;
    border: 1.5px solid var(--black) !important;
    border-radius: 0 !important;
    box-shadow: var(--shadow-md) !important;
}
[role="option"] {
    background: var(--white) !important; color: var(--text) !important;
    min-height: 42px !important;
    font-family: 'Noto Sans JP', sans-serif !important; font-size: 0.85rem !important;
}
[role="option"]:hover { background: var(--bg) !important; color: var(--black) !important; }
[aria-selected="true"] { background: var(--black) !important; color: white !important; font-weight: 600 !important; }
.stSelectbox label, label {
    font-size: 0.58rem !important; font-weight: 700 !important;
    color: var(--text3) !important; letter-spacing: 0.14em !important; text-transform: uppercase !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--black) !important;
    color: white !important; border: none !important;
    border-radius: 0 !important;
    padding: 1rem 2rem !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    width: 100% !important; min-height: 52px !important;
    box-shadow: none !important;
    transition: all 0.15s !important; margin-top: 1.5rem !important;
}
.stButton > button:hover {
    background: var(--red) !important;
    transform: none !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 360px; gap: 1rem;
    border: 2px dashed var(--border2);
    margin-top: 1rem;
}
.empty-icon { font-size: 2.5rem; opacity: 0.35; }
.empty-title { font-size: 0.95rem; font-weight: 700; color: var(--text2); text-align: center; letter-spacing: 0.04em; }
.empty-desc { font-size: 0.78rem; color: var(--text3); text-align: center; line-height: 1.6; max-width: 200px; }

/* ── RESULT CARDS ── */
.result-wrap { display: flex; flex-direction: column; gap: 0.6rem; }
.rcard {
    border: 1.5px solid var(--border); 
    padding: 1.15rem 1.3rem;
    position: relative; transition: all 0.2s;
}
.rcard:hover { border-color: var(--black); box-shadow: 3px 3px 0 rgba(0,0,0,0.08); }
.rcard.best { background: var(--black); border-color: var(--black); }
.rcard:not(.best) { background: var(--white); }
.rcard-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.7rem; }
.rcard-badge {
    font-size: 0.55rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.2rem 0.65rem; display: inline-block; border: 1px solid;
}
.rcard.best .rcard-badge { border-color: rgba(255,255,255,0.3); color: rgba(255,255,255,0.8); background: rgba(255,255,255,0.08); }
.rcard:not(.best) .rcard-badge { border-color: var(--border2); color: var(--text3); background: var(--bg); }
.rcard-rank {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em;
    color: var(--red);
}
.rcard.best .rcard-rank { color: rgba(255,255,255,0.5); }
.rcard-price { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.rcard.best .rcard-price { color: white; }
.rcard:not(.best) .rcard-price { color: var(--black); }
.rcard-per { font-size: 0.65rem; font-weight: 500; margin: 0.3rem 0 0.85rem; letter-spacing: 0.06em; text-transform: uppercase; }
.rcard.best .rcard-per { color: rgba(255,255,255,0.45); }
.rcard:not(.best) .rcard-per { color: var(--text3); }
.rcard-divider { height: 1px; margin-bottom: 0.85rem; }
.rcard.best .rcard-divider { background: rgba(255,255,255,0.15); }
.rcard:not(.best) .rcard-divider { background: var(--border); }
.rcard-stats { display: flex; gap: 2rem; }
.rstat-label { font-size: 0.55rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.2rem; }
.rcard.best .rstat-label { color: rgba(255,255,255,0.45); }
.rcard:not(.best) .rstat-label { color: var(--text3); }
.rstat-val { font-size: 0.9rem; font-weight: 700; letter-spacing: -0.01em; }
.rcard.best .rstat-val { color: white; }
.rcard:not(.best) .rstat-val { color: var(--text); }

/* ── SUMMARY BOX ── */
.summary-box {
    background: var(--bg); border: 1.5px solid var(--border2);
    border-left: 4px solid var(--black);
    padding: 1.1rem 1.25rem; margin-top: 0.6rem;
}
.sb-label {
    font-size: 0.55rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 0.45rem; display: block;
}
.sb-text { font-size: 0.83rem; color: var(--text2); line-height: 1.7; }
.sb-text strong { color: var(--text); font-weight: 700; }

.stSpinner > div { border-color: var(--black) transparent transparent transparent !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
            
/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {

    /* Block container lebih sempit */
    .block-container {
        padding: 0 0.75rem 3rem 0.75rem !important;
    }

    /* Navbar lebih compact */
    .navbar {
        padding: 0 1rem;
        height: 48px;
    }
    .nav-logo { font-size: 1.1rem; }
    .nav-badge { display: none; }

    /* Hero lebih kecil */
    .hero {
        padding: 2rem 1.25rem 1.75rem 1.25rem;
    }
    .hero-title { font-size: 2.2rem; }
    .hero-title-sub { font-size: 1.8rem; }
    .hero-desc { font-size: 0.82rem; }
    .hero-bottom { flex-direction: column; align-items: flex-start; gap: 1rem; }
    .hero-deco { display: none; }
    .hstat-val { font-size: 1.4rem; }

    /* Panel header lebih compact */
    .panel-header { margin-bottom: 1.25rem; }
    .panel-title { font-size: 0.9rem; }

    /* Stepper label disembunyikan di HP kecil */
    .step-label { display: none; }
    .step-sep { max-width: 24px; }

    /* Result cards jadi full width */
    .rcard-stats { gap: 1rem; }
    .rcard-price { font-size: 1.35rem; }

    /* Summary box */
    .summary-box { padding: 0.9rem 1rem; }
    .sb-text { font-size: 0.78rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_excel("data/outfit data.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['product_category'] = df['product_category'].str.strip()
    df['product_type']     = df['product_type'].str.strip()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['hour']      = df['transaction_time'].dt.hour
    df['month']     = df['transaction_date'].dt.month
    df['dayofweek'] = df['transaction_date'].dt.dayofweek
    df = df.dropna(subset=['transaction_qty','unit_price','product_category',
                           'product_type','city_location','subdistrict_name'])
    df_ml = df.copy()
    le_cat  = LabelEncoder(); le_type = LabelEncoder()
    le_city = LabelEncoder(); le_sub  = LabelEncoder()
    df_ml['product_category'] = le_cat.fit_transform(df_ml['product_category'])
    df_ml['product_type']     = le_type.fit_transform(df_ml['product_type'])
    df_ml['city_location']    = le_city.fit_transform(df_ml['city_location'])
    df_ml['subdistrict_name'] = le_sub.fit_transform(df_ml['subdistrict_name'])
    X = df_ml[['city_location','subdistrict_name','product_category',
               'product_type','unit_price','hour','month','dayofweek']]
    y = df_ml['transaction_qty']
    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X, y)
    return mdl, df, le_cat, le_type, le_city, le_sub, X.columns.tolist()


def predict(mdl, le_cat, le_type, le_city, le_sub, df, cols,
            kategori, tipe, city, subdistrict):
    subset = df[(df['product_category']==kategori) & (df['product_type']==tipe) & (df['city_location']==city)]
    if len(subset) < 5:
        subset = df[(df['product_category']==kategori) & (df['product_type']==tipe)]
    if len(subset) < 5:
        subset = df[df['product_category']==kategori]
    if len(subset) < 5:
        subset = df
    hm = subset['unit_price'].mean()
    hasil = []
    for h in [hm * 0.85, hm, hm * 1.15]:
        row = pd.DataFrame([[
            le_city.transform([city])[0],
            le_sub.transform([subdistrict])[0],
            le_cat.transform([kategori])[0],
            le_type.transform([tipe])[0],
            h, 12, 6, 2
        ]], columns=cols)
        qty = mdl.predict(row)[0] * 30
        hasil.append((h, qty, h * qty))
    return sorted(hasil, key=lambda x: x[2], reverse=True)


with st.spinner("Memuat model..."):
    try:
        mdl, df, le_cat, le_type, le_city, le_sub, cols = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        model_loaded = False

if "hasil" not in st.session_state:
    st.session_state.hasil = None
if "last_input" not in st.session_state:
    st.session_state.last_input = {}

# ── NAVBAR ──
st.markdown("""
<div class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">OutfitSight</div>
    <div class="nav-sep"></div>
    <div class="nav-sub">Sales Intelligence</div>
  </div>
  <div class="nav-badge">
    <div class="nav-dot"></div>Model Ready
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.stop()

# ── HERO ──
total_tx    = len(df)
total_city  = df['city_location'].nunique()
total_menu  = df['product_type'].nunique()

st.markdown(f"""
<div class="hero">
  <div class="hero-tag">Pakaian · Machine Learning · Random Forest</div>
  <div class="hero-title">Prediksi Harga &amp;</div>
  <div class="hero-title-sub">Penjualan Produk</div>
  <div class="hero-desc">
    Temukan harga optimal dan estimasi omset bulanan berdasarkan data transaksi nyata
    menggunakan algoritma Random Forest Regression.
  </div>
  <div class="hero-bottom">
    <div class="hero-stats">
      <div class="hstat">
        <div class="hstat-val">{total_tx:,}</div>
        <div class="hstat-label">Total Transaksi</div>
      </div>
      <div class="hstat">
        <div class="hstat-val">{total_city}</div>
        <div class="hstat-label">Kota Aktif</div>
      </div>
      <div class="hstat">
        <div class="hstat-val">{total_menu}</div>
        <div class="hstat-label">Jenis Produk</div>
      </div>
    </div>
    <div class="hero-deco">RF</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DATA MAPS ──
kategori_list = sorted(df['product_category'].unique())
tipe_map = {k: sorted(df[df['product_category']==k]['product_type'].unique()) for k in kategori_list}
city_list = sorted(df['city_location'].unique())
sub_map  = {c: sorted(df[df['city_location']==c]['subdistrict_name'].unique()) for c in city_list}

# ── 2 PANELS (ATAS-BAWAH) ──

# ── ATAS: INPUT ──
with st.container():
    st.markdown("""
    <div class="panel-header">
      <div>
        <span class="panel-label">Input Pencarian</span>
        <span class="panel-title">Pilih Produk &amp; Lokasi</span>
      </div>
      <div class="panel-icon">🔍</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stepper">
      <div class="step">
        <div class="step-num done">✓</div>
        <span class="step-label done">Produk</span>
      </div>
      <div class="step-sep"></div>
      <div class="step">
        <div class="step-num active">2</div>
        <span class="step-label active">Lokasi</span>
      </div>
      <div class="step-sep"></div>
      <div class="step">
        <div class="step-num idle">3</div>
        <span class="step-label idle">Hasil</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Kategori &amp; Jenis Produk</div>', unsafe_allow_html=True)

    sel_kat  = st.selectbox("Kategori Baju", kategori_list, key="kat_select")
    tipe_list = tipe_map[sel_kat]
    sel_tipe = st.selectbox("Jenis Baju", tipe_list, key="tipe_select")

    st.markdown('<div class="section-label" style="margin-top:1rem">Lokasi Penjualan</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        city = st.selectbox("Kota / Kabupaten", city_list, key="city_sel")
    with c2:
        sub_list    = sub_map[city]
        subdistrict = st.selectbox("Kecamatan", sub_list, key="sub_sel")

    st.markdown('<div style="font-size:0.7rem;color:var(--text3);margin-top:-0.5rem;margin-bottom:0.25rem">Kecamatan menentukan estimasi daya beli konsumen lokal.</div>', unsafe_allow_html=True)

    run = st.button("Prediksi Sekarang →")
    if run:
        with st.spinner("Menganalisis..."):
            hasil = predict(mdl, le_cat, le_type, le_city, le_sub,
                            df, cols, sel_kat, sel_tipe, city, subdistrict)
        st.session_state.hasil = hasil
        st.session_state.last_input = {"kat": sel_kat, "tipe": sel_tipe, "city": city, "sub": subdistrict}
        st.rerun()

st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

# ── BAWAH: HASIL ──
with st.container():
    st.markdown("""
    <div class="panel-header">
      <div>
        <span class="panel-label">Hasil Analisis</span>
        <span class="panel-title">3 Skenario Harga</span>
      </div>
      <div class="panel-icon">📊</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.hasil is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📦</div>
          <div class="empty-title">Belum Ada Hasil</div>
          <div class="empty-desc">Pilih produk dan lokasi di atas, lalu klik <strong>Prediksi Sekarang</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        hasil = st.session_state.hasil
        inp   = st.session_state.last_input
        labels = [("🏆 Terbaik", "best"), ("⚖️ Normal", ""), ("💎 Premium", "")]

        cards_html = '<div class="result-wrap">'
        for i, ((h, qty, omzet), (lab, cls)) in enumerate(zip(hasil, labels)):
            h_idr = int(round(h * 1000))
            q_int = int(qty)
            o_idr = int(round(omzet * 1000))
            rank  = ["#1", "#2", "#3"][i]
            cards_html += f"""
            <div class="rcard {cls}">
              <div class="rcard-top">
                <span class="rcard-badge">{lab}</span>
                <span class="rcard-rank">{rank}</span>
              </div>
              <div class="rcard-price">Rp {h_idr:,}</div>
              <div class="rcard-per">per pcs</div>
              <div class="rcard-divider"></div>
              <div class="rcard-stats">
                <div>
                  <div class="rstat-label">Est. Terjual</div>
                  <div class="rstat-val">{q_int:,} pcs/bln</div>
                </div>
                <div>
                  <div class="rstat-label">Est. Omset</div>
                  <div class="rstat-val">Rp {o_idr:,}</div>
                </div>
              </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        best_h   = int(round(hasil[0][0] * 1000))
        best_o   = int(round(hasil[0][2] * 1000))
        best_qty = int(hasil[0][1])
        st.markdown(f"""
        <div class="summary-box">
          <span class="sb-label">📌 Ringkasan Rekomendasi</span>
          <div class="sb-text">
            Harga optimal untuk <strong>{inp.get('tipe','')}</strong>
            (<em>{inp.get('kat','')}</em>) di
            <strong>{inp.get('city','')} — {inp.get('sub','')}</strong> adalah
            <strong>Rp {best_h:,}</strong> per pcs,
            dengan estimasi penjualan <strong>{best_qty:,} pcs/bulan</strong> dan
            omset <strong>Rp {best_o:,}/bulan</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)