"""Streamlit frontend untuk Telco Churn Prediction."""

import streamlit as st
import requests
import os


# === Setup halaman ===
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# === Custom CSS / Theme (monokrom hitam–putih, profesional) ===
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        max-width: 780px;
        padding-top: 2rem;
    }

    h1, h2, h3, h4, p, label, span, div {
        color: #111418;
    }

    /* Header bar hitam */
    .app-header {
        background: #111418;
        border-radius: 14px;
        padding: 26px 30px;
        margin-bottom: 4px;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        color: #c7ccd3;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-top: 6px;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 10px 8px 0 0;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: #ffffff;
        color: #111418;
        border: 1px solid #ffffff;
    }

    /* Section title dengan garis bawah tegas */
    .section-label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111418;
        border-left: 4px solid #111418;
        padding-left: 10px;
        margin: 4px 0 2px 0;
    }

    /* Expander = kartu putih dengan border tegas */
    [data-testid="stExpander"] {
        border: 1px solid #e3e5e8;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(17, 20, 24, 0.05);
        margin-bottom: 0.7rem;
    }
    [data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        color: #111418 !important;
    }

    /* Input fields */
    label {
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #3a3f47 !important;
    }
    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 8px !important;
        border-color: #cfd3d8 !important;
        background: #fafbfc !important;
    }
    div[data-baseweb="select"] > div:hover, .stNumberInput input:hover {
        border-color: #9aa0a8 !important;
    }
    div[data-baseweb="select"] > div:focus-within, .stNumberInput input:focus {
        border-color: #111418 !important;
        box-shadow: 0 0 0 1px #111418 !important;
        background: #ffffff !important;
    }
    /* Sedikit jarak antar input biar napas */
    .stSelectbox, .stNumberInput { margin-bottom: 2px; }

    /* Label pengelompokan kecil di dalam section */
    .group-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        border-bottom: 1px solid #ececef;
        padding-bottom: 4px;
        margin: 6px 0 10px 0;
    }

    /* Tombol prediksi hitam solid */
    .stButton > button,
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.01em;
        padding: 0.65rem 0;
        background: #111418 !important;
        color: #ffffff !important;
        border: 1px solid #111418 !important;
        transition: background 0.15s ease;
    }
    .stButton > button *,
    .stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }
    .stButton > button:hover,
    .stButton > button[kind="primary"]:hover {
        background: #2b2f36 !important;
        color: #ffffff !important;
        border-color: #2b2f36 !important;
    }
    .stButton > button:active,
    .stButton > button:focus,
    .stButton > button[kind="primary"]:focus {
        color: #ffffff !important;
        border-color: #111418 !important;
        box-shadow: none !important;
    }

    /* Metric card */
    [data-testid="stMetric"] {
        background: #111418;
        border: 1px solid #111418;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"] * { color: #ffffff !important; }

    /* Divider lebih lembut */
    hr { border-color: #e3e5e8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Header ===
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📊 Telco Churn Predictor</div>
        <div class="app-subtitle">
            Perkirakan kemungkinan seorang pelanggan akan <b>berhenti berlangganan (churn)</b>
            berdasarkan profil layanannya.
        </div>
        <div>
            <span class="tag">Model: LogisticRegression</span>
            <span class="tag">Threshold: 0.6</span>
            <span class="tag">F1 Score: 0.63</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# === Cek koneksi ke FastAPI ===
API_URL = os.getenv("API_URL", "http://localhost:8000")

try:
    health = requests.get(f"{API_URL}/", timeout=2).json()
    st.success(f"✅ API Connected — Model: {health['model']}, Threshold: {health['threshold']}")
except Exception as e:
    st.error(f"❌ API tidak terhubung. Pastikan FastAPI jalan di {API_URL}")
    st.code(f"uvicorn app.main:app --reload", language="bash")
    st.stop()

    st.divider()
st.markdown('<div class="section-label">📋 Data Pelanggan</div>', unsafe_allow_html=True)
st.caption("Isi profil pelanggan di bawah ini, lalu klik tombol prediksi.")

# === SECTION 1: Demografis ===
with st.expander("👤 Demografis", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Jenis Kelamin",
            options=["Male", "Female"],
        )
        Partner = st.selectbox(
            "Memiliki Pasangan?",
            options=["Yes", "No"],
        )

    with col2:
        SeniorCitizen = st.selectbox(
            "Lansia (Senior Citizen)?",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )
        Dependents = st.selectbox(
            "Memiliki Tanggungan?",
            options=["Yes", "No"],
        )

# === SECTION 2: Layanan ===
with st.expander("🌐 Layanan", expanded=True):
    # Lama langganan & telepon
    st.markdown('<div class="group-label">Langganan & Telepon</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input(
            "Lama Berlangganan (bulan)",
            min_value=0, max_value=100, value=12,
            help="Sudah berapa bulan menjadi pelanggan?",
        )
    with col2:
        PhoneService = st.selectbox(
            "Layanan Telepon?",
            options=["Yes", "No"],
        )
    with col3:
        MultipleLines = st.selectbox(
            "Beberapa Saluran?",
            options=["Yes", "No", "No phone service"],
        )

    # Internet & layanan tambahan
    st.markdown('<div class="group-label">Internet & Layanan Tambahan</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        InternetService = st.selectbox(
            "Layanan Internet",
            options=["DSL", "Fiber optic", "No"],
        )
        OnlineBackup = st.selectbox(
            "Pencadangan Online?",
            options=["Yes", "No", "No internet service"],
        )
        TechSupport = st.selectbox(
            "Dukungan Teknis?",
            options=["Yes", "No", "No internet service"],
        )
        StreamingMovies = st.selectbox(
            "Streaming Film?",
            options=["Yes", "No", "No internet service"],
        )

    with col2:
        OnlineSecurity = st.selectbox(
            "Keamanan Online?",
            options=["Yes", "No", "No internet service"],
        )
        DeviceProtection = st.selectbox(
            "Proteksi Perangkat?",
            options=["Yes", "No", "No internet service"],
        )
        StreamingTV = st.selectbox(
            "Streaming TV?",
            options=["Yes", "No", "No internet service"],
        )

# === SECTION 3: Kontrak & Pembayaran ===
with st.expander("💳 Kontrak & Pembayaran", expanded=True):
    # Detail kontrak
    st.markdown('<div class="group-label">Kontrak</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        Contract = st.selectbox(
            "Jenis Kontrak",
            options=["Month-to-month", "One year", "Two year"],
        )
    with col2:
        PaperlessBilling = st.selectbox(
            "Tagihan Tanpa Kertas?",
            options=["Yes", "No"],
        )
    with col3:
        PaymentMethod = st.selectbox(
            "Metode Pembayaran",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    # Tagihan
    st.markdown('<div class="group-label">Tagihan</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        MonthlyCharges = st.number_input(
            "Tagihan Bulanan ($)",
            min_value=0.0, max_value=200.0, value=50.0, step=0.5,
        )
    with col2:
        TotalCharges = st.number_input(
            "Total Tagihan ($)",
            min_value=0.0, value=600.0, step=10.0,
            help="Total tagihan sejak menjadi pelanggan",
        )

# === TOMBOL PREDIKSI ===
st.divider()

if st.button("🔮 Prediksi Churn", type="primary", use_container_width=True):
    # 1. Kumpulkan semua input jadi payload
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    # 2. Kirim ke FastAPI
    try:
        with st.spinner("Memprediksi..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

        # 3. Tampilkan hasil
        st.divider()
        st.markdown('<div class="section-label">📊 Hasil Prediksi</div>', unsafe_allow_html=True)
        st.write("")

        proba = result["probability"]
        churn = result["churn"]
        threshold = result["threshold"]

        # Layout: 2 kolom — label & probability
        col1, col2 = st.columns(2)

        with col1:
            if churn == 1:
                st.error(f"### ⚠️ CHURN")
                st.caption("Pelanggan berisiko tinggi berhenti langganan")
            else:
                st.success(f"### ✅ TIDAK CHURN")
                st.caption("Pelanggan diprediksi akan tetap berlangganan")

        with col2:
            st.metric(
                label="Probabilitas Churn",
                value=f"{proba:.1%}",
                delta=f"Threshold: {threshold:.0%}",
                delta_color="off",
            )

        # Progress bar visualisasi
        st.progress(proba)

        # Interpretasi bisnis
        st.divider()
        if proba >= 0.75:
            st.warning("🔥 **Risiko sangat tinggi** — perlu intervensi segera (call sales, kasih diskon)")
        elif proba >= threshold:
            st.warning("⚠️ **Risiko menengah-tinggi** — pertimbangkan kirim email retention")
        elif proba >= 0.4:
            st.info("ℹ️ **Risiko menengah** — monitor saja, belum perlu intervensi")
        else:
            st.success("👍 **Risiko rendah** — pelanggan setia, fokus retention biasa")

        # Show raw response (collapsible)
        with st.expander("🔍 Raw API Response"):
            st.json(result)

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.status_code}")
        st.json(e.response.json())
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
