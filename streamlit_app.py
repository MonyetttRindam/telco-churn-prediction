"""Streamlit frontend untuk Telco Churn Prediction."""

import streamlit as st
import requests



# === Setup halaman ===
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# === Header ===
st.title("📊 Telco Churn Predictor")
st.markdown("""
Prediksi kemungkinan pelanggan akan **churn** (berhenti langganan) berdasarkan profil mereka.

Model: LogisticRegression  |  Threshold: 0.6  |  F1 Score: 0.63
""")
st.divider()

# === Cek koneksi ke FastAPI ===
API_URL = "http://127.0.0.1:8000"

try:
    health = requests.get(f"{API_URL}/", timeout=2).json()
    st.success(f"✅ API Connected — Model: {health['model']}, Threshold: {health['threshold']}")
except Exception as e:
    st.error(f"❌ API tidak terhubung. Pastikan FastAPI jalan di {API_URL}")
    st.code(f"uvicorn app.main:app --reload", language="bash")
    st.stop()

    st.divider()
st.subheader("📋 Input Data Pelanggan")

# === SECTION 1: Demografis ===
with st.expander("👤 Demografis", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
        )
        Partner = st.selectbox(
            "Punya Partner?",
            options=["Yes", "No"],
        )
    
    with col2:
        SeniorCitizen = st.selectbox(
            "Senior Citizen?",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )
        Dependents = st.selectbox(
            "Punya Dependents?",
            options=["Yes", "No"],
        )
        # === SECTION 2: Layanan ===
with st.expander("🌐 Layanan", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input(
            "Tenure (bulan)",
            min_value=0, max_value=100, value=12,
            help="Sudah berapa bulan jadi pelanggan?",
        )
        PhoneService = st.selectbox(
            "Phone Service?",
            options=["Yes", "No"],
        )
        MultipleLines = st.selectbox(
            "Multiple Lines?",
            options=["Yes", "No", "No phone service"],
        )
        InternetService = st.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
        )
        OnlineSecurity = st.selectbox(
            "Online Security?",
            options=["Yes", "No", "No internet service"],
        )
    
    with col2:
        OnlineBackup = st.selectbox(
            "Online Backup?",
            options=["Yes", "No", "No internet service"],
        )
        DeviceProtection = st.selectbox(
            "Device Protection?",
            options=["Yes", "No", "No internet service"],
        )
        TechSupport = st.selectbox(
            "Tech Support?",
            options=["Yes", "No", "No internet service"],
        )
        StreamingTV = st.selectbox(
            "Streaming TV?",
            options=["Yes", "No", "No internet service"],
        )
        StreamingMovies = st.selectbox(
            "Streaming Movies?",
            options=["Yes", "No", "No internet service"],
        )
        # === SECTION 3: Kontrak & Pembayaran ===
with st.expander("💳 Kontrak & Pembayaran", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        Contract = st.selectbox(
            "Contract",
            options=["Month-to-month", "One year", "Two year"],
        )
        PaperlessBilling = st.selectbox(
            "Paperless Billing?",
            options=["Yes", "No"],
        )
        PaymentMethod = st.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    
    with col2:
        MonthlyCharges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0, max_value=200.0, value=50.0, step=0.5,
        )
        TotalCharges = st.number_input(
            "Total Charges ($)",
            min_value=0.0, value=600.0, step=10.0,
            help="Total tagihan sejak jadi pelanggan",
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
        st.subheader("📊 Hasil Prediksi")
        
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