# API Usage — Telco Churn Prediction

Base URL (lokal): `http://127.0.0.1:8000` · Interactive docs: `http://127.0.0.1:8000/docs`

## Menjalankan API

```bash
uvicorn app.main:app --reload
```

API memuat artifact dari `models/` (preprocessor, model, config) sekali saat startup.

## Endpoints

| Method | Path | Fungsi |
|---|---|---|
| `GET` | `/` | Healthcheck + info model |
| `POST` | `/predict` | Prediksi churn 1 pelanggan |

---

## GET / — Healthcheck

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "status": "ok",
  "model": "LogisticRegression",
  "threshold": 0.6
}
```

---

## POST /predict

Mengirim profil 1 pelanggan, menerima prediksi churn. **Semua field wajib** dan tervalidasi (nilai di luar daftar `Literal` ditolak dengan HTTP 422).

### Field input

| Field | Tipe | Nilai valid |
|---|---|---|
| `gender` | string | `Male`, `Female` |
| `SeniorCitizen` | int | `0`, `1` |
| `Partner` | string | `Yes`, `No` |
| `Dependents` | string | `Yes`, `No` |
| `tenure` | int | `0`–`100` (bulan langganan) |
| `PhoneService` | string | `Yes`, `No` |
| `MultipleLines` | string | `Yes`, `No`, `No phone service` |
| `InternetService` | string | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | string | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | string | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | string | `Yes`, `No`, `No internet service` |
| `TechSupport` | string | `Yes`, `No`, `No internet service` |
| `StreamingTV` | string | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | string | `Yes`, `No`, `No internet service` |
| `Contract` | string | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | string | `Yes`, `No` |
| `PaymentMethod` | string | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | float | `>= 0` (tagihan bulanan, $) |
| `TotalCharges` | float | `>= 0` (total tagihan, $) |

### Field output

| Field | Tipe | Keterangan |
|---|---|---|
| `churn` | int | `1` = diprediksi churn, `0` = tidak |
| `probability` | float | Probabilitas churn (0–1) |
| `threshold` | float | Threshold decision yang dipakai (0.6) |

---

## Contoh 1 — High-risk customer

Pelanggan baru (tenure 2), Fiber optic, Month-to-month, Electronic check, tanpa add-on → kombinasi paling rawan churn.

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0
  }'
```

**Response:**
```json
{
  "churn": 1,
  "probability": 0.8334,
  "threshold": 0.6
}
```

---

## Contoh 2 — Low-risk customer

Pelanggan loyal (tenure 60), kontrak Two year, auto-payment, banyak add-on → risiko churn rendah.

**Python (`requests`):**
```python
import requests

payload = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 80.0,
    "TotalCharges": 4800.0,
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.json())
```

**Response:**
```json
{
  "churn": 0,
  "probability": 0.0504,
  "threshold": 0.6
}
```

---

## Contoh 3 — Edge case: pelanggan baru (tenure = 0)

Pelanggan yang baru daftar (`tenure = 0`, `TotalCharges = 0`) — kasus yang dulu bikin `TotalCharges` kosong di dataset mentah. API tetap menanganinya dengan benar (di-`fillna(0)` oleh `clean_raw`).

**curl:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 0, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 0.0
  }'
```

**Response:**
```json
{
  "churn": 1,
  "probability": 0.9015,
  "threshold": 0.6
}
```

---

## Error handling — HTTP 422

Jika ada field yang nilainya di luar daftar valid (mis. `Contract: "Yearly"`), FastAPI menolak request **sebelum** sampai ke model:

```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "Contract"],
      "msg": "Input should be 'Month-to-month', 'One year' or 'Two year'",
      "input": "Yearly"
    }
  ]
}
```

Hal yang memicu 422: nilai kategori tidak valid, `tenure` di luar 0–100, `MonthlyCharges`/`TotalCharges` negatif, atau field yang hilang.

---

## Integrasi dengan Streamlit

Frontend [`streamlit_app.py`](../streamlit_app.py) adalah salah satu *consumer* API ini: ia mengumpulkan input via form, mem-POST ke `/predict`, lalu menampilkan hasil + interpretasi bisnis. Jalankan API dulu (`uvicorn app.main:app --reload`), baru `streamlit run streamlit_app.py`.
