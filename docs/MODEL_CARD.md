# Model Card — Telco Churn Classifier

## Model Overview

- **Model type:** Logistic Regression (binary classification)
- **Task:** Memprediksi probabilitas pelanggan Telco akan *churn* (berhenti berlangganan)
- **Version:** 1.0
- **Framework:** scikit-learn 1.5.2
- **Decision threshold:** 0.60 (hasil threshold tuning, bukan default 0.5)
- **Artifacts:** `models/model_final.joblib`, `models/preprocessor.joblib`, `models/model_config.json`

---

## Intended Use

**Use case utama:** Membantu tim retensi memprioritaskan pelanggan berisiko churn untuk diberi intervensi (promo, follow-up). Output berupa probabilitas churn + label biner.

**Pengguna yang dituju:** Tim retensi / CRM, analis bisnis.

**Out-of-scope:**
- **Bukan** untuk keputusan otomatis yang merugikan pelanggan (mis. menaikkan harga, memblokir layanan).
- **Bukan** untuk dataset di luar konteks Telco / fitur yang berbeda.
- Prediksi bersifat *decision-support* — harus didampingi judgment manusia.

---

## Training Data

- **Sumber:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (publik, 7.043 pelanggan, 21 kolom)
- **Target:** `Churn` (Yes/No) — **imbalanced: 26.5% positif** (1.869 churn)
- **Split:** 80/20 stratified → 5.634 train / 1.409 test, `random_state=42` (proporsi churn terjaga di kedua set)
- **Catatan:** dataset bersifat *fiktif* (perusahaan Telco AS); angka & mata uang di analisis bisnis diasumsikan ke konteks Rupiah untuk ilustrasi.

---

## Preprocessing & Features

Logika tersimpan reusable di [`src/preprocessing.py`](../src/preprocessing.py) — dipakai **identik** saat training dan saat inference di API (tidak ada risiko drift).

- **Cleaning:** `TotalCharges` (11 baris kosong untuk pelanggan `tenure=0`) → numeric, `fillna(0)`.
- **Engineered features (4):**
  - `num_addons` — jumlah add-on aktif (0–6)
  - `is_new_customer` — 1 jika `tenure ≤ 12`
  - `has_internet` — 1 jika berlangganan internet
  - `tenure_group` — binning tenure jadi 4 fase
- **Encoding:** numeric → impute median + `StandardScaler`; categorical → `OneHotEncoder(handle_unknown='ignore')`.
- **Output:** 24 kolom mentah → **52 fitur** final.

---

## Model Details

Dipilih **Logistic Regression** dengan hyperparameter:

```python
LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                   solver="lbfgs", random_state=42)
```

`class_weight='balanced'` dipakai untuk menangani imbalance. Dibandingkan 2 alternatif:

| Model | Recall | F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.791 | 0.612 | 0.842 | 0.733 |
| Random Forest | 0.757 | 0.634 | 0.842 | 0.768 |
| XGBoost | 0.668 | 0.597 | 0.818 | 0.761 |

*(metrik di atas pada threshold default 0.5)*

**Alasan memilih Logistic Regression** meski F1 Random Forest sedikit lebih tinggi:
1. **Recall tertinggi (0.791)** — paling sejalan dgn tujuan: menangkap calon churner.
2. **ROC-AUC tertinggi (0.842)** — kualitas ranking terbaik.
3. **Interpretable & murah di-deploy** — koefisien bisa dijelaskan ke tim retensi.

> Hyperparameter tuning (Optuna, 30 trials, 5-fold CV) sudah dicoba tetapi tidak memberi perbaikan berarti, sehingga baseline dipertahankan. Semua eksperimen tercatat di MLflow.

---

## Evaluation

Performa **model final** pada test set dengan **threshold 0.60** (dipilih untuk memaksimalkan F1):

| Metric | Score |
|---|---|
| Precision | 0.561 |
| Recall | 0.717 |
| F1 | 0.629 |
| ROC-AUC | 0.842 |
| Accuracy | 0.776 |

**Confusion matrix (test, n=1.409):**

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | TN = 825 | FP = 210 |
| **Actual Yes** | FN = 106 | TP = 268 |

Model menangkap **268 dari 374** churner sebenarnya (recall 71.7%).

---

## Business Impact

Estimasi berdasarkan confusion matrix @ threshold 0.60, dengan **asumsi** (lihat notebook 03):

| Parameter | Nilai |
|---|---|
| Revenue per customer | Rp 500.000 / bulan |
| Biaya promo (per false alarm) | Rp 50.000 |
| Retention rate (churner di-approach) | 60% |

→ **Net profit ≈ Rp 16,9 juta/bulan (≈ Rp 203 juta/tahun)**.

> Angka ini sensitif terhadap asumsi; ubah parameter di notebook untuk *what-if analysis*.

---

## Limitations & Bias

- **Imbalanced data (26.5%)** — precision relatif rendah (0.561): cukup banyak false alarm. Bisa diterima karena biaya promo (Rp 50rb) jauh lebih kecil dari kerugian kehilangan pelanggan.
- **Static snapshot** — model dilatih pada data satu titik waktu; pola churn bisa berubah → perlu retraining berkala.
- **Asumsi bisnis** — estimasi profit bergantung pada ARPU/biaya/retention yang diasumsikan, bukan data finansial riil.
- **Fairness** — fitur `gender` ikut dilatih tetapi terbukti hampir tidak berpengaruh (churn 26.9% vs 26.2%); tidak ada audit fairness formal terhadap atribut sensitif (`gender`, `SeniorCitizen`).
- **Threshold trade-off** — 0.60 mengutamakan F1; untuk mengejar recall lebih tinggi (menangkap lebih banyak churner), turunkan threshold dengan konsekuensi lebih banyak false alarm.
- **Generalization** — dataset fiktif berbasis AS; tidak otomatis valid untuk operator/pasar lain tanpa kalibrasi ulang.

---

## How to Use

- **REST API:** lihat [API_USAGE.md](API_USAGE.md) untuk contoh request/response.
- **Smoke test lokal:** jalankan [`test_load.py`](../test_load.py) untuk memuat artifact & memprediksi 1 sampel.
- **Training ulang:** jalankan notebook `01` → `02` → `03` (butuh `requirements-dev.txt`).
