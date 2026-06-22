"""
Preprocessing untuk Telco Churn Prediction.

Modul ini dibuat reusable supaya logika yang SAMA dipakai di:
- notebook (eksperimen / training)
- FastAPI nanti (Hari 5) saat memproses request pelanggan baru

Aturan penting (anti data-leakage):
- `add_features()` = feature engineering ROW-WISE (cuma lihat baris itu sendiri),
  jadi AMAN dijalankan sebelum train/test split.
- Hal yang butuh statistik global (mean untuk scaling, kategori untuk encoding)
  diletakkan di dalam `build_preprocessor()` -> hanya di-`fit` pada data train.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 6 layanan add-on; dipakai untuk menghitung jumlah add-on aktif
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TARGET = "Churn"
DROP_COLS = ["customerID"]  # identitas, tidak ada nilai prediktif


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data mentah: perbaiki TotalCharges yang ber-tipe object.

    11 baris punya TotalCharges = ' ' (spasi) karena tenure = 0 (pelanggan
    baru, belum pernah ditagih). Kita ubah ke numeric lalu isi 0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering row-wise. Aman dipanggil sebelum split.

    Fitur baru:
    - num_addons      : jumlah layanan add-on aktif (0-6). EDA: makin sedikit
                        add-on, makin tinggi churn.
    - is_new_customer : 1 jika tenure <= 12 bulan. EDA: churner rata-rata
                        tenure 18 bln -> pelanggan baru paling rawan.
    - has_internet    : 1 jika langganan internet. Kelompok 'No internet'
                        churn-nya sangat rendah (7.4%).
    - tenure_group    : binning tenure jadi 4 fase masa langganan.
    """
    df = df.copy()
    df["num_addons"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)  # str supaya bisa di-one-hot dengan aman
    return df


def split_columns(df: pd.DataFrame):
    """Tentukan kolom numeric / passthrough / categorical untuk transformer.

    Mengembalikan (numeric_cols, passthrough_cols, categorical_cols).
    Dipanggil pada dataframe yang SUDAH lewat add_features().
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons"]
    # sudah 0/1, tidak perlu di-scale / di-encode
    passthrough_cols = ["SeniorCitizen", "is_new_customer", "has_internet"]

    exclude = set(numeric_cols + passthrough_cols + DROP_COLS + [TARGET])
    categorical_cols = [c for c in df.columns if c not in exclude]
    return numeric_cols, passthrough_cols, categorical_cols


def build_preprocessor(numeric_cols, passthrough_cols, categorical_cols) -> ColumnTransformer:
    """Bangun ColumnTransformer.

    - numeric     : isi NaN (median) -> StandardScaler
    - categorical : OneHotEncoder (handle_unknown='ignore' supaya kategori
                    baru saat prediksi tidak bikin error)
    - passthrough : diteruskan apa adanya
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("pass", "passthrough", passthrough_cols),
        ],
        remainder="drop",  # selain di atas (mis. customerID) dibuang
    )


def prepare_xy(df: pd.DataFrame):
    """Pipeline lengkap dari data mentah -> (X, y).

    y dikodekan 1 = Churn 'Yes', 0 = 'No'.
    """
    df = clean_raw(df)
    df = add_features(df)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
