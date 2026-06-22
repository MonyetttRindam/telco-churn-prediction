"""
Smoke test Hari 2 — BUKTIKAN SENDIRI apa itu PREPROCESSING & MLFLOW.

Jalankan dari folder root project (D:\\Coding Vscode\\telco-churn-prediction):

  # PART A — lihat preprocessing bekerja: 1 pelanggan -> 52 angka
  python scripts/smoke_test.py

  # PART B — jalankan EKSPERIMEN kecil ke MLflow. ULANGI dgn nilai beda!
  python scripts/smoke_test.py --experiment --C 1.0
  python scripts/smoke_test.py --experiment --C 0.1
  python scripts/smoke_test.py --experiment --C 10

  # lalu lihat & bandingkan semua eksperimen:
  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def part_a_preprocessing():
    """Tunjukkan transformasi 1 pelanggan: bahasa manusia -> angka."""
    from src.preprocessing import clean_raw, add_features, prepare_xy

    print("=" * 64)
    print("PART A — APA YANG DILAKUKAN PREPROCESSING?")
    print("=" * 64)

    df = pd.read_csv(ROOT / "data/raw/Telco-Customer-Churn.csv")
    cust = df.iloc[[0]]

    print("\n[1] DATA MENTAH 1 pelanggan (manusia bisa baca):")
    print(cust[["gender", "tenure", "InternetService", "Contract",
                "MonthlyCharges", "OnlineSecurity"]].to_string(index=False))

    feat = add_features(clean_raw(cust))
    print("\n[2] SETELAH FEATURE ENGINEERING (kolom baru buatan kita):")
    print(feat[["tenure", "tenure_group", "num_addons",
                "is_new_customer", "has_internet"]].to_string(index=False))

    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    X, _ = prepare_xy(df)
    vec = preprocessor.transform(X.iloc[[0]])
    arr = vec.toarray()[0] if hasattr(vec, "toarray") else np.asarray(vec)[0]

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(arr))]

    print(f"\n[3] SETELAH PREPROCESSOR penuh: {X.shape[1]} kolom -> {len(arr)} ANGKA")
    print("    (model cuma ngerti angka, bukan 'Yes' / 'Fiber optic')")
    print("    8 fitur pertama:")
    preview = pd.DataFrame({"fitur": names, "nilai": np.round(arr, 2)}).head(8)
    print(preview.to_string(index=False))

    print(f"\n>>> INTI: preprocessing = penerjemah bahasa-manusia -> {len(arr)} angka siap-model.")
    print(">>> Objek 'preprocessor.joblib' inilah yang nanti dipakai FastAPI biar")
    print("    pelanggan baru diterjemahkan dengan cara yang PERSIS sama.\n")


def part_b_experiment(C):
    """Latih 1 model kecil & catat hasilnya ke MLflow. Ini satu 'eksperimen'."""
    import mlflow
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)

    print("=" * 64)
    print(f"PART B — EKSPERIMEN ke MLFLOW (LogisticRegression, C={C})")
    print("=" * 64)

    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    X_train, X_test, y_train, y_test = joblib.load(ROOT / "models/split_data.joblib")

    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)

    model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    model.fit(X_tr, y_train)
    pred = model.predict(X_te)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print("\nHasil di data test:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.3f}")

    # Catat eksperimen ke MLflow (buku catatan)
    mlflow.set_tracking_uri(f'sqlite:///{(ROOT / "mlflow.db").as_posix()}')
    mlflow.set_experiment("telco-churn")
    with mlflow.start_run(run_name=f"logreg-C{C}"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("class_weight", "balanced")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    print(f"\n>>> Eksperimen 'logreg-C{C}' TERCATAT di MLflow.")
    print(">>> Ulangi script ini dgn --C lain (mis. 0.1, 10), lalu buka UI:")
    print("    mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(">>> Di UI kamu bisa BANDINGKAN semua run & lihat C mana yg recall-nya terbaik.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="store_true",
                    help="jalankan Part B (eksperimen ke MLflow)")
    ap.add_argument("--C", type=float, default=1.0,
                    help="kekuatan regularisasi LogisticRegression (coba 0.1 / 1 / 10)")
    args = ap.parse_args()

    if args.experiment:
        part_b_experiment(args.C)
    else:
        part_a_preprocessing()
