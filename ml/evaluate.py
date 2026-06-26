"""Evaluasi model di fixed holdout (Phase 1, Step 1.4).

`evaluate_model()` HANYA mengukur performa — TIDAK pernah fit apapun (model
maupun preprocessor). Semua angka dihitung di holdout yang FIXED. Threshold
default 0.60 (lihat MLOPS_PLAN.md > Critical Constraints: Fixed threshold).

Catatan customerID: sama seperti train.py, kita hanya drop 'Churn'.
'customerID' di-buang internal oleh preprocessor (remainder='drop').
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)

TARGET = "Churn"


def evaluate_model(
    model: LogisticRegression,
    preprocessor,
    holdout_df: pd.DataFrame,
    threshold: float = 0.60,
) -> dict:
    """Evaluasi model di holdout. TIDAK pernah fit apapun.

    Args:
        model: LogisticRegression yang sudah di-fit.
        preprocessor: ColumnTransformer yang SUDAH di-fit (transform-only).
        holdout_df: DataFrame holdout hasil clean_raw() + add_features(), HARUS
            menyertakan kolom 'Churn'. 'customerID' boleh ada (di-drop preprocessor).
        threshold: ambang probabilitas untuk decision (default 0.60).

    Returns:
        dict berisi:
            - f1, recall, precision : metrik @ threshold.
            - roc_auc               : threshold-independent (pakai probabilitas).
            - threshold             : threshold yang dipakai.
            - n_samples             : jumlah baris holdout.
            - confusion_matrix      : dict {tn, fp, fn, tp}.
    """
    # 1-2. Pisahkan target; hanya drop 'Churn' (customerID dibiarkan).
    y_true = (holdout_df[TARGET] == "Yes").astype(int)
    X = holdout_df.drop(columns=[TARGET])

    # 3. Transform (NEVER fit).
    X_transformed = preprocessor.transform(X)

    # 4-5. Probabilitas churn -> prediksi @ threshold.
    y_proba = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 6-8. Metrik.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "threshold": float(threshold),
        "n_samples": int(len(holdout_df)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
