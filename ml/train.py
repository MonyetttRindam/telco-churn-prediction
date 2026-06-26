"""Reusable training untuk MLOps retraining system (Phase 1, Step 1.4).

`train_model()` HANYA melatih model — tidak menghitung metrik performa
(itu tanggung jawab `evaluate_model()` di ml/evaluate.py). Lihat MLOPS_PLAN.md
> Decision Log: "train_model() returns metadata only (no performance metrics)".

Constraints (lihat MLOPS_PLAN.md > Critical Constraints):
- Preprocessor FIXED: di-pass sebagai argument, transform-only (TIDAK pernah refit).
- Model params IDENTIK notebook 03 cell `logreg-baseline`.

Catatan customerID:
  preprocessor.feature_names_in_ memuat 'customerID' (preprocessor di-fit di
  notebook 02 pada X yang hanya drop 'Churn'). 'customerID' masuk ke remainder
  dengan remainder='drop' -> di-buang internal oleh preprocessor. Maka di sini
  kita HANYA drop 'Churn'; jangan drop 'customerID' manual (kalau di-drop,
  transform akan gagal karena feature_names_in_ tidak cocok).
"""

from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression

TARGET = "Churn"

# Param IDENTIK notebook 03 (cell params_logreg / logreg-baseline)
MODEL_PARAMS: dict = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}


def train_model(
    train_df: pd.DataFrame,
    preprocessor,
) -> tuple[LogisticRegression, dict]:
    """Train LogisticRegression pada train_df yang sudah di-preprocess.

    Args:
        train_df: DataFrame hasil clean_raw() + add_features(), HARUS menyertakan
            kolom 'Churn'. Kolom 'customerID' boleh ada (di-drop oleh preprocessor).
        preprocessor: ColumnTransformer yang SUDAH di-fit. Dipakai transform-only
            (TIDAK pernah di-fit ulang di sini).

    Returns:
        model: LogisticRegression yang sudah di-fit.
        training_metrics: metadata training (BUKAN metrik performa) berisi
            n_train_samples, n_features, model_params, trained_at.
    """
    # 1. Pisahkan target. Hanya drop 'Churn' -> 'customerID' tetap ada supaya
    #    kolom input cocok dengan preprocessor.feature_names_in_.
    y = (train_df[TARGET] == "Yes").astype(int)
    X = train_df.drop(columns=[TARGET])

    # 2. Transform pakai preprocessor FIXED (NEVER fit_transform).
    X_transformed = preprocessor.transform(X)

    # 3. Latih model dengan param identik notebook 03.
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_transformed, y)

    # 4. Metadata only (no performance metrics).
    training_metrics = {
        "n_train_samples": int(X_transformed.shape[0]),
        "n_features": int(X_transformed.shape[1]),
        "model_params": dict(MODEL_PARAMS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return model, training_metrics
