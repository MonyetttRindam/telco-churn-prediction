"""Upload model production awal (v_initial) ke HuggingFace Hub (Phase 1, Step 1.7).

Membangun staging folder lokal `hf_upload/`, push ke repo
`{HF_USERNAME}/telco-churn-models`, lalu cleanup staging.

SECURITY:
- Token HANYA di-load dari .env (python-dotenv). TIDAK pernah di-print/log/hardcode.
- Error apa pun yang mungkin memuat token akan di-redact jadi 'hf_***' sebelum
  ditampilkan (lihat _redact()).

Jalankan dari root project (venv aktif), HANYA setelah review:
    .venv/Scripts/python.exe scripts/upload_initial_model.py
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "hf_upload"

# Sumber artifact production
SRC_MODEL = ROOT / "models/model_final.joblib"
SRC_PREPROCESSOR = ROOT / "models/preprocessor.joblib"
SRC_SYNTHETIC = ROOT / "ml/data/synthetic"
N_BATCHES = 5

VERSION_ID = "v_initial"


def _redact(msg: str, token: str | None) -> str:
    """Redact token value dari pesan apa pun sebelum ditampilkan."""
    if token and token in msg:
        msg = msg.replace(token, "hf_***")
    return msg


def build_metrics(created_at: str) -> dict:
    """metrics.json — richer dari model_config.json + roc_auc dari notebook 03."""
    return {
        "version_id": VERSION_ID,
        "f1": 0.6291,
        "recall": 0.7166,
        "precision": 0.5607,
        "roc_auc": 0.8419,
        "threshold": 0.60,
        "n_train_samples": 5634,
        "n_holdout_samples": 1409,
        "model_type": "LogisticRegression",
        "model_params": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        },
        "evaluated_on_holdout": True,
        "created_at": created_at,
    }


def build_registry(metrics: dict, created_at: str) -> dict:
    """registry.json — version registry awal dengan history."""
    return {
        "active": VERSION_ID,
        "previous": None,
        "versions": [
            {
                "id": VERSION_ID,
                "created_at": created_at,
                "metrics": metrics,
                "batches_used": [],
                "status": "active",
                "reason": "Initial production model from notebook 03",
            }
        ],
    }


README = """\
---
license: mit
tags:
  - tabular-classification
  - churn-prediction
  - mlops
---

# Telco Customer Churn Prediction Model

ML model untuk prediksi customer churn dengan sistem retraining otomatis.

## Current Version: v_initial

| Metric | Value |
|--------|-------|
| F1 | 0.6291 |
| Recall | 0.7166 |
| Precision | 0.5607 |
| ROC-AUC | 0.8419 |
| Threshold | 0.60 |

## Structure

- `current/` — active model artifacts (model.pkl, preprocessor.pkl, metrics.json)
- `synthetic/` — synthetic data batches for retraining experiments (SDV-generated)
- `registry.json` — version registry with history

## Training

- Algorithm: LogisticRegression (sklearn 1.5.2)
- Features: 52 (after one-hot encoding)
- Train samples: 5634
- Holdout samples: 1409 (fixed, never touched)

## License

MIT
"""


def build_staging() -> None:
    """Bangun struktur hf_upload/ lokal."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "current").mkdir(parents=True)
    (STAGING / "synthetic").mkdir(parents=True)

    # current/
    shutil.copyfile(SRC_MODEL, STAGING / "current/model.pkl")
    shutil.copyfile(SRC_PREPROCESSOR, STAGING / "current/preprocessor.pkl")

    created_at = datetime.now(timezone.utc).isoformat()
    metrics = build_metrics(created_at)
    registry = build_registry(metrics, created_at)

    (STAGING / "current/metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (STAGING / "registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    (STAGING / "README.md").write_text(README, encoding="utf-8")

    # synthetic/
    for i in range(1, N_BATCHES + 1):
        shutil.copyfile(
            SRC_SYNTHETIC / f"batch_{i}.csv", STAGING / f"synthetic/batch_{i}.csv"
        )

    print(f"Staging folder dibangun di {STAGING.relative_to(ROOT)}/")


def main() -> None:
    # 1. Load env (token tidak pernah di-print)
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"

    # 2. Build staging
    build_staging()

    # 3. Push ke HF Hub
    try:
        create_repo(
            repo_id, token=token, exist_ok=True, repo_type="model", private=False
        )
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(STAGING),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial upload: v_initial production model + synthetic data",
        )
        print(f"[SUCCESS] Uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:  # redact token sebelum re-raise/print
        raise RuntimeError(
            f"Upload gagal: {_redact(str(e), token)}"
        ) from None
    finally:
        # 4. Cleanup staging (selalu, sukses maupun gagal)
        if STAGING.exists():
            shutil.rmtree(STAGING)
            print("Local staging folder cleaned up.")


if __name__ == "__main__":
    main()
