"""Sanity check: pastikan train_model + evaluate_model jalan tanpa error.

BUKAN parity check. Pakai sample kecil (200 row) untuk smoke test saja — metrik
yang keluar PASTI berbeda dari production. Parity check (full data vs
model_config.json) ada di Step 1.5.

Jalankan dari root project:
    .venv/Scripts/python.exe ml/test_train_evaluate.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.train import train_model      # noqa: E402
from ml.evaluate import evaluate_model  # noqa: E402

SAMPLE_N = 200
RANDOM_STATE = 42


def main() -> None:
    # 1. Sample kecil dari train.csv (smoke test, bukan full).
    train_df = pd.read_csv(ROOT / "ml/data/train.csv")
    sample = train_df.sample(n=SAMPLE_N, random_state=RANDOM_STATE)
    print(f"Sample train : {len(sample)} rows (dari {len(train_df)})")

    # 2. Load preprocessor FIXED (load-only, tidak ditimpa).
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")

    # 3. train_model -> metadata
    model, training_metrics = train_model(sample, preprocessor)
    print("\n--- training_metrics (metadata only) ---")
    for k, v in training_metrics.items():
        print(f"  {k:16s}: {v}")

    # 4. Holdout full
    holdout_df = pd.read_csv(ROOT / "ml/data/test_holdout.csv")
    print(f"\nHoldout      : {len(holdout_df)} rows")

    # 5. evaluate_model -> metrics
    metrics = evaluate_model(model, preprocessor, holdout_df)
    print("\n--- evaluate_model metrics (DUMMY, 200-sample model) ---")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v}")

    # 6. Done
    print("\n[PASSED] Sanity check - train_model + evaluate_model jalan tanpa error.")
    print("   (Angka di atas dummy; parity check full-data di Step 1.5.)")


if __name__ == "__main__":
    main()
