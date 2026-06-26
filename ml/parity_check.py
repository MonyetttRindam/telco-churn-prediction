"""Parity check (Phase 1, Step 1.5): refactor == production?

Verifikasi `ml/train.py` + `ml/evaluate.py` menghasilkan metrik IDENTIK dengan
production model (`models/model_config.json`) + notebook 03. Data di-load FRESH
dari ml/data/*.csv (BUKAN dari split_data.joblib legacy).

Exit code: 0 = parity achieved, 1 = parity failed (siap dipakai di CI).

Jalankan dari root project (venv aktif):
    .venv/Scripts/python.exe ml/parity_check.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.train import train_model        # noqa: E402
from ml.evaluate import evaluate_model  # noqa: E402

THRESHOLD = 0.60
TOLERANCE = 1e-4  # 4 decimal places

# Expected: model_config.json (f1/recall/precision) + notebook 03 cell 27 (roc_auc)
EXPECTED = {
    "f1": 0.6291,
    "recall": 0.7166,
    "precision": 0.5607,
    "roc_auc": 0.8419,
}


def main() -> int:
    # 1. Artifacts production (read-only)
    preprocessor = joblib.load(ROOT / "models/preprocessor.joblib")
    with open(ROOT / "models/model_config.json") as f:
        config = json.load(f)

    # 2. Data FRESH dari CSV (bukan split_data.joblib)
    train_df = pd.read_csv(ROOT / "ml/data/train.csv")
    holdout_df = pd.read_csv(ROOT / "ml/data/test_holdout.csv")

    # 3. Pipeline refactor
    model, metadata = train_model(train_df, preprocessor)
    metrics = evaluate_model(model, preprocessor, holdout_df, threshold=THRESHOLD)

    # 4-5. Comparison table
    print("=" * 64)
    print("PARITY CHECK  (refactor vs production, threshold=%.2f)" % THRESHOLD)
    print("=" * 64)
    print(f"{'Metric':<12}{'Expected':<12}{'Actual':<12}{'Diff':<12}{'Status'}")
    print("-" * 64)

    all_pass = True
    for name, exp in EXPECTED.items():
        act = metrics[name]
        diff = abs(act - exp)
        passed = diff <= TOLERANCE
        all_pass &= passed
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name:<12}{exp:<12.4f}{act:<12.4f}{diff:<12.4f}{status}")

    print("-" * 64)

    # 6. Verdict
    if all_pass:
        print("[PARITY ACHIEVED] Refactor matches production.")
    else:
        n_fail = sum(abs(metrics[k] - v) > TOLERANCE for k, v in EXPECTED.items())
        print(f"[PARITY FAILED] {n_fail} metric(s) drifted. See above.")

        # 7. Diagnostics (hanya kalau fail)
        print("\n--- diagnostics ---")
        cm = metrics["confusion_matrix"]
        print(f"confusion_matrix : tn={cm['tn']} fp={cm['fp']} "
              f"fn={cm['fn']} tp={cm['tp']}")
        print(f"n_train_samples  : {metadata['n_train_samples']}")
        print(f"n_holdout_samples: {metrics['n_samples']}")
        print(f"n_features       : {metadata['n_features']}")
        print(f"config (expected): f1={config['test_f1']} "
              f"recall={config['test_recall']} precision={config['test_precision']}")
        coefs = model.coef_.ravel()[:5]
        print(f"model.coef_[:5]  : {coefs}")

    print("=" * 64)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
