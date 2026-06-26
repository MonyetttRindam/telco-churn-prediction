"""Generate fixed train / holdout split untuk MLOps retraining system.

Dijalankan SEKALI di Phase 1. Output `ml/data/test_holdout.csv` adalah holdout
yang FIXED selamanya — tidak pernah disentuh lagi setelah ini (lihat MLOPS_PLAN.md
> Critical Constraints).

Parity dengan notebook 02:
  notebook 02 = prepare_xy(df_raw) -> train_test_split(X, y, test_size=0.2,
                stratify=y, random_state=42).
  Di sini kita split satu dataframe utuh (kolom Churn ikut di dalam) tapi
  stratify pakai y = (Churn=='Yes').astype(int) dan random_state=42 yang SAMA.
  train_test_split menentukan partisi dari indeks + stratify array, jadi baris
  train/holdout di sini identik dengan split notebook -> metrics nanti match.

Idempotent: random_state=42, jadi run ulang menghasilkan file identik.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Akses modul preprocessing reusable (root project = parent dari ml/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_raw, add_features, TARGET  # noqa: E402

RAW_CSV = ROOT / "data/raw/Telco-Customer-Churn.csv"
OUT_DIR = ROOT / "ml/data"
TRAIN_CSV = OUT_DIR / "train.csv"
HOLDOUT_CSV = OUT_DIR / "test_holdout.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    # 1. Load raw
    df_raw = pd.read_csv(RAW_CSV)
    n_raw = len(df_raw)

    # 2-3. Cleaning + feature engineering (row-wise, aman pre-split)
    df = clean_raw(df_raw)
    df = add_features(df)

    # 4. Stratified split. y hanya untuk stratify; Churn tetap ada di df.
    y = (df[TARGET] == "Yes").astype(int)
    train_df, holdout_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 5. Save sebagai CSV (self-contained, kolom Churn ikut)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    holdout_df.to_csv(HOLDOUT_CSV, index=False)

    # 6. Summary
    train_rate = (train_df[TARGET] == "Yes").mean() * 100
    holdout_rate = (holdout_df[TARGET] == "Yes").mean() * 100
    diff = abs(train_rate - holdout_rate)

    print("=" * 60)
    print("FIXED HOLDOUT SPLIT")
    print("=" * 60)
    print(f"Source           : {RAW_CSV.relative_to(ROOT)}")
    print(f"Total rows (raw) : {n_raw}")
    print("-" * 60)
    print(f"Train   : {len(train_df):>5} rows | churn rate {train_rate:6.2f}%")
    print(f"Holdout : {len(holdout_df):>5} rows | churn rate {holdout_rate:6.2f}%")
    print(f"Selisih churn rate train vs holdout : {diff:.2f}% "
          f"({'OK <0.5%' if diff < 0.5 else 'WARNING >=0.5%'})")
    print("-" * 60)
    print(f"Kolom output ({len(train_df.columns)}):")
    print(f"  {list(train_df.columns)}")
    new_feats = ["num_addons", "is_new_customer", "has_internet", "tenure_group"]
    present = [c for c in new_feats if c in train_df.columns]
    print(f"Fitur baru add_features() hadir : {present}")
    print("-" * 60)
    print(f"Tersimpan:")
    print(f"  - {TRAIN_CSV.relative_to(ROOT)}")
    print(f"  - {HOLDOUT_CSV.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
