"""Generate synthetic data via SDV GaussianCopulaSynthesizer (Phase 1, Step 1.6).

Fit HANYA di ml/data/train.csv (holdout TIDAK pernah disentuh — lihat MLOPS_PLAN.md
> Critical Constraints: Synthetic fit scope). Menghasilkan 5 batch × 100 rows ke
ml/data/synthetic/ + menyimpan synthesizer & metadata untuk audit trail.

Reproducibility: np.random.seed(SEED) dipanggil SEKALI sebelum fit, lalu 5 batch
di-sample SEKUENSIAL tanpa reset_sampling(). RNG maju tiap sample -> tiap batch
berbeda (no duplication), tapi seluruh urutan deterministik antar run (sudah
diverifikasi: dua run terpisah menghasilkan fingerprint identik).

Catatan API SDV 1.37: reset_sampling() membuat output deterministik tapi
mengabaikan np seed (selalu sama) — JANGAN dipakai untuk batch berbeda.

Jalankan dari root project (venv aktif):
    .venv/Scripts/python.exe ml/generate_synthetic.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import add_features  # noqa: E402

TRAIN_CSV = ROOT / "ml/data/train.csv"
OUT_DIR = ROOT / "ml/data/synthetic"

SEED = 42
N_BATCHES = 5
ROWS_PER_BATCH = 100
ID_COL = "customerID"

# Fitur turunan deterministik (src.preprocessing.add_features). TIDAK di-fit oleh
# SDV — copula tak bisa menjaga invariant antara base & derived (mis. tenure_group
# = fungsi deterministik dari tenure). Di-derive ulang setelah sampling.
ENGINEERED_COLS = ["num_addons", "is_new_customer", "has_internet", "tenure_group"]

# Override categorical untuk BASE columns yang nilainya integer tapi semantic
# categorical. Engineered cols tak perlu di-override (tak di-fit).
CATEGORICAL_OVERRIDES = [
    "SeniorCitizen",    # 0/1
    "Churn",            # Yes/No
]

# Threshold validasi
CHURN_DRIFT_MAX = 5.0   # % (PASS jika < 5%)
DUP_RATE_MAX = 1.0      # % (PASS jika < 1%)


def build_metadata(df: pd.DataFrame) -> Metadata:
    """Auto-detect metadata lalu override kolom categorical."""
    md = Metadata.detect_from_dataframe(data=df)
    for col in CATEGORICAL_OVERRIDES:
        md.update_column(column_name=col, sdtype="categorical")
    return md


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%"


def validation_report(train_df: pd.DataFrame, synth_all: pd.DataFrame) -> bool:
    """Print validation report. Return True jika semua check PASS."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA VALIDATION")
    print("=" * 60)

    # A. Numerical mean & std
    print("\nA. Numerical Columns (mean & std):")
    print(f"{'':<18}{'REAL':<26}{'SYNTHETIC':<26}{'DRIFT(mean)'}")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        rm, rs = train_df[col].mean(), train_df[col].std()
        sm, ss = synth_all[col].mean(), synth_all[col].std()
        drift = (sm - rm) / rm * 100 if rm else float("nan")
        print(f"{col:<18}"
              f"{f'mean={rm:7.2f} std={rs:7.2f}':<26}"
              f"{f'mean={sm:7.2f} std={ss:7.2f}':<26}"
              f"{drift:+.1f}%")

    # B. Categorical distribution (Contract)
    print("\nB. Categorical Distributions (Contract):")
    cats = ["Month-to-month", "One year", "Two year"]
    rc = train_df["Contract"].value_counts(normalize=True)
    sc = synth_all["Contract"].value_counts(normalize=True)
    print("  REAL:      " + "  ".join(f"{c}: {rc.get(c,0)*100:.1f}%" for c in cats))
    print("  SYNTHETIC: " + "  ".join(f"{c}: {sc.get(c,0)*100:.1f}%" for c in cats))

    # C. Churn rate
    real_churn = (train_df["Churn"] == "Yes").mean() * 100
    syn_churn = (synth_all["Churn"] == "Yes").mean() * 100
    churn_drift = abs(syn_churn - real_churn)
    churn_pass = churn_drift < CHURN_DRIFT_MAX
    print("\nC. Churn Rate:")
    print(f"  REAL:       {real_churn:.2f}%")
    print(f"  SYNTHETIC:  {syn_churn:.2f}%")
    print(f"  DRIFT:      {churn_drift:.2f}%  "
          f"[{'PASS' if churn_pass else 'FAIL'} if <{CHURN_DRIFT_MAX}%]")

    # D. Duplicate check (synthetic match train, exclude customerID)
    feat_cols = [c for c in train_df.columns if c != ID_COL]
    train_keys = set(map(tuple, train_df[feat_cols].astype(str).values))
    synth_keys = list(map(tuple, synth_all[feat_cols].astype(str).values))
    n_dup = sum(k in train_keys for k in synth_keys)
    total = len(synth_keys)
    dup_rate = n_dup / total * 100
    dup_pass = dup_rate < DUP_RATE_MAX
    print("\nD. Duplicate Check (synthetic rows match train, exclude customerID):")
    print(f"  Duplicates: {n_dup} / {total} ({dup_rate:.2f}%)  "
          f"[{'PASS' if dup_pass else 'FAIL'} if <{DUP_RATE_MAX}%]")

    # E. Schema check
    real_cols = list(train_df.columns)
    syn_cols = list(synth_all.columns)
    schema_match = real_cols == syn_cols
    print("\nE. Schema Check:")
    print(f"  Real columns ({len(real_cols)}):      {real_cols}")
    print(f"  Synthetic columns ({len(syn_cols)}): {syn_cols}")
    print(f"  Match:             {'YES' if schema_match else 'NO'}")

    print("\n" + "-" * 60)
    all_pass = churn_pass and dup_pass and schema_match
    verdict = "ALL PASS" if all_pass else "CHECK FAILED (lihat di atas)"
    print(f"VERDICT: {verdict}")
    print("=" * 60)
    return all_pass


def consistency_check(synth_all: pd.DataFrame) -> bool:
    """Verifikasi engineered features konsisten dgn base (target: 0 inkonsistensi).

    Mirror logika src.preprocessing.add_features. Setelah re-derive, semua harus 0.
    """
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK (after re-derive)")
    print("=" * 60)
    n = len(synth_all)

    exp_tg = pd.cut(
        synth_all["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    mis_tg = int((exp_tg != synth_all["tenure_group"]).sum())

    exp_hi = (synth_all["InternetService"] != "No").astype(int)
    mis_hi = int((exp_hi != synth_all["has_internet"]).sum())

    exp_nc = (synth_all["tenure"] <= 12).astype(int)
    mis_nc = int((exp_nc != synth_all["is_new_customer"]).sum())

    print(f"tenure_group inconsistent vs tenure          : {mis_tg}/{n} "
          f"({mis_tg / n * 100:.1f}%)  [TARGET: 0]")
    print(f"has_internet inconsistent vs InternetService : {mis_hi}/{n} "
          f"({mis_hi / n * 100:.1f}%)  [TARGET: 0]")
    print(f"is_new_customer inconsistent vs tenure<=12   : {mis_nc}/{n} "
          f"({mis_nc / n * 100:.1f}%)  [TARGET: 0]")

    all_zero = (mis_tg == 0 and mis_hi == 0 and mis_nc == 0)
    print("-" * 60)
    print(f"VERDICT: {'ALL CONSISTENT (0 violations)' if all_zero else 'INCONSISTENT — bug di re-derive'}")
    print("=" * 60)
    return all_zero


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load train; 2. drop customerID + engineered cols sebelum fit.
    #    SDV hanya melihat ~20 kolom dasar; engineered di-derive ulang nanti.
    train_df = pd.read_csv(TRAIN_CSV)
    fit_df = train_df.drop(columns=[ID_COL] + ENGINEERED_COLS)
    print(f"Train (fit) : {len(fit_df)} rows, {len(fit_df.columns)} base cols "
          f"(customerID + {len(ENGINEERED_COLS)} engineered cols di-drop sebelum fit)")

    # 3. Metadata + override
    metadata = build_metadata(fit_df)
    print("\n--- SDV Metadata (sdtype per kolom) ---")
    cols_meta = metadata.tables["table"].columns
    for col, spec in cols_meta.items():
        print(f"  {col:<18}: {spec.get('sdtype')}")

    # 4. Fit (seed sekali sebelum fit -> reproducible)
    np.random.seed(SEED)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(fit_df)
    print(f"\nSynthesizer fitted (seed={SEED}).")

    # 5. Generate 5 batch sekuensial (tanpa reset_sampling).
    #    Per batch: sample base -> re-derive engineered (add_features) -> add ID.
    batches = []
    for i in range(1, N_BATCHES + 1):
        synthetic_base = synthesizer.sample(num_rows=ROWS_PER_BATCH)

        # Re-derive engineered features deterministically (single source of truth).
        synthetic_full = add_features(synthetic_base)

        # Susun ulang kolom agar identik urutan train.csv:
        # [customerID] + base (tanpa engineered) + engineered.
        synthetic_full.insert(
            0, ID_COL, [f"SYNTH_B{i}_{j:03d}" for j in range(ROWS_PER_BATCH)]
        )

        out_path = OUT_DIR / f"batch_{i}.csv"
        synthetic_full.to_csv(out_path, index=False)
        batches.append(synthetic_full)
        print(f"  saved {out_path.relative_to(ROOT)} ({len(synthetic_full)} rows)")

    # 6. Save synthesizer + metadata (overwrite lama; SDV menolak file existing).
    synth_path = OUT_DIR / "synthesizer.pkl"
    meta_path = OUT_DIR / "metadata.json"
    synth_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    synthesizer.save(str(synth_path))
    metadata.save_to_json(str(meta_path))
    print(f"  saved {synth_path.relative_to(ROOT)} (overwrite lama)")
    print(f"  saved {meta_path.relative_to(ROOT)} (overwrite lama)")

    # Validation (gabungan 5 batch = 500 rows)
    synth_all = pd.concat(batches, ignore_index=True)
    quality_pass = validation_report(train_df, synth_all)
    consistent = consistency_check(synth_all)

    if not (quality_pass and consistent):
        print("\n[ACTION] Ada check yang FAIL — laporkan ke reviewer, jangan auto-retry.")


if __name__ == "__main__":
    main()
