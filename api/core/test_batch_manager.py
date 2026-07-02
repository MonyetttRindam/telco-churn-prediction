"""Manual smoke test for the BatchManager class.

Run:
    .venv/Scripts/python.exe api/core/test_batch_manager.py

READ-ONLY with respect to HF Hub: it lists/downloads batches and validates
in-memory DataFrames, but NEVER calls upload_batch (so HF Hub stays clean).
Bad/good fixtures are derived locally from ml/data/synthetic/batch_2.csv and
ml/data/train.csv.

SECURITY: token loaded from .env via python-dotenv; never printed.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from api.core.batch_manager import (  # noqa: E402
    BatchDownloadError,
    BatchManager,
)
from api.core.registry import Registry  # noqa: E402

TRAIN = ROOT / "ml/data/train.csv"
BATCH2 = ROOT / "ml/data/synthetic/batch_2.csv"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    registry = Registry(hf_token=token, hf_username=username, repo_id=repo_id)
    bm = BatchManager(
        hf_token=token,
        hf_username=username,
        repo_id=repo_id,
        registry=registry,
        train_reference_path=TRAIN,
    )

    print("=" * 60)
    print("BATCH MANAGER SMOKE TEST")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    all_pass = True

    # --- Scenario 1: vocabulary extracted ------------------------------------
    vocab = bm.vocabulary
    n_cols = len(vocab["column_names"])
    n_cat = len(vocab["categorical_vocab"])
    # NOTE: spec said "4 categorical vocabs" but the real preprocessor has 16
    # categorical columns (split_columns). We assert the real count.
    vocab_ok = n_cols == 25 and n_cat == 16
    all_pass &= vocab_ok
    print(
        f"[1] vocabulary -> cols={n_cols} categorical_vocabs={n_cat} "
        f"churn_ref={vocab['churn_rate_reference']:.4f} {_fmt(vocab_ok)}"
    )

    # --- Scenario 2: list_batches --------------------------------------------
    batches = bm.list_batches()
    list_ok = len(batches) == 5
    all_pass &= list_ok
    print(f"[2] list_batches -> {len(batches)} batches {_fmt(list_ok)}")

    # --- Scenario 3: download_batch(batch_1) ---------------------------------
    b1 = bm.download_batch("batch_1")
    dl_ok = len(b1) == 100
    all_pass &= dl_ok
    print(f"[3] download_batch('batch_1') -> {len(b1)} rows {_fmt(dl_ok)}")

    # --- Scenario 4: download_batch(batch_99) -> error -----------------------
    try:
        bm.download_batch("batch_99")
        dl_neg = False
    except BatchDownloadError:
        dl_neg = True
    all_pass &= dl_neg
    print(f"[4] download_batch('batch_99') -> BatchDownloadError {_fmt(dl_neg)}")

    # --- Build local fixtures from batch_2 + train ---------------------------
    base = pd.read_csv(BATCH2)  # 100 rows, valid Phase-1 batch
    train = pd.read_csv(TRAIN)

    # good: perturb MonthlyCharges by +0.01 so rows are NOT exact duplicates of
    # batch_2 (which already lives on HF); still valid schema/vocab/churn.
    good = base.copy()
    good["MonthlyCharges"] = good["MonthlyCharges"] + 0.01

    # bad_schema: drop a required column.
    bad_schema = base.drop(columns=["Contract"])

    # bad_churn: force ~70% churn (out of allowed 11.54%-41.54%).
    bad_churn = base.copy()
    k = int(len(bad_churn) * 0.7)
    bad_churn["Churn"] = ["Yes"] * k + ["No"] * (len(bad_churn) - k)

    # has_duplicates: overwrite 5 rows of `good` with real train rows (~5% dup).
    has_dup = good.copy()
    train_rows = train.sample(5, random_state=42)
    has_dup.iloc[:5] = train_rows[has_dup.columns].values

    # new_category: unknown PaymentMethod value.
    new_cat = good.copy()
    new_cat.loc[new_cat.index[0], "PaymentMethod"] = "Cash"

    # --- Scenario 5: good batch -> PASS --------------------------------------
    r5 = bm.validate_batch(good)
    s5 = r5.passed and not r5.errors
    all_pass &= s5
    print(
        f"[5] validate(good) -> passed={r5.passed} errors={len(r5.errors)} "
        f"warnings={len(r5.warnings)} dup={r5.duplicate_info} {_fmt(s5)}"
    )

    # --- Scenario 6: bad schema -> FAIL --------------------------------------
    r6 = bm.validate_batch(bad_schema)
    s6 = (not r6.passed) and not r6.rule_results["schema"]["passed"]
    all_pass &= s6
    print(f"[6] validate(bad_schema) -> passed={r6.passed} schema_detail=\"{r6.rule_results['schema']['detail']}\" {_fmt(s6)}")

    # --- Scenario 7: bad churn -> FAIL ---------------------------------------
    r7 = bm.validate_batch(bad_churn)
    s7 = (not r7.passed) and not r7.rule_results["churn_rate"]["passed"]
    all_pass &= s7
    print(f"[7] validate(bad_churn) -> passed={r7.passed} churn_detail=\"{r7.rule_results['churn_rate']['detail']}\" {_fmt(s7)}")

    # --- Scenario 8: duplicates -> PASS with warning -------------------------
    r8 = bm.validate_batch(has_dup)
    s8 = r8.passed and len(r8.warnings) >= 1 and r8.duplicate_info["count"] >= 5
    all_pass &= s8
    print(
        f"[8] validate(has_duplicates) -> passed={r8.passed} "
        f"warnings={len(r8.warnings)} dup={r8.duplicate_info} {_fmt(s8)}"
    )

    # --- Scenario 9: new category -> FAIL ------------------------------------
    r9 = bm.validate_batch(new_cat)
    s9 = (not r9.passed) and not r9.rule_results["vocabulary"]["passed"]
    all_pass &= s9
    print(f"[9] validate(new_category) -> passed={r9.passed} vocab_detail=\"{r9.rule_results['vocabulary']['detail']}\" {_fmt(s9)}")

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
