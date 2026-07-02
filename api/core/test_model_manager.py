"""Manual smoke test for the ModelManager class.

Run:
    .venv/Scripts/python.exe api/core/test_model_manager.py

Requires ``models/v_initial/{model,preprocessor}.pkl`` on HF Hub — run
``scripts/migrate_hf_structure_v2.py`` first.

This test is READ-ONLY with respect to HF Hub (ModelManager only downloads;
it never uploads), so there is no remote state to clean up. Dummy inference
input is taken from a single row of ``ml/data/test_holdout.csv``.

SECURITY: token loaded from .env via python-dotenv; never printed.
"""

import sys
import threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from api.core.model_manager import (  # noqa: E402
    ModelDownloadError,
    ModelManager,
    ModelNotLoadedError,
)
from api.core.registry import Registry  # noqa: E402

HOLDOUT = ROOT / "ml/data/test_holdout.csv"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _dummy_row() -> pd.DataFrame:
    """One holdout row minus the target, matching the preprocessor schema."""
    df = pd.read_csv(HOLDOUT).head(1)
    return df.drop(columns=["Churn"])


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    registry = Registry(hf_token=token, hf_username=username, repo_id=repo_id)
    mgr = ModelManager(
        hf_token=token, hf_username=username, repo_id=repo_id, registry=registry
    )

    print("=" * 60)
    print("MODEL MANAGER SMOKE TEST")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    all_pass = True

    # --- Pre-check: get_current before load raises ---------------------------
    try:
        mgr.get_current()
        pre_ok = False
    except ModelNotLoadedError:
        pre_ok = True
    all_pass &= pre_ok
    print(f"[0] get_current() before load -> ModelNotLoadedError {_fmt(pre_ok)}")

    # --- Scenario 2: load_active ---------------------------------------------
    mgr.load_active()
    load_ok = mgr.get_current_version_id() == "v_initial"
    all_pass &= load_ok
    print(f"[2] load_active -> version_id={mgr.get_current_version_id()} {_fmt(load_ok)}")

    # --- Scenario 3: get_current + predict_proba on dummy row ----------------
    model, preprocessor, version_id = mgr.get_current()
    X = preprocessor.transform(_dummy_row())
    proba = model.predict_proba(X)
    infer_ok = (
        model is not None
        and preprocessor is not None
        and version_id == "v_initial"
        and proba.shape == (1, 2)
    )
    all_pass &= infer_ok
    print(
        f"[3] get_current + predict_proba -> shape={proba.shape} "
        f"p(churn)={proba[0][1]:.4f} {_fmt(infer_ok)}"
    )

    # --- Scenario 4: self-swap (safe) ----------------------------------------
    mgr.swap_to("v_initial")
    swap_ok = mgr.get_current_version_id() == "v_initial"
    all_pass &= swap_ok
    print(f"[4] swap_to('v_initial') -> version_id={mgr.get_current_version_id()} {_fmt(swap_ok)}")

    # --- Scenario 5: swap to unknown version -> ModelDownloadError -----------
    try:
        mgr.swap_to("v_does_not_exist")
        neg_ok = False
    except ModelDownloadError:
        neg_ok = True
    # After a failed swap, the current model must be untouched.
    neg_ok = neg_ok and mgr.get_current_version_id() == "v_initial"
    all_pass &= neg_ok
    print(f"[5] swap_to('v_does_not_exist') -> ModelDownloadError, current intact {_fmt(neg_ok)}")

    # --- Scenario 6: concurrency ---------------------------------------------
    errors: list[Exception] = []
    got_none = threading.Event()
    stop = threading.Event()

    def reader() -> None:
        try:
            for _ in range(100):
                m, p, v = mgr.get_current()
                if m is None or p is None or v is None:
                    got_none.set()
        except Exception as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    def swapper() -> None:
        try:
            while not stop.is_set():
                mgr.swap_to("v_initial")
                break  # single swap is enough to interleave
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    threads.append(threading.Thread(target=swapper))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()

    conc_ok = not errors and not got_none.is_set()
    all_pass &= conc_ok
    print(
        f"[6] concurrency (3 readers x100 + 1 swapper) -> "
        f"errors={len(errors)} none_seen={got_none.is_set()} {_fmt(conc_ok)}"
    )

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
