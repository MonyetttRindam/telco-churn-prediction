"""Manual test for the Registry class.

Run:
    .venv/Scripts/python.exe api/core/test_registry.py

WARNING: this uses the REAL HF Hub registry and performs WRITE operations.
It snapshots the original registry at the start and force-restores it in a
``finally`` block, so it should never leave garbage behind — but only run it
against a repo where you can tolerate a few extra commits in the history.

SECURITY: token loaded from .env via python-dotenv; never printed.
"""

import copy
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from api.core.registry import (  # noqa: E402
    InvalidStateError,
    Registry,
    VersionNotFoundError,
)

TEST_VERSION = "v_test_dry_run"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    reg = Registry(hf_token=token, hf_username=username, repo_id=repo_id)

    print("=" * 60)
    print("REGISTRY MANUAL TEST")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    # Snapshot the original state for guaranteed restore.
    original = copy.deepcopy(reg.load(force_refresh=True))
    all_pass = True

    try:
        # --- Scenario 1: load + active version --------------------------------
        active = reg.get_active_version()
        print(f"[1] Active version: {active}")

        # --- Scenario 2: list batches ----------------------------------------
        batches = reg.list_batches()
        print(f"[2] Batches available: {len(batches)}")
        for b in batches:
            print(
                f"      - {b['id']}: n_rows={b.get('n_rows')} "
                f"churn={b.get('churn_rate')} source={b.get('source')} "
                f"used_in={b.get('used_in_versions')}"
            )

        # --- Scenario 3: add fake version (status rejected) -------------------
        # Ensure a clean slate if a previous aborted run left the test version.
        if reg._find_version(reg.load(force_refresh=True), TEST_VERSION, required=False):
            print(f"[3] {TEST_VERSION} already present; will be cleaned by restore")
        else:
            reg.add_version(
                version_id=TEST_VERSION,
                metrics={"f1": 0.0, "recall": 0.0, "precision": 0.0, "roc_auc": 0.0},
                batches_used=[],
                status="rejected",
                reason="dry-run test version (safe to delete)",
            )
        info = reg.get_version_info(TEST_VERSION)
        ok = info["status"] == "rejected"
        all_pass &= ok
        print(f"[3] add_version -> status={info['status']} {_fmt(ok)}")

        # --- Scenario 4: reject_version (idempotent) --------------------------
        reg.reject_version(TEST_VERSION)
        ok = reg.get_version_info(TEST_VERSION)["status"] == "rejected"
        all_pass &= ok
        print(f"[4] reject_version -> status=rejected {_fmt(ok)}")

        # --- Scenario 5: promote (rejected is promotable) then rollback -------
        before_active = reg.get_active_version()
        reg.promote_version(TEST_VERSION)
        promoted_ok = (
            reg.get_active_version() == TEST_VERSION
            and reg.get_previous_version() == before_active
        )
        all_pass &= promoted_ok
        print(
            f"[5a] promote -> active={reg.get_active_version()} "
            f"previous={reg.get_previous_version()} {_fmt(promoted_ok)}"
        )

        new_active = reg.rollback()
        rollback_ok = (
            new_active == before_active
            and reg.get_active_version() == before_active
            and reg.get_previous_version() == TEST_VERSION
        )
        all_pass &= rollback_ok
        print(
            f"[5b] rollback -> active={reg.get_active_version()} "
            f"previous={reg.get_previous_version()} {_fmt(rollback_ok)}"
        )

        # --- Negative checks: expected exceptions -----------------------------
        try:
            reg.get_version_info("v_does_not_exist")
            neg_ok = False
        except VersionNotFoundError:
            neg_ok = True
        all_pass &= neg_ok
        print(f"[6a] VersionNotFoundError raised for unknown id {_fmt(neg_ok)}")

        try:
            reg.add_version(
                TEST_VERSION, {}, [], "pending", "dup should fail"
            )
            neg_ok = False
        except InvalidStateError:
            neg_ok = True
        all_pass &= neg_ok
        print(f"[6b] InvalidStateError raised on duplicate version {_fmt(neg_ok)}")

    finally:
        # --- Cleanup: force-restore original registry -------------------------
        reg._cache = copy.deepcopy(original)
        reg._sync_to_hub("Test cleanup: restore original registry")
        reg._invalidate()
        restored = reg.load(force_refresh=True)
        clean = (
            reg._find_version(restored, TEST_VERSION, required=False) is None
            and restored.get("active") == original.get("active")
            and restored.get("previous") == original.get("previous")
        )
        print("-" * 60)
        print(f"[cleanup] original registry restored {_fmt(clean)}")
        all_pass &= clean

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
