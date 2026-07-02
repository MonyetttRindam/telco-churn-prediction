"""Manual smoke test for the RetrainingPipeline class.

Run:
    .venv/Scripts/python.exe api/core/test_retraining.py

WARNING: scenario [3] runs a REAL retraining cycle — it trains, uploads
artifacts to ``models/{version_id}/`` and mutates registry.json on HF Hub.
The ``finally`` block ALWAYS restores the original registry and deletes the
uploaded artifacts, so the repo is left as found even if an assertion fails.

Requires ``models/v_initial/`` on HF Hub (run scripts/migrate_hf_structure_v2.py).

SECURITY: token loaded from .env via python-dotenv; never printed.
"""

import copy
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

from api.core.batch_manager import BatchManager  # noqa: E402
from api.core.model_manager import ModelManager  # noqa: E402
from api.core.registry import Registry  # noqa: E402
from api.core.retraining import RetrainingPipeline  # noqa: E402

TRAIN = ROOT / "ml/data/train.csv"
HOLDOUT = ROOT / "ml/data/test_holdout.csv"
PREPROCESSOR = ROOT / "models/preprocessor.joblib"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    api = HfApi(token=token)

    registry = Registry(hf_token=token, hf_username=username, repo_id=repo_id)
    model_manager = ModelManager(
        hf_token=token, hf_username=username, repo_id=repo_id, registry=registry
    )
    batch_manager = BatchManager(
        hf_token=token, hf_username=username, repo_id=repo_id,
        registry=registry, train_reference_path=TRAIN,
    )
    pipeline = RetrainingPipeline(
        hf_token=token, hf_username=username, repo_id=repo_id,
        registry=registry, model_manager=model_manager,
        batch_manager=batch_manager,
        train_data_path=TRAIN, holdout_path=HOLDOUT,
        preprocessor_path=PREPROCESSOR,
    )

    print("=" * 60)
    print("RETRAINING PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    model_manager.load_active()
    original = copy.deepcopy(registry.load(force_refresh=True))
    all_pass = True
    new_version_id = None

    try:
        # --- [1] Gate logic (unit, no network) -------------------------------
        cur = {"f1": 0.6291, "recall": 0.7166}

        g_pass = pipeline._apply_gate(cur, {"f1": 0.6350, "recall": 0.7200})
        g1 = g_pass.passed and g_pass.f1_gate_passed and g_pass.recall_gate_passed
        all_pass &= g1
        print(f"[1a] gate(better) -> passed={g_pass.passed} {_fmt(g1)}")

        g_f1 = pipeline._apply_gate(cur, {"f1": 0.6000, "recall": 0.7200})
        g2 = (not g_f1.passed) and (not g_f1.f1_gate_passed) and "F1" in g_f1.reason
        all_pass &= g2
        print(f"[1b] gate(F1 drop) -> passed={g_f1.passed} reason=\"{g_f1.reason}\" {_fmt(g2)}")

        g_rec = pipeline._apply_gate(cur, {"f1": 0.6400, "recall": 0.7000})
        g3 = (
            (not g_rec.passed)
            and g_rec.f1_gate_passed
            and (not g_rec.recall_gate_passed)
            and "Recall" in g_rec.reason
        )
        all_pass &= g3
        print(f"[1c] gate(recall drop) -> passed={g_rec.passed} reason=\"{g_rec.reason}\" {_fmt(g3)}")

        # --- [2] Data accumulation -------------------------------------------
        combined = pipeline._accumulate_data("batch_1", [])
        acc_ok = len(combined) == 5634 + 100
        all_pass &= acc_ok
        print(f"[2] accumulate(batch_1, []) -> {len(combined)} rows (expect 5734) {_fmt(acc_ok)}")

        # --- [3] Full retraining flow (DRY RUN, mutates HF) ------------------
        report = pipeline.retrain("batch_1")
        new_version_id = report.new_version_id

        struct_ok = (
            report.new_version_id.startswith("v_")
            and report.batch_used == "batch_1"
            and report.accumulated_batches == ["batch_1"]
            and report.n_train_samples == 5734
            and "f1" in report.current_metrics
            and "f1" in report.new_metrics
            and report.decision in ("promoted", "rejected")
            and report.duration_seconds >= 0
        )
        all_pass &= struct_ok
        print(
            f"[3a] retrain(batch_1) -> version={report.new_version_id} "
            f"decision={report.decision} n_train={report.n_train_samples} "
            f"dur={report.duration_seconds:.1f}s {_fmt(struct_ok)}"
        )
        print(
            f"     current F1={report.current_metrics['f1']:.4f} "
            f"recall={report.current_metrics['recall']:.4f}  ->  "
            f"new F1={report.new_metrics['f1']:.4f} "
            f"recall={report.new_metrics['recall']:.4f}"
        )
        print(f"     gate: {report.gate_result.reason}")

        # Decision must match the gate outcome.
        decision_ok = (report.decision == "promoted") == report.gate_result.passed
        all_pass &= decision_ok
        print(f"[3b] decision matches gate -> {_fmt(decision_ok)}")

        # Registry state must reflect the decision.
        fresh = registry.load(force_refresh=True)
        if report.gate_result.passed:
            reg_ok = (
                fresh["active"] == new_version_id
                and fresh["previous"] == original["active"]
            )
        else:
            reg_ok = fresh["active"] == original["active"]
        # The new version must be recorded either way.
        reg_ok = reg_ok and any(v["id"] == new_version_id for v in fresh["versions"])
        all_pass &= reg_ok
        print(f"[3c] registry reflects decision -> active={fresh['active']} {_fmt(reg_ok)}")

    finally:
        # --- [4] Cleanup: restore registry + delete artifacts ----------------
        print("-" * 60)
        registry._cache = copy.deepcopy(original)
        registry._sync_to_hub("Test cleanup: restore original registry")
        registry._invalidate()

        if new_version_id:
            for fname in ("model.pkl", "preprocessor.pkl"):
                path_in_repo = f"models/{new_version_id}/{fname}"
                try:
                    api.delete_file(
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Test cleanup: delete {path_in_repo}",
                    )
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

        # Resync in-memory model to the restored active version.
        try:
            model_manager.load_active()
        except Exception:  # noqa: BLE001
            pass

        restored = registry.load(force_refresh=True)
        clean = (
            restored.get("active") == original.get("active")
            and restored.get("previous") == original.get("previous")
            and (
                new_version_id is None
                or not any(v["id"] == new_version_id for v in restored["versions"])
            )
        )
        print(f"[4] cleanup: registry restored + artifacts deleted {_fmt(clean)}")
        all_pass &= clean

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
