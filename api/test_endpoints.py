"""Integration test for all Phase 2 FastAPI endpoints.

Run:
    .venv/Scripts/python.exe api/test_endpoints.py

Uses FastAPI's TestClient against the REAL Registry / ModelManager /
BatchManager / RetrainingPipeline (so it hits HF Hub). The API key comes from
the real MLOPS_API_KEY loaded by init_dependencies() from .env.

WARNING: the /retrain and /upload-batch scenarios MUTATE HF Hub (new version +
new batch). The ``finally`` block ALWAYS restores the original registry and
deletes any artifacts/batches created, so the repo is left as found.

SECURITY: token/key loaded from .env; never printed.
"""

import copy
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub import HfApi

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.deps import (  # noqa: E402
    get_model_manager,
    get_registry,
    init_dependencies,
)
from api.routes import (  # noqa: E402
    batches_router,
    jobs_router,
    mlops_router,
    retrain_router,
)

BATCH2 = ROOT / "ml/data/synthetic/batch_2.csv"
HDR = "X-API-Key"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(mlops_router)
    app.include_router(retrain_router)
    app.include_router(batches_router)
    app.include_router(jobs_router)
    return app


def _good_csv_bytes() -> bytes:
    """Valid 'new' batch: batch_2 with MonthlyCharges nudged (avoids self-dup)."""
    df = pd.read_csv(BATCH2)
    df["MonthlyCharges"] = df["MonthlyCharges"] + 0.01
    return df.to_csv(index=False).encode("utf-8")


def _poll_job(client: TestClient, key: str, job_id: str, timeout: float = 120.0):
    """Poll GET /api/jobs/{job_id} until terminal or timeout. Return final JSON."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        last = r.json()
        if last.get("status") in ("completed", "failed"):
            return last
        time.sleep(1.0)
    return last


def main() -> int:
    init_dependencies(ROOT)
    api_key = os.getenv("MLOPS_API_KEY")
    if not api_key:
        raise RuntimeError("MLOPS_API_KEY harus di-set di .env untuk test ini")

    registry = get_registry()
    model_manager = get_model_manager()
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    repo_id = f"{username}/telco-churn-models"
    hf = HfApi(token=token)

    app = _build_app()
    client = TestClient(app)

    print("=" * 60)
    print("ENDPOINTS INTEGRATION TEST")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    original = copy.deepcopy(registry.load(force_refresh=True))
    all_pass = True
    created_version_id = None
    uploaded_batch_id = None

    try:
        # --- [1] GET /status --------------------------------------------------
        r = client.get("/api/status")
        j = r.json()
        s1 = r.status_code == 200 and j["active_version"] == original["active"] and "f1" in j["metrics"]
        all_pass &= s1
        print(f"[1] GET /status -> {r.status_code} active={j.get('active_version')} {_fmt(s1)}")

        # --- [2] GET /history -------------------------------------------------
        r = client.get("/api/history")
        j = r.json()
        s2 = r.status_code == 200 and j["total_count"] >= 1 and any(
            v["id"] == original["active"] for v in j["versions"]
        )
        all_pass &= s2
        print(f"[2] GET /history -> {r.status_code} total={j.get('total_count')} {_fmt(s2)}")

        # --- [3] GET /batches -------------------------------------------------
        r = client.get("/api/batches")
        j = r.json()
        s3 = r.status_code == 200 and j["total_count"] >= 5
        all_pass &= s3
        print(f"[3] GET /batches -> {r.status_code} total={j.get('total_count')} unused={j.get('unused_count')} {_fmt(s3)}")

        # --- [4] GET /verify-key (401 / 403 / 200) ---------------------------
        r_none = client.get("/api/verify-key")
        r_wrong = client.get("/api/verify-key", headers={HDR: "wrong"})
        r_ok = client.get("/api/verify-key", headers={HDR: api_key})
        s4 = (
            r_none.status_code == 401
            and r_wrong.status_code == 403
            and r_ok.status_code == 200
            and r_ok.json()["valid"] is True
        )
        all_pass &= s4
        print(f"[4] verify-key -> none={r_none.status_code} wrong={r_wrong.status_code} ok={r_ok.status_code} {_fmt(s4)}")

        # --- [5] POST /rollback with no previous -> 400 ----------------------
        # Assumes a clean start (original.previous is None).
        r = client.post("/api/rollback", headers={HDR: api_key})
        expect_rollback = 400 if original.get("previous") is None else 200
        s5 = r.status_code == expect_rollback
        all_pass &= s5
        print(f"[5] POST /rollback -> {r.status_code} (expect {expect_rollback}) {_fmt(s5)}")

        # --- [6] POST /retrain invalid batch -> 404 --------------------------
        r = client.post("/api/retrain", headers={HDR: api_key}, json={"batch_id": "batch_999"})
        s6 = r.status_code == 404
        all_pass &= s6
        print(f"[6] POST /retrain (batch_999) -> {r.status_code} (expect 404) {_fmt(s6)}")

        # --- [7] POST /retrain valid batch -> 200 + job_id -------------------
        r = client.post("/api/retrain", headers={HDR: api_key}, json={"batch_id": "batch_1"})
        j = r.json()
        job_id = j.get("job_id")
        s7 = r.status_code == 200 and bool(job_id) and j["status"] == "pending"
        all_pass &= s7
        print(f"[7] POST /retrain (batch_1) -> {r.status_code} job_id={job_id} {_fmt(s7)}")

        # --- [8] Poll job until completed ------------------------------------
        final = _poll_job(client, api_key, job_id) if job_id else None
        s8 = bool(final) and final["status"] == "completed" and final["result"] is not None
        if s8:
            created_version_id = final["result"]["new_version_id"]
        all_pass &= s8
        decision = final["result"]["decision"] if s8 else "?"
        print(f"[8] poll job -> status={final.get('status') if final else None} decision={decision} version={created_version_id} {_fmt(s8)}")

        # --- [9] GET /jobs ----------------------------------------------------
        r = client.get("/api/jobs", params={"job_type": "retrain"})
        j = r.json()
        s9 = r.status_code == 200 and any(job.get("job_id") == job_id for job in j)
        all_pass &= s9
        print(f"[9] GET /jobs (type=retrain) -> {r.status_code} count={len(j)} {_fmt(s9)}")

        # --- [10] POST /upload-batch bad CSV -> 400 --------------------------
        r = client.post(
            "/api/upload-batch",
            headers={HDR: api_key},
            files={"file": ("bad.csv", b"", "text/csv")},
        )
        s10 = r.status_code == 400
        all_pass &= s10
        print(f"[10] POST /upload-batch (empty) -> {r.status_code} (expect 400) {_fmt(s10)}")

        # --- [11] POST /upload-batch good CSV -> 200 + batch_id --------------
        r = client.post(
            "/api/upload-batch",
            headers={HDR: api_key},
            files={"file": ("good.csv", _good_csv_bytes(), "text/csv")},
        )
        j = r.json()
        s11 = r.status_code == 200 and j["validation_passed"] is True and bool(j["batch_id"])
        if s11:
            uploaded_batch_id = j["batch_id"]
        all_pass &= s11
        print(f"[11] POST /upload-batch (good) -> {r.status_code} batch_id={uploaded_batch_id} passed={j.get('validation_passed')} {_fmt(s11)}")

    finally:
        # --- Cleanup ----------------------------------------------------------
        print("-" * 60)
        registry._cache = copy.deepcopy(original)
        registry._sync_to_hub("Test cleanup: restore original registry")
        registry._invalidate()

        if created_version_id:
            for fname in ("model.pkl", "preprocessor.pkl"):
                try:
                    hf.delete_file(
                        path_in_repo=f"models/{created_version_id}/{fname}",
                        repo_id=repo_id, repo_type="model",
                        commit_message=f"Test cleanup: delete models/{created_version_id}/{fname}",
                    )
                except Exception:  # noqa: BLE001
                    pass

        if uploaded_batch_id:
            try:
                hf.delete_file(
                    path_in_repo=f"synthetic/{uploaded_batch_id}.csv",
                    repo_id=repo_id, repo_type="model",
                    commit_message=f"Test cleanup: delete synthetic/{uploaded_batch_id}.csv",
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            model_manager.load_active()
        except Exception:  # noqa: BLE001
            pass

        restored = registry.load(force_refresh=True)
        clean = (
            restored.get("active") == original.get("active")
            and restored.get("previous") == original.get("previous")
            and (created_version_id is None or not any(
                v["id"] == created_version_id for v in restored["versions"]))
        )
        print(f"[cleanup] registry restored + artifacts deleted {_fmt(clean)}")
        all_pass &= clean

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
