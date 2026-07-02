"""One-time migration: add the ``batches`` field to registry.json (v1 -> v2).

The registry uploaded in Phase 1 only has ``active``, ``previous``, and
``versions``. Phase 2 introduces batch tracking. This script:

  1. Downloads the current ``registry.json`` from HF Hub.
  2. Adds ``batches.available`` populated from the 5 existing Phase-1 batches
     (``ml/data/synthetic/batch_1.csv`` .. ``batch_5.csv``), reading real stats
     (n_rows, churn_rate) from each CSV. ``source="sdv_phase1"``.
  3. Sets ``batches.next_batch_number = 6``.
  4. Uploads the updated registry back to HF Hub.
  5. Verifies by downloading again and diffing the ``batches`` field.

Idempotent-ish: if ``batches`` already exists it aborts (to avoid clobbering
real batch history). Re-run only after understanding the current state.

SECURITY: token loaded from .env via python-dotenv; never printed.

Run (venv aktif), HANYA setelah review:
    .venv/Scripts/python.exe scripts/migrate_registry_v2.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = ROOT / "ml/data/synthetic"
N_BATCHES = 5
REPO_TYPE = "model"
REGISTRY_FILENAME = "registry.json"


def batch_stats(batch_csv: Path) -> tuple[int, float, str]:
    """Return (n_rows, churn_rate, created_at) for a batch CSV.

    churn_rate = fraction of rows with Churn == "Yes", rounded to 3 decimals.
    created_at = file mtime as a UTC ISO-8601 string (proxy for creation time
    of the Phase-1 batches, which were not otherwise timestamped).
    """
    df = pd.read_csv(batch_csv)
    n_rows = int(len(df))
    churn_rate = round(float((df["Churn"] == "Yes").mean()), 3)
    mtime = batch_csv.stat().st_mtime
    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return n_rows, churn_rate, created_at


def build_batches_field() -> dict:
    """Build the ``batches`` field from the 5 Phase-1 synthetic batches."""
    available = []
    for i in range(1, N_BATCHES + 1):
        csv_path = SYNTHETIC_DIR / f"batch_{i}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing batch file: {csv_path}")
        n_rows, churn_rate, created_at = batch_stats(csv_path)
        available.append(
            {
                "id": f"batch_{i}",
                "created_at": created_at,  # file mtime (UTC ISO) as proxy
                "n_rows": n_rows,
                "churn_rate": churn_rate,
                "source": "sdv_phase1",
                "used_in_versions": [],
            }
        )
    return {"available": available, "next_batch_number": N_BATCHES + 1}


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    api = HfApi(token=token)

    print("=" * 60)
    print("REGISTRY MIGRATION v1 -> v2 (add batches field)")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    # 1. Download current registry
    local = hf_hub_download(
        repo_id, REGISTRY_FILENAME, repo_type=REPO_TYPE, token=token,
        force_download=True,
    )
    registry = json.loads(Path(local).read_text(encoding="utf-8"))

    if "batches" in registry:
        print("[ABORT] registry already has a 'batches' field. Nothing to do.")
        print("        (Refusing to overwrite existing batch history.)")
        return 1

    # 2 + 3. Build and attach batches field
    batches = build_batches_field()
    registry["batches"] = batches

    print("Batches to add:")
    for b in batches["available"]:
        print(
            f"  - {b['id']}: n_rows={b['n_rows']} churn_rate={b['churn_rate']} "
            f"source={b['source']} created_at={b['created_at']}"
        )
    print(f"  next_batch_number = {batches['next_batch_number']}")
    print("-" * 60)

    # 4. Upload back
    payload = json.dumps(registry, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=REGISTRY_FILENAME,
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message="Migrate registry v2: add batches field (batch_1..5)",
    )
    print("Uploaded updated registry.json")

    # 5. Verify: download again and diff the batches field
    verify_local = hf_hub_download(
        repo_id, REGISTRY_FILENAME, repo_type=REPO_TYPE, token=token,
        force_download=True,
    )
    remote = json.loads(Path(verify_local).read_text(encoding="utf-8"))
    match = remote.get("batches") == batches
    print("-" * 60)
    print(f"Verify batches field round-trip: [{'MATCH' if match else 'MISMATCH'}]")
    print(f"VERDICT: {'SUCCESS' if match else 'FAILED'}")
    print("=" * 60)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
