"""One-time migration: create ``models/v_initial/`` on HF Hub (structure v2).

Phase 1 uploaded the initial model under ``current/`` (model.pkl,
preprocessor.pkl, metrics.json). Phase 2 uses a per-version layout::

    models/{version_id}/model.pkl
    models/{version_id}/preprocessor.pkl

so :class:`ModelManager` can load any version by id. This script:

  1. Downloads ``current/model.pkl`` + ``current/preprocessor.pkl`` from HF Hub.
  2. Re-uploads them to ``models/v_initial/model.pkl`` +
     ``models/v_initial/preprocessor.pkl``.
  3. Verifies both new paths exist in the repo.
  4. Leaves ``current/`` untouched (legacy; not deleted, so any old reference
     keeps working).

Idempotent-ish: if ``models/v_initial/*`` already exists it still re-uploads
(HF skips the commit if bytes are identical) and just re-verifies.

SECURITY: token loaded from .env via python-dotenv; never printed.

Run (venv aktif), HANYA setelah review:
    .venv/Scripts/python.exe scripts/migrate_hf_structure_v2.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO_TYPE = "model"
VERSION_ID = "v_initial"

# (source path in repo, destination path in repo)
FILES = [
    ("current/model.pkl", f"models/{VERSION_ID}/model.pkl"),
    ("current/preprocessor.pkl", f"models/{VERSION_ID}/preprocessor.pkl"),
]


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    api = HfApi(token=token)

    print("=" * 60)
    print("HF STRUCTURE MIGRATION v2 (current/ -> models/v_initial/)")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print("-" * 60)

    # 1 + 2. Download each source, re-upload to destination.
    for src, dst in FILES:
        local = hf_hub_download(
            repo_id, src, repo_type=REPO_TYPE, token=token, force_download=True
        )
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dst,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"Structure v2: copy {src} -> {dst}",
        )
        print(f"  {src}  ->  {dst}  [uploaded]")

    print("-" * 60)

    # 3. Verify both destinations now exist.
    repo_files = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    expected = [dst for _, dst in FILES]
    present = [p for p in expected if p in repo_files]
    ok = len(present) == len(expected)
    missing = [p for p in expected if p not in repo_files]

    print(f"Destinations present: {len(present)}/{len(expected)} "
          f"[{'PASS' if ok else 'FAIL'}]")
    for p in expected:
        print(f"  - {p}: {'OK' if p in repo_files else 'MISSING'}")
    if missing:
        print(f"  Missing: {missing}")
    print("-" * 60)
    print(f"VERDICT: {'SUCCESS' if ok else 'FAILED'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
