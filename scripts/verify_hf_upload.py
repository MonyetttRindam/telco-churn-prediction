"""Verify HF Hub upload: cek file integrity + content match (Phase 1, Step 1.7).

Dijalankan SETELAH scripts/upload_initial_model.py. Karena staging lokal sudah
dihapus, content di-compare terhadap builder kanonik (build_metrics/build_registry)
dengan timestamp dari file yang di-download dinetralkan -> hanya field non-waktu
yang dibandingkan.

SECURITY: token hanya dari .env, tidak pernah di-print.

Jalankan dari root project (venv aktif):
    .venv/Scripts/python.exe scripts/verify_hf_upload.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from upload_initial_model import build_metrics, build_registry  # noqa: E402

EXPECTED_FILES = [
    "current/model.pkl",
    "current/preprocessor.pkl",
    "current/metrics.json",
    "synthetic/batch_1.csv",
    "synthetic/batch_2.csv",
    "synthetic/batch_3.csv",
    "synthetic/batch_4.csv",
    "synthetic/batch_5.csv",
    "registry.json",
    "README.md",
]


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"
    url = f"https://huggingface.co/{repo_id}"
    api = HfApi(token=token)

    # 1. List repo files
    repo_files = set(api.list_repo_files(repo_id, repo_type="model"))
    present = [f for f in EXPECTED_FILES if f in repo_files]
    files_pass = len(present) == len(EXPECTED_FILES)
    missing = [f for f in EXPECTED_FILES if f not in repo_files]

    # 2. Download metrics.json + registry.json, parse
    def _download_json(path: str) -> dict:
        local = hf_hub_download(repo_id, path, repo_type="model", token=token)
        return json.loads(Path(local).read_text(encoding="utf-8"))

    metrics = _download_json("current/metrics.json")
    registry = _download_json("registry.json")

    # 3. Compare terhadap builder kanonik (timestamp dinetralkan)
    expected_metrics = build_metrics(metrics.get("created_at", ""))
    metrics_match = metrics == expected_metrics

    reg_created = registry.get("versions", [{}])[0].get("created_at", "")
    expected_registry = build_registry(expected_metrics, reg_created)
    registry_match = registry == expected_registry

    # 4. Report
    print("=" * 60)
    print("HF UPLOAD VERIFICATION")
    print("=" * 60)
    print(f"Repo: {repo_id}")
    print(f"URL:  {url}")
    print("-" * 60)
    print(f"Files present: {len(present)}/{len(EXPECTED_FILES)}  "
          f"[{'PASS' if files_pass else 'FAIL'}]")
    if missing:
        print(f"  Missing: {missing}")
    print(f"metrics.json content:  [{'MATCH' if metrics_match else 'MISMATCH'} local]")
    print(f"registry.json content: [{'MATCH' if registry_match else 'MISMATCH'} local]")
    print("-" * 60)

    ok = files_pass and metrics_match and registry_match
    print(f"VERDICT: {'SUCCESS' if ok else 'FAILED'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
