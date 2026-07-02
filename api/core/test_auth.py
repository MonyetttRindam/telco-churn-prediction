"""Manual test for API key authentication.

Run:
    .venv/Scripts/python.exe api/core/test_auth.py

Self-contained: it injects a temporary MLOPS_API_KEY into the process
environment for the duration of the test and restores the original value in a
``finally`` block. It does NOT read or depend on the real key in .env, and does
NOT touch HF Hub. Uses FastAPI's TestClient (needs httpx installed).
"""

import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.core.auth import API_KEY_HEADER_NAME, require_api_key  # noqa: E402

TEST_KEY = "unit-test-key-do-not-use-in-prod"


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def main() -> int:
    print("=" * 60)
    print("API KEY AUTH TEST")
    print("=" * 60)

    all_pass = True
    original = os.environ.get("MLOPS_API_KEY")
    os.environ["MLOPS_API_KEY"] = TEST_KEY

    try:
        app = FastAPI()

        @app.get("/protected")
        def protected(_: str = Depends(require_api_key)):
            return {"ok": True}

        client = TestClient(app)

        # --- [1] No header -> 401 --------------------------------------------
        r = client.get("/protected")
        s1 = r.status_code == 401
        all_pass &= s1
        print(f"[1] no header -> {r.status_code} (expect 401) {_fmt(s1)}")

        # --- [2] Wrong key -> 403 --------------------------------------------
        r = client.get("/protected", headers={API_KEY_HEADER_NAME: "wrong-key"})
        s2 = r.status_code == 403
        all_pass &= s2
        print(f"[2] wrong key -> {r.status_code} (expect 403) {_fmt(s2)}")

        # --- [3] Correct key -> 200 ------------------------------------------
        r = client.get("/protected", headers={API_KEY_HEADER_NAME: TEST_KEY})
        s3 = r.status_code == 200 and r.json() == {"ok": True}
        all_pass &= s3
        print(f"[3] correct key -> {r.status_code} body={r.json()} (expect 200) {_fmt(s3)}")

        # --- [4] Server misconfig (no MLOPS_API_KEY) -> 500 ------------------
        os.environ.pop("MLOPS_API_KEY", None)
        r = client.get("/protected", headers={API_KEY_HEADER_NAME: TEST_KEY})
        s4 = r.status_code == 500
        all_pass &= s4
        print(f"[4] server misconfig -> {r.status_code} (expect 500) {_fmt(s4)}")
        os.environ["MLOPS_API_KEY"] = TEST_KEY  # restore for tidiness

    finally:
        # Restore the original env value (or remove it if it wasn't set).
        if original is None:
            os.environ.pop("MLOPS_API_KEY", None)
        else:
            os.environ["MLOPS_API_KEY"] = original

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
