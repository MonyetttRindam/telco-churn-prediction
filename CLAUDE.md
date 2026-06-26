## MLOps Extension (In Progress)
Currently implementing retraining system. See MLOPS_PLAN.md for full plan.
Current phase: Phase 1 - Foundation

## Development Environment
- Use the project venv at `.venv` (Python 3.12). It has the pinned deps
  (scikit-learn 1.5.2 — must match the version used to save `models/*.joblib`).
- **Activate `.venv` before running any script:**
  - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
  - Windows (cmd): `.venv\Scripts\activate.bat`
  - Git Bash: `source .venv/Scripts/activate`
- Or call directly without activating: `.venv/Scripts/python.exe <script>.py`
- Setup from scratch: `py -3.12 -m venv .venv` then
  `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt`
  (`requirements-dev.txt` includes runtime `requirements.txt` via `-r`).
- Note: do NOT use Python 3.14 — pinned deps (pandas 2.2.3, sklearn 1.5.2,
  numpy 1.26.4) have no wheels for it.