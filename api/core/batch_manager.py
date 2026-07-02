"""Synthetic batch manager for the MLOps retraining system.

Single source of truth for all batch access on HF Hub:

* **List** available batches (delegated to :class:`Registry`).
* **Download** batch content as a DataFrame.
* **Validate** an uploaded batch against 7 strict rules (Keputusan 5).
* **Upload** a new batch (called from the ``/upload-batch`` endpoint).

Validation reference (column names, dtypes, categorical vocabulary, numeric
stats, churn-rate reference) is extracted once from ``ml/data/train.csv`` at
construction and cached in memory.

HF Hub path convention::

    synthetic/{batch_id}.csv

SECURITY: the HF token lives in memory only; never printed or logged.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

# Project root on sys.path so ``src`` is importable when run standalone.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocessing import TARGET, split_columns  # noqa: E402

from api.core.registry import Registry  # noqa: E402

REPO_TYPE = "model"

# Validation constants (Keputusan 5).
EXPECTED_N_COLS = 25
MIN_ROWS = 50
MAX_ROWS = 500
CHURN_REF = 0.2654
CHURN_TOLERANCE = 0.15  # absolute (percentage points): 11.54% - 41.54%
DRIFT_THRESHOLD = 0.20  # relative mean drift per numeric column
DRIFT_MIN_COLS = 3  # number of drifted columns needed to warn

# Duplicate-percentage bands -> action.
DUP_ACCEPT = 0.01
DUP_WARN = 0.10
DUP_CONFIRM = 0.50


# === Exceptions ===============================================================


class BatchManagerError(Exception):
    """Base class for all batch-manager errors."""


class BatchNotFoundError(BatchManagerError):
    """Raised when a batch_id is not present in the registry."""


class BatchDownloadError(BatchManagerError):
    """Raised when downloading/parsing a batch CSV from HF Hub fails."""


class BatchValidationError(BatchManagerError):
    """Raised for unexpected failures during validation (not rule failures)."""


# === ValidationResult =========================================================


@dataclass
class ValidationResult:
    """Outcome of :meth:`BatchManager.validate_batch`."""

    passed: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    duplicate_info: dict = field(default_factory=dict)
    rule_results: dict = field(default_factory=dict)


# === BatchManager =============================================================


class BatchManager:
    """Manages synthetic batches on HF Hub (list / download / validate / upload)."""

    def __init__(
        self,
        hf_token: str,
        hf_username: str,
        repo_id: str,
        registry: Registry,
        train_reference_path: Path,
    ) -> None:
        """Initialize the manager and extract the validation vocabulary.

        Args:
            hf_token: HuggingFace API token (kept in memory only).
            hf_username: HF username.
            repo_id: Full repo id, e.g. ``"MonyetttRindam/telco-churn-models"``.
            registry: Shared :class:`Registry` instance (composition).
            train_reference_path: Path to ``ml/data/train.csv`` — the reference
                for schema, dtypes, categorical vocabulary and numeric stats.
        """
        self._hf_token = hf_token
        self._hf_username = hf_username
        self._repo_id = repo_id
        self._registry = registry
        self._train_path = Path(train_reference_path)

        self._api = HfApi(token=hf_token)

        # Extracted once, cached in memory.
        self._vocab = self._extract_vocabulary()
        self._numeric_cols = self._vocab["numeric_cols"]
        self._passthrough_cols = self._vocab["passthrough_cols"]
        self._categorical_cols = self._vocab["categorical_cols"]

    # === Path convention ======================================================

    def _hf_batch_path(self, batch_id: str) -> str:
        return f"synthetic/{batch_id}.csv"

    # === Vocabulary extraction ================================================

    def _extract_vocabulary(self) -> dict:
        """Read train.csv and extract the validation reference (cached)."""
        if not self._train_path.exists():
            raise BatchManagerError(
                f"Train reference not found: {self._train_path}"
            )
        df = pd.read_csv(self._train_path)
        numeric_cols, passthrough_cols, categorical_cols = split_columns(df)

        categorical_vocab = {
            c: set(df[c].astype(str).unique()) for c in categorical_cols
        }
        numeric_stats = {
            c: {"mean": float(df[c].mean()), "std": float(df[c].std())}
            for c in numeric_cols
        }
        churn_rate_reference = float((df[TARGET] == "Yes").mean())

        return {
            "column_names": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "numeric_cols": numeric_cols,
            "passthrough_cols": passthrough_cols,
            "categorical_cols": categorical_cols,
            "categorical_vocab": categorical_vocab,
            "numeric_stats": numeric_stats,
            "churn_rate_reference": churn_rate_reference,
        }

    @property
    def vocabulary(self) -> dict:
        """Expose the cached validation vocabulary (read-only use)."""
        return self._vocab

    # === READ operations ======================================================

    def list_batches(self, only_unused: bool = False) -> list:
        """List available batches (delegated to the registry)."""
        return self._registry.list_batches(only_unused=only_unused)

    def get_batch_info(self, batch_id: str) -> dict:
        """Return registry metadata for ``batch_id``.

        Raises:
            BatchNotFoundError: if the batch is unknown.
        """
        for b in self._registry.list_batches():
            if b.get("id") == batch_id:
                return b
        raise BatchNotFoundError(f"Batch '{batch_id}' not found in registry")

    def download_batch(self, batch_id: str) -> pd.DataFrame:
        """Download a batch CSV from HF Hub as a DataFrame.

        Raises:
            BatchDownloadError: if the file is missing or fails to parse.
        """
        try:
            local = hf_hub_download(
                self._repo_id,
                self._hf_batch_path(batch_id),
                repo_type=REPO_TYPE,
                token=self._hf_token,
            )
            return pd.read_csv(local)
        except Exception as exc:  # noqa: BLE001 - normalize to our error type
            raise BatchDownloadError(
                f"Failed to download batch '{batch_id}': {exc}"
            ) from None

    # === WRITE operations =====================================================

    def upload_batch(self, df: pd.DataFrame, source: str = "user_upload") -> str:
        """Upload a new batch to HF Hub and register it.

        Does NOT validate — call :meth:`validate_batch` first (separation of
        concerns). Auto-names the batch via ``registry.next_batch_number``.

        Returns:
            The newly-created batch_id (e.g. ``"batch_6"``).
        """
        n = self._registry.get_next_batch_number()
        batch_id = f"batch_{n}"

        tmp = Path(tempfile.gettempdir()) / f"{batch_id}.csv"
        df.to_csv(tmp, index=False)
        try:
            self._api.upload_file(
                path_or_fileobj=str(tmp),
                path_in_repo=self._hf_batch_path(batch_id),
                repo_id=self._repo_id,
                repo_type=REPO_TYPE,
                commit_message=f"Add batch {batch_id} ({source})",
            )
        except Exception as exc:  # noqa: BLE001
            raise BatchManagerError(
                f"Failed to upload batch '{batch_id}': {exc}"
            ) from None
        finally:
            tmp.unlink(missing_ok=True)

        churn_rate = round(self._churn_rate(df), 3)
        self._registry.add_batch(batch_id, len(df), churn_rate, source)
        return batch_id

    # === VALIDATION ===========================================================

    def validate_batch(self, df: pd.DataFrame, verbose: bool = True) -> ValidationResult:
        """Run the 7 validation rules and return a :class:`ValidationResult`."""
        errors: list = []
        warnings: list = []
        rule_results: dict = {}

        # Rules 1-5: hard failures.
        for name, check in (
            ("schema", self._check_schema),
            ("dtypes", self._check_dtypes),
            ("row_count", self._check_row_count),
            ("vocabulary", self._check_vocabulary),
            ("churn_rate", self._check_churn_rate),
        ):
            res = check(df)
            rule_results[name] = res
            if not res["passed"]:
                errors.append(res["detail"])

        # Rule 6: duplicates (interactive — warn or reject).
        dup = self._check_duplicates(df, verbose=verbose)
        rule_results["duplicates"] = dup["rule"]
        duplicate_info = dup["info"]
        if duplicate_info["action"] == "reject":
            errors.append(dup["rule"]["detail"])
        elif duplicate_info["action"] in ("warn", "confirm"):
            warnings.append(dup["rule"]["detail"])

        # Rule 7: distribution drift (soft warning, non-blocking).
        drift = self._check_drift(df)
        rule_results["drift"] = drift
        if not drift["passed"]:
            warnings.append(drift["detail"])

        passed = len(errors) == 0
        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            duplicate_info=duplicate_info,
            rule_results=rule_results,
        )

    # --- Rule 1: schema -------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> dict:
        expected = set(self._vocab["column_names"])
        actual = set(df.columns)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        n_ok = len(df.columns) == EXPECTED_N_COLS
        churn_ok = TARGET in df.columns
        passed = n_ok and not missing and not extra and churn_ok
        detail = (
            f"columns={len(df.columns)} (expected {EXPECTED_N_COLS}); "
            f"missing={missing}; extra={extra}; churn_present={churn_ok}"
        )
        return {"passed": passed, "detail": detail}

    # --- Rule 2: dtypes -------------------------------------------------------

    def _check_dtypes(self, df: pd.DataFrame) -> dict:
        problems: list = []
        numeric_like = self._numeric_cols + self._passthrough_cols
        for c in numeric_like:
            if c in df.columns and not is_numeric_dtype(df[c]):
                problems.append(f"'{c}' expected numeric, got {df[c].dtype}")
        for c in self._categorical_cols:
            if c in df.columns and not (
                is_object_dtype(df[c]) or is_string_dtype(df[c])
            ):
                problems.append(f"'{c}' expected object/string, got {df[c].dtype}")
        if TARGET in df.columns:
            vals = set(df[TARGET].dropna().unique())
            ok_labels = vals <= {"Yes", "No"}
            ok_binary = vals <= {0, 1} or vals <= {0.0, 1.0}
            if not (ok_labels or ok_binary):
                problems.append(f"'{TARGET}' values invalid: {sorted(vals)}")
        passed = not problems
        detail = "ok" if passed else "; ".join(problems)
        return {"passed": passed, "detail": detail}

    # --- Rule 3: row count ----------------------------------------------------

    def _check_row_count(self, df: pd.DataFrame) -> dict:
        n = len(df)
        passed = MIN_ROWS <= n <= MAX_ROWS
        detail = f"rows={n} (allowed {MIN_ROWS}-{MAX_ROWS})"
        return {"passed": passed, "detail": detail}

    # --- Rule 4: categorical vocabulary --------------------------------------

    def _check_vocabulary(self, df: pd.DataFrame) -> dict:
        unknown: list = []
        for c in self._categorical_cols:
            if c not in df.columns:
                continue
            vals = set(df[c].astype(str).unique())
            bad = vals - self._vocab["categorical_vocab"][c]
            if bad:
                unknown.append(
                    f"Kolom '{c}' ada value {sorted(bad)} yang tidak dikenal"
                )
        passed = not unknown
        detail = "ok" if passed else "; ".join(unknown)
        return {"passed": passed, "detail": detail}

    # --- Rule 5: churn rate ---------------------------------------------------

    def _check_churn_rate(self, df: pd.DataFrame) -> dict:
        rate = self._churn_rate(df)
        low = CHURN_REF - CHURN_TOLERANCE
        high = CHURN_REF + CHURN_TOLERANCE
        passed = low <= rate <= high
        detail = f"churn_rate={rate:.4f} (allowed {low:.4f}-{high:.4f})"
        return {"passed": passed, "detail": detail}

    # --- Rule 6: duplicates ---------------------------------------------------

    def _check_duplicates(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        reference = self._duplicate_reference(verbose=verbose)
        # Compare on all shared columns except the (unique) customerID.
        compare_cols = [
            c
            for c in self._vocab["column_names"]
            if c != "customerID" and c in df.columns and c in reference.columns
        ]
        ref_set = set(map(tuple, reference[compare_cols].astype(str).values))
        cand = df[compare_cols].astype(str).values
        dup_count = sum(1 for row in cand if tuple(row) in ref_set)
        pct = dup_count / len(df) if len(df) else 0.0

        if pct <= DUP_ACCEPT:
            action = "accept"
        elif pct <= DUP_WARN:
            action = "warn"
        elif pct <= DUP_CONFIRM:
            action = "confirm"
        else:
            action = "reject"

        info = {
            "count": int(dup_count),
            "percentage": round(pct * 100, 2),
            "action": action,
        }
        detail = (
            f"{dup_count}/{len(df)} rows ({info['percentage']}%) duplicate vs "
            f"train + existing batches -> action={action}"
        )
        passed = action in ("accept", "warn", "confirm")
        return {"rule": {"passed": passed, "detail": detail}, "info": info}

    def _duplicate_reference(self, verbose: bool = True) -> pd.DataFrame:
        """Build the reference set: train.csv + all existing batches (from HF)."""
        frames = [pd.read_csv(self._train_path)]
        batches = self._registry.list_batches()
        if verbose:
            print(f"  [dup] building reference: train + {len(batches)} batches")
        for b in batches:
            bid = b["id"]
            if verbose:
                print(f"        - downloading {bid} ...")
            frames.append(self.download_batch(bid))
        # Align on the union of columns; missing columns become NaN (won't match).
        return pd.concat(frames, ignore_index=True, sort=False)

    # --- Rule 7: distribution drift ------------------------------------------

    def _check_drift(self, df: pd.DataFrame) -> dict:
        drifted: list = []
        for c in self._numeric_cols:
            if c not in df.columns or not is_numeric_dtype(df[c]):
                continue
            ref_mean = self._vocab["numeric_stats"][c]["mean"]
            if ref_mean == 0:
                continue
            batch_mean = float(df[c].mean())
            rel = abs(batch_mean - ref_mean) / abs(ref_mean)
            if rel > DRIFT_THRESHOLD:
                drifted.append(f"{c} (rel drift {rel:.2%})")
        drift_detected = len(drifted) >= DRIFT_MIN_COLS
        passed = not drift_detected  # non-blocking; only feeds a warning
        detail = (
            "ok"
            if not drifted
            else f"drift in {len(drifted)} numeric col(s): {'; '.join(drifted)}"
        )
        return {"passed": passed, "detail": detail}

    # === Helpers ==============================================================

    def _churn_rate(self, df: pd.DataFrame) -> float:
        """Compute churn rate, handling both "Yes"/"No" and 0/1 encodings."""
        if TARGET not in df.columns:
            return float("nan")
        s = df[TARGET].dropna()
        uniq = set(s.unique())
        if uniq <= {"Yes", "No"}:
            return float((s == "Yes").mean())
        if uniq <= {0, 1} or uniq <= {0.0, 1.0}:
            return float(s.astype(float).mean())
        # Mixed / unexpected: best-effort on the "Yes" label.
        return float((s.astype(str) == "Yes").mean())
