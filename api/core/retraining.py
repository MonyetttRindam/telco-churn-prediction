"""Retraining pipeline: the core Phase 2 orchestrator.

Combines :class:`Registry`, :class:`ModelManager`, :class:`BatchManager`, and
Phase 1's ``ml/`` pipeline (``train_model`` / ``evaluate_model``) into one full
retraining cycle guarded by a validation gate.

Flow:
  1. Load context (current model version, metrics, batches_used).
  2. Download target batch + accumulate previously-used batches.
  3. Combine train.csv + accumulated batches.
  4. Train new model (reuse ``ml/train.py``).
  5. Evaluate on the FIXED holdout (reuse ``ml/evaluate.py``, threshold 0.60).
  6. Validation gate: ``F1_new >= F1_current - tol  AND  recall_new >= recall_current``.
  7. Promote (gate pass) or reject (gate fail) — update registry.
  8. Upload artifacts to HF Hub (``models/{version_id}/``).
  9. Return a :class:`RetrainingReport`.

Constraints (MLOPS_PLAN.md):
* FIXED holdout — only ever read for evaluation, never for training.
* FIXED preprocessor — loaded once, transform-only, never refit.
* FIXED threshold — 0.60.

SECURITY: the HF token lives in memory only; never printed or logged.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import HfApi

# Project root on sys.path so ``ml`` and ``src`` are importable standalone.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.evaluate import evaluate_model  # noqa: E402
from ml.train import train_model  # noqa: E402

from api.core.batch_manager import BatchManager  # noqa: E402
from api.core.model_manager import ModelManager  # noqa: E402
from api.core.registry import Registry  # noqa: E402

REPO_TYPE = "model"
HOLDOUT_THRESHOLD = 0.60


# === Exceptions ===============================================================


class RetrainingError(Exception):
    """Base class for all retraining errors."""


class BatchAlreadyUsedError(RetrainingError):
    """Raised when the target batch was already used by the current version."""


class DataAccumulationError(RetrainingError):
    """Raised when combining train + batches fails."""


class ArtifactUploadError(RetrainingError):
    """Raised when uploading model/preprocessor artifacts to HF Hub fails."""


# === Data structures ==========================================================


@dataclass
class ValidationGateResult:
    """Outcome of the validation gate."""

    passed: bool
    f1_diff: float  # new - current (positive = better)
    recall_diff: float
    f1_gate_passed: bool  # F1_new >= F1_current - tolerance
    recall_gate_passed: bool  # recall_new >= recall_current
    reason: str  # human-readable


@dataclass
class RetrainingReport:
    """Full report of a retraining cycle."""

    new_version_id: str
    batch_used: str
    accumulated_batches: list
    n_train_samples: int

    current_metrics: dict
    new_metrics: dict

    gate_result: ValidationGateResult
    decision: str  # "promoted" or "rejected"

    started_at: str
    completed_at: str
    duration_seconds: float


# === RetrainingPipeline =======================================================


class RetrainingPipeline:
    """Orchestrates a full retraining cycle with validation gate."""

    def __init__(
        self,
        hf_token: str,
        hf_username: str,
        repo_id: str,
        registry: Registry,
        model_manager: ModelManager,
        batch_manager: BatchManager,
        train_data_path: Path,
        holdout_path: Path,
        preprocessor_path: Path,
        f1_tolerance: float = 0.02,
    ) -> None:
        """Initialize the pipeline.

        Args:
            train_data_path: ``ml/data/train.csv`` (real training data).
            holdout_path: ``ml/data/test_holdout.csv`` (FIXED, eval-only).
            preprocessor_path: ``models/preprocessor.joblib`` (FIXED).
            f1_tolerance: how much F1 may drop and still pass the gate (0.02).
        """
        self._hf_token = hf_token
        self._hf_username = hf_username
        self._repo_id = repo_id
        self._registry = registry
        self._model_manager = model_manager
        self._batch_manager = batch_manager
        self._train_path = Path(train_data_path)
        self._holdout_path = Path(holdout_path)
        self._preprocessor_path = Path(preprocessor_path)
        self._f1_tolerance = f1_tolerance

        self._api = HfApi(token=hf_token)
        # FIXED preprocessor: load once, transform-only, never refit.
        self._preprocessor = joblib.load(self._preprocessor_path)

    # === Public API ===========================================================

    def retrain(self, batch_id: str) -> RetrainingReport:
        """Execute a full retraining cycle for ``batch_id``.

        Raises:
            BatchNotFoundError: if batch_id doesn't exist (from BatchManager).
            BatchAlreadyUsedError: if the current version already used it.
            RetrainingError: if any pipeline step fails.
        """
        started = datetime.now(timezone.utc)

        # 1. Context.
        ctx = self._load_context()
        # Validate batch exists (raises BatchNotFoundError) + not already used.
        self._batch_manager.get_batch_info(batch_id)
        if batch_id in ctx["batches_used"]:
            raise BatchAlreadyUsedError(
                f"Batch '{batch_id}' already used by version "
                f"'{ctx['current_version_id']}'"
            )
        accumulated = ctx["batches_used"] + [batch_id]

        # 2-3. Accumulate data.
        combined = self._accumulate_data(batch_id, ctx["batches_used"])

        # 4. Train.
        model, train_meta = self._train_new_model(combined)

        # 5. Evaluate on fixed holdout.
        new_eval = self._evaluate_on_holdout(model, self._preprocessor)

        # 6. Gate.
        gate = self._apply_gate(ctx["current_metrics"], new_eval)

        # 7-8. Version id + metrics + upload artifacts.
        version_id = self._generate_version_id()
        new_metrics = self._build_metrics(
            version_id, new_eval, train_meta, started.isoformat()
        )
        self._upload_artifacts(version_id, model, self._preprocessor)

        # 9. Promote or reject.
        if gate.passed:
            self._promote(version_id, new_metrics, accumulated, gate)
            decision = "promoted"
        else:
            self._reject(version_id, new_metrics, accumulated, gate)
            decision = "rejected"

        completed = datetime.now(timezone.utc)
        return RetrainingReport(
            new_version_id=version_id,
            batch_used=batch_id,
            accumulated_batches=accumulated,
            n_train_samples=train_meta["n_train_samples"],
            current_metrics=ctx["current_metrics"],
            new_metrics=new_metrics,
            gate_result=gate,
            decision=decision,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=(completed - started).total_seconds(),
        )

    # === Internal steps =======================================================

    def _load_context(self) -> dict:
        """Return current version_id, its metrics, and its batches_used."""
        version_id = self._registry.get_active_version()
        info = self._registry.get_version_info(version_id)
        return {
            "current_version_id": version_id,
            "current_metrics": info["metrics"],
            "batches_used": list(info.get("batches_used", [])),
        }

    def _accumulate_data(
        self, batch_id: str, previous_batches: list
    ) -> pd.DataFrame:
        """Combine train.csv + previously-used batches + the new batch.

        Batches are appended in registry order (deterministic). The new batch
        is appended last.
        """
        try:
            frames = [pd.read_csv(self._train_path)]
            for bid in previous_batches:
                frames.append(self._batch_manager.download_batch(bid))
            frames.append(self._batch_manager.download_batch(batch_id))
            return pd.concat(frames, ignore_index=True, sort=False)
        except Exception as exc:  # noqa: BLE001
            raise DataAccumulationError(
                f"Failed to accumulate data for '{batch_id}': {exc}"
            ) from None

    def _train_new_model(self, combined_data: pd.DataFrame) -> tuple:
        """Train a new model on the combined data (reuse ml.train.train_model)."""
        return train_model(combined_data, self._preprocessor)

    def _evaluate_on_holdout(self, model, preprocessor) -> dict:
        """Evaluate on the FIXED holdout at threshold 0.60 (reuse ml.evaluate)."""
        holdout_df = pd.read_csv(self._holdout_path)
        return evaluate_model(
            model, preprocessor, holdout_df, threshold=HOLDOUT_THRESHOLD
        )

    def _apply_gate(self, current: dict, new: dict) -> ValidationGateResult:
        """Apply the validation gate.

        F1: ``new >= current - tolerance``. Recall: ``new >= current``.
        """
        f1_diff = new["f1"] - current["f1"]
        recall_diff = new["recall"] - current["recall"]

        f1_gate = new["f1"] >= (current["f1"] - self._f1_tolerance)
        recall_gate = new["recall"] >= current["recall"]
        passed = f1_gate and recall_gate

        if passed:
            reason = (
                f"Passed: F1 {new['f1']:.4f} >= "
                f"{current['f1'] - self._f1_tolerance:.4f} AND "
                f"Recall {new['recall']:.4f} >= {current['recall']:.4f}"
            )
        elif not f1_gate and not recall_gate:
            reason = (
                f"Failed both: F1 dropped {abs(f1_diff):.4f} (>tolerance) "
                f"AND Recall dropped {abs(recall_diff):.4f}"
            )
        elif not f1_gate:
            reason = (
                f"Failed F1: {new['f1']:.4f} < "
                f"{current['f1'] - self._f1_tolerance:.4f} "
                f"(drop {abs(f1_diff):.4f})"
            )
        else:
            reason = (
                f"Failed Recall: {new['recall']:.4f} < {current['recall']:.4f}"
            )

        return ValidationGateResult(
            passed=passed,
            f1_diff=f1_diff,
            recall_diff=recall_diff,
            f1_gate_passed=f1_gate,
            recall_gate_passed=recall_gate,
            reason=reason,
        )

    def _generate_version_id(self) -> str:
        """Return ``v_YYYYMMDD_HHMMSS`` (UTC) — unique at 1-second resolution."""
        return datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")

    def _build_metrics(
        self,
        version_id: str,
        eval_result: dict,
        train_meta: dict,
        created_at: str,
    ) -> dict:
        """Assemble the metrics dict stored in the registry."""
        return {
            "version_id": version_id,
            "f1": eval_result["f1"],
            "recall": eval_result["recall"],
            "precision": eval_result["precision"],
            "roc_auc": eval_result["roc_auc"],
            "threshold": eval_result["threshold"],
            "n_train_samples": train_meta["n_train_samples"],
            "n_holdout_samples": eval_result["n_samples"],
            "model_type": "LogisticRegression",
            "model_params": train_meta["model_params"],
            "evaluated_on_holdout": True,
            "confusion_matrix": eval_result["confusion_matrix"],
            "created_at": created_at,
        }

    def _upload_artifacts(self, version_id: str, model, preprocessor) -> None:
        """Upload model + preprocessor to ``models/{version_id}/``.

        The preprocessor is FIXED, so the same file is uploaded for every
        version. This is redundant by design: it makes each version fully
        self-contained (loadable independently by :class:`ModelManager`).
        """
        tmp_model = Path(tempfile.gettempdir()) / f"{version_id}_model.pkl"
        joblib.dump(model, tmp_model)
        try:
            self._api.upload_file(
                path_or_fileobj=str(tmp_model),
                path_in_repo=f"models/{version_id}/model.pkl",
                repo_id=self._repo_id,
                repo_type=REPO_TYPE,
                commit_message=f"Retraining: upload model for {version_id}",
            )
            # FIXED preprocessor — re-upload the canonical file every version.
            self._api.upload_file(
                path_or_fileobj=str(self._preprocessor_path),
                path_in_repo=f"models/{version_id}/preprocessor.pkl",
                repo_id=self._repo_id,
                repo_type=REPO_TYPE,
                commit_message=f"Retraining: upload preprocessor for {version_id}",
            )
        except Exception as exc:  # noqa: BLE001
            raise ArtifactUploadError(
                f"Failed to upload artifacts for '{version_id}': {exc}"
            ) from None
        finally:
            tmp_model.unlink(missing_ok=True)

    def _promote(
        self,
        version_id: str,
        metrics: dict,
        batches_used: list,
        gate: ValidationGateResult,
    ) -> None:
        """Promote a passing version to active + swap the in-memory model.

        Order (prefer registry consistency over in-memory consistency): add
        (pending) -> promote (atomic in registry) -> swap in-memory -> mark
        batches used.
        """
        self._registry.add_version(
            version_id=version_id,
            metrics=metrics,
            batches_used=batches_used,
            status="pending",
            reason=f"Retraining candidate. {gate.reason}",
        )
        self._registry.promote_version(version_id)
        self._model_manager.swap_to(version_id)
        for bid in batches_used:
            self._registry.mark_batch_used(bid, version_id)

    def _reject(
        self,
        version_id: str,
        metrics: dict,
        batches_used: list,
        gate: ValidationGateResult,
    ) -> None:
        """Archive a failing version as rejected. Active model is unchanged.

        Rejected versions still consumed the batches, so mark them used.
        """
        self._registry.add_version(
            version_id=version_id,
            metrics=metrics,
            batches_used=batches_used,
            status="rejected",
            reason=gate.reason,
        )
        for bid in batches_used:
            self._registry.mark_batch_used(bid, version_id)
