"""Dependency injection providers for FastAPI endpoints.

Singletons are initialized once at startup (via :func:`init_dependencies`) and
injected into endpoints via ``Depends()`` — this avoids re-initializing
Registry / ModelManager / BatchManager / RetrainingPipeline / JobQueue on every
request (which would re-hit HF Hub and reload the model each time).

SECURITY: the HF token is loaded from ``.env`` and held only in memory by the
managers; never printed or logged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from api.core.batch_manager import BatchManager
from api.core.job_queue import JobQueue
from api.core.model_manager import ModelManager
from api.core.registry import Registry
from api.core.retraining import RetrainingPipeline

# === Module-level singletons ==================================================

_registry: Optional[Registry] = None
_model_manager: Optional[ModelManager] = None
_batch_manager: Optional[BatchManager] = None
_pipeline: Optional[RetrainingPipeline] = None
_job_queue: Optional[JobQueue] = None


def init_dependencies(project_root: Path) -> None:
    """Initialize all singletons. Call once at FastAPI startup.

    Raises:
        RuntimeError: if HF_TOKEN / HF_USERNAME are missing from the env.
    """
    global _registry, _model_manager, _batch_manager, _pipeline, _job_queue

    load_dotenv(project_root / ".env")
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    if not token or not username:
        raise RuntimeError("HF_TOKEN dan HF_USERNAME harus ada di .env")

    repo_id = f"{username}/telco-churn-models"

    _registry = Registry(hf_token=token, hf_username=username, repo_id=repo_id)

    _model_manager = ModelManager(
        hf_token=token,
        hf_username=username,
        repo_id=repo_id,
        registry=_registry,
    )
    _model_manager.load_active()  # load active model into memory at startup

    _batch_manager = BatchManager(
        hf_token=token,
        hf_username=username,
        repo_id=repo_id,
        registry=_registry,
        train_reference_path=project_root / "ml/data/train.csv",
    )

    _pipeline = RetrainingPipeline(
        hf_token=token,
        hf_username=username,
        repo_id=repo_id,
        registry=_registry,
        model_manager=_model_manager,
        batch_manager=_batch_manager,
        train_data_path=project_root / "ml/data/train.csv",
        holdout_path=project_root / "ml/data/test_holdout.csv",
        preprocessor_path=project_root / "models/preprocessor.joblib",
    )

    _job_queue = JobQueue(max_jobs=100)


# === Dependency provider functions (for FastAPI Depends) ======================


def get_registry() -> Registry:
    if _registry is None:
        raise RuntimeError("Dependencies not initialized")
    return _registry


def get_model_manager() -> ModelManager:
    if _model_manager is None:
        raise RuntimeError("Dependencies not initialized")
    return _model_manager


def get_batch_manager() -> BatchManager:
    if _batch_manager is None:
        raise RuntimeError("Dependencies not initialized")
    return _batch_manager


def get_pipeline() -> RetrainingPipeline:
    if _pipeline is None:
        raise RuntimeError("Dependencies not initialized")
    return _pipeline


def get_job_queue() -> JobQueue:
    if _job_queue is None:
        raise RuntimeError("Dependencies not initialized")
    return _job_queue
