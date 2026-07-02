"""MLOps endpoints: /status, /history, /rollback, /verify-key."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from api.core.auth import require_api_key
from api.core.model_manager import ModelDownloadError, ModelManager
from api.core.registry import (
    InvalidStateError,
    Registry,
    RegistryConnectionError,
    VersionNotFoundError,
)
from api.deps import get_model_manager, get_registry
from api.schemas import (
    HistoryResponse,
    MetricsSchema,
    RollbackResponse,
    StatusResponse,
    VerifyKeyResponse,
    VersionInfo,
)

router = APIRouter(prefix="/api", tags=["mlops"])

_METRIC_KEYS = ("f1", "recall", "precision", "roc_auc", "threshold")


def _metrics(metrics: dict) -> MetricsSchema:
    """Extract the 5 public metric fields from a full metrics dict."""
    return MetricsSchema(**{k: metrics[k] for k in _METRIC_KEYS})


@router.get("/status", response_model=StatusResponse)
def get_status(registry: Registry = Depends(get_registry)) -> StatusResponse:
    """Return the current active model status + batch counts."""
    try:
        active_id = registry.get_active_version()
        active_info = registry.get_version_info(active_id)
        batches = registry.list_batches()
        unused = registry.list_batches(only_unused=True)
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    return StatusResponse(
        active_version=active_id,
        previous_version=registry.get_previous_version(),
        metrics=_metrics(active_info["metrics"]),
        n_batches_available=len(batches),
        n_batches_unused=len(unused),
    )


@router.get("/history", response_model=HistoryResponse)
def get_history(
    status_filter: Optional[str] = None,
    registry: Registry = Depends(get_registry),
) -> HistoryResponse:
    """Return all model versions (optionally filtered by status)."""
    try:
        versions = registry.list_versions(status=status_filter)
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    version_infos = [
        VersionInfo(
            id=v["id"],
            created_at=v["created_at"],
            metrics=_metrics(v["metrics"]),
            batches_used=v["batches_used"],
            status=v["status"],
            reason=v["reason"],
        )
        for v in versions
    ]
    return HistoryResponse(versions=version_infos, total_count=len(version_infos))


@router.get("/verify-key", response_model=VerifyKeyResponse)
def verify_key(_: str = Depends(require_api_key)) -> VerifyKeyResponse:
    """Verify the admin API key (used by the Streamlit UI to unlock actions)."""
    return VerifyKeyResponse(valid=True, message="Key is valid")


@router.post("/rollback", response_model=RollbackResponse)
def rollback(
    _: str = Depends(require_api_key),
    registry: Registry = Depends(get_registry),
    model_manager: ModelManager = Depends(get_model_manager),
) -> RollbackResponse:
    """Roll back to the previous active version and sync the in-memory model."""
    try:
        previous_active = registry.get_active_version()  # currently active
        new_active = registry.rollback()  # swap active <-> previous
        model_manager.swap_to(new_active)  # sync in-memory model
    except InvalidStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from None
    except VersionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from None
    except ModelDownloadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to swap model: {exc}",
        ) from None
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    return RollbackResponse(
        new_active_version=new_active,
        previous_active_version=previous_active,
        message=f"Rolled back from {previous_active} to {new_active}",
    )
