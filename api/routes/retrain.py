"""Retraining endpoint: POST /retrain (async via JobQueue)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.core.auth import require_api_key
from api.core.batch_manager import BatchManager, BatchNotFoundError
from api.core.job_queue import JobQueue
from api.core.registry import Registry, RegistryConnectionError
from api.core.retraining import RetrainingPipeline
from api.deps import (
    get_batch_manager,
    get_job_queue,
    get_pipeline,
    get_registry,
)
from api.schemas import RetrainRequest, RetrainResponse

router = APIRouter(prefix="/api", tags=["retrain"])


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retrain(
    request: RetrainRequest,
    _: str = Depends(require_api_key),
    pipeline: RetrainingPipeline = Depends(get_pipeline),
    batch_manager: BatchManager = Depends(get_batch_manager),
    registry: Registry = Depends(get_registry),
    job_queue: JobQueue = Depends(get_job_queue),
) -> RetrainResponse:
    """Trigger an async retraining run for ``batch_id``.

    Synchronous pre-checks (before the background job is submitted):
      * batch must exist            -> 404
      * batch not already used by the active version -> 409
    The actual training runs in the background; poll ``/api/jobs/{job_id}``.
    """
    # 1. Batch must exist.
    try:
        batch_manager.get_batch_info(request.batch_id)
    except BatchNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from None
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    # 2. Batch must not already be used by the active version (409).
    try:
        active_id = registry.get_active_version()
        active_info = registry.get_version_info(active_id)
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None
    if request.batch_id in active_info.get("batches_used", []):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Batch '{request.batch_id}' already used by active version "
                f"'{active_id}'"
            ),
        )

    # 3. Submit background job.
    job_id = job_queue.submit(
        job_type="retrain",
        target=pipeline.retrain,
        batch_id=request.batch_id,
        metadata={"batch_id": request.batch_id},
    )

    return RetrainResponse(
        job_id=job_id,
        status="pending",
        message=f"Retraining job submitted. Poll /api/jobs/{job_id} for status.",
        batch_id=request.batch_id,
    )
