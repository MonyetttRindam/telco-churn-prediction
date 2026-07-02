"""Batch endpoints: GET /batches, POST /upload-batch."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.core.auth import require_api_key
from api.core.batch_manager import BatchManager, BatchManagerError
from api.core.registry import Registry, RegistryConnectionError
from api.deps import get_batch_manager, get_registry
from api.schemas import BatchesResponse, BatchInfo, UploadBatchResponse

router = APIRouter(prefix="/api", tags=["batches"])


@router.get("/batches", response_model=BatchesResponse)
def list_batches(registry: Registry = Depends(get_registry)) -> BatchesResponse:
    """List all available batches with usage counts."""
    try:
        batches = registry.list_batches()
        unused = registry.list_batches(only_unused=True)
    except RegistryConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None

    batch_infos = [BatchInfo(**b) for b in batches]
    return BatchesResponse(
        batches=batch_infos,
        total_count=len(batches),
        unused_count=len(unused),
    )


@router.post("/upload-batch", response_model=UploadBatchResponse)
async def upload_batch(
    file: UploadFile = File(...),
    _: str = Depends(require_api_key),
    batch_manager: BatchManager = Depends(get_batch_manager),
) -> UploadBatchResponse:
    """Upload a new synthetic batch CSV. Validates (7 rules) before uploading.

    Validation failure returns 200 with ``validation_passed=False`` and the
    error/warning detail (the batch is NOT uploaded). A malformed CSV is 400.
    """
    # 1. Parse the uploaded CSV.
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV: {exc}",
        ) from None

    # 2. Validate (verbose=False -> no server-log noise).
    validation = batch_manager.validate_batch(df, verbose=False)

    if not validation.passed:
        return UploadBatchResponse(
            batch_id="",
            validation_passed=False,
            warnings=validation.warnings,
            errors=validation.errors,
            duplicate_info=validation.duplicate_info,
            message="Validation failed. Batch NOT uploaded.",
        )

    # 3. Passed -> upload + register.
    try:
        batch_id = batch_manager.upload_batch(df, source="user_upload")
    except BatchManagerError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        ) from None

    return UploadBatchResponse(
        batch_id=batch_id,
        validation_passed=True,
        warnings=validation.warnings,
        errors=[],
        duplicate_info=validation.duplicate_info,
        message=f"Batch uploaded as {batch_id}",
    )
