"""Pydantic response/request models for FastAPI endpoints (Pydantic v2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MetricsSchema(BaseModel):
    f1: float
    recall: float
    precision: float
    roc_auc: float
    threshold: float


class VersionInfo(BaseModel):
    id: str
    created_at: str
    metrics: MetricsSchema
    batches_used: list[str]
    status: str
    reason: str


class BatchInfo(BaseModel):
    id: str
    created_at: Optional[str]
    n_rows: int
    churn_rate: float
    source: str
    used_in_versions: list[str]


class StatusResponse(BaseModel):
    active_version: str
    previous_version: Optional[str]
    metrics: MetricsSchema
    n_batches_available: int
    n_batches_unused: int


class HistoryResponse(BaseModel):
    versions: list[VersionInfo]
    total_count: int


class BatchesResponse(BaseModel):
    batches: list[BatchInfo]
    total_count: int
    unused_count: int


class RetrainRequest(BaseModel):
    batch_id: str = Field(..., description="Batch ID to use for retraining")


class RetrainResponse(BaseModel):
    job_id: str
    status: str
    message: str
    batch_id: str


class RollbackResponse(BaseModel):
    new_active_version: str
    previous_active_version: str
    message: str


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[dict]
    error_message: Optional[str]
    metadata: dict


class UploadBatchResponse(BaseModel):
    batch_id: str
    validation_passed: bool
    warnings: list[str]
    errors: list[str]
    duplicate_info: dict
    message: str


class VerifyKeyResponse(BaseModel):
    valid: bool
    message: str
