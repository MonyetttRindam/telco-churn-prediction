"""Job endpoints: GET /jobs/{job_id}, GET /jobs."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from api.core.job_queue import Job, JobNotFoundError, JobQueue
from api.deps import get_job_queue
from api.schemas import JobResponse

router = APIRouter(prefix="/api", tags=["jobs"])


def _to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error_message=job.error_message,
        metadata=job.metadata,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str, job_queue: JobQueue = Depends(get_job_queue)
) -> JobResponse:
    """Return the status/result of a specific job."""
    try:
        job = job_queue.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from None
    return _to_response(job)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(
    job_type: Optional[str] = None,
    limit: int = 20,
    job_queue: JobQueue = Depends(get_job_queue),
) -> list[JobResponse]:
    """List recent jobs (most-recent first), optionally filtered by type."""
    jobs = job_queue.list_jobs(job_type=job_type, status=None, limit=limit)
    return [_to_response(j) for j in jobs]
