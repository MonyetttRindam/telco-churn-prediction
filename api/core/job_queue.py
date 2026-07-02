"""In-memory background job queue for long-running MLOps operations.

Simple thread-based queue for a single-instance deployment. Jobs are stored
in-memory (dict); history is bounded by ``max_jobs`` (LRU eviction of finished
jobs). Not persistent — a restart loses pending/running jobs (completed jobs'
real results live in the registry anyway).

For multi-instance scale, replace with Redis + Celery/RQ.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional


def _utcnow() -> str:
    """Current time as a UTC ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# === Exceptions ===============================================================


class JobQueueError(Exception):
    """Base class for all job-queue errors."""


class JobNotFoundError(JobQueueError):
    """Raised when a job_id is unknown."""


# === Data structures ==========================================================


class JobStatus(str, Enum):
    """Lifecycle states of a job."""

    PENDING = "pending"  # queued, not started
    RUNNING = "running"  # currently executing
    COMPLETED = "completed"  # success, result available
    FAILED = "failed"  # error, error_message available


# Terminal states — safe to evict.
_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED)


@dataclass
class Job:
    """A single background job and its outcome."""

    job_id: str
    job_type: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# === JobQueue =================================================================


class JobQueue:
    """Thread-based, in-memory background job queue (single-instance)."""

    def __init__(self, max_jobs: int = 100) -> None:
        """Initialize the queue.

        Args:
            max_jobs: max jobs kept in memory; oldest finished jobs are evicted
                (LRU) once this is exceeded.
        """
        self._max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}  # insertion-ordered == submission order
        self._lock = threading.Lock()

    # === Public API ===========================================================

    def submit(
        self,
        job_type: str,
        target: Callable,
        *args,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Submit a job for background execution.

        Args:
            job_type: category (e.g. "retrain", "batch_validate").
            target: callable to run in a daemon thread.
            *args, **kwargs: forwarded to ``target``.
            metadata: optional context (e.g. ``{"batch_id": "batch_1"}``).

        Returns:
            The generated job_id.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=_utcnow(),
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._execute,
            args=(job_id, target, args, kwargs),
            daemon=True,
            name=f"jobqueue-{job_id}",
        )
        thread.start()

        # Evict old finished jobs if we're over capacity.
        self.cleanup_old_jobs()
        return job_id

    def get_job(self, job_id: str) -> Job:
        """Return the job with ``job_id``.

        Raises:
            JobNotFoundError: if unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job '{job_id}' not found")
            return job

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 20,
    ) -> list:
        """List recent jobs (most-recent first), optionally filtered."""
        with self._lock:
            jobs = list(self._jobs.values())
        if job_type is not None:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        jobs.reverse()  # most recent (last inserted) first
        return jobs[:limit]

    def cleanup_old_jobs(self) -> int:
        """Evict oldest finished jobs beyond ``max_jobs``. Return count evicted."""
        with self._lock:
            return self._cleanup_locked()

    # === Internal =============================================================

    def _execute(self, job_id: str, target: Callable, args: tuple, kwargs: dict) -> None:
        """Run ``target`` in a daemon thread, updating job status/result."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # evicted before it even started (shouldn't happen)
                return
            job.status = JobStatus.RUNNING
            job.started_at = _utcnow()

        try:
            result = target(*args, **kwargs)
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = _utcnow()
                job.result = self._serialize_result(result)
        except Exception as exc:  # noqa: BLE001 - record any failure
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = _utcnow()
                job.error_message = str(exc)
        finally:
            # A job just reached a terminal state — try to trim history.
            self.cleanup_old_jobs()

    @staticmethod
    def _serialize_result(result) -> Optional[dict]:
        """Normalize a target's return value into a JSON-friendly dict.

        Dataclasses (incl. nested, e.g. RetrainingReport -> ValidationGateResult)
        are converted via ``asdict``; dicts pass through; anything else is
        wrapped as ``{"value": result}``.
        """
        if result is None:
            return None
        if is_dataclass(result) and not isinstance(result, type):
            return asdict(result)
        if isinstance(result, dict):
            return dict(result)
        return {"value": result}

    def _cleanup_locked(self) -> int:
        """Evict oldest terminal jobs beyond capacity. Assumes lock held."""
        n_over = len(self._jobs) - self._max_jobs
        if n_over <= 0:
            return 0
        evicted = 0
        # dict preserves submission order -> iterate oldest-first.
        for job_id in list(self._jobs.keys()):
            if evicted >= n_over:
                break
            if self._jobs[job_id].status in _TERMINAL:
                del self._jobs[job_id]
                evicted += 1
        return evicted
