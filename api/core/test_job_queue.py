"""Manual smoke test for the JobQueue class.

Run:
    .venv/Scripts/python.exe api/core/test_job_queue.py

Pure local: NO HF Hub, NO filesystem. Uses sleepy sample tasks to exercise the
background threading, status transitions, filtering, concurrency and LRU
eviction.
"""

import sys
import threading
import time
from pathlib import Path

# Allow running as a standalone script (add project root to sys.path).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.core.job_queue import (  # noqa: E402
    JobNotFoundError,
    JobQueue,
    JobStatus,
)


def _fmt(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


# --- Sample tasks -------------------------------------------------------------


def sample_fast_task(x, y):
    time.sleep(0.5)
    return {"sum": x + y}


def sample_slow_task():
    time.sleep(2)
    return {"result": "done"}


def sample_failing_task():
    time.sleep(0.5)
    raise ValueError("intentional failure")


def _wait_for(queue: JobQueue, job_id: str, timeout: float = 5.0):
    """Poll until the job reaches a terminal state, is evicted, or times out.

    Returns the terminal JobStatus, or None if the job was evicted (which, since
    only terminal jobs are evictable, means it finished then got trimmed).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = queue.get_job(job_id).status
        except JobNotFoundError:
            return None  # completed then evicted (LRU)
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return status
        time.sleep(0.05)
    try:
        return queue.get_job(job_id).status
    except JobNotFoundError:
        return None


def main() -> int:
    print("=" * 60)
    print("JOB QUEUE SMOKE TEST")
    print("=" * 60)

    all_pass = True
    q = JobQueue(max_jobs=100)

    # --- [1] Submit + poll fast task -----------------------------------------
    jid = q.submit("test_fast", sample_fast_task, 3, 5, metadata={"note": "fast"})
    immediate = q.get_job(jid).status
    s_imm = immediate in (JobStatus.PENDING, JobStatus.RUNNING)
    final = _wait_for(q, jid)
    job = q.get_job(jid)
    s1 = s_imm and final == JobStatus.COMPLETED and job.result == {"sum": 8}
    all_pass &= s1
    print(f"[1] fast task -> immediate={immediate.value} final={final.value} result={job.result} {_fmt(s1)}")

    # --- [2] Slow task status progression ------------------------------------
    jid = q.submit("test_slow", sample_slow_task)
    time.sleep(0.5)
    mid = q.get_job(jid).status  # should be RUNNING (task sleeps 2s)
    final = _wait_for(q, jid, timeout=5.0)
    s2 = mid == JobStatus.RUNNING and final == JobStatus.COMPLETED
    all_pass &= s2
    print(f"[2] slow task -> mid={mid.value} final={final.value} {_fmt(s2)}")

    # --- [3] Failing task ----------------------------------------------------
    jid = q.submit("test_fail", sample_failing_task)
    final = _wait_for(q, jid)
    job = q.get_job(jid)
    s3 = final == JobStatus.FAILED and "intentional failure" in (job.error_message or "")
    all_pass &= s3
    print(f"[3] failing task -> final={final.value} error=\"{job.error_message}\" {_fmt(s3)}")

    # --- [4] Get non-existent job --------------------------------------------
    try:
        q.get_job("job_fake")
        s4 = False
    except JobNotFoundError:
        s4 = True
    all_pass &= s4
    print(f"[4] get_job('job_fake') -> JobNotFoundError {_fmt(s4)}")

    # --- [5] List jobs with filter -------------------------------------------
    a = q.submit("test_fast", sample_fast_task, 1, 1)
    b = q.submit("test_fast", sample_fast_task, 2, 2)
    c = q.submit("test_slow", sample_slow_task)
    fast_jobs = q.list_jobs(job_type="test_fast")
    fast_ids = {j.job_id for j in fast_jobs}
    s5 = a in fast_ids and b in fast_ids and c not in fast_ids
    all_pass &= s5
    print(f"[5] list_jobs(type='test_fast') -> {len(fast_jobs)} jobs, filtered correctly {_fmt(s5)}")
    _wait_for(q, c, timeout=5.0)  # let the slow one finish before moving on

    # --- [6] Concurrent submission -------------------------------------------
    submitted: list = []
    sub_lock = threading.Lock()

    def _submit_one():
        j = q.submit("test_concurrent", sample_fast_task, 10, 20)
        with sub_lock:
            submitted.append(j)

    threads = [threading.Thread(target=_submit_one) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    unique_ok = len(set(submitted)) == 5
    for j in submitted:
        _wait_for(q, j)
    all_done = all(
        q.get_job(j).status == JobStatus.COMPLETED for j in submitted
    )
    s6 = unique_ok and all_done
    all_pass &= s6
    print(f"[6] concurrent submit x5 -> unique_ids={unique_ok} all_completed={all_done} {_fmt(s6)}")

    # --- [7] LRU eviction ----------------------------------------------------
    # Submit + wait SEQUENTIALLY so completion order == submission order; this
    # makes LRU age deterministic (parallel completion would evict whichever
    # finished-job is oldest-by-insertion, not necessarily the oldest submitted).
    q2 = JobQueue(max_jobs=5)
    ids = []
    for i in range(8):
        j = q2.submit("evict", sample_fast_task, i, i)
        ids.append(j)
        _wait_for(q2, j)  # finish before next submit
    q2.cleanup_old_jobs()  # deterministic final trim
    with q2._lock:
        remaining = set(q2._jobs.keys())
    len_ok = len(remaining) == 5
    oldest_evicted = all(j not in remaining for j in ids[:3])
    newest_kept = all(j in remaining for j in ids[3:])
    s7 = len_ok and oldest_evicted and newest_kept
    all_pass &= s7
    print(f"[7] LRU evict (max=5, submit 8) -> remaining={len(remaining)} oldest3_evicted={oldest_evicted} newest5_kept={newest_kept} {_fmt(s7)}")

    print("=" * 60)
    print(f"VERDICT: {'SUCCESS' if all_pass else 'FAILED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
