"""Registry manager for the MLOps retraining system.

Single interface for reading/writing ``registry.json`` on HuggingFace Hub.
Thread-safe via an internal lock; reads are cached in-memory with a TTL.

The registry tracks two things:

* **versions** — every trained model (active / previous / archived / rejected /
  pending) with its holdout metrics.
* **batches** — synthetic/user data batches available for retraining, plus a
  monotonically increasing ``next_batch_number`` used for auto-naming.

Schema (v2)::

    {
      "active": "v_initial",
      "previous": null,
      "versions": [
        {"id", "created_at", "metrics", "batches_used", "status", "reason"}
      ],
      "batches": {
        "available": [
          {"id", "created_at", "n_rows", "churn_rate", "source",
           "used_in_versions"}
        ],
        "next_batch_number": 6
      }
    }

SECURITY: the HF token is only ever held in memory (passed to the constructor,
typically from ``.env``); it is never printed or logged.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download

REGISTRY_FILENAME = "registry.json"
REPO_TYPE = "model"

# Statuses a version may hold. Only "pending"/"rejected" versions may be promoted.
PROMOTABLE_STATUSES = ("pending", "rejected")


# === Exceptions ===============================================================


class RegistryError(Exception):
    """Base class for all registry errors."""


class RegistryConnectionError(RegistryError):
    """Raised when communication with HF Hub fails (up/download)."""


class VersionNotFoundError(RegistryError):
    """Raised when a requested version_id does not exist in the registry."""


class InvalidStateError(RegistryError):
    """Raised when an operation is invalid for the current registry state."""


# === Helpers ==================================================================


def _utcnow() -> str:
    """Return the current time as a UTC ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# === Registry =================================================================


class Registry:
    """Manages the model version registry on HuggingFace Hub.

    All WRITE operations are serialized through an internal lock and always
    re-read the latest registry from the Hub before mutating (to avoid
    clobbering concurrent changes), then sync the new state back and invalidate
    the in-memory cache so the next read is fresh.
    """

    def __init__(
        self,
        hf_token: str,
        hf_username: str,
        repo_id: str,
        cache_ttl: int = 60,
    ) -> None:
        """Initialize the registry client.

        Args:
            hf_token: HuggingFace API token (kept in memory only).
            hf_username: HF username (kept for reference / logging-free use).
            repo_id: Full repo id, e.g. ``"MonyetttRindam/telco-churn-models"``.
            cache_ttl: In-memory cache time-to-live in seconds.
        """
        self._hf_token = hf_token
        self._hf_username = hf_username
        self._repo_id = repo_id
        self._cache_ttl = cache_ttl

        self._api = HfApi(token=hf_token)
        self._lock = Lock()

        self._cache: Optional[dict] = None
        self._cache_time: float = 0.0

    # === READ operations ======================================================

    def load(self, force_refresh: bool = False) -> dict:
        """Load the registry from HF Hub (or the in-memory cache).

        Args:
            force_refresh: Bypass the cache and re-download from the Hub.

        Returns:
            The full registry dict.
        """
        now = time.monotonic()
        cache_fresh = (
            self._cache is not None and (now - self._cache_time) < self._cache_ttl
        )
        if not force_refresh and cache_fresh:
            return self._cache

        reg = self._load_from_hub()
        self._cache = reg
        self._cache_time = now
        return reg

    def get_active_version(self) -> str:
        """Return the active version_id."""
        return self.load()["active"]

    def get_previous_version(self) -> Optional[str]:
        """Return the previous version_id (rollback target), or None."""
        return self.load().get("previous")

    def get_version_info(self, version_id: str) -> dict:
        """Return the metadata dict for a specific version.

        Raises:
            VersionNotFoundError: if version_id is unknown.
        """
        return self._find_version(self.load(), version_id)

    def list_versions(self, status: Optional[str] = None) -> list:
        """List versions, optionally filtered by ``status``."""
        versions = self.load().get("versions", [])
        if status is None:
            return list(versions)
        return [v for v in versions if v.get("status") == status]

    def list_batches(self, only_unused: bool = False) -> list:
        """List available batches, optionally only those not yet used.

        A batch is "unused" when its ``used_in_versions`` list is empty.
        """
        batches = self.load().get("batches", {}).get("available", [])
        if not only_unused:
            return list(batches)
        return [b for b in batches if not b.get("used_in_versions")]

    def get_next_batch_number(self) -> int:
        """Return the next batch number for auto-naming (``batch_N``)."""
        return self.load().get("batches", {}).get("next_batch_number", 1)

    # === WRITE operations =====================================================

    def add_version(
        self,
        version_id: str,
        metrics: dict,
        batches_used: list,
        status: str,
        reason: str,
    ) -> None:
        """Add a new version to the registry.

        Raises:
            InvalidStateError: if version_id already exists.
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            if self._find_version(reg, version_id, required=False) is not None:
                raise InvalidStateError(f"Version '{version_id}' already exists")

            reg.setdefault("versions", []).append(
                {
                    "id": version_id,
                    "created_at": _utcnow(),
                    "metrics": metrics,
                    "batches_used": list(batches_used),
                    "status": status,
                    "reason": reason,
                }
            )
            self._cache = reg
            self._sync_to_hub(f"Add version {version_id} ({status})")
            self._invalidate()

    def promote_version(self, version_id: str) -> None:
        """Promote a version to active. The old active becomes ``archived``.

        The old active is recorded as ``previous`` so it can be rolled back to.

        Raises:
            VersionNotFoundError: if version_id is unknown.
            InvalidStateError: if the version's status is not promotable
                (must be "pending" or "rejected").
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            ver = self._find_version(reg, version_id)
            if ver.get("status") not in PROMOTABLE_STATUSES:
                raise InvalidStateError(
                    f"Version '{version_id}' has status '{ver.get('status')}'; "
                    f"only {PROMOTABLE_STATUSES} can be promoted"
                )

            old_active = reg.get("active")
            if old_active and old_active != version_id:
                old_ver = self._find_version(reg, old_active, required=False)
                if old_ver is not None:
                    old_ver["status"] = "archived"

            reg["previous"] = old_active
            reg["active"] = version_id
            ver["status"] = "active"

            self._cache = reg
            self._sync_to_hub(
                f"Promote {version_id} to active (from {old_active})"
            )
            self._invalidate()

    def reject_version(self, version_id: str) -> None:
        """Mark a version as ``rejected``. Does not change the active version.

        Raises:
            VersionNotFoundError: if version_id is unknown.
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            ver = self._find_version(reg, version_id)
            ver["status"] = "rejected"
            self._cache = reg
            self._sync_to_hub(f"Reject version {version_id}")
            self._invalidate()

    def rollback(self) -> str:
        """Swap ``active`` and ``previous``. Returns the new active version_id.

        Raises:
            InvalidStateError: if there is no previous version to roll back to.
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            previous = reg.get("previous")
            if not previous:
                raise InvalidStateError("Cannot rollback: no previous version")

            old_active = reg.get("active")
            reg["active"] = previous
            reg["previous"] = old_active

            new_active_ver = self._find_version(reg, previous, required=False)
            if new_active_ver is not None:
                new_active_ver["status"] = "active"
            if old_active:
                old_active_ver = self._find_version(reg, old_active, required=False)
                if old_active_ver is not None:
                    old_active_ver["status"] = "archived"

            self._cache = reg
            self._sync_to_hub(f"Rollback: swap {old_active} <-> {previous}")
            self._invalidate()
            return previous

    def add_batch(
        self,
        batch_id: str,
        n_rows: int,
        churn_rate: float,
        source: str,
    ) -> None:
        """Register a new batch (called after upload validation).

        Increments ``batches.next_batch_number``.

        Raises:
            InvalidStateError: if batch_id already exists.
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            batches = reg.setdefault(
                "batches", {"available": [], "next_batch_number": 1}
            )
            available = batches.setdefault("available", [])
            if any(b.get("id") == batch_id for b in available):
                raise InvalidStateError(f"Batch '{batch_id}' already exists")

            available.append(
                {
                    "id": batch_id,
                    "created_at": _utcnow(),
                    "n_rows": int(n_rows),
                    "churn_rate": float(churn_rate),
                    "source": source,
                    "used_in_versions": [],
                }
            )
            batches["next_batch_number"] = batches.get("next_batch_number", 1) + 1

            self._cache = reg
            self._sync_to_hub(f"Add batch {batch_id} ({source})")
            self._invalidate()

    def mark_batch_used(self, batch_id: str, in_version: str) -> None:
        """Record that ``batch_id`` was used in ``in_version``.

        Raises:
            InvalidStateError: if batch_id is unknown.
        """
        with self._lock:
            reg = self.load(force_refresh=True)
            available = reg.get("batches", {}).get("available", [])
            for batch in available:
                if batch.get("id") == batch_id:
                    used = batch.setdefault("used_in_versions", [])
                    if in_version not in used:
                        used.append(in_version)
                    break
            else:
                raise InvalidStateError(f"Batch '{batch_id}' not found")

            self._cache = reg
            self._sync_to_hub(f"Mark batch {batch_id} used in {in_version}")
            self._invalidate()

    # === Internal =============================================================

    def _find_version(
        self, reg: dict, version_id: str, required: bool = True
    ) -> Optional[dict]:
        """Return the version dict for ``version_id`` from ``reg``.

        Args:
            required: if True, raise VersionNotFoundError when missing;
                otherwise return None.
        """
        for ver in reg.get("versions", []):
            if ver.get("id") == version_id:
                return ver
        if required:
            raise VersionNotFoundError(f"Version '{version_id}' not found")
        return None

    def _invalidate(self) -> None:
        """Invalidate the in-memory cache so the next read re-fetches."""
        self._cache_time = 0.0

    def _sync_to_hub(self, commit_message: str) -> None:
        """Upload the current in-memory registry state to HF Hub.

        Raises:
            RegistryConnectionError: if the upload fails.
        """
        payload = json.dumps(self._cache, indent=2).encode("utf-8")
        try:
            self._api.upload_file(
                path_or_fileobj=payload,
                path_in_repo=REGISTRY_FILENAME,
                repo_id=self._repo_id,
                repo_type=REPO_TYPE,
                commit_message=commit_message,
            )
        except Exception as exc:  # noqa: BLE001 - normalize to our error type
            raise RegistryConnectionError(
                f"Failed to sync registry to hub: {exc}"
            ) from None

    def _load_from_hub(self) -> dict:
        """Download and parse ``registry.json`` from HF Hub.

        Always forces a fresh download (the TTL cache lives one layer up).

        Raises:
            RegistryConnectionError: if the download or parse fails.
        """
        try:
            local = hf_hub_download(
                self._repo_id,
                REGISTRY_FILENAME,
                repo_type=REPO_TYPE,
                token=self._hf_token,
                force_download=True,
            )
            return json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - normalize to our error type
            raise RegistryConnectionError(
                f"Failed to load registry from hub: {exc}"
            ) from None
