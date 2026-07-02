"""Thread-safe model + preprocessor manager for FastAPI.

Loads the active model from HF Hub based on ``registry.active``, keeps it in
memory, and allows an atomic swap when a new version is promoted.

Design:

* ``/predict`` is hot — it reads the in-memory (model, preprocessor) once per
  request under a shared lock and runs inference; no HF Hub round-trip.
* ``/retrain`` / rollback are rare — they call :meth:`ModelManager.swap_to`,
  which downloads the new artifacts *before* taking the lock, then swaps the
  three references (model / preprocessor / version_id) in one short critical
  section. In-flight requests that already grabbed the old references keep
  running safely (Python object references stay valid; the GIL makes the
  attribute rebinding atomic).

HF Hub storage convention (Registry Reference strategy)::

    models/{version_id}/model.pkl
    models/{version_id}/preprocessor.pkl

SECURITY: the HF token lives in memory only; never printed or logged.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Optional

import joblib
from huggingface_hub import hf_hub_download

from api.core.registry import Registry

REPO_TYPE = "model"


# === Exceptions ===============================================================


class ModelManagerError(Exception):
    """Base class for all model-manager errors."""


class ModelNotLoadedError(ModelManagerError):
    """Raised when the current model is requested before it has been loaded."""


class ModelDownloadError(ModelManagerError):
    """Raised when downloading/deserializing artifacts from HF Hub fails."""


# === ModelManager =============================================================


class ModelManager:
    """Holds the active model + preprocessor in memory, swappable atomically."""

    def __init__(
        self,
        hf_token: str,
        hf_username: str,
        repo_id: str,
        registry: Registry,
    ) -> None:
        """Initialize the manager.

        Args:
            hf_token: HuggingFace API token (kept in memory only).
            hf_username: HF username.
            repo_id: Full repo id, e.g. ``"MonyetttRindam/telco-churn-models"``.
            registry: Shared :class:`Registry` instance (composition — the
                manager never re-implements registry logic).
        """
        self._hf_token = hf_token
        self._hf_username = hf_username
        self._repo_id = repo_id
        self._registry = registry

        # RLock: swap_to may read state (e.g. current version) while holding it.
        self._lock = RLock()

        self._model: Optional[object] = None
        self._preprocessor: Optional[object] = None
        self._version_id: Optional[str] = None

    # === Path convention ======================================================

    def _hf_model_path(self, version_id: str) -> str:
        return f"models/{version_id}/model.pkl"

    def _hf_preprocessor_path(self, version_id: str) -> str:
        return f"models/{version_id}/preprocessor.pkl"

    # === Download helper ======================================================

    def _download_artifacts(self, version_id: str) -> tuple[object, object]:
        """Download + deserialize (model, preprocessor) for ``version_id``.

        Runs outside the swap lock so a slow download never blocks readers.

        Raises:
            ModelDownloadError: if any file is missing or fails to load.
        """
        try:
            model_local = hf_hub_download(
                self._repo_id,
                self._hf_model_path(version_id),
                repo_type=REPO_TYPE,
                token=self._hf_token,
                force_download=True,
            )
            preproc_local = hf_hub_download(
                self._repo_id,
                self._hf_preprocessor_path(version_id),
                repo_type=REPO_TYPE,
                token=self._hf_token,
                force_download=True,
            )
            model = joblib.load(model_local)
            preprocessor = joblib.load(preproc_local)
        except Exception as exc:  # noqa: BLE001 - normalize to our error type
            raise ModelDownloadError(
                f"Failed to download artifacts for '{version_id}': {exc}"
            ) from None
        return model, preprocessor

    # === Public API ===========================================================

    def load_active(self) -> None:
        """Load the active model + preprocessor from HF Hub.

        Reads ``registry.active``, downloads that version's artifacts, and
        stores them in memory. Called once at FastAPI startup.

        Raises:
            ModelDownloadError: if the active version's artifacts can't load.
        """
        version_id = self._registry.get_active_version()
        model, preprocessor = self._download_artifacts(version_id)
        with self._lock:
            self._model = model
            self._preprocessor = preprocessor
            self._version_id = version_id

    def get_current(self) -> tuple[object, object, str]:
        """Return (model, preprocessor, version_id) — thread-safe.

        Raises:
            ModelNotLoadedError: if no model has been loaded yet.
        """
        with self._lock:
            if self._model is None or self._preprocessor is None:
                raise ModelNotLoadedError(
                    "Model not loaded. Call load_active() first."
                )
            return self._model, self._preprocessor, self._version_id

    def get_current_version_id(self) -> str:
        """Return the version_id currently active in memory.

        Raises:
            ModelNotLoadedError: if no model has been loaded yet.
        """
        with self._lock:
            if self._version_id is None:
                raise ModelNotLoadedError(
                    "Model not loaded. Call load_active() first."
                )
            return self._version_id

    def swap_to(self, version_id: str) -> None:
        """Atomically swap the in-memory model to ``version_id``.

        Downloads the new artifacts first (outside the lock), then swaps the
        three references in one short critical section. If the download fails,
        the current model is left untouched.

        Raises:
            ModelDownloadError: if the new version's artifacts can't load.
        """
        # 1. Download to temporaries (may be slow; not holding the lock).
        model, preprocessor = self._download_artifacts(version_id)
        # 2-4. Swap under the lock (fast; readers see old-or-new, never partial).
        with self._lock:
            self._model = model
            self._preprocessor = preprocessor
            self._version_id = version_id
