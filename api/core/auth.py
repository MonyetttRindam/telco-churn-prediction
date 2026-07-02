"""API key authentication for WRITE endpoints.

A single API key is stored in ``.env`` as ``MLOPS_API_KEY``. Clients pass it via
the ``X-API-Key`` header.

READ endpoints (status, history, batches, jobs list) stay public. WRITE
endpoints (upload-batch, retrain, rollback) depend on :func:`require_api_key`.

SECURITY: the key is compared in constant time (``hmac.compare_digest``) to
avoid leaking it through response-timing differences.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"

# auto_error=False -> returns None instead of raising, so we can distinguish
# "missing header" (401) from "wrong key" (403) ourselves.
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


# === Exceptions ===============================================================


class AuthError(Exception):
    """Base class for auth errors."""


class MissingAPIKeyError(AuthError):
    """Raised when the server hasn't configured MLOPS_API_KEY."""


# === Helpers ==================================================================


def get_expected_key() -> str:
    """Return the expected API key from the environment.

    Raises:
        MissingAPIKeyError: if MLOPS_API_KEY is not set.
    """
    key = os.getenv("MLOPS_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "MLOPS_API_KEY not configured in environment. "
            "Set it in .env or Railway env vars."
        )
    return key


def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string comparison to avoid timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


# === FastAPI dependency =======================================================


def require_api_key(provided: Optional[str] = Security(api_key_header)) -> str:
    """FastAPI dependency: validate the ``X-API-Key`` header.

    Usage::

        @app.post("/retrain")
        def retrain(_: str = Depends(require_api_key)):
            ...

    Returns:
        The validated key (usually unused, but available to the endpoint).

    Raises:
        HTTPException 401: header missing.
        HTTPException 403: header present but wrong.
        HTTPException 500: server misconfigured (no MLOPS_API_KEY).
    """
    try:
        expected = get_expected_key()
    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from None

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER_NAME} header",
        )

    if not _constant_time_eq(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return provided
