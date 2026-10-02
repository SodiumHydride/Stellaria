"""Errors that reach the UI as ``{code, message, detail}`` (``code`` is stable and localized)."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        detail: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail
        self.params = params or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail, "params": self.params}


class JobCancelled(Exception):
    """Raised inside a job function when the job has been cancelled."""


def not_found(kind: str, ident: str) -> ApiError:
    return ApiError(f"{kind}_not_found", f"{kind.capitalize()} '{ident}' not found", status=404)
