"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from app.core.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime
