"""Convenience entrypoint so `uvicorn api:app` works from the repository root."""

from src.api import app

__all__ = ["app"]
