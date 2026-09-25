"""Thin entry point so `uvicorn main:app` keeps working after the module split."""
from api import app  # noqa: F401

__all__ = ["app"]