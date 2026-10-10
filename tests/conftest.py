"""Shared pytest fixtures for the DiabetaBot test suite."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def app() -> FastAPI:
    """Return a fresh application instance per test."""
    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a test client wired to the application under test."""
    with TestClient(app) as test_client:
        yield test_client
