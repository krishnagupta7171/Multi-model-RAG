import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.routes.health import router
from src.api.deps import get_cache_dependency, get_settings_dependency


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_liveness_probe(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_check_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "dependencies" in data
    assert "version" in data


def test_health_check_with_overrides():
    app = FastAPI()
    app.include_router(router)

    mock_cache = MagicMock()
    mock_cache.is_available = True

    mock_settings = MagicMock()
    mock_settings.version = "1.0.0"

    app.dependency_overrides[get_cache_dependency] = lambda: mock_cache
    app.dependency_overrides[get_settings_dependency] = lambda: mock_settings

    test_client = TestClient(app)
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["dependencies"]["cache"] == "healthy"