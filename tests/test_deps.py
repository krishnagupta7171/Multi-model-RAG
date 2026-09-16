import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from src.api.deps import ( get_request_id,get_settings_dependency,verify_api_key,)


def test_get_request_id_from_header():
    req_id = get_request_id(x_request_id="client-req-999")
    assert req_id == "client-req-999"


def test_get_request_id_generated():
    req_id = get_request_id(x_request_id=None)
    assert req_id is not None
    assert len(req_id) > 10


def test_get_settings_dependency():
    settings = get_settings_dependency()
    assert settings is not None


@pytest.mark.asyncio
async def test_verify_api_key_dev_mode():
    mock_settings = MagicMock()
    mock_settings.is_production = False
    mock_settings.environment = "development"

    # In dev mode, missing key returns 'development' without error
    key = await verify_api_key(x_api_key=None, settings=mock_settings)
    assert key == "development"


@pytest.mark.asyncio
async def test_verify_api_key_prod_missing():
    mock_settings = MagicMock()
    mock_settings.is_production = True
    mock_settings.api_key = "secret_key_123"

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(x_api_key=None, settings=mock_settings)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_prod_valid():
    mock_settings = MagicMock()
    mock_settings.is_production = True
    mock_settings.api_key = "secret_key_123"

    key = await verify_api_key(x_api_key="secret_key_123", settings=mock_settings)
    assert key == "secret_key_123"