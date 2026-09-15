import os
from unittest.mock import MagicMock, patch

from src.observability.trace import setup_langsmith, trace_event


def test_setup_langsmith_enabled():
    mock_settings = MagicMock()
    mock_settings.langsmith_tracing = True
    mock_settings.langsmith_api_key = "lsv2_pt_testkey_123"
    mock_settings.langsmith_project = "rag-eval-suite"

    with patch("src.observability.trace.get_settings", return_value=mock_settings):
        # Clear env to ensure clean run
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)

        status = setup_langsmith()

        assert status is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_pt_testkey_123"
        assert os.environ.get("LANGCHAIN_PROJECT") == "rag-eval-suite"


def test_setup_langsmith_disabled():
    mock_settings = MagicMock()
    mock_settings.langsmith_tracing = False
    mock_settings.langsmith_api_key = None

    with patch("src.observability.trace.get_settings", return_value=mock_settings):
        status = setup_langsmith()
        assert status is False


def test_trace_event_logging():
    with patch("src.observability.trace.logger.debug") as mock_debug:
        trace_event("query_retrieval_start", {"query": "test", "top_k": 5})
        mock_debug.assert_called_once_with(
            "Trace event: query_retrieval_start",
            extra={"query": "test", "top_k": 5},
        )