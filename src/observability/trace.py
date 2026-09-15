import os
from typing import Any, Dict, Optional

from ..observability.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)


def setup_langsmith() -> bool:
    
    settings = get_settings()

    tracing_enabled = getattr(settings, "langsmith_tracing", False)
    api_key = getattr(settings, "langsmith_api_key", None)
    project = getattr(settings, "langsmith_project", "default-rag")

    if tracing_enabled and api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project

        logger.info(f"LangSmith tracing enabled for project: {project}")
        return True
    else:
        logger.info("LangSmith tracing disabled")
        return False


def trace_event(event_name: str,metadata: Optional[Dict[str, Any]] = None,) -> None:
   
    extra_data = metadata or {}
    logger.debug(f"Trace event: {event_name}", extra=extra_data)


setup_langsmith()