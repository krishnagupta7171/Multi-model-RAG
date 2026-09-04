
import sys

from pathlib import Path
from typing import Optional

from loguru import logger

from ..utils.config import Settings, get_settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    
    if settings is None:
        settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )

    # File handler with JSON format for production
    if settings.is_production:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger.add(
            log_dir / "rag_system_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level=settings.log_level,
            serialize=True,
        )

    # Add request ID context
    logger.configure(
        patcher=lambda record: record.update(
            {
                "request_id": record.get("extra", {})
                .get("request_id", "N/A"),
                "environment": settings.environment,
            }
        )
    )


def get_logger(name: str):
    
    return logger.bind(module=name)