# battery_dashboard/utils/logging.py
import logging
import structlog
from typing import Any, Dict
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup structured logging configuration"""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if not log_file else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file)])
        ]
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    # Ensure the named logger uses the same configuration as root
    logger = logging.getLogger(name)

    # If no handlers, inherit from root
    if not logger.handlers:
        logger.handlers = logging.getLogger().handlers
        logger.setLevel(logging.getLogger().level)

    return structlog.get_logger(name)







