"""Logging configuration and utilities using structlog"""

import logging
import sys
from typing import Optional

import structlog


def setup_logging(
    name: str, level: int = logging.INFO, log_file: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Configure and return a structured logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs

    Returns:
        Configured structlog logger instance
    """
    # Configure standard logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get or create a structured logger with standard naming convention.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structlog logger instance
    """
    return structlog.get_logger(f"finetune.{name}")
