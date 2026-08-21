"""
src/utils/logger.py

Central logging configuration for the AI Digital Twin project. Every
module that wants to log should call get_logger(__name__) rather than
configuring its own logger. Format, date format, and default level
are sourced from config/settings.py, so logging behavior for the
entire project changes in one place.
"""

import logging
import sys

from config.settings import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def get_logger(name: str, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Get a configured logger for a given module.

    Args:
        name: The logger's name — callers should always pass
            __name__.
        level: The minimum severity level this logger will emit.
            Defaults to config.settings.LOG_LEVEL.

    Returns:
        logging.Logger: A logger writing to stdout, with formatting
            sourced from config.settings.LOG_FORMAT /
            LOG_DATE_FORMAT.

    Note:
        Uses logger.handlers to avoid attaching duplicate handlers if
        get_logger() is called more than once for the same name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

    return logger