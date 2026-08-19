"""
src/utils/logger.py

Central logging configuration for the AI Digital Twin project. Every
module that wants to log should call get_logger(__name__) rather than
configuring its own logger — this guarantees consistent formatting
and a single place to change logging behavior (e.g. adding a file
handler, changing the format, or changing the default level) for the
entire project.
"""

import logging
import sys

# Default format: timestamp, level, logger name (which module), message.
# Example output:
#   2026-08-18 14:03:12 | INFO     | src.ingestion.imd_loader | Dataset loaded: data/raw/rainfall.nc
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_DEFAULT_LEVEL = logging.INFO


def get_logger(name: str, level: int = _DEFAULT_LEVEL) -> logging.Logger:
    """
    Get a configured logger for a given module.

    Args:
        name: The logger's name — callers should always pass
            __name__, so the logger is named after the module that's
            using it (e.g. "src.ingestion.imd_loader"). This is what
            makes %(name)s in the log output tell you exactly which
            file a message came from.
        level: The minimum severity level this logger will emit.
            Defaults to logging.INFO, meaning DEBUG messages are
            suppressed unless a caller explicitly requests DEBUG.

    Returns:
        logging.Logger: A logger writing to stdout, with consistent
            timestamp/level/name/message formatting.

    Note:
        Uses logger.handlers to avoid attaching duplicate handlers if
        get_logger() is called more than once for the same name
        (e.g. if a module is reloaded, or get_logger(__name__) is
        called from multiple functions within the same module) —
        without this guard, log messages would be printed multiple
        times, once per attached handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent messages from also being handled by the root
        # logger, which would otherwise cause duplicate output if
        # anything else in the environment (e.g. a Jupyter kernel)
        # has its own root logging configured.
        logger.propagate = False

    return logger