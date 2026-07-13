"""
utils/logger.py
===============
Centralised logging configuration for the project.

Creates a root logger that simultaneously writes to:
    - The console (colourised via ``rich`` if available)
    - A rotating file handler (optional, configured via YAML)

Usage
-----
>>> from utils.logger import get_logger, setup_logging
>>> setup_logging(log_level="INFO", log_file="logs/train.log")
>>> logger = get_logger(__name__)
>>> logger.info("Training started.")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Rich console handler (optional dependency)
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per log file
_BACKUP_COUNT = 5               # Keep 5 rotated files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_file: bool = True,
) -> None:
    """Configure the root logger for the project.

    This function is idempotent: calling it multiple times will not add
    duplicate handlers.

    Parameters
    ----------
    log_level : str
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.  Defaults to ``"INFO"``.
    log_file : Optional[str]
        Absolute or relative path to the log file.  If ``None`` and
        ``log_to_file`` is ``True``, defaults to ``"logs/train.log"``.
    log_to_file : bool
        Whether to attach a rotating file handler.  Defaults to ``True``.

    Examples
    --------
    >>> setup_logging(log_level="DEBUG", log_file="logs/debug.log")
    """
    root_logger = logging.getLogger()
    numeric_level = _parse_level(log_level)

    # Guard against duplicate handler registration
    if root_logger.handlers:
        return

    root_logger.setLevel(numeric_level)

    # Console handler
    root_logger.addHandler(_build_console_handler(numeric_level))

    # File handler (optional)
    if log_to_file:
        resolved_path = log_file or "logs/train.log"
        root_logger.addHandler(_build_file_handler(resolved_path, numeric_level))

    root_logger.info(
        "Logging initialised — level=%s, file=%s",
        log_level,
        log_file if log_to_file else "disabled",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger inheriting from the root logger.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.debug("Debug message.")
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_level(level_str: str) -> int:
    """Convert a level string to a ``logging`` integer constant."""
    level = getattr(logging, level_str.upper(), None)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid log level: '{level_str}'. "
            "Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return level


def _build_console_handler(level: int) -> logging.Handler:
    """Build a console handler, using ``rich`` when available."""
    if _RICH_AVAILABLE:
        handler = RichHandler(
            level=level,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True,
        )
        # Rich formats its own output; suppress the default formatter
        handler.setFormatter(logging.Formatter(datefmt=_DATE_FORMAT))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )
    return handler


def _build_file_handler(log_file: str, level: int) -> logging.Handler:
    """Build a ``RotatingFileHandler`` for the given path."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    return handler
