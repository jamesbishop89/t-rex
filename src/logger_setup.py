"""
Logger Setup Module.

Configure logging for the T-Rex reconciliation tool without clobbering
non-T-Rex handlers already attached to the process.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_TREX_HANDLER_FLAG = "_t_rex_managed"
_TREX_FILE_PATH_ATTR = "_t_rex_log_file"


def _build_formatter() -> logging.Formatter:
    """Create the shared formatter used by managed handlers."""
    return logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _is_trex_handler(handler: logging.Handler) -> bool:
    """Return whether the handler is managed by T-Rex logging setup."""
    return bool(getattr(handler, _TREX_HANDLER_FLAG, False))


def _configure_console_handler(root_logger: logging.Logger,
                               formatter: logging.Formatter,
                               numeric_level: int) -> None:
    """Add or update the single managed console handler."""
    console_handler = next(
        (
            handler for handler in root_logger.handlers
            if _is_trex_handler(handler)
            and isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ),
        None,
    )

    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        setattr(console_handler, _TREX_HANDLER_FLAG, True)
        root_logger.addHandler(console_handler)

    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)


def _configure_file_handler(root_logger: logging.Logger,
                            formatter: logging.Formatter,
                            numeric_level: int,
                            log_file: str) -> None:
    """Add or update the managed file handler for the requested path."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = str(log_path.resolve())

    for handler in list(root_logger.handlers):
        if not (_is_trex_handler(handler) and isinstance(handler, logging.FileHandler)):
            continue
        if getattr(handler, _TREX_FILE_PATH_ATTR, None) == resolved_path:
            continue
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = next(
        (
            handler for handler in root_logger.handlers
            if _is_trex_handler(handler)
            and isinstance(handler, logging.FileHandler)
            and getattr(handler, _TREX_FILE_PATH_ATTR, None) == resolved_path
        ),
        None,
    )

    if file_handler is None:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        setattr(file_handler, _TREX_HANDLER_FLAG, True)
        setattr(file_handler, _TREX_FILE_PATH_ATTR, resolved_path)
        root_logger.addHandler(file_handler)

    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)


def setup_logger(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for T-Rex application.

    This setup is idempotent for T-Rex-managed handlers and preserves unrelated
    handlers already attached to the root logger.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = _build_formatter()

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    _configure_console_handler(root_logger, formatter, numeric_level)

    if log_file:
        try:
            _configure_file_handler(root_logger, formatter, numeric_level, log_file)
            root_logger.info(f"File logging enabled: {log_file}")
        except Exception as exc:
            root_logger.warning(f"Could not setup file logging: {exc}")

    root_logger.info(f"Logging initialized at {logging.getLevelName(numeric_level)} level")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to provide logging capabilities to other classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        if not hasattr(self, '_logger'):
            class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(class_name)
        return self._logger
