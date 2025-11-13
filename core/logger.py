from __future__ import annotations
import logging
import sys

_LOGGER_NAME = "adcs"


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _configure_logger()


def log(msg: str, *args, **kwargs) -> None:
    """Info-level log with timestamp."""
    logger.info(msg, *args, **kwargs)


def warn(msg: str, *args, **kwargs) -> None:
    """Warning-level log with timestamp."""
    logger.warning(msg, *args, **kwargs)
