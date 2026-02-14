"""
Logging Setup
=============
Configures Python's built-in logging for the entire project.

Usage
-----
    from log_config import get_logger
    logger = get_logger(__name__)

    logger.info("Model loaded in %.2fs", elapsed)
    logger.warning("SHAP failed, using manual fallback")
    logger.error("Failed to load model: %s", err)

What happens
------------
- Every log message goes to the **console** (coloured by level).
- Every log message also goes to **logs/app.log** (rotated at 5 MB, 3 backups).
- The log level is controlled by `settings.LOG_LEVEL` (default: INFO).
- In production, set LOG_LEVEL=WARNING to only see problems.
"""

import logging
from logging.handlers import RotatingFileHandler

from config import settings


def _setup_logging() -> None:
    """Configure the root logger once (idempotent)."""

    root = logging.getLogger()

    # Avoid duplicate handlers if our setup already ran.
    # Check for our specific handler (not just any handler, because
    # pytest or other tools may add their own before us).
    if any(getattr(h, "_churnguard", False) for h in root.handlers):
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root.setLevel(level)

    # ── Format ──────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    console._churnguard = True  # type: ignore[attr-defined]
    root.addHandler(console)

    # ── File handler (rotating) ─────────────────────────────────────────────
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,               # keep 3 old files (app.log.1, .2, .3)
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    file_handler._churnguard = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)


# Run setup on import
_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Parameters
    ----------
    name : str
        Usually ``__name__`` — e.g. ``"app.churn_service"``

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
