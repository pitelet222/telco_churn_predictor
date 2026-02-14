"""
Tests for log_config.py — logging setup and get_logger utility.

Covers:
    - get_logger returns a Logger
    - Logger name matches module name
    - Root logger has handlers (console + file)
    - Log file is created in LOG_DIR
    - Log messages are written to file
    - Third-party loggers are suppressed
"""

import logging
from pathlib import Path

import pytest

from config import settings
from log_config import get_logger


# ═══════════════════════════════════════════════════════════════════════════
#  get_logger TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestGetLogger:
    """Tests for the get_logger() factory function."""

    def test_returns_logger_instance(self):
        """get_logger must return a logging.Logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        """Logger name should match the argument."""
        logger = get_logger("my.custom.module")
        assert logger.name == "my.custom.module"

    def test_different_names_give_different_loggers(self):
        """Each module gets its own logger instance."""
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a is not b
        assert a.name != b.name


# ═══════════════════════════════════════════════════════════════════════════
#  ROOT LOGGER SETUP
# ═══════════════════════════════════════════════════════════════════════════


class TestRootLogger:
    """Verify the root logger was configured correctly on import."""

    def test_has_churnguard_handlers(self):
        """Root logger should have our custom handlers (tagged _churnguard)."""
        root = logging.getLogger()
        tagged = [h for h in root.handlers if getattr(h, "_churnguard", False)]
        assert len(tagged) >= 2, f"Expected >=2 tagged handlers, got {len(tagged)}"

    def test_has_stream_handler(self):
        """One tagged handler should be a StreamHandler (console)."""
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and getattr(h, "_churnguard", False)
        ]
        assert len(stream_handlers) >= 1

    def test_has_file_handler(self):
        """One tagged handler should be a file-based handler."""
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler)
            and getattr(h, "_churnguard", False)
        ]
        assert len(file_handlers) >= 1

    def test_log_level_at_most_info(self):
        """Root log level should be INFO or lower (i.e. more verbose)."""
        root = logging.getLogger()
        # Our setup sets INFO; root.level may be overridden but
        # effective level should allow INFO messages through.
        assert root.getEffectiveLevel() <= logging.INFO


# ═══════════════════════════════════════════════════════════════════════════
#  LOG FILE
# ═══════════════════════════════════════════════════════════════════════════


class TestLogFile:
    """Verify log messages are written to disk."""

    def test_log_dir_exists(self):
        """LOG_DIR should be created automatically."""
        assert settings.LOG_DIR.exists()

    def test_log_file_exists(self):
        """app.log should exist in LOG_DIR."""
        log_file = settings.LOG_DIR / "app.log"
        assert log_file.exists()

    def test_messages_written_to_file(self):
        """Writing a log message should appear in the file."""
        logger = get_logger("test_file_write")
        marker = "TEST_LOG_MARKER_abc123"
        logger.info(marker)

        # Flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_file = settings.LOG_DIR / "app.log"
        content = log_file.read_text(encoding="utf-8")
        assert marker in content


# ═══════════════════════════════════════════════════════════════════════════
#  THIRD-PARTY SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════


class TestThirdPartySuppression:
    """Noisy third-party loggers should be suppressed to WARNING+."""

    @pytest.mark.parametrize("lib_name", ["httpx", "openai", "shap"])
    def test_suppressed_to_warning(self, lib_name):
        """Third-party loggers' effective level should be WARNING or higher."""
        lib_logger = logging.getLogger(lib_name)
        assert lib_logger.getEffectiveLevel() >= logging.WARNING
