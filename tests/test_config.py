"""
Tests for config.py — centralized configuration management.

Covers:
    - Settings singleton loads correctly
    - All expected settings exist with correct types
    - Default values match documented expectations
    - Paths resolve to real directories
    - Environment variable override works
"""

import os
import pytest
from pathlib import Path

from config import Settings, settings


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT VALUES
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaults:
    """The singleton `settings` should have the correct defaults."""

    def test_openai_model_default(self):
        assert settings.OPENAI_MODEL == "gpt-4o-mini"

    def test_openai_temperature_default(self):
        assert settings.OPENAI_TEMPERATURE == 0.7

    def test_openai_max_tokens_default(self):
        assert settings.OPENAI_MAX_TOKENS == 600

    def test_test_size_default(self):
        assert settings.TEST_SIZE == 0.3

    def test_random_state_default(self):
        assert settings.RANDOM_STATE == 42

    def test_cv_folds_default(self):
        assert settings.CV_FOLDS == 5

    def test_shap_top_n_default(self):
        assert settings.SHAP_TOP_N == 5

    def test_app_title_default(self):
        assert settings.APP_TITLE == "ChurnGuard AI"

    def test_app_icon_default(self):
        assert settings.APP_ICON == "🛡️"

    def test_log_level_default(self):
        assert settings.LOG_LEVEL == "INFO"

    def test_log_dir_default(self):
        assert str(settings.LOG_DIR).endswith("logs")


# ═══════════════════════════════════════════════════════════════════════════
#  TYPES
# ═══════════════════════════════════════════════════════════════════════════


class TestTypes:
    """Ensure settings have the correct Python types."""

    def test_openai_model_is_str(self):
        assert isinstance(settings.OPENAI_MODEL, str)

    def test_openai_temperature_is_float(self):
        assert isinstance(settings.OPENAI_TEMPERATURE, float)

    def test_openai_max_tokens_is_int(self):
        assert isinstance(settings.OPENAI_MAX_TOKENS, int)

    def test_data_path_is_path(self):
        assert isinstance(settings.DATA_PATH, Path)

    def test_models_dir_is_path(self):
        assert isinstance(settings.MODELS_DIR, Path)

    def test_test_size_is_float(self):
        assert isinstance(settings.TEST_SIZE, float)

    def test_random_state_is_int(self):
        assert isinstance(settings.RANDOM_STATE, int)

    def test_log_level_is_str(self):
        assert isinstance(settings.LOG_LEVEL, str)

    def test_log_dir_is_path(self):
        assert isinstance(settings.LOG_DIR, Path)


# ═══════════════════════════════════════════════════════════════════════════
#  PATH VALIDITY
# ═══════════════════════════════════════════════════════════════════════════


class TestPaths:
    """Default paths should point to real locations in the project."""

    def test_data_path_exists(self):
        assert settings.DATA_PATH.exists(), (
            f"DATA_PATH does not exist: {settings.DATA_PATH}"
        )

    def test_models_dir_exists(self):
        assert settings.MODELS_DIR.exists(), (
            f"MODELS_DIR does not exist: {settings.MODELS_DIR}"
        )

    def test_data_path_is_csv(self):
        assert settings.DATA_PATH.suffix == ".csv"


# ═══════════════════════════════════════════════════════════════════════════
#  VALUE CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════


class TestConstraints:
    """Settings should have values within valid ranges."""

    def test_temperature_range(self):
        assert 0.0 <= settings.OPENAI_TEMPERATURE <= 2.0

    def test_max_tokens_positive(self):
        assert settings.OPENAI_MAX_TOKENS > 0

    def test_test_size_between_0_and_1(self):
        assert 0.0 < settings.TEST_SIZE < 1.0

    def test_cv_folds_at_least_2(self):
        assert settings.CV_FOLDS >= 2

    def test_log_level_is_valid(self):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert settings.LOG_LEVEL.upper() in valid

    def test_shap_top_n_positive(self):
        assert settings.SHAP_TOP_N > 0


# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvOverride:
    """Environment variables should override defaults (12-factor pattern)."""

    def test_model_override(self, monkeypatch):
        """Setting OPENAI_MODEL env var should override the default."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        fresh = Settings()
        assert fresh.OPENAI_MODEL == "gpt-4o"

    def test_temperature_override(self, monkeypatch):
        """Setting OPENAI_TEMPERATURE env var should override and cast to float."""
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
        fresh = Settings()
        assert fresh.OPENAI_TEMPERATURE == 0.2

    def test_shap_top_n_override(self, monkeypatch):
        """Setting SHAP_TOP_N env var should override and cast to int."""
        monkeypatch.setenv("SHAP_TOP_N", "10")
        fresh = Settings()
        assert fresh.SHAP_TOP_N == 10

    def test_case_insensitive(self, monkeypatch):
        """Env vars should work regardless of case."""
        monkeypatch.setenv("openai_model", "gpt-3.5-turbo")
        fresh = Settings()
        assert fresh.OPENAI_MODEL == "gpt-3.5-turbo"
