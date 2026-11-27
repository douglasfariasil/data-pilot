from pathlib import Path

from data_pilot.config import PROJECT_ROOT, Settings


def test_settings_uses_default_values(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    settings = Settings()

    assert settings.raw_sales_path == PROJECT_ROOT / "data" / "raw" / "sales.csv"
    assert settings.clean_sales_path == PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"
    assert settings.openrouter_api_key == ""
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_model == "openrouter/free"
    assert settings.app_title == "Data Pilot"


def test_settings_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.com/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")

    settings = Settings()

    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_base_url == "https://example.com/api/v1"
    assert settings.openrouter_model == "test-model"


def test_settings_strips_environment_variable_whitespace(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  test-key  ")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "  https://example.com/api/v1  ")
    monkeypatch.setenv("OPENROUTER_MODEL", "  test-model  ")

    settings = Settings()

    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_base_url == "https://example.com/api/v1"
    assert settings.openrouter_model == "test-model"


def test_settings_accepts_custom_values():
    settings = Settings(
        raw_sales_path=Path("/tmp/raw.csv"),
        clean_sales_path=Path("/tmp/clean.csv"),
        openrouter_api_key="custom-key",
        openrouter_base_url="https://custom.example.com/api/v1",
        openrouter_model="custom-model",
    )

    assert settings.raw_sales_path == Path("/tmp/raw.csv")
    assert settings.clean_sales_path == Path("/tmp/clean.csv")
    assert settings.openrouter_api_key == "custom-key"
    assert settings.openrouter_base_url == "https://custom.example.com/api/v1"
    assert settings.openrouter_model == "custom-model"


def test_settings_is_immutable():
    settings = Settings(openrouter_api_key="test-key")

    try:
        settings.openrouter_api_key = "another-key"
    except AttributeError:
        pass
    else:
        raise AssertionError("Settings deveria ser imutável.")
