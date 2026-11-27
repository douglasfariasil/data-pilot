import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    raw_sales_path: Path = PROJECT_ROOT / "data" / "raw" / "sales.csv"
    clean_sales_path: Path = PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"

    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip()
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free",
        ).strip()
    )

    app_title: str = "Data Pilot"


settings = Settings()
