"""
Centralized configuration for the Part 1 backend.

All environment variables are loaded once at startup via pydantic-settings.
Copy `.env.example` to `.env` and fill in your real API keys before running.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: backend/ directory (parent of this file)
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a `.env` file.

    Each field maps to one config value. Defaults are safe for local dev
    but you must set secrets (API keys, SMTP password) in production.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = Field(default="AI Agent Pipeline", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # --- CORS (comma-separated origins in .env) ---
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    # --- Google Gemini ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_rag_temperature: float = Field(default=0.2, alias="GEMINI_RAG_TEMPERATURE")
    # Content Agent uses a separate model entry so RAG and formatting can be tuned independently
    gemini_content_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_CONTENT_MODEL")

    # --- RAG retrieval ---
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # --- Pinecone ---
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="phaseb", alias="PINECONE_INDEX")

    # --- Documents ---
    upload_dir: str = Field(default=str(BASE_DIR / "uploads"), alias="UPLOAD_DIR")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # --- Gmail SMTP (placeholders until Email Agent is wired) ---
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from_email: str = Field(default="", alias="SMTP_FROM_EMAIL")

    # --- SQLite (reserved for Part 1 persistence / Part 2 expansion) ---
    sqlite_db_path: str = Field(
        default=str(BASE_DIR / "data" / "app.db"),
        alias="SQLITE_DB_PATH",
    )

    # --- News API (Part 2 — Disaster Management) ---
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list for FastAPI middleware."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """
  Return a cached Settings instance (singleton pattern).

  Use `get_settings()` everywhere instead of creating Settings() repeatedly.
  """
    return Settings()
