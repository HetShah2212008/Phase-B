"""
Filesystem path helpers.

Ensures data/, uploads/, and Chroma directories exist before services use them.
"""

from pathlib import Path

from config import BASE_DIR, get_settings


def ensure_app_directories() -> None:
    """
    Create standard directories if they are missing.

    Call from FastAPI lifespan or before ingesting documents.
    """
    settings = get_settings()
    paths = [
        BASE_DIR / "data",
        Path(settings.chroma_persist_dir),
        Path(settings.upload_dir),
        Path(settings.sqlite_db_path).parent,
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
