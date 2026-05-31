"""
Filesystem path helpers.

Resolves storage directories at import time so every module that needs
UPLOADS_DIR can import it directly.

Priority:
  1. RENDER_PERSISTENT_DIR env var — set this on Render.com to a persistent
     disk mount path (e.g. /var/data). Uploads will live there so they
     survive deploys.
  2. Project directory fallback — used locally and in any environment that
     does not set RENDER_PERSISTENT_DIR.

Exported constants:
  UPLOADS_DIR — Path to the PDF uploads directory
"""

import os
from pathlib import Path

# ── Resolve base directory ────────────────────────────────────────────────────
# On Render (or any cloud with a persistent disk), set RENDER_PERSISTENT_DIR
# to the mount path. Locally it is unset and we fall back to the project root.

_render_base = os.getenv("RENDER_PERSISTENT_DIR", "").strip()

if _render_base:
    BASE_STORAGE = Path(_render_base)
else:
    # Fall back to the backend/ directory (parent of this file's parent)
    BASE_STORAGE = Path(__file__).resolve().parent.parent

# ── Resolved paths ────────────────────────────────────────────────────────────

UPLOADS_DIR = BASE_STORAGE / "uploads"

# ── Auto-create on import ─────────────────────────────────────────────────────

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper (called from FastAPI lifespan) ─────────────────────────────────────

def ensure_app_directories() -> None:
    """
    Ensure all application directories exist.

    UPLOADS_DIR is already created at import time above.
    This function handles any additional directories (sqlite, etc.) that
    config.py may reference, and is safe to call multiple times.
    """
    from config import BASE_DIR, get_settings

    settings = get_settings()

    extra_paths = [
        BASE_DIR / "data",
        Path(settings.sqlite_db_path).parent,
    ]
    for path in extra_paths:
        path.mkdir(parents=True, exist_ok=True)

    # Re-ensure uploads dir in case it was deleted at runtime
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
