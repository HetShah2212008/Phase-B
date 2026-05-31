"""
FastAPI application entry point for Part 1 — AI Agent Pipeline.

Run locally:
  cd backend
  uvicorn main:app --reload

Endpoints will grow as agents and pipelines are implemented.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.disaster_routes import router as disaster_router
from api.routes.pipeline_routes import router as pipeline_router
from api.routes.rag_routes import router as rag_router
from config import get_settings
from utils.paths import ensure_app_directories

# Configure logging once at import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown hooks.

    Use this to warm up DB connections, Chroma, or shared service instances
    when you add heavier initialization in later milestones.
    """
    settings = get_settings()
    ensure_app_directories()
    logger.info("Starting %s (debug=%s)", settings.app_name, settings.debug)
    yield
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """Application factory — keeps tests able to import a fresh app instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Part 1 foundation: RAG, Content, Email agents with LangGraph and ADK pipelines.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS: allow frontend dev servers to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RAG endpoints: /upload-pdf, /query
    app.include_router(rag_router)

    # Disaster Management endpoints: /api/disaster/run, /api/disaster/review, etc.
    app.include_router(disaster_router, prefix="/api")

    # Pipeline orchestration endpoints: /api/pipeline/langgraph, /api/pipeline/adk
    app.include_router(pipeline_router, prefix="/api")

    # Static files + frontend UI
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    """Attach all API routes to the FastAPI instance."""

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, Any]:
        """
        Liveness probe for Docker, Kubernetes, or manual checks.

        Returns service name and whether Gemini key is configured (not the key itself).
        """
        settings = get_settings()
        return {
            "status": "healthy",
            "app": settings.app_name,
            "gemini_configured": bool(settings.gemini_api_key),
            "smtp_configured": bool(settings.smtp_user and settings.smtp_password),
        }

    @app.get("/", tags=["System"])
    async def root() -> dict[str, str]:
        """Simple welcome route pointing to docs and health."""
        return {
            "message": "AI Agent Pipeline API — Part 1 RAG enabled",
            "docs": "/docs",
            "health": "/health",
            "upload_pdf": "/upload-pdf",
            "query": "/query",
        }

    @app.get("/ui", tags=["System"])
    async def serve_frontend() -> FileResponse:
        """Serve the dashboard frontend."""
        return FileResponse("static/index.html")


# Uvicorn imports `app` from this module
app = create_app()

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
