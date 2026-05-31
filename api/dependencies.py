"""
FastAPI dependency injection for shared services and agents.

Using dependencies keeps routes thin and makes unit tests easy (override deps).
"""

from functools import lru_cache

from agents.rag_agent import RAGAgent
from services.document_service import DocumentService
from services.gemini_service import GeminiService
from services.vector_service import VectorService


@lru_cache
def get_gemini_service() -> GeminiService:
    return GeminiService()


@lru_cache
def get_vector_service() -> VectorService:
    # First call loads the embedding model — may take a few seconds
    return VectorService()


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService()


def get_rag_agent() -> RAGAgent:
    return RAGAgent(
        gemini=get_gemini_service(),
        vector=get_vector_service(),
    )
