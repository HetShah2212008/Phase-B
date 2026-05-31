"""
Pydantic models for FastAPI request/response bodies.

Separate from PipelineState so HTTP contracts stay stable when graph state evolves.
"""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query body."""

    query: str = Field(..., min_length=1, description="User question for RAG")


class RetrievedChunkInfo(BaseModel):
    """One retrieved chunk with similarity metadata."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity: float = 0.0
    distance: float | None = None


class QueryResponse(BaseModel):
    """POST /query response."""

    answer: str
    retrieved_chunks: list[str]
    sources: list[RetrievedChunkInfo] = Field(
        default_factory=list,
        description="Chunks with metadata and similarity scores",
    )
    used_fallback_response: bool = Field(
        default=False,
        description="True when Gemini was unavailable and the answer was built from raw chunks",
    )


class UploadPdfResponse(BaseModel):
    """POST /upload-pdf response."""

    success: bool
    filename: str
    chunks_created: int
    message: str = ""
