"""
Shared pipeline state schema for LangGraph-style orchestration.

LangGraph passes a single state object between nodes (agents). Each node
reads what it needs, updates its fields, and returns the state for the next node.

Part 2 (disaster system) can extend this schema with extra fields without
breaking Part 1 — add new optional keys here when that phase starts.
"""

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """
    Typed dictionary representing the full pipeline state.

    `total=False` means every key is optional at creation time; agents fill
    them in as the graph runs. This matches LangGraph's StateGraph pattern.
    """

    # User's original question or task description
    query: str

    # Text chunks retrieved from Pinecone by the RAG Agent
    retrieved_chunks: list[str]

    # Full retrieval hits (metadata + similarity) for APIs and debugging
    retrieval_details: list[dict[str, Any]]

    # Answer produced by the RAG Agent using retrieved context
    rag_response: str

    # Polished output from the Content Agent (reports, summaries, etc.)
    formatted_content: str

    # Target address for the Email Agent
    email_recipient: str

    # Result of send attempt: e.g. "sent", "failed", "skipped"
    email_status: str

    # Prophet forecasting output (dates, values, metadata) from ML module
    forecast_data: dict[str, Any]

    # Set to True when the RAG agent activates its Gemini-quota fallback path
    used_fallback_response: bool


def create_initial_state(query: str = "") -> PipelineState:
    """
    Build a fresh state dict with empty defaults.

    LangGraph pipelines and API handlers should start from this shape so every
    downstream agent knows which keys exist.
    """
    return PipelineState(
        query=query,
        retrieved_chunks=[],
        retrieval_details=[],
        rag_response="",
        formatted_content="",
        email_recipient="",
        email_status="",
        forecast_data={},
        used_fallback_response=False,
    )
