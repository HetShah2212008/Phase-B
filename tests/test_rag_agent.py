"""Unit tests for RAGAgent with mocked services (no real API calls)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.rag_agent import RAGAgent, _FALLBACK_PREFIX
from schemas.pipeline_state import create_initial_state


@pytest.mark.asyncio
async def test_rag_agent_retrieves_and_generates():
    mock_vector = MagicMock()
    mock_vector.similarity_search.return_value = [
        {
            "id": "1",
            "document": "The capital of France is Paris.",
            "metadata": {"page": 1},
            "distance": 0.1,
            "similarity": 0.9,
        }
    ]

    mock_gemini = MagicMock()
    mock_gemini.generate_rag_response = AsyncMock(
        return_value="Paris is the capital of France."
    )

    agent = RAGAgent(gemini=mock_gemini, vector=mock_vector)
    state = create_initial_state(query="What is the capital of France?")

    result = await agent.run(state)

    mock_vector.similarity_search.assert_called_once()
    mock_gemini.generate_rag_response.assert_awaited_once()
    call_kwargs = mock_gemini.generate_rag_response.await_args.kwargs
    assert call_kwargs["query"] == "What is the capital of France?"
    assert call_kwargs["context_chunks"] == ["The capital of France is Paris."]

    assert result["rag_response"] == "Paris is the capital of France."
    assert result["retrieved_chunks"] == ["The capital of France is Paris."]
    assert len(result["retrieval_details"]) == 1
    # Successful generation must not set the fallback flag
    assert result["used_fallback_response"] is False


@pytest.mark.asyncio
async def test_rag_agent_empty_query():
    agent = RAGAgent(gemini=MagicMock(), vector=MagicMock())
    state = create_initial_state(query="   ")

    result = await agent.run(state)

    assert result["rag_response"] == ""
    assert result["retrieved_chunks"] == []
    assert result["used_fallback_response"] is False


@pytest.mark.asyncio
async def test_rag_agent_fallback_on_gemini_error():
    """
    When Gemini raises ANY exception, the agent must:
    - return HTTP-200-safe state (no re-raise)
    - set used_fallback_response = True
    - include the fallback prefix in rag_response
    - preserve retrieved_chunks from the vector search
    """
    mock_vector = MagicMock()
    mock_vector.similarity_search.return_value = [
        {
            "id": "1",
            "document": "FastAPI is a modern Python web framework.",
            "metadata": {"page": 0},
            "distance": 0.15,
            "similarity": 0.85,
        }
    ]

    mock_gemini = MagicMock()
    mock_gemini.generate_rag_response = AsyncMock(
        side_effect=RuntimeError("Gemini RAG API error: 429 RESOURCE_EXHAUSTED")
    )

    agent = RAGAgent(gemini=mock_gemini, vector=mock_vector)
    state = create_initial_state(query="What is FastAPI?")

    result = await agent.run(state)

    # Must not raise — fallback activates silently
    assert result["used_fallback_response"] is True
    assert _FALLBACK_PREFIX in result["rag_response"]
    # Chunks must still be present for the sources section of the API response
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0] == "FastAPI is a modern Python web framework."


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    RuntimeError("Gemini RAG API error: API key not valid"),
    RuntimeError("Gemini RAG API error: models/bad-model not found"),
    ValueError("The response was blocked due to safety settings"),
    AttributeError("object has no attribute 'text'"),
    ConnectionError("network unreachable"),
    Exception("unexpected SDK error"),
])
async def test_rag_agent_fallback_on_any_gemini_exception(exc):
    """Every exception type from Gemini must activate fallback, never propagate."""
    mock_vector = MagicMock()
    mock_vector.similarity_search.return_value = [
        {
            "id": "1",
            "document": "Some retrieved document text.",
            "metadata": {},
            "distance": 0.2,
            "similarity": 0.8,
        }
    ]

    mock_gemini = MagicMock()
    mock_gemini.generate_rag_response = AsyncMock(side_effect=exc)

    agent = RAGAgent(gemini=mock_gemini, vector=mock_vector)
    state = create_initial_state(query="test query")

    result = await agent.run(state)

    assert result["used_fallback_response"] is True
    assert _FALLBACK_PREFIX in result["rag_response"]
    assert len(result["retrieved_chunks"]) > 0
