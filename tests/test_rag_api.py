"""
API tests for RAG endpoints with dependency overrides.

These tests do NOT call Gemini or load sentence-transformers — they mock
services to verify routing, request validation, and response shape.

Manual integration test (real stack):
  1. Set GEMINI_API_KEY in .env
  2. pip install -r requirements.txt
  3. uvicorn main:app --reload
  4. POST /upload-pdf with a real PDF
  5. POST /query with {"query": "your question"}
"""

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document

from agents.rag_agent import _FALLBACK_PREFIX
from api.dependencies import get_document_service, get_rag_agent, get_vector_service
from main import app


@pytest.fixture
def mock_vector_service():
    service = MagicMock()
    service.add_documents.return_value = ["id-1", "id-2"]
    return service


@pytest.fixture
def mock_document_service(tmp_path):
    service = MagicMock()
    service.upload_dir = tmp_path
    service.load_pdf.return_value = [
        Document(page_content="Sample PDF text about machine learning.", metadata={"page": 0})
    ]
    service.split_documents.return_value = [
        Document(page_content="Sample PDF text about machine learning.", metadata={"page": 0})
    ]
    return service


@pytest.fixture
def mock_rag_agent():
    """Normal success path — Gemini returns an answer."""
    agent = MagicMock()

    async def _run(state):
        state["retrieved_chunks"] = ["Sample PDF text about machine learning."]
        state["retrieval_details"] = [
            {
                "id": "abc",
                "document": "Sample PDF text about machine learning.",
                "metadata": {"page": 0},
                "distance": 0.12,
                "similarity": 0.88,
            }
        ]
        state["rag_response"] = "The document discusses machine learning."
        state["used_fallback_response"] = False
        return state

    agent.run = _run
    return agent


@pytest.fixture
def mock_rag_agent_fallback():
    """Fallback path — Gemini failed, agent returns chunk-based summary."""
    agent = MagicMock()

    async def _run(state):
        state["retrieved_chunks"] = ["Sample PDF text about machine learning."]
        state["retrieval_details"] = [
            {
                "id": "abc",
                "document": "Sample PDF text about machine learning.",
                "metadata": {"page": 0},
                "distance": 0.12,
                "similarity": 0.88,
            }
        ]
        state["rag_response"] = (
            f"{_FALLBACK_PREFIX}\n\n• Sample PDF text about machine learning."
        )
        state["used_fallback_response"] = True
        return state

    agent.run = _run
    return agent


@pytest.fixture(autouse=True)
def override_dependencies(mock_vector_service, mock_document_service, mock_rag_agent):
    app.dependency_overrides[get_vector_service] = lambda: mock_vector_service
    app.dependency_overrides[get_document_service] = lambda: mock_document_service
    app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_pdf_success(mock_vector_service, mock_document_service):
    transport = ASGITransport(app=app)
    pdf_bytes = b"%PDF-1.4 minimal"

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload-pdf",
            files={"file": ("test.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "test.pdf"
    assert data["chunks_created"] == 1
    mock_document_service.load_pdf.assert_called_once()
    mock_document_service.split_documents.assert_called_once()
    mock_vector_service.add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_upload_pdf_rejects_non_pdf():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload-pdf",
            files={"file": ("notes.txt", BytesIO(b"hello"), "text/plain")},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_query_endpoint():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"query": "What is this document about?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The document discusses machine learning."
    assert len(data["retrieved_chunks"]) == 1
    assert data["sources"][0]["similarity"] == 0.88
    assert data["sources"][0]["metadata"]["page"] == 0
    # Normal path must report no fallback
    assert data["used_fallback_response"] is False


@pytest.mark.asyncio
async def test_query_rejects_empty_query():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_fallback_returns_200(mock_vector_service, mock_document_service, mock_rag_agent_fallback):
    """
    When Gemini fails, the route must still return HTTP 200.
    The response body must contain used_fallback_response=True and the
    fallback prefix in the answer field.
    """
    # Override only the RAG agent for this test
    app.dependency_overrides[get_rag_agent] = lambda: mock_rag_agent_fallback

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"query": "What is this document about?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["used_fallback_response"] is True
    assert _FALLBACK_PREFIX in data["answer"]
    # Sources must still be populated from retrieval
    assert len(data["retrieved_chunks"]) == 1
    assert len(data["sources"]) == 1
