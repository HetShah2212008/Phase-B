"""Tests for PDF chunking logic (no vector DB or LLM)."""

from langchain_core.documents import Document

from services.document_service import DocumentService


def test_split_documents_respects_chunk_settings():
    service = DocumentService()
    # One long page of text
    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={"page": 0})]

    chunks = service.split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= service._chunk_size + 50  # small margin for separators


def test_preprocess_text_collapses_whitespace():
    service = DocumentService()
    assert service.preprocess_text("  hello   world  \n\n") == "hello world"
