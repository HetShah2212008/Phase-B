"""Tests for shared LangGraph pipeline state."""

from schemas.pipeline_state import create_initial_state


def test_create_initial_state_has_all_keys():
    state = create_initial_state(query="What is RAG?")

    assert state["query"] == "What is RAG?"
    assert state["retrieved_chunks"] == []
    assert state["retrieval_details"] == []
    assert state["rag_response"] == ""
    assert state["formatted_content"] == ""
    assert state["email_recipient"] == ""
    assert state["email_status"] == ""
    assert state["forecast_data"] == {}
    # Fallback flag must exist and default to False
    assert state["used_fallback_response"] is False
