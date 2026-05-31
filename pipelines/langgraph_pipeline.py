"""
LangGraph orchestration pipeline.

Graph topology:
  START → rag → content → email → END

Node responsibilities:
  rag     — ChromaDB retrieval + grounded Gemini answer  (RAGAgent)
  content — Format rag_response into a Markdown report   (ContentAgent)
  email   — Deliver formatted_content via SMTP           (EmailAgent)

Each node reads from shared PipelineState, updates its own fields, and
passes the state to the next node unchanged otherwise.

Usage:
    pipeline = LangGraphPipeline()

    # Full pipeline (RAG + format + email)
    result = await pipeline.run(
        query="What are the key findings?",
        email_recipient="user@example.com",
    )

    # RAG only (no formatting, no email)
    result = await pipeline.run_rag_only("What are the key findings?")
"""

import logging
from typing import Any

from agents.content_agent import ContentAgent
from agents.email_agent import EmailAgent
from agents.rag_agent import RAGAgent
from schemas.pipeline_state import PipelineState, create_initial_state

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore[misc, assignment]
    END = None  # type: ignore[misc, assignment]


class LangGraphPipeline:
    """
    Builds and runs the three-node LangGraph workflow.

    The compiled graph is cached after the first call to build_graph() so
    subsequent pipeline.run() calls reuse the same compiled object.
    """

    def __init__(
        self,
        rag: RAGAgent | None = None,
        content: ContentAgent | None = None,
        email: EmailAgent | None = None,
    ) -> None:
        self._rag     = rag     or RAGAgent()
        self._content = content or ContentAgent()
        self._email   = email   or EmailAgent()
        self._compiled_graph: Any = None

    def build_graph(self) -> Any:
        """
        Compile the StateGraph and cache it.

        Called automatically by run() — you only need to call this directly
        if you want to inspect the compiled graph object.
        """
        if StateGraph is None:
            raise ImportError(
                "langgraph is not installed. Run: pip install langgraph"
            )

        if self._compiled_graph is not None:
            return self._compiled_graph

        graph = StateGraph(PipelineState)

        # --- Node definitions ---

        async def rag_node(state: PipelineState) -> PipelineState:
            """Retrieve relevant chunks and generate a grounded Gemini answer."""
            return await self._rag.run(state)

        async def content_node(state: PipelineState) -> PipelineState:
            """Format rag_response into a structured Markdown report."""
            return await self._content.run(state)

        async def email_node(state: PipelineState) -> PipelineState:
            """Send formatted_content to email_recipient via SMTP."""
            return await self._email.run(state)

        # --- Wire the graph ---
        graph.add_node("rag",     rag_node)
        graph.add_node("content", content_node)
        graph.add_node("email",   email_node)

        graph.set_entry_point("rag")
        graph.add_edge("rag",     "content")
        graph.add_edge("content", "email")
        graph.add_edge("email",   END)

        self._compiled_graph = graph.compile()
        logger.info("LangGraphPipeline graph compiled: rag → content → email → END")
        return self._compiled_graph

    async def run(
        self,
        query: str,
        email_recipient: str = "",
    ) -> PipelineState:
        """
        Execute the full pipeline: RAG → Content → Email.

        Args:
            query:            User question or task description.
            email_recipient:  If provided, the formatted report is emailed here.
                              If omitted, the email node skips delivery gracefully.

        Returns:
            Final PipelineState with all fields populated:
              - rag_response        — grounded Gemini answer
              - formatted_content   — structured Markdown report
              - email_status        — "sent" | "skipped" | "failed"
              - used_fallback_response — True if Gemini quota/auth failed
        """
        initial = create_initial_state(query=query)
        if email_recipient:
            initial["email_recipient"] = email_recipient

        logger.info(
            "[LangGraphPipeline] starting — query=%r, recipient=%r",
            query[:80] if query else "",
            email_recipient or "(none)",
        )

        compiled = self.build_graph()
        result: PipelineState = await compiled.ainvoke(initial)

        logger.info(
            "[LangGraphPipeline] complete — "
            "rag_len=%d, content_len=%d, email_status=%r, fallback=%s",
            len(result.get("rag_response") or ""),
            len(result.get("formatted_content") or ""),
            result.get("email_status", ""),
            result.get("used_fallback_response", False),
        )
        return result

    async def run_rag_only(self, query: str) -> PipelineState:
        """
        Run only the RAG node — no formatting, no email.

        Useful for the /query API endpoint which only needs the RAG answer
        and does not need the full pipeline overhead.
        """
        state = create_initial_state(query=query)
        return await self._rag.run(state)
