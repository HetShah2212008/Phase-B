"""
Financial Analysis Pipeline — LangGraph end-to-end workflow.

Graph topology:
  START → rag_agent → content_agent → email_agent → END

Node responsibilities:
  rag_agent     — Retrieve relevant document chunks and generate a grounded
                  Gemini answer using the existing, untouched RAGAgent.
  content_agent — Format the raw RAG answer into a structured Markdown
                  executive report using generate_content().
  email_agent   — Deliver the polished report to the recipient using
                  send_email_report().

State schema (FinancialPipelineState):
  user_query      — the user's original question
  recipient_email — where to send the final report (optional)
  rag_raw_output  — raw answer from the RAG agent
  polished_report — formatted Markdown report from the content agent
  email_status    — "sent" | "skipped" | "failed"

Usage:
    import asyncio
    from pipelines.financial_pipeline import financial_analysis_pipeline

    result = asyncio.run(financial_analysis_pipeline.ainvoke({
        "user_query": "What are the key financial risks?",
        "recipient_email": "analyst@example.com",
    }))
    print(result["polished_report"])
    print(result["email_status"])

Note on RAGAgent:
    The existing RAGAgent (agents/rag_agent.py) is imported and used as-is.
    Its internal logic, fallback handling, and ChromaDB integration are
    completely untouched. This pipeline only calls agent.run(state) and
    maps the result into FinancialPipelineState.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.content_agent import generate_content
from agents.rag_agent import RAGAgent
from schemas.pipeline_state import create_initial_state
from services.email_service import send_email_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline state schema
# ---------------------------------------------------------------------------

class FinancialPipelineState(TypedDict, total=False):
    """
    Shared state passed between all three pipeline nodes.

    total=False means every key is optional at creation time — nodes fill
    in their own fields and leave the rest unchanged.
    """
    user_query:      str   # Input: the user's question
    recipient_email: str   # Input: where to email the final report (optional)
    rag_raw_output:  str   # Set by rag_agent node
    polished_report: str   # Set by content_agent node
    email_status:    str   # Set by email_agent node: "sent" | "skipped" | "failed"


# ---------------------------------------------------------------------------
# Shared RAGAgent instance — lazy, created on first pipeline invocation.
# ---------------------------------------------------------------------------
_rag_agent: "RAGAgent | None" = None


def _get_rag_agent() -> "RAGAgent":
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent


# ---------------------------------------------------------------------------
# Node 1: RAG Agent
# ---------------------------------------------------------------------------

async def rag_node(state: FinancialPipelineState) -> dict:
    """
    Node 1/3 — Retrieve relevant chunks and generate a grounded answer.

    Wraps the existing RAGAgent.run() without modifying it.
    Maps the result from PipelineState into FinancialPipelineState.
    """
    print("🤖 [Node 1/3] Executing RAG Agent...")

    query = (state.get("user_query") or "").strip()
    if not query:
        logger.warning("[rag_node] user_query is empty — returning empty output")
        return {"rag_raw_output": ""}

    # Build a PipelineState compatible with RAGAgent.run()
    rag_state = create_initial_state(query=query)

    try:
        result = await _get_rag_agent().run(rag_state)
        raw_output = result.get("rag_response") or ""
        logger.info("[rag_node] RAG complete — output len=%d", len(raw_output))
        print(f"   ✅ RAG complete — {len(raw_output)} chars retrieved")
        return {"rag_raw_output": raw_output}

    except Exception as exc:
        logger.error("[rag_node] RAGAgent raised %s: %s", type(exc).__name__, exc)
        print(f"   ❌ RAG failed: {exc}")
        return {"rag_raw_output": f"RAG retrieval failed: {exc}"}


# ---------------------------------------------------------------------------
# Node 2: Content Agent
# ---------------------------------------------------------------------------

async def content_node(state: FinancialPipelineState) -> dict:
    """
    Node 2/3 — Format the raw RAG answer into a polished executive report.

    Calls generate_content() from agents/content_agent.py.
    Falls back to the raw output if formatting fails so the pipeline
    always has something to send.
    """
    print("✍️  [Node 2/3] Executing Content Agent...")

    raw_output = (state.get("rag_raw_output") or "").strip()

    if not raw_output:
        logger.warning("[content_node] rag_raw_output is empty — skipping formatting")
        print("   ⚠️  No RAG output to format — skipping")
        return {"polished_report": ""}

    try:
        report = await generate_content(raw_output, tone="professional")
        logger.info("[content_node] report generated — len=%d", len(report))
        print(f"   ✅ Report generated — {len(report)} chars")
        return {"polished_report": report}

    except Exception as exc:
        logger.error(
            "[content_node] generate_content failed (%s: %s) — "
            "using raw output as fallback",
            type(exc).__name__, exc,
        )
        print(f"   ⚠️  Formatting failed ({exc}) — using raw RAG output")
        return {"polished_report": raw_output}


# ---------------------------------------------------------------------------
# Node 3: Email Agent
# ---------------------------------------------------------------------------

async def email_node(state: FinancialPipelineState) -> dict:
    """
    Node 3/3 — Deliver the polished report to the recipient.

    Calls send_email_report() from services/email_service.py.
    Skips gracefully when no recipient is set or SMTP is not configured.
    """
    print("📧 [Node 3/3] Executing Email Agent...")

    recipient = (state.get("recipient_email") or "").strip()
    report    = (state.get("polished_report") or "").strip()

    if not recipient:
        logger.info("[email_node] no recipient_email set — skipping delivery")
        print("   ℹ️  No recipient set — skipping email delivery")
        return {"email_status": "skipped"}

    if not report:
        logger.warning("[email_node] polished_report is empty — skipping delivery")
        print("   ⚠️  No report content — skipping email delivery")
        return {"email_status": "skipped"}

    success = await send_email_report(
        report_content=report,
        recipient_email=recipient,
    )

    if success:
        logger.info("[email_node] report delivered to %s", recipient)
        print(f"   ✅ Email delivered to {recipient}")
        return {"email_status": "sent"}
    else:
        logger.error("[email_node] delivery failed to %s", recipient)
        print(f"   ❌ Email delivery failed to {recipient}")
        return {"email_status": "failed"}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_financial_pipeline():
    """Compile and return the financial analysis pipeline graph."""
    graph = StateGraph(FinancialPipelineState)

    graph.add_node("rag_agent",     rag_node)
    graph.add_node("content_agent", content_node)
    graph.add_node("email_agent",   email_node)

    graph.add_edge(START,           "rag_agent")
    graph.add_edge("rag_agent",     "content_agent")
    graph.add_edge("content_agent", "email_agent")
    graph.add_edge("email_agent",   END)

    compiled = graph.compile()
    logger.info(
        "financial_analysis_pipeline compiled: "
        "START → rag_agent → content_agent → email_agent → END"
    )
    return compiled


# Lazy-loaded pipeline — compiled on first use, not at import time.
_financial_pipeline_instance = None


def get_financial_pipeline():
    global _financial_pipeline_instance
    if _financial_pipeline_instance is None:
        _financial_pipeline_instance = _build_financial_pipeline()
    return _financial_pipeline_instance


# Backward-compatible name: accessing this attribute triggers lazy compilation.
class _LazyPipeline:
    """Proxy that compiles the pipeline on first attribute/call access."""
    def __getattr__(self, name):
        return getattr(get_financial_pipeline(), name)

    async def ainvoke(self, *args, **kwargs):
        return await get_financial_pipeline().ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return get_financial_pipeline().invoke(*args, **kwargs)


# Exported pipeline — import this in routes or scripts
financial_analysis_pipeline = _LazyPipeline()
