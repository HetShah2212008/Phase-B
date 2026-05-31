"""
Content Agent — formats raw RAG output into a structured executive report.

Pipeline position:
  RAG Agent → [Content Agent] → Email Agent

This module provides two interfaces:

1. Module-level function (used by financial_pipeline.py):
       await generate_content(rag_output, tone="professional")

2. Class-based agent (used by langgraph_pipeline.py):
       await ContentAgent().run(state)

Both call the same underlying Gemini logic — the class just wraps the function
so it can read/write PipelineState for the LangGraph graph.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

from schemas.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model — financial_pipeline uses gemini-2.5-flash as specified
# Falls back to GEMINI_CONTENT_MODEL env var, then GEMINI_MODEL, then default
# ---------------------------------------------------------------------------
_CONTENT_MODEL = (
    os.getenv("GEMINI_CONTENT_MODEL")
    or os.getenv("GEMINI_MODEL")
    or "gemini-2.5-flash"
)

# ---------------------------------------------------------------------------
# Formatting system prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a professional report writer.
Your job is to take raw AI-generated analysis and reformat it into a clean,
well-structured executive report in Markdown.

Formatting rules:
1. Start with a bold title: ## Executive Summary
2. Use clear section headings (##) for each major topic.
3. Use bullet points (- ) for lists and key findings.
4. Keep language professional, concise, and factual.
5. Do NOT add information that was not in the original text.
6. End with a ## Conclusion section summarising the key takeaways."""

_TONE_INSTRUCTIONS = {
    "professional": "Use formal, precise business language.",
    "friendly":     "Use warm, approachable language while remaining factual.",
    "technical":    "Use technical terminology appropriate for a specialist audience.",
}


# ---------------------------------------------------------------------------
# Module-level async function — primary interface for financial_pipeline.py
# ---------------------------------------------------------------------------

async def generate_content(rag_output: str, tone: str = "professional") -> str:
    """
    Transform raw RAG text into a beautifully formatted executive report.

    Uses gemini-2.5-flash with a structured formatting prompt to produce
    clean Markdown with headings, bullet points, and an executive summary.

    Args:
        rag_output: Raw text answer produced by the RAG agent.
        tone:       Writing style — "professional" | "friendly" | "technical".
                    Defaults to "professional".

    Returns:
        Formatted Markdown report string.

    Raises:
        ValueError: If rag_output is empty.
        RuntimeError: If the Gemini API call fails.
    """
    if not rag_output.strip():
        raise ValueError("rag_output cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Configure it in .env before calling Gemini."
        )

    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS["professional"])
    system_instruction = f"{_SYSTEM_PROMPT}\n\nTone instruction: {tone_instruction}"

    full_prompt = (
        f"{system_instruction}\n\n"
        f"Please format the following analysis into a structured executive report:\n\n"
        f"{rag_output}"
    )

    client = genai.Client(api_key=api_key)

    try:
        response = await client.aio.models.generate_content(
            model=_CONTENT_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
        report = response.text
        logger.info(
            "[generate_content] report generated — model=%s, tone=%s, len=%d",
            _CONTENT_MODEL, tone, len(report),
        )
        return report

    except Exception as exc:
        logger.exception("[generate_content] Gemini call failed")
        raise RuntimeError(f"Content generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Class-based agent — used by langgraph_pipeline.py (reads/writes PipelineState)
# ---------------------------------------------------------------------------

class ContentAgent:
    """
    Wraps generate_content() as a LangGraph-compatible node.

    Reads  : state["rag_response"]
    Writes : state["formatted_content"]

    Graceful degradation: if Gemini formatting fails, falls back to the raw
    rag_response so the email step still has content to send.
    """

    async def run(self, state: PipelineState) -> PipelineState:
        rag_response = (state.get("rag_response") or "").strip()

        if not rag_response:
            logger.warning("[ContentAgent] rag_response is empty — skipping formatting")
            state["formatted_content"] = ""
            return state

        try:
            report = await generate_content(rag_response, tone="professional")
            state["formatted_content"] = report
            logger.info("[ContentAgent] formatted_content set — len=%d", len(report))

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[ContentAgent] formatting failed (%s: %s) — "
                "falling back to raw rag_response",
                type(exc).__name__, exc,
            )
            state["formatted_content"] = rag_response

        return state
