"""
RAG Agent — Retrieval-Augmented Generation orchestration.

This agent coordinates services only; it does not load PDFs or talk to Pinecone
directly beyond VectorService / GeminiService abstractions.

LangGraph node flow:
  query → similarity_search → generate_rag_response → updated state

Fallback contract
-----------------
ANY exception raised by GeminiService is caught here and converted into a
graceful fallback response built from the already-retrieved chunks.

The exception is NEVER re-raised after Gemini is called — FastAPI will never
see a 500/502 caused by a Gemini problem.

The ONLY exceptions that propagate out of this agent are those raised during
vector retrieval (Pinecone failures), which are genuine infrastructure errors
that the route layer should surface as 500.
"""

import logging
import re
from typing import Any

from config import get_settings
from schemas.pipeline_state import PipelineState
from services.gemini_service import GeminiService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback tunables
# ---------------------------------------------------------------------------
_FALLBACK_MAX_CHUNKS = 3          # deduplicated chunks shown in fallback
_FALLBACK_MAX_CHARS  = 1_200      # hard cap on the body (prefix excluded)
_DEDUP_THRESHOLD     = 0.6        # Jaccard similarity → treat as duplicate
_CHUNK_MAX_SENTENCES = 2          # sentences kept per chunk bullet
_CHUNK_MAX_CHARS     = 300        # hard cap per chunk bullet

_FALLBACK_PREFIX = "[Fallback mode: generated directly from retrieved context]"
_NO_CHUNKS_MSG   = (
    "The AI generation service is currently unavailable and no document "
    "context was retrieved. Please try again later."
)


# ---------------------------------------------------------------------------
# Chunk helpers (pure functions — no I/O, easy to unit-test)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase word-token set for Jaccard comparison."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _deduplicate(chunks: list[str]) -> list[str]:
    """
    Drop chunks whose token-Jaccard similarity to any already-kept chunk
    meets or exceeds _DEDUP_THRESHOLD.  Order is preserved.
    """
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for chunk in chunks:
        tokens = _tokenize(chunk)
        if any(_jaccard(tokens, t) >= _DEDUP_THRESHOLD for t in kept_tokens):
            continue
        kept.append(chunk)
        kept_tokens.append(tokens)
    return kept


def _shorten(text: str) -> str:
    """
    Normalise whitespace, keep the first _CHUNK_MAX_SENTENCES sentences,
    hard-cap at _CHUNK_MAX_CHARS, and ensure terminal punctuation.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    # Sentence-boundary positions (after ". ", "! ", "? ")
    ends = [m.end() for m in re.finditer(r"[.!?]\s", cleaned)]
    if len(ends) >= _CHUNK_MAX_SENTENCES:
        cleaned = cleaned[: ends[_CHUNK_MAX_SENTENCES - 1]].strip()
    elif len(ends) == 1:
        cleaned = cleaned[: ends[0]].strip()

    if len(cleaned) > _CHUNK_MAX_CHARS:
        cut = cleaned[:_CHUNK_MAX_CHARS].rsplit(" ", 1)[0]
        cleaned = cut.rstrip(",.;:") + "..."

    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _build_fallback(chunks: list[str]) -> str:
    """
    Build a bullet-point fallback answer from retrieved chunks.

    Steps:
      1. Deduplicate near-identical chunks.
      2. Take the top _FALLBACK_MAX_CHUNKS.
      3. Shorten each to ≤ _CHUNK_MAX_SENTENCES / _CHUNK_MAX_CHARS.
      4. Render as bullet list.
      5. Hard-cap the body at _FALLBACK_MAX_CHARS.
      6. Prepend _FALLBACK_PREFIX.
    """
    if not chunks:
        return f"{_FALLBACK_PREFIX}\n\n{_NO_CHUNKS_MSG}"

    unique = _deduplicate(chunks)
    top    = unique[:_FALLBACK_MAX_CHUNKS]

    bullets = [f"• {_shorten(c)}" for c in top if _shorten(c)]
    if not bullets:
        return f"{_FALLBACK_PREFIX}\n\n{_NO_CHUNKS_MSG}"

    body = "\n".join(bullets)

    if len(body) > _FALLBACK_MAX_CHARS:
        truncated = body[:_FALLBACK_MAX_CHARS]
        last = max(
            truncated.rfind(". "), truncated.rfind(".\n"),
            truncated.rfind("! "), truncated.rfind("? "),
        )
        if last > _FALLBACK_MAX_CHARS // 2:
            truncated = truncated[: last + 1]
        body = truncated.rstrip() + " [...]"

    return f"{_FALLBACK_PREFIX}\n\n{body}"


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:
    """
    Retrieves relevant document chunks and generates a grounded Gemini answer.

    Wired into LangGraph as the first node after START.
    """

    def __init__(
        self,
        gemini: GeminiService | None = None,
        vector: VectorService | None = None,
    ) -> None:
        self._gemini = gemini or GeminiService()
        self._vector = vector or VectorService()

    async def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the full RAG step.

        Retrieval errors propagate (infrastructure failure).
        ALL Gemini errors are caught and converted to a fallback response —
        the method always returns a valid PipelineState.
        """
        query = (state.get("query") or "").strip()
        if not query:
            logger.warning("[RAGAgent] empty query received — returning empty state")
            state["rag_response"]        = ""
            state["retrieved_chunks"]    = []
            state["retrieval_details"]   = []
            state["used_fallback_response"] = False
            return state

        settings = get_settings()

        # ------------------------------------------------------------------
        # Step 1: Vector retrieval — allowed to raise (infra failure)
        # ------------------------------------------------------------------
        hits = self._vector.similarity_search(query, n_results=settings.rag_top_k)
        chunk_texts = [h["document"] for h in hits if h.get("document")]

        state["retrieved_chunks"]  = chunk_texts
        state["retrieval_details"] = hits

        logger.info(
            "[RAGAgent] retrieval succeeded — %d chunk(s) for query=%r",
            len(chunk_texts), query[:80],
        )

        # ------------------------------------------------------------------
        # Step 2: Gemini generation — NEVER allowed to raise
        # ------------------------------------------------------------------
        try:
            answer = await self._gemini.generate_rag_response(
                query=query,
                context_chunks=chunk_texts,
                temperature=settings.gemini_rag_temperature,
            )
            state["rag_response"]           = answer
            state["used_fallback_response"] = False
            logger.info(
                "[RAGAgent] Gemini succeeded — response len=%d", len(answer)
            )

        except Exception as exc:  # noqa: BLE001 — universal Gemini catch
            # Log the full exception so we can diagnose the exact error type
            # without it ever reaching FastAPI.
            logger.error(
                "[RAGAgent] Gemini raised %s — activating fallback. Error: %s",
                type(exc).__name__, exc, exc_info=True,
            )

            fallback = _build_fallback(chunk_texts)
            chunks_used = min(len(_deduplicate(chunk_texts)), _FALLBACK_MAX_CHUNKS)

            logger.warning(
                "[RAGAgent] fallback activated — built from %d/%d chunk(s). "
                "Exception type: %s",
                chunks_used, len(chunk_texts), type(exc).__name__,
            )

            state["rag_response"]           = fallback
            state["used_fallback_response"] = True
            # Return normally — do NOT re-raise under any circumstances
            return state

        return state

    def get_dependencies(self) -> dict[str, Any]:
        """Expose services for testing and dependency injection."""
        return {"gemini": self._gemini, "vector": self._vector}
