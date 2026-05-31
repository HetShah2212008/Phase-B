"""
Reusable wrapper around the Google Gemini SDK (google.genai).

Agents call this service for text generation instead of talking to the API
directly. That keeps prompts, models, and error handling in one place.

SDK: google-genai (the current, supported package)
Async: uses client.aio.models.generate_content() — natively async, no thread needed.
"""

import logging
from typing import Any

from google import genai
from google.genai import types

from config import get_settings

logger = logging.getLogger(__name__)

# Grounded RAG system instructions — model must not invent facts outside context
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
1. Answer ONLY using information from the context below.
2. If the context does not contain enough information to answer, respond clearly with:
   "I could not find that information in the provided documents."
3. Do not use outside knowledge or make assumptions beyond the context.
4. Keep answers concise, factual, and well-structured.
5. If partial information exists, state what is known and what is missing."""


class GeminiService:
    """
    Thin abstraction over google.genai.Client.

    Instantiate once (e.g. in FastAPI lifespan or agent __init__) and reuse.
    One Client is created per instance; async calls go through client.aio.models
    which is natively async — no asyncio.to_thread wrapper needed.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Generation calls will fail until configured."
            )

        self._model_name = settings.gemini_model
        logger.info(
            f"Gemini initialized with model={self._model_name}, "
            f"api_key_set={bool(settings.gemini_api_key)}, "
            f"api_key_prefix={settings.gemini_api_key[:8] + '...' if settings.gemini_api_key else 'MISSING'}"
        )
        self._default_temperature = settings.gemini_temperature
        self._rag_temperature = settings.gemini_rag_temperature

        # One client instance shared across all calls from this service.
        self._client = genai.Client(api_key=settings.gemini_api_key or None)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _make_config(self, temperature: float) -> types.GenerateContentConfig:
        """Build a GenerateContentConfig for a single call."""
        return types.GenerateContentConfig(temperature=temperature)

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a text completion from a user prompt.

        Args:
            prompt: Main user message / instruction.
            system_prompt: Optional system role instructions prepended to the prompt.
            temperature: Override default generation temperature.
            **kwargs: Ignored (kept for interface compatibility).

        Raises:
            ValueError: If prompt is empty or API key is missing.
            RuntimeError: If the model call fails.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Set it in .env before calling Gemini."
            )

        # Combine system instructions + user prompt into a single string.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        temp = temperature if temperature is not None else self._default_temperature
        config = self._make_config(temp)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=full_prompt,
                config=config,
            )

            # Diagnostic logging — fires before .text access so it always runs
            logger.debug("[GeminiService.generate_text] raw response: %r", response)
            try:
                for ci, candidate in enumerate(response.candidates or []):
                    logger.debug(
                        "[GeminiService.generate_text] candidate[%d] "
                        "finish_reason=%s safety=%s",
                        ci,
                        getattr(candidate, "finish_reason", "?"),
                        getattr(candidate, "safety_ratings", "?"),
                    )
                    content = getattr(candidate, "content", None)
                    if content:
                        for pi, part in enumerate(getattr(content, "parts", [])):
                            logger.debug(
                                "[GeminiService.generate_text] "
                                "candidate[%d].part[%d]: %r",
                                ci, pi, getattr(part, "text", part),
                            )
            except Exception as log_exc:  # noqa: BLE001
                logger.debug(
                    "[GeminiService.generate_text] could not inspect candidates: %s",
                    log_exc,
                )

            text = response.text
            logger.debug(
                "[GeminiService.generate_text] response.text (len=%d): %r",
                len(text), text[:200],
            )
            return text

        except Exception as exc:
            logger.exception("Gemini generation failed")
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def generate_rag_response(
        self,
        query: str,
        context_chunks: list[str],
        *,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate an answer grounded in retrieved document chunks.

        Combines the RAG system prompt, numbered context chunks, and the user
        question into a single string prompt so the model:
          - cites only provided context
          - admits when information is not found
          - avoids hallucinating facts

        Args:
            query: User question from pipeline state.
            context_chunks: Text retrieved from the vector database.
            temperature: Lower values (e.g. 0.2) reduce creative hallucination.
            **kwargs: Ignored (kept for interface compatibility).

        Raises:
            ValueError: If query is empty or API key is missing.
            RuntimeError: If the model call fails.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty for RAG generation.")

        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Set it in .env before calling Gemini."
            )

        # Number chunks so the model can refer to sources consistently.
        if context_chunks:
            context_body = "\n\n".join(
                f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
            )
        else:
            context_body = "(No context chunks were retrieved from the document store.)"

        # Single-string prompt: system instructions + context + question.
        full_prompt = f"""{RAG_SYSTEM_PROMPT}

Context:
---
{context_body}
---

Question: {query}

Answer based only on the context above."""

        rag_temp = temperature if temperature is not None else self._rag_temperature
        config = self._make_config(rag_temp)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=full_prompt,
                config=config,
            )

            # Diagnostic logging — fires before .text access so it always runs
            logger.debug(
                "[GeminiService.generate_rag_response] raw response: %r", response
            )
            try:
                for ci, candidate in enumerate(response.candidates or []):
                    logger.debug(
                        "[GeminiService.generate_rag_response] candidate[%d] "
                        "finish_reason=%s safety=%s",
                        ci,
                        getattr(candidate, "finish_reason", "?"),
                        getattr(candidate, "safety_ratings", "?"),
                    )
                    content = getattr(candidate, "content", None)
                    if content:
                        for pi, part in enumerate(getattr(content, "parts", [])):
                            logger.debug(
                                "[GeminiService.generate_rag_response] "
                                "candidate[%d].part[%d]: %r",
                                ci, pi, getattr(part, "text", part),
                            )
            except Exception as log_exc:  # noqa: BLE001
                logger.debug(
                    "[GeminiService.generate_rag_response] "
                    "could not inspect candidates: %s",
                    log_exc,
                )

            text = response.text
            logger.debug(
                "[GeminiService.generate_rag_response] response.text (len=%d): %r",
                len(text), text[:200],
            )
            return text

        except Exception as exc:
            logger.exception("Gemini RAG generation failed")
            raise RuntimeError(f"Gemini RAG API error: {exc}") from exc
