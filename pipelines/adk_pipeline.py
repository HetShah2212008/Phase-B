"""
Google ADK pipeline — lazy loaded to avoid opentelemetry conflicts with chromadb.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from schemas.pipeline_state import PipelineState, create_initial_state

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class ADKPipeline:
    """
    Runs RAG → Content → Email pipeline using Google ADK SequentialAgent.

    Imports are lazy to avoid opentelemetry version conflicts at startup.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def _build(self):
        if self._pipeline is not None:
            return self._pipeline

        from google.adk.agents import LlmAgent, SequentialAgent

        rag_agent = LlmAgent(
            name="RAGAgent",
            model=GEMINI_MODEL,
            instruction="You are a research assistant. Answer the user query clearly and factually. Store your answer in output_key 'rag_response'.",
            description="Answers questions from the knowledge base.",
            output_key="rag_response",
        )

        content_agent = LlmAgent(
            name="ContentAgent",
            model=GEMINI_MODEL,
            instruction="You are a professional report writer. Take the research answer in 'rag_response' and reformat it into a clean executive report in Markdown with ## Executive Summary, ## section headings, bullet points, and ## Conclusion. Store output in 'formatted_content'.",
            description="Formats the RAG response into a structured executive report.",
            output_key="formatted_content",
        )

        email_agent = LlmAgent(
            name="EmailAgent",
            model=GEMINI_MODEL,
            instruction="You are an email composition assistant. Take the formatted report in 'formatted_content' and write a professional email body with greeting and sign-off. Store output in 'email_body'.",
            description="Composes the final email from the formatted report.",
            output_key="email_body",
        )

        self._pipeline = SequentialAgent(
            name="ReportPipeline",
            sub_agents=[rag_agent, content_agent, email_agent],
        )
        return self._pipeline

    async def run(self, query: str, email_recipient: str = "") -> PipelineState:
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        state = create_initial_state(query=query)
        if email_recipient:
            state["email_recipient"] = email_recipient

        pipeline = self._build()
        runner = InMemoryRunner(agent=pipeline, app_name="adk_pipeline")

        # Must create session first before running
        session = await runner.session_service.create_session(
            app_name="adk_pipeline",
            user_id="user",
        )

        final_text = ""
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=query)],
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""

        state["rag_response"] = final_text[:500] if final_text else ""
        state["formatted_content"] = final_text
        state["email_status"] = "composed"
        return state

    def get_agent_registry(self) -> dict[str, Any]:
        pipeline = self._build()
        return {
            "rag": pipeline.sub_agents[0],
            "content": pipeline.sub_agents[1],
            "email": pipeline.sub_agents[2],
        }
