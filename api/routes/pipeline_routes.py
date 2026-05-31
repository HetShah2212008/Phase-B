"""
Pipeline orchestration endpoints.

POST /api/pipeline/langgraph  — Run the full LangGraph pipeline (RAG → Content → Email)
POST /api/pipeline/adk        — Run the ADK SequentialAgent pipeline
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipelines.langgraph_pipeline import LangGraphPipeline
from pipelines.adk_pipeline import ADKPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Pipeline"])


class PipelineRequest(BaseModel):
    query: str
    email_recipient: str = ""


@router.post("/pipeline/langgraph")
async def run_langgraph(body: PipelineRequest):
    """
    Run the LangGraph pipeline: RAG → Content Agent → Email Agent.

    Returns the full pipeline state including rag_response, formatted_content,
    and email_status.
    """
    try:
        p = LangGraphPipeline()
        result = await p.run(body.query, body.email_recipient)
        return dict(result)
    except Exception as exc:
        logger.exception("LangGraph pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pipeline/adk")
async def run_adk(body: PipelineRequest):
    """
    Run the ADK SequentialAgent pipeline: RAG → Content → Email.

    Returns the full pipeline state including rag_response, formatted_content,
    and email_status.
    """
    try:
        p = ADKPipeline()
        result = await p.run(body.query, body.email_recipient)
        return dict(result)
    except Exception as exc:
        logger.exception("ADK pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
