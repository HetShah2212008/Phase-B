"""
Disaster Management API endpoints.

Endpoints:
  POST /api/disaster/run              — Start pipeline for a location
  POST /api/disaster/review           — Human approve / reject the alert
  GET  /api/disaster/session/{id}     — Retrieve session state
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipelines.disaster_pipeline import DisasterPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Disaster"])

# In-memory session store for HITL
_sessions: dict[str, Any] = {}
_pipeline_instance: "DisasterPipeline | None" = None


def _get_pipeline() -> "DisasterPipeline":
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DisasterPipeline()
    return _pipeline_instance


class LocationRequest(BaseModel):
    location: str


class ReviewRequest(BaseModel):
    session_id: str
    approved: bool
    feedback: str = ""


@router.post("/disaster/run")
async def run_disaster_pipeline(body: LocationRequest):
    """
    Start the disaster pipeline for a given location.

    Runs all 7 nodes (ingest → forecast → predict → news → assess →
    generate_alert → human_review) and returns the state awaiting approval.
    """
    try:
        state = await _get_pipeline().run(body.location)
        session_id = f"disaster_{body.location.replace(' ', '_')}_{id(state)}"
        _sessions[session_id] = state

        return {
            "session_id": session_id,
            "location": body.location,
            "disaster_type": state.get("disaster_prediction", {}).get("disaster_type"),
            "confidence": state.get("disaster_prediction", {}).get("confidence"),
            "severity": state.get("severity"),
            "department": state.get("department"),
            "action_plan": state.get("action_plan"),
            "alert_message": state.get("alert_message"),
            "news_articles": state.get("news_articles", [])[:3],
            "forecast": state.get("forecast_data", {}),
            "awaiting_approval": True,
            "learned_insights": state.get("learned_insights", []),
            "iteration": state.get("iteration", 0),
        }

    except Exception as e:
        logger.exception("Disaster pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disaster/review")
async def review_alert(body: ReviewRequest):
    """
    Human approves or rejects the generated alert.

    If approved  → marks email_status as alert_approved and returns.
    If rejected  → triggers self-improvement loop (up to 3 iterations)
                   and returns the regenerated alert.
    """
    state = _sessions.get(body.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    updated = await _get_pipeline().run_with_approval(state, body.approved, body.feedback)
    _sessions[body.session_id] = updated

    return {
        "session_id": body.session_id,
        "approved": body.approved,
        "email_status": updated.get("email_status", ""),
        "action_plan": updated.get("action_plan", ""),
        "alert_message": updated.get("alert_message", ""),
        "learned_insights": updated.get("learned_insights", []),
        "iteration": updated.get("iteration", 0),
        "message": (
            "Alert approved and sent!"
            if body.approved
            else "Regenerating with improvements..."
        ),
    }


@router.get("/disaster/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve the full state for an existing disaster session."""
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state
