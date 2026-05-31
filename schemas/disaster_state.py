"""
Shared state schema for the Disaster Management Pipeline (Part 2).

Each LangGraph node reads what it needs and writes its own fields.
total=False means every key is optional at creation time.
"""

from typing import Any, TypedDict


class DisasterState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    location: str                        # City / region name supplied by user
    latitude: float                      # Resolved by geocoding
    longitude: float                     # Resolved by geocoding

    # ── Environmental data ─────────────────────────────────────────────────
    weather_data: dict[str, Any]         # Raw Open-Meteo API response
    historical_df: Any                   # pandas DataFrame for Prophet
    forecast_data: dict[str, Any]        # Prophet forecast summary

    # ── ML prediction ──────────────────────────────────────────────────────
    disaster_prediction: dict[str, Any]  # {disaster_type, confidence, features}

    # ── News monitoring ────────────────────────────────────────────────────
    news_articles: list[dict[str, Any]]  # Top articles from NewsAPI

    # ── LLM assessment ─────────────────────────────────────────────────────
    severity: str                        # "low" | "medium" | "high" | "critical"
    department: str                      # Routed department name

    # ── Alert generation ───────────────────────────────────────────────────
    action_plan: str                     # Detailed action plan from dept agent
    alert_message: str                   # Public-facing alert message

    # ── Human-in-the-loop ──────────────────────────────────────────────────
    human_approved: bool                 # True once a human approves the alert
    rejection_feedback: str             # Reviewer's reason for rejection

    # ── Self-improvement loop ──────────────────────────────────────────────
    learned_insights: list[str]          # Accumulated rules from past rejections
    agent_prompt_addendum: str           # Injected into next generation attempt
    iteration: int                       # How many regeneration cycles have run

    # ── Delivery ───────────────────────────────────────────────────────────
    email_status: str                    # "alert_approved" | "sent" | "skipped"


def create_disaster_state(location: str) -> DisasterState:
    """Return a fresh DisasterState with safe defaults for all fields."""
    return DisasterState(
        location=location,
        latitude=0.0,
        longitude=0.0,
        weather_data={},
        historical_df=None,
        forecast_data={},
        disaster_prediction={},
        news_articles=[],
        severity="low",
        department="",
        action_plan="",
        alert_message="",
        human_approved=False,
        rejection_feedback="",
        learned_insights=[],
        agent_prompt_addendum="",
        email_status="",
        iteration=0,
    )
