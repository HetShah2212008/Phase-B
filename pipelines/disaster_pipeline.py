"""
Disaster Management Pipeline — LangGraph state-driven system.

Nodes:
1. ingest_environment  — fetch weather from Open-Meteo API
2. forecast_weather    — Prophet time series forecast
3. predict_disaster    — sklearn ML classification
4. monitor_news        — NewsAPI search
5. assess_and_route    — Gemini LLM severity + department routing
6. generate_alert      — Department agent drafts action plan
7. human_review        — HITL pause for approve/reject
8. self_improve        — Reflect, generate insight, update prompt, retry
"""

import logging
import os
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph

from schemas.disaster_state import DisasterState, create_disaster_state

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


# ─── GEMINI HELPER ────────────────────────────────────────────────────────────

async def call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text or ""


# ─── NODE 1: ENVIRONMENTAL DATA INGESTION ─────────────────────────────────────

async def ingest_environment(state: DisasterState) -> DisasterState:
    location = state["location"]
    logger.info("[Node1] Fetching weather for: %s", location)

    # Geocode location using Open-Meteo geocoding API
    async with httpx.AsyncClient(timeout=10) as client:
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_data = geo_resp.json()

    if not geo_data.get("results"):
        # Default to New York if geocoding fails
        lat, lon = 40.7128, -74.0060
    else:
        r = geo_data["results"][0]
        lat, lon = r["latitude"], r["longitude"]

    state["latitude"] = lat
    state["longitude"] = lon

    # Fetch current + historical weather from Open-Meteo
    async with httpx.AsyncClient(timeout=15) as client:
        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation,windspeed_10m,relative_humidity_2m",
                "past_days": 7,
                "forecast_days": 3,
                "timezone": "auto",
            },
        )
        weather_data = weather_resp.json()

    state["weather_data"] = weather_data

    # Build historical dataframe for forecasting
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    precip = hourly.get("precipitation", [])

    if times and precip:
        df = pd.DataFrame({
            "ds": pd.to_datetime(times),
            "y": [float(p) if p is not None else 0.0 for p in precip],
        })
        state["historical_df"] = df

    logger.info("[Node1] Weather fetched — lat=%.2f, lon=%.2f, rows=%d", lat, lon, len(times))
    return state


# ─── NODE 2: TIME SERIES FORECASTING ──────────────────────────────────────────

def forecast_weather(state: DisasterState) -> DisasterState:
    logger.info("[Node2] Running time series forecast")
    df = state.get("historical_df")

    if df is None or len(df) < 10:
        state["forecast_data"] = {"status": "insufficient_data", "forecast": []}
        return state

    try:
        from prophet import Prophet

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            changepoint_prior_scale=0.1,
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=48, freq="h")
        forecast = model.predict(future)

        future_only = forecast[forecast["ds"] > df["ds"].max()]
        max_precip = float(future_only["yhat"].max())
        avg_precip = float(future_only["yhat"].mean())

        state["forecast_data"] = {
            "status": "ok",
            "max_precipitation_48h": round(max_precip, 2),
            "avg_precipitation_48h": round(avg_precip, 2),
            "forecast_summary": (
                f"Max rainfall: {max_precip:.1f}mm, "
                f"Avg: {avg_precip:.1f}mm over next 48h"
            ),
        }
        logger.info("[Node2] Forecast done — max_precip=%.2f", max_precip)

    except Exception as e:
        logger.error("[Node2] Forecast failed: %s", e)
        state["forecast_data"] = {"status": "error", "message": str(e), "max_precipitation_48h": 0}

    return state


# ─── NODE 3: DISASTER PREDICTION MODEL ────────────────────────────────────────
# Features: [max_precip, avg_precip, max_wind, humidity] → disaster_type

from sklearn.ensemble import RandomForestClassifier as _RFC

_X_TRAIN = np.array([
    [0,   0,    10,  40],  [2,   1,    15,  50],  [5,   2,    20,  60],
    [80,  40,   25,  90],  [120, 60,   30,  95],  [150, 80,   35,  98],
    [10,  5,    80,  30],  [15,  8,    100, 25],  [20,  10,   120, 20],
    [5,   3,    20,  85],  [8,   4,    25,  88],  [12,  6,    30,  90],
    [0,   0,    5,   75],  [1,   0.5,  8,   60],  [3,   1,    12,  55],
    [35,  18,   28,  80],  [60,  30,   22,  92],  [90,  45,   18,  96],
])
_Y_TRAIN = [
    "None",      "None",      "None",
    "Flood",     "Flood",     "Flood",
    "Hurricane", "Hurricane", "Hurricane",
    "Heatwave",  "Heatwave",  "Heatwave",
    "None",      "None",      "None",
    "Flood",     "Flood",     "Flood",
]
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = _RFC(n_estimators=50, random_state=42)
        _classifier.fit(_X_TRAIN, _Y_TRAIN)
        logger.info("Disaster ML classifier trained and cached")
    return _classifier


def predict_disaster(state: DisasterState) -> DisasterState:
    logger.info("[Node3] Running disaster prediction ML model")

    # Extract features from current state
    weather = state.get("weather_data", {})
    hourly = weather.get("hourly", {})
    winds = hourly.get("windspeed_10m", [0])
    humidity = hourly.get("relative_humidity_2m", [50])
    forecast = state.get("forecast_data", {})

    max_precip = forecast.get("max_precipitation_48h", 0)
    avg_precip = forecast.get("avg_precipitation_48h", 0)
    max_wind = max(winds[-24:]) if winds else 0
    avg_humidity = sum(humidity[-24:]) / max(len(humidity[-24:]), 1) if humidity else 50

    features = np.array([[max_precip, avg_precip, max_wind, avg_humidity]])
    prediction = get_classifier().predict(features)[0]
    probabilities = get_classifier().predict_proba(features)[0]
    confidence = float(max(probabilities))

    state["disaster_prediction"] = {
        "disaster_type": prediction,
        "confidence": round(confidence, 2),
        "features": {
            "max_precipitation": max_precip,
            "avg_precipitation": avg_precip,
            "max_wind_speed": max_wind,
            "avg_humidity": avg_humidity,
        },
    }
    logger.info("[Node3] Prediction: %s (confidence=%.2f)", prediction, confidence)
    return state


# ─── NODE 4: NEWS MONITORING ───────────────────────────────────────────────────

async def monitor_news(state: DisasterState) -> DisasterState:
    location = state["location"]
    disaster_type = state.get("disaster_prediction", {}).get("disaster_type", "disaster")
    logger.info("[Node4] Fetching news for %s + %s", location, disaster_type)

    if not NEWS_API_KEY:
        state["news_articles"] = [{"title": "News API not configured", "description": ""}]
        return state

    query = f"{location} {disaster_type} weather emergency"
    if disaster_type == "None":
        query = f"{location} weather"

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": 5,
                "language": "en",
                "apiKey": NEWS_API_KEY,
            },
        )
        data = resp.json()

    articles = data.get("articles", [])
    state["news_articles"] = [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "publishedAt": a.get("publishedAt", ""),
        }
        for a in articles[:5]
    ]
    logger.info("[Node4] Found %d news articles", len(state["news_articles"]))
    return state


# ─── NODE 5: COGNITIVE ASSESSOR & ROUTER ──────────────────────────────────────

async def assess_and_route(state: DisasterState) -> DisasterState:
    logger.info("[Node5] LLM assessing severity and routing")

    prediction = state.get("disaster_prediction", {})
    news = state.get("news_articles", [])
    forecast = state.get("forecast_data", {})
    news_summary = "\n".join([
        f"- {a['title']}: {a['description']}" for a in news[:3]
    ])

    prompt = f"""You are a disaster management cognitive assessor.

Location: {state['location']}
ML Prediction: {prediction.get('disaster_type')} (confidence: {prediction.get('confidence')})
Forecast: {forecast.get('forecast_summary', 'N/A')}
Recent News:
{news_summary}

Based on the ML prediction and news context, assess:
1. Severity level: must be exactly one of: "low", "medium", "high", "critical"
2. Which department should handle this: must be exactly one of: "Emergency Response", "Civil Defense", "Public Works"

Respond in this exact format:
SEVERITY: <level>
DEPARTMENT: <department name>
REASONING: <2-3 sentences explaining your assessment>"""

    response = await call_gemini(prompt)

    severity = "medium"
    department = "Emergency Response"
    for line in response.split("\n"):
        if line.startswith("SEVERITY:"):
            severity = line.replace("SEVERITY:", "").strip().lower()
        elif line.startswith("DEPARTMENT:"):
            department = line.replace("DEPARTMENT:", "").strip()

    state["severity"] = severity
    state["department"] = department
    logger.info("[Node5] Severity=%s, Department=%s", severity, department)
    return state


# ─── NODE 6: DEPARTMENT AGENTS ────────────────────────────────────────────────

async def generate_alert(state: DisasterState) -> DisasterState:
    department = state.get("department", "Emergency Response")
    severity = state.get("severity", "medium")
    prediction = state.get("disaster_prediction", {})
    addendum = state.get("agent_prompt_addendum", "")
    iteration = state.get("iteration", 0)
    logger.info("[Node6] %s generating alert (iteration=%d)", department, iteration)

    dept_personas = {
        "Emergency Response": (
            "You are the Emergency Response coordinator. "
            "Focus on immediate evacuation, rescue operations, and emergency shelters."
        ),
        "Civil Defense": (
            "You are the Civil Defense director. "
            "Focus on infrastructure protection, military coordination, and civilian safety protocols."
        ),
        "Public Works": (
            "You are the Public Works director. "
            "Focus on drainage systems, road closures, utility protection, and infrastructure repair."
        ),
    }
    persona = dept_personas.get(department, dept_personas["Emergency Response"])

    prompt = f"""{persona}

Disaster Alert Request:
- Location: {state['location']}
- Disaster Type: {prediction.get('disaster_type', 'Unknown')}
- Severity: {severity.upper()}
- Confidence: {prediction.get('confidence', 0)*100:.0f}%
- Weather Forecast: {state.get('forecast_data', {}).get('forecast_summary', 'N/A')}
{f'Additional requirements from previous review: {addendum}' if addendum else ''}

Generate a specific action plan and public alert message. Include:
1. Immediate actions (next 6 hours)
2. Precautionary measures for residents
3. Resources being deployed
4. Emergency contact: 911 and local emergency management

Format your response as:
ACTION PLAN:
<detailed action plan>
ALERT MESSAGE:
<public alert message>"""

    response = await call_gemini(prompt)

    action_plan = ""
    alert_message = ""
    if "ALERT MESSAGE:" in response:
        parts = response.split("ALERT MESSAGE:")
        action_plan = parts[0].replace("ACTION PLAN:", "").strip()
        alert_message = parts[1].strip()
    else:
        action_plan = response
        alert_message = response[:500]

    state["action_plan"] = action_plan
    state["alert_message"] = alert_message
    logger.info("[Node6] Alert generated — len=%d", len(alert_message))
    return state


# ─── NODE 7: HUMAN-IN-THE-LOOP GATEKEEPER ─────────────────────────────────────

def human_review(state: DisasterState) -> DisasterState:
    """
    Pauses pipeline for human approval.

    In API mode this node marks state as pending — the frontend calls
    POST /api/disaster/review to approve or reject.
    """
    logger.info("[Node7] Awaiting human review")
    state["human_approved"] = False
    state["_awaiting_human"] = True
    return state


# ─── NODE 8: SELF-IMPROVING LOOP ──────────────────────────────────────────────

async def self_improve(state: DisasterState) -> DisasterState:
    feedback = state.get("rejection_feedback", "The output was not satisfactory.")
    existing_insights = state.get("learned_insights", [])
    iteration = state.get("iteration", 0)
    logger.info("[Node8] Self-improving based on feedback: %s", feedback)

    prompt = f"""You are a meta-learning AI that improves disaster alert generation.

The human reviewer rejected the previous alert with this feedback:
"{feedback}"

Previous learned insights:
{chr(10).join(f'- {i}' for i in existing_insights) if existing_insights else 'None yet'}

Generate ONE new concrete rule/insight that would prevent this rejection in future.
Format: Start with "Always" or "Never" — make it specific and actionable.
Example: "Always include specific emergency contact numbers for high-severity alerts."

Respond with ONLY the insight, nothing else."""

    new_insight = await call_gemini(prompt)
    new_insight = new_insight.strip()

    insights = existing_insights + [new_insight]
    state["learned_insights"] = insights
    state["agent_prompt_addendum"] = "\n".join(insights)
    state["iteration"] = iteration + 1
    state["human_approved"] = False
    logger.info("[Node8] New insight: %s", new_insight)
    return state


# ─── ROUTING FUNCTIONS ─────────────────────────────────────────────────────────

def route_after_review(state: DisasterState) -> str:
    if state.get("human_approved"):
        return "approved"
    return "rejected"


def route_after_improve(state: DisasterState) -> str:
    if state.get("iteration", 0) >= 3:
        return "max_retries"
    return "retry"


# ─── PIPELINE CLASS ────────────────────────────────────────────────────────────

class DisasterPipeline:
    def __init__(self):
        self._graph = None

    def build_graph(self):
        if self._graph:
            return self._graph

        graph = StateGraph(DisasterState)

        graph.add_node("ingest_environment", ingest_environment)
        graph.add_node("forecast_weather",   forecast_weather)
        graph.add_node("predict_disaster",   predict_disaster)
        graph.add_node("monitor_news",       monitor_news)
        graph.add_node("assess_and_route",   assess_and_route)
        graph.add_node("generate_alert",     generate_alert)
        graph.add_node("human_review",       human_review)
        graph.add_node("self_improve",       self_improve)

        graph.set_entry_point("ingest_environment")
        graph.add_edge("ingest_environment", "forecast_weather")
        graph.add_edge("forecast_weather",   "predict_disaster")
        graph.add_edge("predict_disaster",   "monitor_news")
        graph.add_edge("monitor_news",       "assess_and_route")
        graph.add_edge("assess_and_route",   "generate_alert")
        graph.add_edge("generate_alert",     "human_review")

        graph.add_conditional_edges(
            "human_review",
            route_after_review,
            {"approved": END, "rejected": "self_improve"},
        )
        graph.add_conditional_edges(
            "self_improve",
            route_after_improve,
            {"retry": "generate_alert", "max_retries": END},
        )

        self._graph = graph.compile()
        return self._graph

    async def run(self, location: str) -> DisasterState:
        """Run the full pipeline up to the human_review node."""
        state = create_disaster_state(location=location)
        graph = self.build_graph()
        result = await graph.ainvoke(state)
        return result

    async def run_with_approval(
        self,
        state: DisasterState,
        approved: bool,
        feedback: str = "",
    ) -> DisasterState:
        """Continue pipeline after human decision."""
        state["human_approved"] = approved
        state["rejection_feedback"] = feedback
        state["_awaiting_human"] = False

        if approved:
            state["email_status"] = "alert_approved"
            return state

        # Rejected — run self-improve → generate_alert loop
        graph = self.build_graph()
        result = await graph.ainvoke(state)
        return result
