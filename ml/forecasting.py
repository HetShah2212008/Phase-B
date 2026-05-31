"""
Time-series forecasting — cash flow prediction with Prophet.
Produces a chart showing when the company runs out of cash.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt
    PROPHET_AVAILABLE = True
except ImportError:
    plt = None
    PROPHET_AVAILABLE = False


def generate_synthetic_cashflow(
    start_date: str = "2024-01-01",
    months: int = 12,
    starting_balance: float = 500_000,
    monthly_burn: float = 45_000,
    noise_scale: float = 8_000,
) -> pd.DataFrame:
    """
    Generate realistic synthetic cash flow data.
    Balance decreases monthly with some random noise.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    dates = [
        datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30 * i)
        for i in range(months)
    ]
    balance = starting_balance
    records = []
    for d in dates:
        balance = balance - monthly_burn + rng.normal(0, noise_scale)
        records.append({"ds": d, "y": max(balance, 0)})

    return pd.DataFrame(records)


def run_cashflow_forecast(
    historical_df: pd.DataFrame | None = None,
    forecast_periods: int = 12,
    output_path: str = "cashflow_forecast.png",
) -> dict[str, Any]:
    """
    Train Prophet on cash flow data, forecast future months,
    find the zero-crossing date, and save a chart.

    Returns dict with:
      - forecast: list of {ds, yhat, yhat_lower, yhat_upper}
      - zero_date: estimated date balance hits zero (or None)
      - chart_path: path to saved PNG chart
      - status: "ok" | "error"
    """
    if not PROPHET_AVAILABLE:
        return {"status": "error", "message": "prophet/matplotlib not installed"}

    if historical_df is None:
        historical_df = generate_synthetic_cashflow()

    # Train Prophet
    from prophet import Prophet
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
    )
    model.fit(historical_df)

    future = model.make_future_dataframe(periods=forecast_periods, freq="MS")
    forecast_df = model.predict(future)

    # Find zero-crossing (when yhat first goes <= 0)
    zero_date = None
    future_only = forecast_df[forecast_df["ds"] > historical_df["ds"].max()]
    zero_rows = future_only[future_only["yhat"] <= 0]
    if not zero_rows.empty:
        zero_date = zero_rows.iloc[0]["ds"].strftime("%Y-%m-%d")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical
    ax.plot(
        historical_df["ds"], historical_df["y"],
        "o-", color="#2563eb", label="Historical Balance", linewidth=2, markersize=4,
    )

    # Forecast line
    ax.plot(
        future_only["ds"], future_only["yhat"],
        "--", color="#f59e0b", label="Forecasted Balance", linewidth=2,
    )

    # Confidence band
    ax.fill_between(
        future_only["ds"],
        future_only["yhat_lower"],
        future_only["yhat_upper"],
        alpha=0.2, color="#f59e0b", label="Confidence Interval",
    )

    # Zero line
    ax.axhline(y=0, color="#ef4444", linestyle="-", linewidth=1.5, label="Zero Balance")

    # Zero crossing marker
    if zero_date:
        ax.axvline(
            x=pd.to_datetime(zero_date),
            color="#ef4444", linestyle=":", linewidth=2,
            label=f"Estimated Zero: {zero_date}",
        )

    ax.set_title("Cash Flow Forecast — Runway Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cash Balance ($)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    # Serialize forecast
    records = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_periods)
    forecast_list = [
        {
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(float(row["yhat"]), 2),
            "yhat_lower": round(float(row["yhat_lower"]), 2),
            "yhat_upper": round(float(row["yhat_upper"]), 2),
        }
        for _, row in records.iterrows()
    ]

    logger.info(
        "[ForecastingService] chart saved to %s, zero_date=%s",
        output_path, zero_date,
    )

    return {
        "status": "ok",
        "forecast": forecast_list,
        "zero_date": zero_date,
        "chart_path": output_path,
    }


# Keep class-based API for backward compatibility
class ForecastingService:
    def run_forecast(
        self,
        historical_df: pd.DataFrame | None = None,
        forecast_periods: int = 12,
        output_path: str = "cashflow_forecast.png",
    ) -> dict[str, Any]:
        return run_cashflow_forecast(historical_df, forecast_periods, output_path)