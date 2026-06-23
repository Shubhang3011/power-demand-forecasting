"""FastAPI backend for the Power Demand Forecasting dashboard.

Serves a small JSON API (under ``/api``) on top of the trained ``src`` core and
mounts the static frontend at ``/`` so the whole app runs from a single server:

    cd "d:/Assignments placement/power-demand-forecasting"
    python -m uvicorn backend.main:app

Endpoints (all return JSON):
    GET /api/health     -> liveness + whether the model artifact is loadable
    GET /api/forecast   -> full 24h / 144-block 10-minute demand forecast
    GET /api/weather    -> next-24h hourly weather aligned to the forecast start
    GET /api/holidays   -> localized holidays in [start, end] (defaults to +30d)
    GET /api/dashboard  -> {forecast, weather, holidays} in one call

The frontend is mounted LAST (after the API routes) so ``/api/*`` always wins.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable defensively. This file lives at
# <root>/backend/main.py, so the project root is parents[1]. uvicorn is normally
# launched from the root (so ``import src`` already works), but inserting the
# root on sys.path means the app also works no matter the current directory.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src import config  # noqa: E402
from src.holidays_data import get_holidays  # noqa: E402
from src.predict import generate_forecast, load_model  # noqa: E402
from src.weather import get_weather_payload  # noqa: E402

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Power Demand Forecasting API",
    description=(
        "24-hour, 10-minute-resolution electrical demand forecast for three "
        "132KV feeders in Dhanbad, Jharkhand, plus localized weather and "
        "holiday context."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_is_loaded() -> bool:
    """True if the trained model artifact exists and loads cleanly."""
    try:
        load_model()
        return True
    except Exception:
        return False


def _forecast_start_from(forecast: dict) -> str | None:
    """ISO timestamp of the first forecast block (its window start), if any."""
    blocks = forecast.get("forecast") or []
    if blocks:
        return blocks[0].get("timestamp")
    return None


# ---------------------------------------------------------------------------
# API routes (registered BEFORE the static mount so /api/* always resolves)
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Liveness probe with model + location context."""
    return {
        "status": "ok",
        "model_loaded": _model_is_loaded(),
        "location": config.LOCATION,
        "latitude": config.LAT,
        "longitude": config.LON,
        "timezone": config.TIMEZONE,
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/forecast")
def forecast(hours: int = Query(config.HORIZON_HOURS, ge=1, le=72)) -> dict:
    """Full demand forecast (144 ten-minute blocks for the default 24h)."""
    try:
        return generate_forecast(target_start=None, hours=hours)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Failed to generate forecast: {exc}"
        ) from exc


@app.get("/api/weather")
def weather(hours: int = Query(config.HORIZON_HOURS, ge=1, le=72)) -> dict:
    """Next-24h hourly weather, aligned to the current forecast start hour."""
    try:
        # Align to the same start the forecast uses: the floor-hour of the next
        # 10-minute block. Passing start_dt=None lets get_weather_payload floor
        # the current local hour, which matches generate_forecast's window.
        return get_weather_payload(start_dt=None, hours=hours)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch weather: {exc}"
        ) from exc


@app.get("/api/holidays")
def holidays(
    start: str | None = Query(None, description="Inclusive start date YYYY-MM-DD"),
    end: str | None = Query(None, description="Inclusive end date YYYY-MM-DD"),
) -> dict:
    """Localized holidays in ``[start, end]`` (defaults: today .. today+30d)."""
    today = _dt.date.today()
    start_date = start or today.isoformat()
    end_date = end or (today + _dt.timedelta(days=30)).isoformat()
    try:
        records = get_holidays(start_date, end_date)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid date range: {exc}"
        ) from exc
    return {
        "location": config.LOCATION,
        "start": start_date,
        "end": end_date,
        "count": len(records),
        "holidays": records,
    }


@app.get("/api/dashboard")
def dashboard(hours: int = Query(config.HORIZON_HOURS, ge=1, le=72)) -> dict:
    """One-shot payload combining forecast, weather and holidays.

    Weather and holidays are aligned to the same forecast window so the
    frontend can render everything from a single request.
    """
    try:
        fc = generate_forecast(target_start=None, hours=hours)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Failed to generate forecast: {exc}"
        ) from exc

    start_iso = _forecast_start_from(fc)
    try:
        wx = get_weather_payload(start_dt=start_iso, hours=hours)
    except Exception:
        wx = {"location": config.LOCATION, "source": "unavailable", "hourly": []}

    # Holidays spanning the forecast window (and a small look-ahead so a 24h
    # forecast that crosses midnight still surfaces the upcoming day's holiday).
    if start_iso:
        start_day = _dt.datetime.fromisoformat(start_iso).date()
    else:
        start_day = _dt.date.today()
    end_day = start_day + _dt.timedelta(days=max(1, (hours + 23) // 24))
    try:
        hol = get_holidays(start_day.isoformat(), end_day.isoformat())
    except Exception:
        hol = []

    return {
        "forecast": fc,
        "weather": wx,
        "holidays": {
            "location": config.LOCATION,
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "count": len(hol),
            "holidays": hol,
        },
    }


# ---------------------------------------------------------------------------
# Static frontend mount (LAST). Serves index.html at GET / via html=True.
# Computed relative to this file with pathlib so it is cwd-independent.
# ---------------------------------------------------------------------------
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIR), html=True),
        name="frontend",
    )
else:  # pragma: no cover - frontend is expected to exist in the final app
    @app.get("/")
    def _no_frontend() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "message": (
                    "Power Demand Forecasting API is running. The frontend "
                    f"directory ({_FRONTEND_DIR}) was not found; build it to "
                    "serve the dashboard at /. The JSON API is available under "
                    "/api (see /docs)."
                ),
                "api": ["/api/health", "/api/forecast", "/api/weather",
                        "/api/holidays", "/api/dashboard"],
            },
        )
