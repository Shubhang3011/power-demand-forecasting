"""Weather access for forecasting.

Primary source is the FREE, no-key Open-Meteo API:
  * forecast : https://api.open-meteo.com/v1/forecast
  * archive  : https://archive-api.open-meteo.com/v1/archive

If the network is unavailable (or the API errors / returns nothing), we
degrade gracefully to a *climatology* fallback built from the cleaned training
CSV: monthly-by-hour means of Temperature / Humidity / WindSpeed. Cloud cover
is not in the training data, so its fallback is a reasonable monthly mean
(seasonal) defaulting to 50%.

All returned frames are indexed by tz-naive local (Asia/Kolkata) wall-clock
hourly timestamps with columns: temperature, humidity, cloud_cover, wind_speed.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

try:  # requests is a hard dependency, but stay defensive.
    import requests
except Exception:  # pragma: no cover
    requests = None

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m"
_REQUEST_TIMEOUT = 8  # seconds

# Reasonable monthly mean cloud cover (%) for the Dhanbad region: dry winters,
# heavy monsoon cloud Jun-Sep. Used only when the API can't supply it.
_MONTHLY_CLOUD_FALLBACK = {
    1: 25, 2: 25, 3: 30, 4: 35, 5: 45,
    6: 70, 7: 80, 8: 80, 9: 70, 10: 40,
    11: 25, 12: 20,
}


# ---------------------------------------------------------------------------
# Climatology (offline fallback)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _build_climatology() -> pd.DataFrame:
    """Monthly-by-hour climatology of weather from the cleaned training CSV.

    Returns a DataFrame indexed by (month, hour) with columns
    temperature, humidity, wind_speed. Cached for the process lifetime.
    """
    # Imported lazily to avoid a circular import at module load time.
    from .data_cleaning import clean_load_data, load_raw

    try:
        clean, _ = clean_load_data(load_raw())
    except Exception:
        # Absolute fallback if the CSV is unreadable: a flat, plausible table.
        idx = pd.MultiIndex.from_product(
            [range(1, 13), range(24)], names=["month", "hour"]
        )
        return pd.DataFrame(
            {"temperature": 25.0, "humidity": 60.0, "wind_speed": 1.0}, index=idx
        )

    g = pd.DataFrame(
        {
            "month": clean.index.month,
            "hour": clean.index.hour,
            "temperature": clean["Temperature"].to_numpy(),
            "humidity": clean["Humidity"].to_numpy(),
            "wind_speed": clean["WindSpeed"].to_numpy(),
        }
    )
    clim = g.groupby(["month", "hour"]).mean()
    return clim


def _climatology_frame(start_dt: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Construct an hourly weather frame from climatology for the given window."""
    clim = _build_climatology()
    index = pd.date_range(start=start_dt, periods=hours, freq="h")
    rows = []
    for ts in index:
        try:
            base = clim.loc[(ts.month, ts.hour)]
            temp = float(base["temperature"])
            hum = float(base["humidity"])
            wind = float(base["wind_speed"])
        except KeyError:
            temp, hum, wind = 25.0, 60.0, 1.0
        cloud = float(_MONTHLY_CLOUD_FALLBACK.get(ts.month, 50))
        rows.append((temp, hum, cloud, wind))

    df = pd.DataFrame(
        rows,
        columns=["temperature", "humidity", "cloud_cover", "wind_speed"],
        index=index,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
def _parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo hourly payload into our standard weather frame."""
    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "cloud_cover": hourly.get("cloud_cover"),
            "wind_speed": hourly.get("wind_speed_10m"),
        }
    )
    df = df.set_index("timestamp")
    # The API is asked for local time (timezone param), so the index is already
    # local wall-clock and tz-naive.
    return df


def _fetch_open_meteo_forecast(start_dt: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Fetch an hourly forecast from Open-Meteo for the requested window.

    Raises on any failure so the caller can fall back to climatology.
    """
    if requests is None:
        raise RuntimeError("requests is unavailable")

    end_dt = start_dt + pd.Timedelta(hours=hours - 1)
    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "hourly": HOURLY_VARS,
        "timezone": config.TIMEZONE,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = _parse_open_meteo(resp.json())

    # Slice to exactly the requested window.
    index = pd.date_range(start=start_dt, periods=hours, freq="h")
    df = df.reindex(df.index.union(index)).interpolate(method="time")
    df = df.reindex(index)
    if df[["temperature", "humidity", "wind_speed"]].isna().any().any():
        raise ValueError("Open-Meteo returned insufficient data for window")
    # Cloud cover may legitimately be missing for far-future hours; backfill.
    df["cloud_cover"] = df["cloud_cover"].interpolate(limit_direction="both")
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_forecast_weather(start_dt=None, hours: int = config.HORIZON_HOURS) -> pd.DataFrame:
    """Hourly weather for a forecast window starting at ``start_dt``.

    Returns a DataFrame indexed by tz-naive local hourly timestamps with
    columns ``temperature, humidity, cloud_cover, wind_speed``. The frame
    carries an attribute ``df.attrs['source']`` set to either ``"open-meteo"``
    or ``"climatology-fallback"``.
    """
    if start_dt is None:
        start_dt = pd.Timestamp.now().floor("h")
    start_dt = pd.Timestamp(start_dt).floor("h")

    try:
        df = _fetch_open_meteo_forecast(start_dt, hours)
        df.attrs["source"] = "open-meteo"
        return df
    except Exception:
        df = _climatology_frame(start_dt, hours)
        df.attrs["source"] = "climatology-fallback"
        return df


def get_weather_payload(start_dt=None, hours: int = config.HORIZON_HOURS) -> dict:
    """JSON-friendly weather payload for the dashboard/API.

    Returns ``{location, latitude, longitude, source, hourly:[...]}`` where each
    hourly entry has ``timestamp(ISO), temperature_c, humidity_pct,
    cloud_cover_pct, wind_speed_kmh``.
    """
    df = get_forecast_weather(start_dt, hours)
    source = df.attrs.get("source", "climatology-fallback")

    hourly = []
    for ts, row in df.iterrows():
        hourly.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "temperature_c": round(float(row["temperature"]), 2),
                "humidity_pct": round(float(row["humidity"]), 2),
                "cloud_cover_pct": round(float(row["cloud_cover"]), 2),
                "wind_speed_kmh": round(float(row["wind_speed"]), 2),
            }
        )

    return {
        "location": config.LOCATION,
        "latitude": config.LAT,
        "longitude": config.LON,
        "source": source,
        "hourly": hourly,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json

    payload = get_weather_payload(hours=6)
    print("source:", payload["source"])
    print(json.dumps(payload["hourly"][:2], indent=2))
