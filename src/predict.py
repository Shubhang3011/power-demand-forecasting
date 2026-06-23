"""Inference: generate a 24-hour, 10-minute-resolution demand forecast.

Loads the trained artifact, fetches hourly weather (Open-Meteo with a
climatology fallback), expands it onto the 10-minute forecast grid, builds the
exact training feature matrix, and predicts each feeder plus the total load.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config, weather
from .features import build_features


@lru_cache(maxsize=1)
def load_model() -> dict:
    """Load (and cache) the trained model artifact from ``MODEL_PATH``.

    Raises a clear ``FileNotFoundError`` if the model has not been trained yet.
    """
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {config.MODEL_PATH}. "
            f"Train it first with:  python -m src.train"
        )
    return joblib.load(config.MODEL_PATH)


def _next_block_start(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Next full 10-minute block boundary at/after the current local time."""
    if now is None:
        now = pd.Timestamp.now(tz=config.TIMEZONE).tz_localize(None)
    now = pd.Timestamp(now)
    # Ceil to the next 10-minute boundary.
    return now.ceil(f"{config.INTERVAL_MIN}min")


def _weather_to_grid(weather_hourly: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Upsample hourly weather onto the 10-minute forecast grid.

    Interpolates linearly between hourly points and forward/back fills the
    edges so every grid timestamp has Temperature/Humidity/WindSpeed.
    """
    w = weather_hourly.rename(
        columns={
            "temperature": "Temperature",
            "humidity": "Humidity",
            "wind_speed": "WindSpeed",
        }
    )[["Temperature", "Humidity", "WindSpeed"]]

    union = w.index.union(grid)
    w = w.reindex(union).interpolate(method="time", limit_direction="both")
    w = w.reindex(grid).ffill().bfill()
    return w


def generate_forecast(target_start=None, hours: int = config.HORIZON_HOURS) -> dict:
    """Generate a demand forecast for ``hours`` ahead at 10-minute resolution.

    If ``target_start`` is None, starts at the next full 10-minute block after
    the current local (Asia/Kolkata) time. Returns a JSON-friendly dict with a
    per-block forecast list (``hours*6`` blocks; 144 for the default 24h).
    """
    artifact = load_model()
    models = artifact["models"]
    feeders = artifact["feeders"]
    metrics = artifact.get("metrics", {})

    if target_start is None:
        start = _next_block_start()
    else:
        start = pd.Timestamp(target_start).floor(f"{config.INTERVAL_MIN}min")

    n_blocks = hours * (60 // config.INTERVAL_MIN)  # 144 for 24h @ 10min
    grid = pd.date_range(start=start, periods=n_blocks, freq=f"{config.INTERVAL_MIN}min")

    # Hourly weather (fetch one extra hour so interpolation covers the tail).
    weather_hourly = weather.get_forecast_weather(start.floor("h"), hours=hours + 1)
    weather_source = weather_hourly.attrs.get("source", "climatology-fallback")

    weather_grid = _weather_to_grid(weather_hourly, grid)

    # Feature frame: weather on the grid index drives build_features.
    feat_input = pd.DataFrame(
        {
            "Temperature": weather_grid["Temperature"],
            "Humidity": weather_grid["Humidity"],
            "WindSpeed": weather_grid["WindSpeed"],
        },
        index=grid,
    )
    X = build_features(feat_input)

    # Predict each feeder.
    feeder_preds = {}
    for feeder in feeders:
        yp = models[feeder].predict(X)
        feeder_preds[feeder] = np.asarray(yp, dtype=float)

    total = np.sum([feeder_preds[f] for f in feeders], axis=0)

    forecast = []
    for i, ts in enumerate(grid):
        block = (ts.hour * 60 + ts.minute) // config.INTERVAL_MIN
        entry = {
            "block": int(block),
            "timestamp": ts.isoformat(),
            "hour": int(ts.hour),
        }
        for feeder in feeders:
            entry[feeder] = round(float(feeder_preds[feeder][i]), 2)
        entry["total_load_kw"] = round(float(total[i]), 2)
        forecast.append(entry)

    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "location": config.LOCATION,
        "latitude": config.LAT,
        "longitude": config.LON,
        "interval_minutes": config.INTERVAL_MIN,
        "n_blocks": int(n_blocks),
        "horizon_hours": int(hours),
        "weather_source": weather_source,
        "model_metrics": metrics,
        "forecast": forecast,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json

    out = generate_forecast()
    print("n_blocks:", out["n_blocks"], "len:", len(out["forecast"]))
    print("weather_source:", out["weather_source"])
    print(json.dumps(out["forecast"][0], indent=2))
