"""Central configuration for the Power Demand Forecasting project.

All paths are resolved relative to this file so the package works from any
current working directory (backend server, notebook, CLI, tests).

Location context: Apex Power & Utilities operates three 132KV feeders in the
Dhanbad coal/steel belt of Jharkhand, India.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Geography / locale
# ---------------------------------------------------------------------------
LAT: float = 23.7957
LON: float = 86.4304
LOCATION: str = "Dhanbad, Jharkhand, India"
TIMEZONE: str = "Asia/Kolkata"

# ---------------------------------------------------------------------------
# Feeders
# ---------------------------------------------------------------------------
FEEDERS: list[str] = ["F1", "F2", "F3"]
FEEDER_COLS: dict[str, str] = {
    "F1": "F1_132KV_PowerConsumption",
    "F2": "F2_132KV_PowerConsumption",
    "F3": "F3_132KV_PowerConsumption",
}

# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------
INTERVAL_MIN: int = 10            # readings are taken every 10 minutes
BLOCKS_PER_DAY: int = 144         # 24h * 60min / 10min
HORIZON_HOURS: int = 24           # default forecast horizon

# ---------------------------------------------------------------------------
# Filesystem paths (absolute, derived from this file's location)
# config.py lives at <root>/src/config.py, so the project root is parents[1].
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"

MODEL_PATH: Path = MODELS_DIR / "demand_model.pkl"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

RAW_CSV: Path = DATA_DIR / "Utility_consumption.csv"
HOLIDAYS_CSV: Path = DATA_DIR / "holidays_dhanbad.csv"

# Ensure the models directory exists (cheap, idempotent).
MODELS_DIR.mkdir(parents=True, exist_ok=True)
