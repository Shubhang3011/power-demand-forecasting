"""Feature engineering for the demand-forecasting models.

The model consumes calendar/temporal features (with cyclical encodings),
weather, and localized holiday flags. The exact feature ordering is a hard
contract shared by training and inference - see ``FEATURE_COLUMNS``.

NOTE on cloud_cover: cloud cover is intentionally *not* a model feature
because it is absent from the 2017 training CSV (the dataset only carries
Temperature/Humidity/WindSpeed). Cloud cover is still fetched from the
weather API and surfaced on the dashboard, but training without it keeps the
train-time and serve-time feature spaces identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .holidays_data import holiday_flags_for_index

# Hard contract: exact columns, in this exact order.
FEATURE_COLUMNS = [
    "block_of_day",
    "hour",
    "minute",
    "dayofweek",
    "day",
    "month",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "block_sin",
    "block_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "Temperature",
    "Humidity",
    "WindSpeed",
    "is_holiday",
    "is_festive_holiday",
    "is_industrial_holiday",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from a DatetimeIndexed frame.

    ``df`` must have a ``DatetimeIndex`` and the weather columns
    ``Temperature``, ``Humidity``, ``WindSpeed``. Returns a DataFrame with
    exactly ``FEATURE_COLUMNS`` and the same index as ``df``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_features requires a DatetimeIndex.")
    for col in ("Temperature", "Humidity", "WindSpeed"):
        if col not in df.columns:
            raise KeyError(f"build_features requires weather column '{col}'.")

    idx = df.index
    feat = pd.DataFrame(index=idx)

    hour = idx.hour
    minute = idx.minute
    block_of_day = (hour * 60 + minute) // config.INTERVAL_MIN  # 0..143
    dayofweek = idx.dayofweek

    feat["block_of_day"] = block_of_day.astype(int)
    feat["hour"] = hour.astype(int)
    feat["minute"] = minute.astype(int)
    feat["dayofweek"] = dayofweek.astype(int)
    feat["day"] = idx.day.astype(int)
    feat["month"] = idx.month.astype(int)
    feat["dayofyear"] = idx.dayofyear.astype(int)
    feat["weekofyear"] = idx.isocalendar().week.astype(int).to_numpy()
    feat["is_weekend"] = (dayofweek >= 5).astype(int)

    # Cyclical encodings.
    block = block_of_day.astype(float)
    feat["block_sin"] = np.sin(2 * np.pi * block / config.BLOCKS_PER_DAY)
    feat["block_cos"] = np.cos(2 * np.pi * block / config.BLOCKS_PER_DAY)

    dow = dayofweek.astype(float)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = idx.month.astype(float)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Weather (passed straight through).
    feat["Temperature"] = df["Temperature"].to_numpy()
    feat["Humidity"] = df["Humidity"].to_numpy()
    feat["WindSpeed"] = df["WindSpeed"].to_numpy()

    # Holiday flags.
    flags = holiday_flags_for_index(idx)
    feat["is_holiday"] = flags["is_holiday"].to_numpy()
    feat["is_festive_holiday"] = flags["is_festive_holiday"].to_numpy()
    feat["is_industrial_holiday"] = flags["is_industrial_holiday"].to_numpy()

    # Enforce exact column order / set.
    feat = feat[FEATURE_COLUMNS]
    return feat


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .data_cleaning import clean_load_data, load_raw

    clean, _ = clean_load_data(load_raw())
    X = build_features(clean)
    print(X.shape)
    print(list(X.columns) == FEATURE_COLUMNS)
    print(X.head())
