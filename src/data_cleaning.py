"""Loading and cleaning of the raw 10-minute utility consumption data.

Known data issues handled here (and surfaced in the cleaning ``report``):

1. MIXED DATE FORMATS - some rows are ``DD-MM-YYYY HH:MM`` and some are
   ``MM/DD/YYYY HH:MM``. We disambiguate purely by the date separator:
       * a "-" in the string  -> day-first  (``%d-%m-%Y %H:%M``)
       * a "/" in the string  -> month-first (``%m/%d/%Y %H:%M``)
   (the two formats are interleaved in the file, not cleanly split early/late,
   so a single ``pd.to_datetime`` guess is unreliable - the separator rule is).

2. MISSING TIMESTAMPS - the series is reindexed onto a *complete* 10-minute
   grid between the first and last reading so any gaps become explicit NaNs.

3. OUTLIERS / SENSOR ERRORS - non-positive feeder readings (<= 0) are treated
   as missing, and statistical outliers are detected with a robust
   rolling-median + MAD filter and set to NaN. Everything is then filled with
   time interpolation and a time-of-day/seasonal pattern as a last resort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns we expect to carry through cleaning.
WEATHER_COLS = ["Temperature", "Humidity", "WindSpeed"]
FEEDER_VALUE_COLS = list(config.FEEDER_COLS.values())


# ---------------------------------------------------------------------------
# Datetime parsing
# ---------------------------------------------------------------------------
def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse a Series of date strings using the separator disambiguation rule.

    Rows containing "-" are parsed day-first (``%d-%m-%Y %H:%M``); rows
    containing "/" are parsed month-first (``%m/%d/%Y %H:%M``). The hour part
    may be zero-padded ("00:00") or not ("0:00"); ``%H`` accepts both.

    Returns a ``datetime64[ns]`` Series aligned to the input index.
    """
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    is_slash = s.str.contains("/", regex=False)
    is_dash = s.str.contains("-", regex=False) & ~is_slash

    if is_dash.any():
        out.loc[is_dash] = pd.to_datetime(
            s.loc[is_dash], format="%d-%m-%Y %H:%M", errors="coerce"
        )
    if is_slash.any():
        out.loc[is_slash] = pd.to_datetime(
            s.loc[is_slash], format="%m/%d/%Y %H:%M", errors="coerce"
        )

    return out


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and attach a parsed ``Datetime`` column (not yet index)."""
    df = pd.read_csv(path)
    df["Datetime"] = parse_datetime_series(df["Datetime"])
    return df


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------
def _mad_outlier_mask(series: pd.Series, window: int = 145, n_sigmas: float = 6.0) -> pd.Series:
    """Robust rolling-median + MAD outlier mask for a single feeder column.

    A point is flagged when it deviates from the local (one-day, centered)
    rolling median by more than ``n_sigmas`` scaled MADs. The scale factor
    1.4826 converts MAD to a std-equivalent for normally distributed data.
    Window defaults to 145 (~one day of 10-min blocks, made odd for centering).
    """
    if window % 2 == 0:
        window += 1
    med = series.rolling(window, center=True, min_periods=window // 4).median()
    abs_dev = (series - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=window // 4).median()
    # Guard against a zero MAD (perfectly flat regions): fall back to global MAD.
    global_mad = (series - series.median()).abs().median()
    scale = mad.replace(0, np.nan).fillna(global_mad if global_mad > 0 else 1.0)
    robust_z = abs_dev / (1.4826 * scale)
    return (robust_z > n_sigmas) & med.notna()


def _fill_with_seasonal_pattern(series: pd.Series) -> pd.Series:
    """Fill any remaining NaNs using a (month, block-of-day) climatology.

    This handles long contiguous gaps where plain interpolation would smear a
    flat line across many blocks; we instead borrow the typical value for that
    time-of-day in that month.
    """
    if not series.isna().any():
        return series
    idx = series.index
    key = pd.DataFrame(
        {
            "month": idx.month,
            "block": (idx.hour * 60 + idx.minute) // config.INTERVAL_MIN,
            "val": series.to_numpy(),
        },
        index=idx,
    )
    pattern = key.groupby(["month", "block"])["val"].transform("mean")
    filled = series.fillna(pattern)
    # Absolute last resort: global mean.
    return filled.fillna(series.mean())


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_load_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw loaded frame.

    Returns ``(clean_df, report)`` where ``clean_df`` is indexed by a complete
    10-minute ``DatetimeIndex`` with columns ``Temperature, Humidity,
    WindSpeed`` plus the three feeder columns and NO NaNs.
    """
    report: dict = {}
    df = df.copy()
    report["rows_in"] = int(len(df))

    # Drop rows whose datetime failed to parse, then sort and de-duplicate.
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    report["unparseable_datetimes"] = int(report["rows_in"] - len(df))

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Datetime"], keep="first")
    report["duplicate_timestamps"] = int(before_dedup - len(df))

    df = df.set_index("Datetime")

    keep_cols = WEATHER_COLS + FEEDER_VALUE_COLS
    df = df[keep_cols].astype(float)

    # Build the complete 10-minute grid and reindex onto it (exposes gaps).
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{config.INTERVAL_MIN}min",
    )
    n_unique = df.index.nunique()
    df = df.reindex(full_index)
    df.index.name = "Datetime"
    report["rows_after_reindex"] = int(len(df))
    # Timestamps present on the grid but absent from the data = pure gaps.
    report["missing_timestamps"] = int(len(full_index) - n_unique)

    # --- Feeder cleaning -----------------------------------------------------
    negatives_or_zeros = 0
    outliers_removed = 0
    for col in FEEDER_VALUE_COLS:
        nonpositive = (df[col] <= 0)
        negatives_or_zeros += int(nonpositive.sum())
        df.loc[nonpositive, col] = np.nan

        outlier_mask = _mad_outlier_mask(df[col])
        # Don't double-count points already nulled as non-positive.
        outlier_mask = outlier_mask & df[col].notna()
        outliers_removed += int(outlier_mask.sum())
        df.loc[outlier_mask, col] = np.nan

    report["negatives_or_zeros"] = int(negatives_or_zeros)
    report["outliers_removed"] = int(outliers_removed)

    # Count how many feeder cells are missing (gaps + nulled) before filling.
    feeder_nan_before = int(df[FEEDER_VALUE_COLS].isna().sum().sum())

    # --- Fill feeders: time interpolation, then seasonal pattern, then ffill/bfill
    for col in FEEDER_VALUE_COLS:
        df[col] = df[col].interpolate(method="time", limit_direction="both")
        df[col] = _fill_with_seasonal_pattern(df[col])
        df[col] = df[col].ffill().bfill()

    feeder_nan_after = int(df[FEEDER_VALUE_COLS].isna().sum().sum())
    report["gaps_filled"] = int(feeder_nan_before - feeder_nan_after)

    # --- Weather: interpolate then forward/back fill -------------------------
    weather_nan_before = int(df[WEATHER_COLS].isna().sum().sum())
    for col in WEATHER_COLS:
        df[col] = df[col].interpolate(method="time", limit_direction="both")
        df[col] = df[col].ffill().bfill()
    report["weather_values_filled"] = int(weather_nan_before)

    # Final safety net: no NaNs may remain.
    if df.isna().any().any():
        df = df.ffill().bfill()

    report["final_rows"] = int(len(df))
    report["remaining_nans"] = int(df.isna().sum().sum())
    report["date_min"] = df.index.min().isoformat()
    report["date_max"] = df.index.max().isoformat()

    return df, report


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    raw = load_raw()
    clean, rep = clean_load_data(raw)
    import json

    print(json.dumps(rep, indent=2))
    print(clean.head())
