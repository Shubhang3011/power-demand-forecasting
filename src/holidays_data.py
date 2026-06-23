"""Localized holiday calendar for Dhanbad / Jharkhand.

The CSV (``data/holidays_dhanbad.csv``) is self-sourced and covers
Jharkhand-specific festivals (Sarhul, Karma, Tusu, Chhath), pan-Indian
religious festivals, national holidays, and coal/steel-belt *industrial*
holidays (Vishwakarma Puja, Labour Day). Years covered: 2017 and 2024-2026.

These functions expose the calendar both as records (for the API/dashboard)
and as per-timestamp integer flags (for model features).
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from . import config


def load_holidays() -> pd.DataFrame:
    """Load the holiday calendar with a parsed ``date`` column (datetime64)."""
    df = pd.read_csv(config.HOLIDAYS_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    # Normalise the boolean-ish int columns.
    for col in ["is_national", "is_festive", "is_industrial"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = 0
    if "type" not in df.columns:
        df["type"] = ""
    if "name" not in df.columns:
        df["name"] = ""
    return df


def _to_timestamp(value) -> pd.Timestamp:
    """Coerce a date / datetime / string into a normalized (midnight) Timestamp."""
    ts = pd.to_datetime(value)
    return ts.normalize()


def get_holidays(start, end) -> list[dict]:
    """Return holiday records within ``[start, end]`` inclusive.

    ``start`` / ``end`` may be ``date``, ``datetime``, ``Timestamp`` or string.
    Each record: ``{date(YYYY-MM-DD), name, type, is_festive(bool),
    is_industrial(bool), is_national(bool)}``.
    """
    df = load_holidays()
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end)
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    sub = df.loc[mask].sort_values("date")

    records: list[dict] = []
    for _, row in sub.iterrows():
        records.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "name": str(row["name"]),
                "type": str(row["type"]),
                "is_festive": bool(int(row["is_festive"])),
                "is_industrial": bool(int(row["is_industrial"])),
                "is_national": bool(int(row["is_national"])),
            }
        )
    return records


def holiday_flags_for_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Build per-timestamp holiday flags for a DatetimeIndex.

    Returns a DataFrame indexed exactly like ``idx`` with integer columns
    ``is_holiday``, ``is_festive_holiday``, ``is_industrial_holiday``. Matching
    is done on the calendar *date* (any reading on a holiday date is flagged).
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)

    df = load_holidays()

    # Map each holiday date -> its flags (dedupe on date, OR-ing flags).
    grouped = (
        df.groupby(df["date"].dt.normalize())
        .agg(
            is_festive=("is_festive", "max"),
            is_industrial=("is_industrial", "max"),
        )
    )
    holiday_dates = set(grouped.index)

    dates = idx.normalize()
    is_holiday = pd.Series(
        [1 if d in holiday_dates else 0 for d in dates], index=idx, dtype=int
    )

    fest_map = grouped["is_festive"].to_dict()
    ind_map = grouped["is_industrial"].to_dict()
    is_festive = pd.Series(
        [int(fest_map.get(d, 0)) for d in dates], index=idx, dtype=int
    )
    is_industrial = pd.Series(
        [int(ind_map.get(d, 0)) for d in dates], index=idx, dtype=int
    )

    out = pd.DataFrame(
        {
            "is_holiday": is_holiday,
            "is_festive_holiday": is_festive,
            "is_industrial_holiday": is_industrial,
        },
        index=idx,
    )
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    print(get_holidays("2017-09-01", "2017-10-31"))
    sample = pd.date_range("2017-09-16", "2017-09-18", freq="6h")
    print(holiday_flags_for_index(sample))
