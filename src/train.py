"""Train per-feeder LightGBM demand-forecasting models.

Run with::

    cd power-demand-forecasting
    python -m src.train

Pipeline: load_raw -> clean_load_data -> build_features (X) and the three
feeder columns (y). A *time-based* split is used (first ~92% train, last ~8%
test, which is roughly the final four weeks of 2017) so evaluation reflects
genuine out-of-time forecasting performance. One LGBMRegressor is trained per
feeder; metrics are reported per feeder and for the TOTAL load (sum of feeders).

Artifacts written:
  * MODEL_PATH      (joblib pickle) - models + contract metadata
  * METADATA_PATH   (JSON) - human-readable metrics, features, windows, report
"""
from __future__ import annotations

import datetime as _dt
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data_cleaning import clean_load_data, load_raw
from .features import FEATURE_COLUMNS, build_features

# Fraction of the (time-ordered) data reserved for the out-of-time test set.
TEST_FRACTION = 0.08

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=48,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%), guarding against zero denominators."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true, y_pred) -> dict:
    """Compute MAE / RMSE / MAPE / R2 for one target series."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(_mape(y_true, y_pred), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
    }


def train(verbose: bool = True) -> dict:
    """Run the full training pipeline; return the saved artifact dict."""
    # --- Load & clean --------------------------------------------------------
    raw = load_raw()
    clean, report = clean_load_data(raw)

    # --- Features & targets --------------------------------------------------
    X = build_features(clean)
    y = clean[[config.FEEDER_COLS[f] for f in config.FEEDERS]].copy()

    assert list(X.columns) == FEATURE_COLUMNS, "Feature column contract violated."

    # --- Time-based split ----------------------------------------------------
    n = len(X)
    n_test = int(round(n * TEST_FRACTION))
    n_train = n - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    train_window = (X_train.index.min().isoformat(), X_train.index.max().isoformat())
    test_window = (X_test.index.min().isoformat(), X_test.index.max().isoformat())

    if verbose:
        print(f"Rows: {n}  train={n_train}  test={n_test}")
        print(f"Train window: {train_window[0]} -> {train_window[1]}")
        print(f"Test  window: {test_window[0]} -> {test_window[1]}")

    # --- Train one model per feeder ------------------------------------------
    models: dict[str, lgb.LGBMRegressor] = {}
    preds_test: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}

    for feeder in config.FEEDERS:
        col = config.FEEDER_COLS[feeder]
        reg = lgb.LGBMRegressor(**LGBM_PARAMS)
        reg.fit(X_train, y_train[col])
        models[feeder] = reg

        yp = reg.predict(X_test)
        preds_test[feeder] = yp
        metrics[feeder] = _metrics(y_test[col].to_numpy(), yp)
        if verbose:
            m = metrics[feeder]
            print(
                f"  {feeder}: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} "
                f"MAPE={m['MAPE']:.2f}% R2={m['R2']:.4f}"
            )

    # --- Total-load metrics --------------------------------------------------
    total_true = y_test.sum(axis=1).to_numpy()
    total_pred = np.sum([preds_test[f] for f in config.FEEDERS], axis=0)
    metrics["TOTAL"] = _metrics(total_true, total_pred)
    if verbose:
        m = metrics["TOTAL"]
        print(
            f"  TOTAL: MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} "
            f"MAPE={m['MAPE']:.2f}% R2={m['R2']:.4f}"
        )

    # --- Build & persist artifact -------------------------------------------
    trained_at = _dt.datetime.now().isoformat(timespec="seconds")
    artifact = {
        "models": models,
        "feature_columns": FEATURE_COLUMNS,
        "feeders": config.FEEDERS,
        "feeder_cols": config.FEEDER_COLS,
        "metrics": metrics,
        "trained_at": trained_at,
        "interval_minutes": config.INTERVAL_MIN,
        "blocks_per_day": config.BLOCKS_PER_DAY,
        "lightgbm_version": lgb.__version__,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_window": train_window,
        "test_window": test_window,
        "test_fraction": TEST_FRACTION,
        "lgbm_params": LGBM_PARAMS,
        "cleaning_report": report,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)

    # Human-readable metadata (no pickled estimators).
    metadata = {
        "trained_at": trained_at,
        "location": config.LOCATION,
        "latitude": config.LAT,
        "longitude": config.LON,
        "lightgbm_version": lgb.__version__,
        "interval_minutes": config.INTERVAL_MIN,
        "blocks_per_day": config.BLOCKS_PER_DAY,
        "horizon_hours": config.HORIZON_HOURS,
        "feeders": config.FEEDERS,
        "feeder_cols": config.FEEDER_COLS,
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "test_fraction": TEST_FRACTION,
        "train_window": {"start": train_window[0], "end": train_window[1]},
        "test_window": {"start": test_window[0], "end": test_window[1]},
        "lgbm_params": LGBM_PARAMS,
        "metrics": metrics,
        "data_cleaning_report": report,
        "notes": {
            "cloud_cover": (
                "cloud_cover is intentionally excluded as a model feature "
                "because it is absent from the 2017 training CSV; it is still "
                "served by the weather API for the dashboard."
            ),
            "date_parsing": (
                "Mixed date formats handled by separator: '-' => %d-%m-%Y %H:%M, "
                "'/' => %m/%d/%Y %H:%M."
            ),
        },
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    if verbose:
        print(f"\nSaved model    -> {config.MODEL_PATH}")
        print(f"Saved metadata -> {config.METADATA_PATH}")

    return artifact


def main() -> None:
    train(verbose=True)


if __name__ == "__main__":
    main()
