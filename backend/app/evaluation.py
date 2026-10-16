"""Rolling-origin backtesting and forecast metrics."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

QUANTILE_COLS = {0.1: "p10", 0.5: "p50", 0.9: "p90"}


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual > 0
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask]))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sum(np.abs(pred - actual)) / np.sum(np.abs(actual)))


def coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    return float(np.mean((actual >= lo) & (actual <= hi)))


def pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = np.asarray(actual, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(actual: np.ndarray, frame: pd.DataFrame) -> dict:
    """Metrics for one forecast frame (p10/p50/p90) against actuals."""
    scale = float(np.mean(actual))
    return {
        "mape": mape(actual, frame["p50"]),
        "wape": wape(actual, frame["p50"]),
        "coverage_p10_p90": coverage(actual, frame["p10"].to_numpy(), frame["p90"].to_numpy()),
        # Scaled pinball loss averaged over the three quantiles (lower is better).
        "scaled_pinball": float(np.mean([pinball(actual, frame[c], q) for q, c in QUANTILE_COLS.items()]) / scale),
        "interval_width_pct": float(np.mean((frame["p90"] - frame["p10"]).to_numpy() / actual)),
    }


def rolling_origins(df: pd.DataFrame, folds: int, horizon: int, step: int | None = None) -> list[pd.Timestamp]:
    step = step or horizon
    last = pd.Timestamp(df["date"].max())
    first_origin = last - pd.Timedelta(days=horizon - 1) - pd.Timedelta(days=step * (folds - 1))
    return [first_origin + pd.Timedelta(days=step * i) for i in range(folds)]


def backtest(
    df: pd.DataFrame,
    fit_forecast: Callable[[pd.DataFrame, pd.DataFrame], dict],
    origins: list[pd.Timestamp],
    horizon: int,
) -> dict:
    """Run a model over rolling origins. `fit_forecast(train_df, future_spend)` returns a forecast dict."""
    channels = sorted(df["channel"].unique())
    rows, fold_rows = [], []
    runtimes = []
    for origin in origins:
        train = df[df["date"] < origin]
        test = df[(df["date"] >= origin) & (df["date"] < origin + pd.Timedelta(days=horizon))]
        t0 = time.perf_counter()
        fc = fit_forecast(train, test[["date", "channel", "spend"]])
        runtimes.append(time.perf_counter() - t0)
        for ch in channels:
            a = test[test["channel"] == ch].sort_values("date")
            frame = fc["channels"][ch].reset_index(drop=True)
            rows.append({"origin": origin, "series": ch, "actual": a["revenue"].to_numpy(), "frame": frame})
        tot = test.groupby("date")["revenue"].sum().sort_index()
        rows.append({"origin": origin, "series": "total", "actual": tot.to_numpy(),
                     "frame": fc["total"].reset_index(drop=True)})
        fold_rows.append({"origin": str(origin.date()),
                          "total_actual": float(tot.sum()), **{f"total_{k}": v for k, v in fc["horizon_total"].items()}})

    metrics: dict[str, dict] = {}
    for series in channels + ["total"]:
        sel = [r for r in rows if r["series"] == series]
        actual = np.concatenate([r["actual"] for r in sel])
        frame = pd.concat([r["frame"] for r in sel], ignore_index=True)
        metrics[series] = {k: round(v, 4) for k, v in score(actual, frame).items()}
    # Pooled across channel-days ("all channels" view)
    sel = [r for r in rows if r["series"] != "total"]
    actual = np.concatenate([r["actual"] for r in sel])
    frame = pd.concat([r["frame"] for r in sel], ignore_index=True)
    metrics["channel_avg"] = {k: round(v, 4) for k, v in score(actual, frame).items()}

    return {
        "metrics": metrics,
        "folds": fold_rows,
        "avg_fit_forecast_seconds": round(float(np.mean(runtimes)), 3),
        "origins": [str(o.date()) for o in origins],
        "horizon": horizon,
    }
