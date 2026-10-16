"""Application service layer: datasets, cached model fits, forecasts, optimisation, backtests."""
from __future__ import annotations

import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.cache import ModelCache, fingerprint_df
from app.data.generator import generate_dataset
from app.evaluation import backtest, rolling_origins
from app.models.baselines import MODEL_REGISTRY
from app.models.roas_quantile import ROASForecaster
from app.optimizer import ChannelBounds, optimize_budget, response_curves, scale_plan

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MRF_DATA_DIR", BASE_DIR / "data"))
REPORT_PATH = Path(os.getenv("MRF_REPORT_PATH", BASE_DIR / "reports" / "benchmark.json"))
CACHE_DIR = os.getenv("MRF_CACHE_DIR", str(BASE_DIR / ".cache"))
DEFAULT_DATASET = "synthetic"
MODELS = ["roas_quantile", *MODEL_REGISTRY]
REQUIRED_COLUMNS = {"date", "channel", "spend", "revenue"}
MIN_DAYS = 450

cache = ModelCache(maxsize=int(os.getenv("MRF_CACHE_SIZE", "64")),
                   disk_dir=CACHE_DIR if os.getenv("MRF_DISK_CACHE", "1") == "1" else None)
_datasets: dict[str, pd.DataFrame] = {}
_lock = threading.Lock()


# -- datasets -----------------------------------------------------------------
def _load_default() -> pd.DataFrame:
    path = DATA_DIR / "media_daily.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = generate_dataset()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def get_dataset(dataset_id: str = DEFAULT_DATASET) -> pd.DataFrame:
    with _lock:
        if dataset_id not in _datasets:
            if dataset_id != DEFAULT_DATASET:
                raise KeyError(dataset_id)
            _datasets[dataset_id] = _load_default()
        return _datasets[dataset_id]


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["channel"] = df["channel"].astype(str).str.strip().str.lower()
    for c in ("spend", "revenue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "channel", "spend", "revenue"])
    df = df.groupby(["date", "channel"], as_index=False).sum(numeric_only=True)
    if (df["spend"] < 0).any() or (df["revenue"] < 0).any():
        raise ValueError("spend and revenue must be non-negative")
    out = []
    for ch, g in df.groupby("channel"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full)
        g["channel"] = ch
        g[["spend", "revenue"]] = g[["spend", "revenue"]].interpolate(limit_direction="both")
        if len(g) < MIN_DAYS:
            raise ValueError(f"channel '{ch}' has {len(g)} days of history; at least {MIN_DAYS} are required")
        g.index.name = "date"
        out.append(g.reset_index())
    df = pd.concat(out, ignore_index=True)
    # Keep the common date range so channel totals are comparable.
    start = df.groupby("channel")["date"].min().max()
    end = df.groupby("channel")["date"].max().min()
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df["revenue"] = df["revenue"].clip(lower=1.0)
    return df.sort_values(["date", "channel"]).reset_index(drop=True)


def register_csv(content: bytes, name: str | None = None) -> dict:
    df = validate_frame(pd.read_csv(io.BytesIO(content)))
    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    with _lock:
        _datasets[dataset_id] = df
    return dataset_summary(dataset_id) | {"name": name}


def list_datasets() -> list[dict]:
    get_dataset(DEFAULT_DATASET)
    return [dataset_summary(k) for k in list(_datasets)]


def dataset_summary(dataset_id: str) -> dict:
    df = get_dataset(dataset_id)
    by_ch = df.groupby("channel").agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
    return {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "channels": sorted(df["channel"].unique().tolist()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "totals": {ch: {"spend": round(r.spend, 2), "revenue": round(r.revenue, 2),
                        "roas": round(r.revenue / r.spend, 4) if r.spend else None} for ch, r in by_ch.iterrows()},
    }


def history(dataset_id: str, start: str | None = None, end: str | None = None, freq: str = "D") -> dict:
    df = get_dataset(dataset_id)
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    wide = df.pivot_table(index="date", columns="channel", values=["spend", "revenue"], aggfunc="sum")
    if freq != "D":
        wide = wide.resample(freq).sum()
    records = []
    for d, row in wide.iterrows():
        rec = {"date": str(d.date())}
        for (metric, ch), v in row.items():
            rec[f"{ch}_{metric}"] = round(float(v), 2)
        rec["total_spend"] = round(float(row["spend"].sum()), 2)
        rec["total_revenue"] = round(float(row["revenue"].sum()), 2)
        records.append(rec)
    return {"dataset_id": dataset_id, "freq": freq, "channels": sorted(df["channel"].unique().tolist()),
            "records": records}


# -- spend plans --------------------------------------------------------------
def default_plan(df: pd.DataFrame, horizon: int, multipliers: dict[str, float] | None = None,
                 channels: list[str] | None = None) -> pd.DataFrame:
    """Business-as-usual media plan: recent 28-day weekday-average spend, optionally scaled per channel."""
    last = pd.Timestamp(df["date"].max())
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    channels = channels or sorted(df["channel"].unique())
    recent = df[df["date"] > last - pd.Timedelta(days=28)]
    rows = []
    for ch in channels:
        r = recent[recent["channel"] == ch]
        by_dow = r.groupby(r["date"].dt.dayofweek)["spend"].mean()
        mult = (multipliers or {}).get(ch, 1.0)
        for d in dates:
            rows.append({"date": d, "channel": ch, "spend": float(by_dow.get(d.dayofweek, r["spend"].mean())) * mult})
    return pd.DataFrame(rows)


def custom_plan(df: pd.DataFrame, daily_spend: dict[str, list[float]]) -> pd.DataFrame:
    last = pd.Timestamp(df["date"].max())
    rows = []
    for ch, values in daily_spend.items():
        for i, v in enumerate(values):
            rows.append({"date": last + pd.Timedelta(days=i + 1), "channel": ch, "spend": max(float(v), 0.0)})
    return pd.DataFrame(rows)


# -- fitting ------------------------------------------------------------------
def _build(model: str):
    if model == "roas_quantile":
        return ROASForecaster()
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]()
    raise KeyError(model)


def fit_model(df: pd.DataFrame, model: str, channels: list[str] | None = None, use_cache: bool = True):
    channels = sorted(channels or df["channel"].unique())
    train = df[df["channel"].isin(channels)]

    def compute():
        return _build(model).fit(train, channels)

    if not use_cache:
        return compute()
    key = (fingerprint_df(train), model, str(train["date"].max().date()), channels, "v1")
    return cache.get_or_compute(f"fit-{model}", key, compute)


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return json.loads(out.round(4).to_json(orient="records"))


def forecast(dataset_id: str, model: str = "roas_quantile", horizon: int = 28,
             channels: list[str] | None = None, multipliers: dict[str, float] | None = None,
             daily_spend: dict[str, list[float]] | None = None, use_cache: bool = True) -> dict:
    df = get_dataset(dataset_id)
    channels = sorted(channels or df["channel"].unique())
    if daily_spend:
        plan = custom_plan(df, {c: v[:horizon] for c, v in daily_spend.items() if c in channels})
        missing = set(channels) - set(plan["channel"])
        if missing:
            plan = pd.concat([plan, default_plan(df, horizon, multipliers, sorted(missing))], ignore_index=True)
    else:
        plan = default_plan(df, horizon, multipliers, channels)
    fc_model = fit_model(df, model, channels, use_cache)
    fc = fc_model.forecast(plan)
    return {
        "dataset_id": dataset_id,
        "model": model,
        "horizon": horizon,
        "train_end": str(df["date"].max().date()),
        "channels": {ch: _frame_records(f) for ch, f in fc["channels"].items()},
        "total": _frame_records(fc["total"]),
        "horizon_total": {k: round(v, 2) for k, v in fc["horizon_total"].items()},
        "planned_spend": {ch: round(float(plan.loc[plan["channel"] == ch, "spend"].sum()), 2) for ch in channels},
    }


def model_details(dataset_id: str) -> dict:
    df = get_dataset(dataset_id)
    return fit_model(df, "roas_quantile").describe()


def curves(dataset_id: str, horizon: int = 28) -> dict:
    df = get_dataset(dataset_id)
    fc = fit_model(df, "roas_quantile")
    plan = default_plan(df, horizon)
    return {"dataset_id": dataset_id, "horizon": horizon, "curves": response_curves(fc, plan)}


def optimize(dataset_id: str, total_budget: float | None, horizon: int = 28, objective: str = "p50",
             bounds: dict[str, dict] | None = None, max_change: float | None = None) -> dict:
    df = get_dataset(dataset_id)
    fc = fit_model(df, "roas_quantile")
    plan = default_plan(df, horizon)
    planned_total = float(plan["spend"].sum())
    total_budget = float(total_budget or planned_total)
    q = {"p10": 0.1, "p50": 0.5, "p90": 0.9}[objective]
    b = {ch: ChannelBounds(min_spend=v.get("min_spend") or 0.0, max_spend=v.get("max_spend") or float("inf"))
         for ch, v in (bounds or {}).items()}
    if max_change is not None:
        # Trust region: keep each channel within +/- max_change of its budget-scaled current plan, where
        # the response curve is supported by historical data.
        for ch in plan["channel"].unique():
            cur = float(plan.loc[plan["channel"] == ch, "spend"].sum()) * total_budget / planned_total
            lo, hi = cur * (1 - max_change), cur * (1 + max_change)
            bb = b.get(ch, ChannelBounds())
            b[ch] = ChannelBounds(max(bb.min_spend, lo), min(bb.max_spend, hi))
    result = optimize_budget(fc, plan, total_budget, b, quantile=q)
    # Full probabilistic forecast for both allocations (joint residuals -> proper total intervals).
    opt_plan = scale_plan(plan, {ch: a["spend"] for ch, a in result["allocation"].items()})
    base_plan = scale_plan(plan, {ch: a["baseline_spend"] for ch, a in result["allocation"].items()})
    result["optimized_forecast"] = {k: round(v, 2) for k, v in fc.forecast(opt_plan)["horizon_total"].items()}
    result["baseline_forecast"] = {k: round(v, 2) for k, v in fc.forecast(base_plan)["horizon_total"].items()}
    result["planned_budget"] = round(planned_total, 2)
    result["horizon"] = horizon
    return result


# -- backtests ----------------------------------------------------------------
def run_backtest(dataset_id: str, models: list[str] | None = None, folds: int = 6, horizon: int = 28,
                 use_cache: bool = True) -> dict:
    df = get_dataset(dataset_id)
    origins = rolling_origins(df, folds, horizon)
    results = {}
    for model in models or MODELS:
        def fit_forecast(train, future, model=model):
            return fit_model(train, model, use_cache=use_cache).forecast(future)
        results[model] = backtest(df, fit_forecast, origins, horizon)
    return {"dataset_id": dataset_id, "folds": folds, "horizon": horizon, "results": results}


def benchmark_report() -> dict | None:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text())
    return None


def weekly_kpis(dataset_id: str) -> dict:
    df = get_dataset(dataset_id)
    last = df["date"].max()
    cur = df[df["date"] > last - pd.Timedelta(days=28)]
    prev = df[(df["date"] <= last - pd.Timedelta(days=28)) & (df["date"] > last - pd.Timedelta(days=56))]

    def agg(x):
        s, r = float(x["spend"].sum()), float(x["revenue"].sum())
        return {"spend": round(s, 2), "revenue": round(r, 2), "roas": round(r / s, 4) if s else None}

    out = {"window_days": 28, "as_of": str(last.date()), "total": agg(cur), "previous_total": agg(prev), "channels": {}}
    for ch in sorted(df["channel"].unique()):
        out["channels"][ch] = {"current": agg(cur[cur["channel"] == ch]), "previous": agg(prev[prev["channel"] == ch])}
    return out

