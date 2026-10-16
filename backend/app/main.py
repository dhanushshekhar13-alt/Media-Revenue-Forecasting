"""FastAPI entry point for the Media Revenue Forecasting Platform."""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from app import service

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the default dataset and pre-fit the ROAS model in the background so the first request is fast."""
    if os.getenv("MRF_WARMUP", "1") == "1":
        def _warm():
            try:
                service.fit_model(service.get_dataset(service.DEFAULT_DATASET), "roas_quantile")
            except Exception:  # pragma: no cover - warm-up is best effort
                pass

        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Media Revenue Forecasting Platform",
    description="Probabilistic (P10/P50/P90) revenue forecasting and budget allocation across Google, Meta and Bing.",
    version="1.0.0",
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=origins != ["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

ModelName = Literal["roas_quantile", "prophet", "prophet_spend", "arima"]


@app.middleware("http")
async def timing_header(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _dataset_or_404(dataset_id: str):
    try:
        return service.get_dataset(dataset_id)
    except KeyError:
        raise HTTPException(404, f"dataset '{dataset_id}' not found")


# -- schemas -------------------------------------------------------------------
class ForecastRequest(BaseModel):
    dataset_id: str = service.DEFAULT_DATASET
    model: ModelName = "roas_quantile"
    horizon: int = Field(28, ge=7, le=90)
    channels: list[str] | None = None
    spend_multipliers: dict[str, float] | None = Field(
        None, description="Scale the business-as-usual plan per channel, e.g. {'meta': 1.2}")
    daily_spend: dict[str, list[float]] | None = Field(None, description="Explicit daily spend plan per channel")


class Bound(BaseModel):
    min_spend: float | None = Field(None, ge=0)
    max_spend: float | None = Field(None, ge=0)


class OptimizeRequest(BaseModel):
    dataset_id: str = service.DEFAULT_DATASET
    total_budget: float | None = Field(None, gt=0, description="Horizon budget; defaults to the current plan total")
    horizon: int = Field(28, ge=7, le=90)
    objective: Literal["p10", "p50", "p90"] = "p50"
    bounds: dict[str, Bound] | None = None
    max_change: float | None = Field(None, gt=0, le=1, description="Optional trust region, e.g. 0.3 = each channel "
                                     "stays within +/-30% of its current share")


class BacktestRequest(BaseModel):
    dataset_id: str = service.DEFAULT_DATASET
    models: list[ModelName] | None = None
    folds: int = Field(4, ge=1, le=12)
    horizon: int = Field(28, ge=7, le=56)


# -- routes ----------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "Media Revenue Forecasting Platform", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets():
    return service.list_datasets()


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "file too large (max 20MB)")
    try:
        return service.register_csv(content, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"could not parse CSV: {e}")


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str):
    _dataset_or_404(dataset_id)
    return service.dataset_summary(dataset_id)


@app.get("/api/history")
def history(dataset_id: str = service.DEFAULT_DATASET, start: str | None = None, end: str | None = None,
            freq: Literal["D", "W", "MS"] = "D"):
    _dataset_or_404(dataset_id)
    return service.history(dataset_id, start, end, freq)


@app.get("/api/kpis")
def kpis(dataset_id: str = service.DEFAULT_DATASET):
    _dataset_or_404(dataset_id)
    return service.weekly_kpis(dataset_id)


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    df = _dataset_or_404(req.dataset_id)
    unknown = set(req.channels or []) - set(df["channel"].unique())
    if unknown:
        raise HTTPException(422, f"unknown channels: {sorted(unknown)}")
    return service.forecast(req.dataset_id, req.model, req.horizon, req.channels,
                            req.spend_multipliers, req.daily_spend)


@app.get("/api/models/roas")
def roas_model(dataset_id: str = service.DEFAULT_DATASET):
    _dataset_or_404(dataset_id)
    return service.model_details(dataset_id)


@app.get("/api/response-curves")
def response_curves(dataset_id: str = service.DEFAULT_DATASET, horizon: int = Query(28, ge=7, le=90)):
    _dataset_or_404(dataset_id)
    return service.curves(dataset_id, horizon)


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    _dataset_or_404(req.dataset_id)
    bounds = {ch: b.model_dump() for ch, b in (req.bounds or {}).items()}
    try:
        return service.optimize(req.dataset_id, req.total_budget, req.horizon, req.objective, bounds,
                                req.max_change)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/benchmark")
def benchmark():
    report = service.benchmark_report()
    if report is None:
        raise HTTPException(404, "benchmark report not generated yet; run `python -m scripts.benchmark`")
    return report


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    _dataset_or_404(req.dataset_id)
    return service.run_backtest(req.dataset_id, req.models, req.folds, req.horizon)


@app.get("/api/cache")
def cache_stats():
    return service.cache.stats()


@app.delete("/api/cache")
def cache_clear():
    service.cache.clear()
    return service.cache.stats()
