"""Reproducible benchmark: model accuracy, caching speed-up and optimiser validation.

    python -m scripts.benchmark            # full run (12 folds, all models)
    python -m scripts.benchmark --quick    # 4 folds

Writes reports/benchmark.json (served by GET /api/benchmark) and reports/benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import service
from app.cache import ModelCache
from app.data.generator import true_expected_revenue
from app.models.roas_quantile import ROASForecaster
from app.optimizer import scale_plan

MODEL_LABELS = {
    "roas_quantile": "ROAS-driven quantile (ours)",
    "prophet": "Prophet (univariate)",
    "prophet_spend": "Prophet + spend regressor",
    "arima": "SARIMA(1,1,1)(1,0,1)[7]",
}


def accuracy(folds: int, horizon: int) -> dict:
    out = {}
    for model in service.MODELS:
        t0 = time.perf_counter()
        res = service.run_backtest(service.DEFAULT_DATASET, [model], folds, horizon, use_cache=False)
        r = res["results"][model]
        r["wall_seconds"] = round(time.perf_counter() - t0, 2)
        r["label"] = MODEL_LABELS[model]
        out[model] = r
        m = r["metrics"]
        print(f"{model:15s} total MAPE {m['total']['mape']:.4f} cov {m['total']['coverage_p10_p90']:.3f} | "
              f"channel MAPE {m['channel_avg']['mape']:.4f} cov {m['channel_avg']['coverage_p10_p90']:.3f} | "
              f"{r['wall_seconds']}s")
    return out


def caching(requests: int = 25) -> dict:
    """Simulate a dashboard session: many scenario forecasts + optimiser runs on one history."""
    df = service.get_dataset(service.DEFAULT_DATASET)
    rng = np.random.default_rng(0)
    scenarios = [{ch: float(rng.uniform(0.6, 1.6)) for ch in ("google", "meta", "bing")} for _ in range(requests)]

    def session(use_cache: bool) -> float:
        t0 = time.perf_counter()
        for mult in scenarios:
            service.forecast(service.DEFAULT_DATASET, multipliers=mult, use_cache=use_cache)
        return time.perf_counter() - t0

    service.cache.clear()
    uncached = session(False)
    service.cache.clear()
    cached = session(True)            # first request fits, the rest reuse it
    warm = session(True)              # everything warm (e.g. after start-up pre-fit)

    # Backtest re-run (e.g. benchmark page refresh, CI) with fitted folds cached on disk.
    tmp = ModelCache(disk_dir=None)
    orig, service.cache = service.cache, tmp
    t0 = time.perf_counter(); service.run_backtest(service.DEFAULT_DATASET, ["roas_quantile"], 4, 28); bt_cold = time.perf_counter() - t0
    t0 = time.perf_counter(); service.run_backtest(service.DEFAULT_DATASET, ["roas_quantile"], 4, 28); bt_warm = time.perf_counter() - t0
    service.cache = orig
    del df
    return {
        "scenario_requests": requests,
        "uncached_seconds": round(uncached, 3),
        "cached_seconds": round(cached, 3),
        "warm_seconds": round(warm, 3),
        "speedup_cold_start": round(uncached / cached, 1),
        "speedup_warm": round(uncached / warm, 1),
        "per_request_ms_uncached": round(1000 * uncached / requests, 1),
        "per_request_ms_warm": round(1000 * warm / requests, 1),
        "backtest_cold_seconds": round(bt_cold, 3),
        "backtest_warm_seconds": round(bt_warm, 3),
        "backtest_speedup": round(bt_cold / bt_warm, 1),
    }


def optimizer_validation(folds: int = 6, horizon: int = 28) -> dict:
    """At several historical origins, compare the optimiser's allocation with the BAU split
    under the *true* data-generating process (possible only because the data is synthetic)."""
    df = service.get_dataset(service.DEFAULT_DATASET)
    rows = []
    for origin in pd.date_range("2025-02-03", periods=folds, freq="56D"):
        train = df[df["date"] < origin]
        fc = ROASForecaster(horizon=horizon).fit(train)
        plan = service.default_plan(train, horizon)
        from app.optimizer import optimize_budget
        res = optimize_budget(fc, plan, float(plan["spend"].sum()))
        alloc = {ch: a["spend"] for ch, a in res["allocation"].items()}
        opt_plan = scale_plan(plan, alloc)

        def truth(p):
            total = 0.0
            for ch in alloc:
                pc = p[p["channel"] == ch].sort_values("date")
                hist = train[train["channel"] == ch].sort_values("date")["spend"].to_numpy()
                total += true_expected_revenue(ch, pd.DatetimeIndex(pc["date"]), pc["spend"].to_numpy(), hist)
            return total

        t_base, t_opt = truth(plan), truth(opt_plan)
        rows.append({"origin": str(origin.date()), "predicted_uplift_pct": res["uplift_pct"],
                     "true_uplift_pct": round(100 * (t_opt / t_base - 1), 3),
                     "allocation_share": {ch: a["share"] for ch, a in res["allocation"].items()}})
        print(f"optimizer {origin.date()} predicted {res['uplift_pct']:.2f}% true {rows[-1]['true_uplift_pct']:.2f}%")
    return {"origins": rows,
            "mean_predicted_uplift_pct": round(float(np.mean([r["predicted_uplift_pct"] for r in rows])), 3),
            "mean_true_uplift_pct": round(float(np.mean([r["true_uplift_pct"] for r in rows])), 3)}


def to_markdown(rep: dict) -> str:
    lines = [f"# Benchmark ({rep['generated_at']})", "",
             f"Dataset: synthetic Google/Meta/Bing daily, {rep['dataset']['start']} to {rep['dataset']['end']}. "
             f"Rolling origin backtest: {rep['folds']} folds x {rep['horizon']}-day horizon, planned spend known.", "",
             "| Model | Total MAPE | Total P10-P90 coverage | Channel MAPE | Channel coverage | Scaled pinball | Fit+forecast (s) |",
             "|---|---|---|---|---|---|---|"]
    for m, r in rep["accuracy"].items():
        t, c = r["metrics"]["total"], r["metrics"]["channel_avg"]
        lines.append(f"| {r['label']} | {t['mape']:.2%} | {t['coverage_p10_p90']:.1%} | {c['mape']:.2%} | "
                     f"{c['coverage_p10_p90']:.1%} | {c['scaled_pinball']:.4f} | {r['avg_fit_forecast_seconds']} |")
    lines += ["", "Per channel (ROAS model):", "", "| Series | MAPE | Coverage |", "|---|---|---|"]
    for s, v in rep["accuracy"]["roas_quantile"]["metrics"].items():
        lines.append(f"| {s} | {v['mape']:.2%} | {v['coverage_p10_p90']:.1%} |")
    c = rep["caching"]
    lines += ["", f"Caching: {c['scenario_requests']} scenario forecasts took {c['uncached_seconds']}s uncached vs "
              f"{c['cached_seconds']}s cached from cold ({c['speedup_cold_start']}x) and {c['warm_seconds']}s warm "
              f"({c['speedup_warm']}x). Backtest re-run: {c['backtest_cold_seconds']}s -> {c['backtest_warm_seconds']}s "
              f"({c['backtest_speedup']}x).", ""]
    o = rep["optimizer"]
    lines += [f"Optimiser: mean predicted uplift {o['mean_predicted_uplift_pct']}% vs true uplift under the "
              f"data-generating process {o['mean_true_uplift_pct']}% (same total budget).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=28)
    args = ap.parse_args()
    folds = 4 if args.quick else args.folds
    df = service.get_dataset(service.DEFAULT_DATASET)
    rep = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "dataset": service.dataset_summary(service.DEFAULT_DATASET),
        "folds": folds, "horizon": args.horizon,
        "accuracy": accuracy(folds, args.horizon),
        "caching": caching(),
        "optimizer": optimizer_validation(),
    }
    del df
    service.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    service.REPORT_PATH.write_text(json.dumps(rep, indent=2, default=str))
    md = to_markdown(rep)
    service.REPORT_PATH.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
