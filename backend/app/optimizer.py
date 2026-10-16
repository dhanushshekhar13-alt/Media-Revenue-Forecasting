"""Budget allocation across channels using the fitted ROAS response curves.

Revenue as a function of channel budget is concave (diminishing returns from the
saturation term), so a greedy marginal-return allocation in small increments is
optimal up to the step size. A risk-averse objective uses the P10 residual
quantile instead of the median, which shifts budget toward less volatile channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.models.roas_quantile import ROASForecaster


@dataclass
class ChannelBounds:
    min_spend: float = 0.0
    max_spend: float = float("inf")


def response_functions(fc: ROASForecaster, plan: pd.DataFrame, quantile: float = 0.5) -> dict:
    fns = {}
    for ch, m in fc.channel_models.items():
        p = plan[plan["channel"] == ch].sort_values("date")
        if p.empty:
            continue
        shift = float(np.nanquantile(fc.residuals[ch], quantile))
        shape = np.maximum(p["spend"].to_numpy(dtype=float), 1e-9)
        fns[ch] = m.response_fn(pd.DatetimeIndex(p["date"]), shape, shift)
    return fns


def response_curves(fc: ROASForecaster, plan: pd.DataFrame, points: int = 40, max_multiple: float = 3.0) -> dict:
    """Revenue (P10/P50/P90) vs horizon spend for each channel, around the planned budget."""
    curves = {}
    fns = {q: response_functions(fc, plan, q) for q in (0.1, 0.5, 0.9)}
    for ch in fns[0.5]:
        planned = float(plan.loc[plan["channel"] == ch, "spend"].sum())
        grid = np.linspace(0, planned * max_multiple, points + 1)[1:]
        rows = []
        for x in grid:
            p10, p50, p90 = (fns[q][ch](x) for q in (0.1, 0.5, 0.9))
            rows.append({"spend": round(float(x), 2), "p10": round(p10, 2), "p50": round(p50, 2), "p90": round(p90, 2),
                         "roas_p50": round(p50 / x, 4)})
        # Marginal ROAS at planned spend (finite difference)
        f = fns[0.5][ch]
        eps = max(planned * 0.01, 1.0)
        curves[ch] = {"planned_spend": round(planned, 2), "points": rows,
                      "marginal_roas_at_plan": round((f(planned + eps) - f(planned - eps)) / (2 * eps), 4)}
    return curves


def optimize_budget(
    fc: ROASForecaster,
    plan: pd.DataFrame,
    total_budget: float,
    bounds: dict[str, ChannelBounds] | None = None,
    quantile: float = 0.5,
    steps: int = 1000,
) -> dict:
    """Allocate `total_budget` over the horizon to maximise revenue at `quantile`."""
    fns = response_functions(fc, plan, quantile)
    channels = list(fns)
    bounds = {ch: (bounds or {}).get(ch, ChannelBounds()) for ch in channels}
    min_total = sum(b.min_spend for b in bounds.values())
    max_total = sum(b.max_spend for b in bounds.values())
    if min_total > total_budget + 1e-6:
        raise ValueError(f"channel minimums ({min_total:,.0f}) exceed total budget ({total_budget:,.0f})")
    if max_total < total_budget - 1e-6:
        raise ValueError(f"channel maximums ({max_total:,.0f}) are below total budget ({total_budget:,.0f})")

    alloc = {ch: bounds[ch].min_spend for ch in channels}
    current = {ch: fns[ch](alloc[ch]) for ch in channels}
    remaining = total_budget - min_total
    step = remaining / steps if steps else remaining
    while remaining > 1e-6:
        d = min(step, remaining)
        best, best_gain = None, -np.inf
        for ch in channels:
            if alloc[ch] + d > bounds[ch].max_spend + 1e-6:
                continue
            gain = fns[ch](alloc[ch] + d) - current[ch]
            if gain > best_gain:
                best, best_gain = ch, gain
        if best is None:  # every channel capped
            break
        alloc[best] += d
        current[best] += best_gain
        remaining -= d

    # Baseline: today's plan mix scaled to the same budget.
    planned = {ch: float(plan.loc[plan["channel"] == ch, "spend"].sum()) for ch in channels}
    planned_total = sum(planned.values())
    baseline_alloc = {ch: total_budget * planned[ch] / planned_total for ch in channels}
    baseline_rev = {ch: fns[ch](baseline_alloc[ch]) for ch in channels}

    def marginal(ch: str, x: float) -> float:
        eps = max(x * 0.01, 1.0)
        return (fns[ch](x + eps) - fns[ch](max(x - eps, 0))) / (x + eps - max(x - eps, 0))

    opt_total = sum(current.values())
    base_total = sum(baseline_rev.values())
    return {
        "objective_quantile": quantile,
        "total_budget": total_budget,
        "allocation": {
            ch: {
                "spend": round(alloc[ch], 2),
                "share": round(alloc[ch] / total_budget, 4) if total_budget else 0.0,
                "expected_revenue": round(current[ch], 2),
                "roas": round(current[ch] / alloc[ch], 4) if alloc[ch] > 0 else None,
                "marginal_roas": round(marginal(ch, alloc[ch]), 4),
                "baseline_spend": round(baseline_alloc[ch], 2),
                "baseline_revenue": round(baseline_rev[ch], 2),
            }
            for ch in channels
        },
        "expected_revenue": round(opt_total, 2),
        "baseline_revenue": round(base_total, 2),
        "uplift_pct": round(100 * (opt_total / base_total - 1), 3) if base_total else None,
    }


def scale_plan(plan: pd.DataFrame, totals: dict[str, float]) -> pd.DataFrame:
    """Rescale each channel's daily plan so it sums to `totals[channel]`."""
    out = plan.copy()
    for ch, total in totals.items():
        mask = out["channel"] == ch
        cur = out.loc[mask, "spend"].sum()
        out.loc[mask, "spend"] = out.loc[mask, "spend"] * (total / cur if cur > 0 else 0)
    return out
