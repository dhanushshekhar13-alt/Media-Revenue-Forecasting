"""Baseline forecasters the ROAS approach replaced: Prophet and (S)ARIMA.

All baselines share the interface `fit(df) -> self` and `forecast(future_spend) -> dict`
matching `ROASForecaster.forecast` (per-channel frames + a total frame), so the
evaluation harness and API can swap them freely.

* `prophet`         - univariate Prophet on daily revenue (the typical baseline)
* `prophet_spend`   - Prophet with planned spend as an extra regressor (a stronger,
                      fairer baseline that also sees the media plan)
* `arima`           - SARIMA(1,1,1)(1,0,1)[7] on log revenue
The channel total is modelled directly on the aggregated revenue series.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def _series_by_channel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {ch: g[["date", "spend", "revenue"]].sort_values("date").reset_index(drop=True)
           for ch, g in df.groupby("channel")}
    out["__total__"] = df.groupby("date", as_index=False)[["spend", "revenue"]].sum()
    return out


def _future_by_channel(future_spend: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {ch: g[["date", "spend"]].sort_values("date").reset_index(drop=True)
           for ch, g in future_spend.groupby("channel")}
    out["__total__"] = future_spend.groupby("date", as_index=False)["spend"].sum()
    return out


def _package(per_series: dict[str, pd.DataFrame]) -> dict:
    total = per_series.pop("__total__")
    return {
        "channels": per_series,
        "total": total,
        "horizon_total": {  # sum of daily quantiles (approximation - baselines lack joint draws)
            "p10": float(total["p10"].sum()), "p50": float(total["p50"].sum()), "p90": float(total["p90"].sum()),
        },
    }


@dataclass
class ProphetForecaster:
    use_spend_regressor: bool = False
    interval_width: float = 0.8
    models: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, channels: list[str] | None = None) -> "ProphetForecaster":
        from prophet import Prophet

        logging.getLogger("cmdstanpy").disabled = True

        for key, s in _series_by_channel(df).items():
            if channels and key != "__total__" and key not in channels:
                continue
            m = Prophet(interval_width=self.interval_width, weekly_seasonality=True,
                        yearly_seasonality=True, daily_seasonality=False, uncertainty_samples=300)
            m.add_country_holidays(country_name="US")
            train = pd.DataFrame({"ds": s["date"], "y": s["revenue"]})
            if self.use_spend_regressor:
                train["spend"] = s["spend"].to_numpy()
                m.add_regressor("spend")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(train)
            self.models[key] = m
        return self

    def forecast(self, future_spend: pd.DataFrame) -> dict:
        out = {}
        for key, plan in _future_by_channel(future_spend).items():
            if key not in self.models:
                continue
            fut = pd.DataFrame({"ds": plan["date"]})
            if self.use_spend_regressor:
                fut["spend"] = plan["spend"].to_numpy()
            p = self.models[key].predict(fut)
            out[key] = pd.DataFrame({
                "date": plan["date"], "spend": plan["spend"],
                "p10": np.maximum(p["yhat_lower"].to_numpy(), 0),
                "p50": np.maximum(p["yhat"].to_numpy(), 0),
                "p90": np.maximum(p["yhat_upper"].to_numpy(), 0),
            })
            out[key]["mean"] = out[key]["p50"]
        return _package(out)


@dataclass
class ARIMAForecaster:
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7)
    alpha: float = 0.2  # 80% interval -> P10/P90
    train_days: int = 730
    models: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, channels: list[str] | None = None) -> "ARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        for key, s in _series_by_channel(df).items():
            if channels and key != "__total__" and key not in channels:
                continue
            y = np.log(s["revenue"].clip(lower=1.0).to_numpy()[-self.train_days:])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[key] = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        return self

    def forecast(self, future_spend: pd.DataFrame) -> dict:
        out = {}
        for key, plan in _future_by_channel(future_spend).items():
            if key not in self.models:
                continue
            fc = self.models[key].get_forecast(len(plan))
            ci = np.asarray(fc.conf_int(alpha=self.alpha))
            out[key] = pd.DataFrame({
                "date": plan["date"], "spend": plan["spend"],
                "p10": np.exp(ci[:, 0]), "p50": np.exp(np.asarray(fc.predicted_mean)), "p90": np.exp(ci[:, 1]),
            })
            out[key]["mean"] = out[key]["p50"]
        return _package(out)


MODEL_REGISTRY = {
    "prophet": lambda: ProphetForecaster(use_spend_regressor=False),
    "prophet_spend": lambda: ProphetForecaster(use_spend_regressor=True),
    "arima": lambda: ARIMAForecaster(),
}
