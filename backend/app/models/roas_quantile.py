"""ROAS-driven probabilistic forecaster (P10 / P50 / P90).

Instead of extrapolating revenue as a univariate time series (what Prophet/ARIMA
do), we forecast *return on ad spend* and multiply it by the planned spend:

    revenue_t = spend_t * ROAS_t

ROAS is far more stable than revenue because most revenue variance comes from
budget decisions that are known in advance (the media plan). Each channel gets
its own model, with channel-specific hyper-parameters chosen by time-series
validation:

    log ROAS_t = b0 + b1 * log(adstock_t / spend_t)          (carry-over)
                    + b2 * log(1 + adstock_t / half_sat)       (saturation)
                    + calendar (day-of-week, yearly Fourier, retail events, trend)

fitted with recency-weighted ridge regression. An AR(1) correction on recent
residuals tracks short-lived efficiency shocks.

Uncertainty comes from *out-of-sample* residuals collected by an inner rolling
backtest on the training window (split-conformal style). Residuals are stored
jointly across channels by (origin, horizon) so that channel-total intervals
keep the cross-channel correlation instead of assuming independence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.features import adstock, adstock_carry, calendar_features

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class ChannelConfig:
    decay_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    half_sat_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    recency_halflife_days: float = 365.0
    ridge_alpha: float = 1.0
    validation_days: int = 56
    fourier_order: int = 3


# Channel-specific search spaces / settings. Meta has longer carry-over and noisier
# response, Bing is low-volume and noisy so it gets stronger regularisation.
CHANNEL_CONFIGS: dict[str, ChannelConfig] = {
    "google": ChannelConfig(decay_grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), ridge_alpha=0.5),
    "meta": ChannelConfig(decay_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), ridge_alpha=1.0,
                          recency_halflife_days=270.0),
    "bing": ChannelConfig(decay_grid=(0.0, 0.1, 0.2, 0.3, 0.4), ridge_alpha=3.0),
}


def config_for(channel: str) -> ChannelConfig:
    return CHANNEL_CONFIGS.get(channel, ChannelConfig())


@dataclass
class ChannelROASModel:
    """Point model of log-ROAS for one channel."""

    channel: str
    origin: pd.Timestamp
    config: ChannelConfig = field(default_factory=ChannelConfig)
    decay: float = 0.3
    half_sat: float = 1.0
    model: Ridge | None = None
    columns: list[str] = field(default_factory=list)
    carry: float = 0.0
    last_resid: float = 0.0
    ar_phi: float = 0.0
    train_end: pd.Timestamp | None = None

    # -- design matrix -------------------------------------------------
    def _design(self, dates: pd.DatetimeIndex, spend: np.ndarray, carry: float) -> pd.DataFrame:
        spend = np.maximum(np.asarray(spend, dtype=float), 1.0)
        ad = np.maximum(adstock(spend, self.decay, carry), 1.0)
        X = calendar_features(dates, self.origin, self.config.fourier_order)
        X["carry_ratio"] = np.log(ad / spend)
        X["saturation"] = np.log1p(ad / self.half_sat)
        return X

    def _weights(self, dates: pd.DatetimeIndex) -> np.ndarray:
        age = (dates.max() - dates).days.to_numpy()
        return 0.5 ** (age / self.config.recency_halflife_days)

    def _fit_once(self, hist: pd.DataFrame) -> tuple[Ridge, pd.DataFrame, np.ndarray]:
        dates = pd.DatetimeIndex(hist["date"])
        X = self._design(dates, hist["spend"].to_numpy(), 0.0)
        y = np.log(hist["revenue"].to_numpy() / np.maximum(hist["spend"].to_numpy(), 1.0))
        m = Ridge(alpha=self.config.ridge_alpha)
        m.fit(X.to_numpy(), y, sample_weight=self._weights(dates))
        return m, X, y

    # -- public API ----------------------------------------------------
    def fit(self, hist: pd.DataFrame, search: bool = True) -> "ChannelROASModel":
        hist = hist.sort_values("date").reset_index(drop=True)
        if search:
            self._search(hist)
        m, X, y = self._fit_once(hist)
        self.model, self.columns = m, list(X.columns)
        resid = y - m.predict(X.to_numpy())
        r = resid[-180:]
        denom = float(np.dot(r[:-1], r[:-1]))
        self.ar_phi = float(np.clip(np.dot(r[1:], r[:-1]) / denom, 0.0, 0.95)) if denom > 0 else 0.0
        self.last_resid = float(np.mean(resid[-3:]))
        self.carry = adstock_carry(np.maximum(hist["spend"].to_numpy(), 1.0), self.decay)
        self.train_end = pd.Timestamp(hist["date"].max())
        return self

    def _search(self, hist: pd.DataFrame) -> None:
        v = self.config.validation_days
        train, valid = hist.iloc[:-v], hist.iloc[-v:]
        median_spend = float(np.median(train["spend"]))
        best = (np.inf, self.decay, self.half_sat)
        for decay in self.config.decay_grid:
            for mult in self.config.half_sat_multipliers:
                self.decay, self.half_sat = decay, median_spend * mult
                m, _, _ = self._fit_once(train)
                carry = adstock_carry(np.maximum(train["spend"].to_numpy(), 1.0), decay)
                Xv = self._design(pd.DatetimeIndex(valid["date"]), valid["spend"].to_numpy(), carry)
                pred = np.exp(m.predict(Xv.to_numpy())) * valid["spend"].to_numpy()
                mape = float(np.mean(np.abs(pred - valid["revenue"].to_numpy()) / valid["revenue"].to_numpy()))
                if mape < best[0]:
                    best = (mape, decay, median_spend * mult)
        _, self.decay, self.half_sat = best

    def predict_log_roas(self, dates: pd.DatetimeIndex, spend: np.ndarray, ar_correction: bool = True) -> np.ndarray:
        assert self.model is not None, "model not fitted"
        X = self._design(pd.DatetimeIndex(dates), spend, self.carry)
        pred = self.model.predict(X[self.columns].to_numpy())
        if ar_correction:
            h = np.arange(1, len(pred) + 1)
            pred = pred + self.last_resid * self.ar_phi ** h
        return pred

    def response_fn(self, dates: pd.DatetimeIndex, shape: np.ndarray, residual_shift: float = 0.0):
        """Fast closure: horizon total spend -> horizon revenue, with spend spread by `shape`.

        The calendar part of the design is computed once; only the two spend-dependent
        terms are re-evaluated, which makes the budget optimiser ~100x cheaper than
        calling `predict_log_roas` per candidate allocation.
        """
        assert self.model is not None
        dates = pd.DatetimeIndex(dates)
        shape = np.asarray(shape, dtype=float)
        shape = shape / shape.sum()
        coef = dict(zip(self.columns, self.model.coef_))
        X = self._design(dates, np.ones(len(dates)), 0.0)[self.columns]
        spend_cols = ["carry_ratio", "saturation"]
        cal_cols = [c for c in self.columns if c not in spend_cols]
        base = X[cal_cols].to_numpy() @ np.array([coef[c] for c in cal_cols]) + self.model.intercept_
        base = base + self.last_resid * self.ar_phi ** np.arange(1, len(dates) + 1) + residual_shift
        c_carry, c_sat = coef["carry_ratio"], coef["saturation"]
        decay, half_sat, carry = self.decay, self.half_sat, self.carry

        def f(total: float) -> float:
            if total <= 0:
                return 0.0
            spend = np.maximum(total * shape, 1.0)
            ad = np.maximum(adstock(spend, decay, carry), 1.0)
            log_roas = base + c_carry * np.log(ad / spend) + c_sat * np.log1p(ad / half_sat)
            return float(np.sum(np.exp(log_roas) * spend))

        return f

    def coefficients(self) -> dict[str, float]:
        assert self.model is not None
        return {c: float(v) for c, v in zip(self.columns, self.model.coef_)} | {"intercept": float(self.model.intercept_)}


@dataclass
class ROASForecaster:
    """Multi-channel ROAS forecaster with joint conformal residuals."""

    horizon: int = 28
    inner_folds: int = 12
    channel_models: dict[str, ChannelROASModel] = field(default_factory=dict)
    residuals: dict[str, np.ndarray] = field(default_factory=dict)  # channel -> (folds, horizon)
    train_end: pd.Timestamp | None = None
    origin: pd.Timestamp | None = None

    def fit(self, df: pd.DataFrame, channels: list[str] | None = None) -> "ROASForecaster":
        df = df.sort_values("date")
        channels = channels or sorted(df["channel"].unique())
        self.origin = pd.Timestamp(df["date"].min())
        self.train_end = pd.Timestamp(df["date"].max())
        for ch in channels:
            hist = df[df["channel"] == ch][["date", "spend", "revenue"]].reset_index(drop=True)
            cm = ChannelROASModel(ch, self.origin, config_for(ch)).fit(hist, search=True)
            self.channel_models[ch] = cm
            self.residuals[ch] = self._inner_backtest(cm, hist)
        return self

    def _inner_backtest(self, cm: ChannelROASModel, hist: pd.DataFrame) -> np.ndarray:
        """Out-of-sample log residuals at horizons 1..H from `inner_folds` rolling origins."""
        H = self.horizon
        res = np.full((self.inner_folds, H), np.nan)
        for k in range(self.inner_folds):
            cut = len(hist) - H * (k + 1)
            if cut < 365:
                break
            train, test = hist.iloc[:cut], hist.iloc[cut:cut + H]
            inner = ChannelROASModel(cm.channel, cm.origin, cm.config, decay=cm.decay, half_sat=cm.half_sat)
            inner.fit(train, search=False)
            pred = inner.predict_log_roas(pd.DatetimeIndex(test["date"]), test["spend"].to_numpy())
            actual = np.log(test["revenue"].to_numpy() / np.maximum(test["spend"].to_numpy(), 1.0))
            res[k, :len(test)] = actual - pred
        return res

    # -- forecasting ------------------------------------------------------
    def _residual_pool(self, channels: list[str]) -> np.ndarray:
        """Joint residual draws, shape (n_draws, n_channels). Rows with NaN dropped."""
        stacked = np.stack([self.residuals[c].ravel() for c in channels], axis=1)
        return stacked[~np.isnan(stacked).any(axis=1)]

    def forecast(self, future_spend: pd.DataFrame, quantiles: tuple[float, ...] = QUANTILES) -> dict:
        """future_spend: long df with date, channel, spend (planned media)."""
        channels = [c for c in self.channel_models if c in set(future_spend["channel"])]
        pool = self._residual_pool(channels)
        per_channel: dict[str, pd.DataFrame] = {}
        point_rev = []
        for j, ch in enumerate(channels):
            plan = future_spend[future_spend["channel"] == ch].sort_values("date")
            dates = pd.DatetimeIndex(plan["date"])
            spend = plan["spend"].to_numpy(dtype=float)
            log_roas = self.channel_models[ch].predict_log_roas(dates, spend)
            qs = np.quantile(pool[:, j], quantiles)
            out = pd.DataFrame({"date": dates, "spend": spend})
            for q, r in zip(quantiles, qs):
                out[f"roas_p{int(q * 100)}"] = np.exp(log_roas + r)
                out[f"p{int(q * 100)}"] = out[f"roas_p{int(q * 100)}"] * spend
            out["mean"] = (np.exp(log_roas[:, None] + pool[:, j][None, :]).mean(axis=1)) * spend
            per_channel[ch] = out
            point_rev.append(np.exp(log_roas) * np.maximum(spend, 0))
        # Channel total: sum joint residual draws so cross-channel correlation is kept.
        base = np.stack(point_rev, axis=1)                     # (days, channels)
        draws = (base[:, None, :] * np.exp(pool[None, :, :])).sum(axis=2)  # (days, n_draws)
        total = pd.DataFrame({"date": per_channel[channels[0]]["date"],
                              "spend": sum(per_channel[c]["spend"] for c in channels)})
        for q in quantiles:
            total[f"p{int(q * 100)}"] = np.quantile(draws, q, axis=1)
        total["mean"] = draws.mean(axis=1)
        # Horizon totals (sum over days) - again via joint draws per day-index.
        horizon_draws = draws.sum(axis=0)
        summary = {f"p{int(q * 100)}": float(np.quantile(horizon_draws, q)) for q in quantiles}
        return {"channels": per_channel, "total": total, "horizon_total": summary}

    def expected_revenue(self, channel: str, dates: pd.DatetimeIndex, spend: np.ndarray, quantile: float = 0.5) -> float:
        """Horizon revenue for one channel at a given quantile of the residual distribution."""
        log_roas = self.channel_models[channel].predict_log_roas(dates, spend)
        r = self.residuals[channel]
        q = float(np.nanquantile(r, quantile))
        return float(np.sum(np.exp(log_roas + q) * spend))

    def describe(self) -> dict:
        return {
            ch: {
                "adstock_decay": m.decay,
                "half_saturation": round(m.half_sat, 2),
                "ar_phi": round(m.ar_phi, 3),
                "residual_sd": round(float(np.nanstd(self.residuals[ch])), 4),
                "coefficients": {k: round(v, 4) for k, v in m.coefficients().items()},
            }
            for ch, m in self.channel_models.items()
        }
