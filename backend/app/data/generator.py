"""Synthetic multi-channel media dataset.

Produces daily spend / revenue for Google, Meta and Bing with the dynamics that
make real paid-media revenue hard to forecast from revenue history alone:

* spend is set by humans: budget step changes, flighted campaigns, dark days
* revenue responds to *adstocked* spend through a saturating (Hill) curve
* weekly + yearly seasonality and retail peaks (Black Friday / Cyber Monday,
  Christmas, Prime-Day-style July event, Valentine's)
* slowly drifting channel efficiency (auction pressure, creative fatigue)
* multiplicative log-normal noise that is correlated across channels

The generator is fully deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.features import adstock, holiday_frame


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    base_daily_spend: float
    peak_roas: float          # ROAS at very low spend (before saturation)
    half_saturation: float    # adstocked daily spend at which ROAS halves
    hill_shape: float
    adstock_decay: float
    noise_sigma: float
    weekday_profile: tuple[float, ...]  # Mon..Sun revenue multipliers
    drift_per_year: float     # log-ROAS drift per year
    spend_volatility: float


CHANNEL_SPECS: dict[str, ChannelSpec] = {
    "google": ChannelSpec(
        name="google", base_daily_spend=12_000, peak_roas=6.0, half_saturation=26_000,
        hill_shape=1.0, adstock_decay=0.25, noise_sigma=0.07,
        weekday_profile=(1.05, 1.06, 1.04, 1.02, 0.97, 0.92, 0.94),
        drift_per_year=-0.06, spend_volatility=0.08,
    ),
    "meta": ChannelSpec(
        name="meta", base_daily_spend=8_000, peak_roas=4.4, half_saturation=15_000,
        hill_shape=0.9, adstock_decay=0.55, noise_sigma=0.10,
        weekday_profile=(0.96, 0.97, 0.98, 1.00, 1.03, 1.04, 1.02),
        drift_per_year=-0.10, spend_volatility=0.12,
    ),
    "bing": ChannelSpec(
        name="bing", base_daily_spend=2_500, peak_roas=5.0, half_saturation=7_000,
        hill_shape=1.1, adstock_decay=0.20, noise_sigma=0.12,
        weekday_profile=(1.08, 1.07, 1.05, 1.02, 0.96, 0.90, 0.92),
        drift_per_year=0.03, spend_volatility=0.10,
    ),
}

CHANNELS = tuple(CHANNEL_SPECS)


def _yearly_seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.to_numpy()
    t = 2 * np.pi * doy / 365.25
    return 0.06 * np.sin(t - 1.2) + 0.04 * np.cos(2 * t)


def _spend_plan(spec: ChannelSpec, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    n = len(dates)
    # Budget regime: piecewise-constant multipliers that change every 3-8 weeks.
    regime = np.ones(n)
    i = 0
    level = 1.0
    while i < n:
        length = int(rng.integers(21, 57))
        level = float(np.clip(level * np.exp(rng.normal(0, 0.18)), 0.5, 2.2))
        regime[i:i + length] = level
        i += length
    # Long-run growth of the media program
    growth = np.exp(0.12 * (np.arange(n) / 365.25))
    # Seasonal budget lift into Q4 and around retail events
    hol = holiday_frame(dates)
    seasonal = 1 + 0.35 * hol["bfcm_window"].to_numpy() + 0.20 * hol["holiday_season"].to_numpy() \
        + 0.25 * hol["summer_event"].to_numpy()
    # Flighted promo bursts (~6 per year, 4-10 days each)
    bursts = np.ones(n)
    for _ in range(int(n / 365 * 6)):
        start = int(rng.integers(0, n))
        bursts[start:start + int(rng.integers(4, 11))] *= rng.uniform(1.3, 1.9)
    # Weekday pacing + daily jitter
    dow = np.array([1.02, 1.03, 1.02, 1.0, 0.98, 0.96, 0.99])[dates.dayofweek]
    jitter = np.exp(rng.normal(0, spec.spend_volatility, n))
    spend = spec.base_daily_spend * regime * growth * seasonal * bursts * dow * jitter
    # Occasional dark days (budget cap hit / campaign paused)
    dark = rng.random(n) < 0.004
    spend[dark] *= 0.15
    return np.round(spend, 2)


def generate_dataset(start: str = "2023-01-01", end: str = "2025-12-31", seed: int = 7) -> pd.DataFrame:
    """Return a long dataframe: date, channel, spend, revenue, conversions, impressions, clicks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    hol = holiday_frame(dates)
    yearly = _yearly_seasonality(dates)
    event_lift = (
        0.45 * hol["bfcm"].to_numpy()
        + 0.18 * hol["bfcm_window"].to_numpy()
        + 0.12 * hol["holiday_season"].to_numpy()
        - 0.35 * hol["christmas_shutdown"].to_numpy()
        + 0.20 * hol["summer_event"].to_numpy()
        + 0.08 * hol["valentines"].to_numpy()
    )
    # Common demand shock shared by all channels (market-wide), AR(1)
    common = np.zeros(n)
    eps = rng.normal(0, 0.035, n)
    for t in range(1, n):
        common[t] = 0.7 * common[t - 1] + eps[t]

    rows = []
    for spec in CHANNEL_SPECS.values():
        spend = _spend_plan(spec, dates, rng)
        ad = adstock(spend, spec.adstock_decay)
        sat = 1.0 / (1.0 + (ad / spec.half_saturation) ** spec.hill_shape)
        drift = spec.drift_per_year * np.arange(n) / 365.25
        weekday = np.log(np.array(spec.weekday_profile))[dates.dayofweek]
        log_roas = (
            np.log(spec.peak_roas) + np.log(sat) + drift + weekday + yearly + event_lift + common
            + rng.normal(0, spec.noise_sigma, n)
        )
        roas = np.exp(log_roas)
        # ROAS is defined on *same-day* spend; carry-over revenue makes it "ad / spend" scaled.
        revenue = roas * ad
        aov = {"google": 92.0, "meta": 74.0, "bing": 105.0}[spec.name] * np.exp(rng.normal(0, 0.05, n))
        conversions = np.maximum(0, np.round(revenue / aov))
        cpc = {"google": 1.35, "meta": 0.82, "bing": 1.05}[spec.name] * np.exp(rng.normal(0, 0.06, n))
        clicks = np.round(spend / cpc)
        ctr = {"google": 0.045, "meta": 0.011, "bing": 0.038}[spec.name]
        impressions = np.round(clicks / (ctr * np.exp(rng.normal(0, 0.08, n))))
        rows.append(pd.DataFrame({
            "date": dates, "channel": spec.name, "spend": spend,
            "revenue": np.round(revenue, 2), "conversions": conversions,
            "clicks": clicks, "impressions": impressions,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "channel"]).reset_index(drop=True)


def true_expected_revenue(channel: str, dates: pd.DatetimeIndex, spend: np.ndarray,
                          history_spend: np.ndarray, start: str = "2023-01-01") -> float:
    """Ground-truth expected revenue of a spend plan under the data-generating process.

    Only available for synthetic data; used to check that the optimiser's recommended
    allocation really earns more than the business-as-usual split.
    """
    from app.features import adstock_carry

    spec = CHANNEL_SPECS[channel]
    dates = pd.DatetimeIndex(dates)
    spend = np.asarray(spend, dtype=float)
    carry = adstock_carry(history_spend, spec.adstock_decay)
    ad = adstock(spend, spec.adstock_decay, carry)
    sat = 1.0 / (1.0 + (ad / spec.half_saturation) ** spec.hill_shape)
    hol = holiday_frame(dates)
    event_lift = (
        0.45 * hol["bfcm"].to_numpy() + 0.18 * hol["bfcm_window"].to_numpy()
        + 0.12 * hol["holiday_season"].to_numpy() - 0.35 * hol["christmas_shutdown"].to_numpy()
        + 0.20 * hol["summer_event"].to_numpy() + 0.08 * hol["valentines"].to_numpy()
    )
    drift = spec.drift_per_year * (dates - pd.Timestamp(start)).days.to_numpy() / 365.25
    weekday = np.log(np.array(spec.weekday_profile))[dates.dayofweek]
    log_roas = np.log(spec.peak_roas) + np.log(sat) + drift + weekday + _yearly_seasonality(dates) + event_lift
    noise_var = spec.noise_sigma ** 2 + 0.035 ** 2 / (1 - 0.7 ** 2)
    return float(np.sum(np.exp(log_roas + noise_var / 2) * ad))


if __name__ == "__main__":  # pragma: no cover
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/media_daily.csv"
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")
