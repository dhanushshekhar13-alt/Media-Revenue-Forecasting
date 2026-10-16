"""Feature engineering shared by the data generator and the forecasting models."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAY_COLUMNS = ["bfcm", "bfcm_window", "holiday_season", "christmas_shutdown", "summer_event", "valentines"]


def adstock(spend: np.ndarray, decay: float, carry: float = 0.0) -> np.ndarray:
    """Geometric adstock: a_t = s_t + decay * a_{t-1}, normalised so a constant spend s maps to s.

    `carry` seeds the recursion (the un-normalised adstock of the day before the series),
    which lets a forecast continue the adstock of the training history.
    """
    spend = np.asarray(spend, dtype=float)
    out = np.empty_like(spend)
    acc = carry
    for i, s in enumerate(spend):
        acc = s + decay * acc
        out[i] = acc
    return out * (1 - decay)


def adstock_carry(spend: np.ndarray, decay: float) -> float:
    """Un-normalised adstock state after the last element of `spend`."""
    acc = 0.0
    for s in np.asarray(spend, dtype=float):
        acc = s + decay * acc
    return acc


def _thanksgiving(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    return first_thu + pd.Timedelta(weeks=3)


def _summer_event(year: int) -> pd.Timestamp:
    jul1 = pd.Timestamp(year=year, month=7, day=1)
    first_tue = jul1 + pd.Timedelta(days=(1 - jul1.dayofweek) % 7)
    return first_tue + pd.Timedelta(weeks=1)


def holiday_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Retail calendar flags (0/1) for each date."""
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(0, index=dates, columns=HOLIDAY_COLUMNS, dtype=float)
    for year in sorted(set(dates.year) | {dates.year.min() - 1, dates.year.max() + 1}):
        tg = _thanksgiving(year)
        bf, cm = tg + pd.Timedelta(days=1), tg + pd.Timedelta(days=4)
        out.loc[(dates >= bf) & (dates <= cm), "bfcm"] = 1
        out.loc[(dates >= bf - pd.Timedelta(days=14)) & (dates <= cm), "bfcm_window"] = 1
        out.loc[(dates >= pd.Timestamp(year, 12, 1)) & (dates <= pd.Timestamp(year, 12, 22)), "holiday_season"] = 1
        shutdown = [pd.Timestamp(year, 12, 24), pd.Timestamp(year, 12, 25), pd.Timestamp(year, 12, 26),
                    pd.Timestamp(year, 1, 1)]
        out.loc[dates.isin(shutdown), "christmas_shutdown"] = 1
        se = _summer_event(year)
        out.loc[(dates >= se - pd.Timedelta(days=1)) & (dates <= se + pd.Timedelta(days=1)), "summer_event"] = 1
        out.loc[(dates >= pd.Timestamp(year, 2, 7)) & (dates <= pd.Timestamp(year, 2, 14)), "valentines"] = 1
    return out


def calendar_features(dates: pd.DatetimeIndex, origin: pd.Timestamp, fourier_order: int = 3) -> pd.DataFrame:
    """Day-of-week dummies, yearly Fourier terms, trend and holiday flags."""
    dates = pd.DatetimeIndex(dates)
    feats = {}
    for d in range(1, 7):
        feats[f"dow_{d}"] = (dates.dayofweek == d).astype(float)
    t = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    for k in range(1, fourier_order + 1):
        feats[f"sin_{k}"] = np.sin(k * t)
        feats[f"cos_{k}"] = np.cos(k * t)
    feats["trend"] = (dates - origin).days.to_numpy() / 365.25
    df = pd.DataFrame(feats, index=dates)
    return pd.concat([df, holiday_frame(dates)], axis=1)
