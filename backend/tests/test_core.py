import numpy as np
import pandas as pd
import pytest

from app.cache import ModelCache
from app.data.generator import generate_dataset, true_expected_revenue
from app.evaluation import coverage, mape, pinball, rolling_origins
from app.features import adstock, adstock_carry, holiday_frame
from app.models.roas_quantile import ROASForecaster
from app.optimizer import ChannelBounds, optimize_budget, response_curves


@pytest.fixture(scope="module")
def data():
    return generate_dataset(start="2023-01-01", end="2024-12-31", seed=3)


@pytest.fixture(scope="module")
def fitted(data):
    train = data[data["date"] < "2024-12-01"]
    return train, ROASForecaster(inner_folds=4).fit(train)


def _plan(train, horizon=28):
    last = train["date"].max()
    dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    rows = []
    for ch, g in train.groupby("channel"):
        level = g["spend"].tail(28).mean()
        rows += [{"date": d, "channel": ch, "spend": level} for d in dates]
    return pd.DataFrame(rows)


def test_generator_deterministic_and_complete(data):
    again = generate_dataset(start="2023-01-01", end="2024-12-31", seed=3)
    pd.testing.assert_frame_equal(data, again)
    assert set(data["channel"]) == {"google", "meta", "bing"}
    assert data.groupby("channel").size().nunique() == 1
    assert (data["spend"] > 0).all() and (data["revenue"] > 0).all()


def test_adstock_steady_state_and_carry():
    s = np.full(200, 100.0)
    assert adstock(s, 0.6)[-1] == pytest.approx(100.0, rel=1e-6)
    a, b = s[:150], s[150:]
    continued = adstock(b, 0.6, adstock_carry(a, 0.6))
    np.testing.assert_allclose(continued, adstock(s, 0.6)[150:])


def test_holiday_calendar_black_friday():
    hol = holiday_frame(pd.date_range("2024-11-01", "2024-12-31"))
    assert hol.loc["2024-11-29", "bfcm"] == 1  # Black Friday 2024
    assert hol.loc["2024-12-02", "bfcm"] == 1  # Cyber Monday 2024
    assert hol.loc["2024-12-25", "christmas_shutdown"] == 1


def test_metrics():
    a = np.array([100.0, 200.0])
    assert mape(a, np.array([110.0, 180.0])) == pytest.approx(0.1)
    assert coverage(a, np.array([90, 210]), np.array([110, 220])) == 0.5
    assert pinball(a, a, 0.9) == 0


def test_rolling_origins_end_at_data_end(data):
    origins = rolling_origins(data, folds=3, horizon=28)
    assert origins[-1] + pd.Timedelta(days=27) == data["date"].max()
    assert (origins[1] - origins[0]).days == 28


def test_forecast_quantiles_ordered_and_accurate(data, fitted):
    train, fc = fitted
    test = data[(data["date"] >= "2024-12-01")]
    out = fc.forecast(test[["date", "channel", "spend"]])
    for ch, f in out["channels"].items():
        assert (f["p10"] <= f["p50"]).all() and (f["p50"] <= f["p90"]).all()
        actual = test[test["channel"] == ch]["revenue"].to_numpy()
        assert mape(actual, f["p50"].to_numpy()) < 0.25
    tot = out["total"]
    assert (tot["p10"] <= tot["p90"]).all()
    h = out["horizon_total"]
    assert h["p10"] < h["p50"] < h["p90"]


def test_more_spend_more_revenue_but_lower_roas(fitted):
    train, fc = fitted
    plan = _plan(train)
    lo = fc.forecast(plan)
    hi_plan = plan.assign(spend=plan["spend"] * 1.5)
    hi = fc.forecast(hi_plan)
    for ch in lo["channels"]:
        r_lo, r_hi = lo["channels"][ch]["p50"].sum(), hi["channels"][ch]["p50"].sum()
        assert r_hi > r_lo
        assert r_hi / hi_plan[hi_plan.channel == ch]["spend"].sum() < r_lo / plan[plan.channel == ch]["spend"].sum()


def test_response_fn_matches_forecast(fitted):
    train, fc = fitted
    plan = _plan(train)
    ch = "google"
    p = plan[plan.channel == ch]
    m = fc.channel_models[ch]
    f = m.response_fn(pd.DatetimeIndex(p["date"]), p["spend"].to_numpy())
    direct = np.sum(np.exp(m.predict_log_roas(pd.DatetimeIndex(p["date"]), p["spend"].to_numpy())) * p["spend"])
    assert f(p["spend"].sum()) == pytest.approx(direct, rel=1e-6)


def test_optimizer_respects_budget_and_bounds(fitted):
    train, fc = fitted
    plan = _plan(train)
    budget = float(plan["spend"].sum())
    bounds = {"bing": ChannelBounds(min_spend=50_000, max_spend=60_000)}
    res = optimize_budget(fc, plan, budget, bounds)
    alloc = res["allocation"]
    assert sum(a["spend"] for a in alloc.values()) == pytest.approx(budget, rel=1e-6)
    assert 50_000 - 1 <= alloc["bing"]["spend"] <= 60_000 + 1
    assert res["expected_revenue"] >= res["baseline_revenue"] - 1e-6
    free = optimize_budget(fc, plan, budget)
    mr = [a["marginal_roas"] for a in free["allocation"].values() if a["spend"] > 0]
    assert max(mr) - min(mr) < 0.1 * np.mean(mr)  # equalised marginal returns at the optimum


def test_optimizer_infeasible(fitted):
    train, fc = fitted
    plan = _plan(train)
    with pytest.raises(ValueError):
        optimize_budget(fc, plan, 1000.0, {"google": ChannelBounds(min_spend=5000)})


def test_response_curves_concave(fitted):
    train, fc = fitted
    c = response_curves(fc, _plan(train), points=20)
    for curve in c.values():
        p50 = np.array([p["p50"] for p in curve["points"]])
        assert np.all(np.diff(p50) > 0)
        assert np.all(np.diff(p50, 2) < 1e-6 * p50.max())


def test_true_expected_revenue_increasing(data):
    hist = data[data.channel == "meta"]["spend"].to_numpy()
    dates = pd.date_range("2025-01-01", periods=28)
    a = true_expected_revenue("meta", dates, np.full(28, 5000.0), hist)
    b = true_expected_revenue("meta", dates, np.full(28, 10000.0), hist)
    assert b > a and b / 10000 < a / 5000


def test_cache_hits_and_disk(tmp_path):
    calls = []
    c = ModelCache(maxsize=2, disk_dir=tmp_path)
    fn = lambda: calls.append(1) or {"x": 1}
    assert c.get_or_compute("ns", ("a",), fn) == {"x": 1}
    assert c.get_or_compute("ns", ("a",), fn) == {"x": 1}
    assert len(calls) == 1 and c.stats()["hits"] == 1
    c2 = ModelCache(disk_dir=tmp_path)  # new process, same disk
    assert c2.get_or_compute("ns", ("a",), fn) == {"x": 1}
    assert len(calls) == 1 and c2.stats()["disk_hits"] == 1
    c.get_or_compute("ns", ("b",), fn)
    c.get_or_compute("ns", ("c",), fn)
    assert c.stats()["entries"] == 2  # LRU eviction
