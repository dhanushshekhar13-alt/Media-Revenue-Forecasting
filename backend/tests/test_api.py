import io
import os

os.environ.setdefault("MRF_WARMUP", "0")
os.environ.setdefault("MRF_DISK_CACHE", "0")

import pytest
from fastapi.testclient import TestClient

from app.data.generator import generate_dataset
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_datasets_and_history():
    ds = client.get("/api/datasets").json()
    assert ds[0]["dataset_id"] == "synthetic"
    assert ds[0]["channels"] == ["bing", "google", "meta"]
    h = client.get("/api/history", params={"freq": "W"}).json()
    assert len(h["records"]) > 100 and "google_revenue" in h["records"][0]


def test_kpis():
    k = client.get("/api/kpis").json()
    assert k["total"]["revenue"] > 0 and set(k["channels"]) == {"bing", "google", "meta"}


def test_forecast_roas_and_scenario():
    base = client.post("/api/forecast", json={"horizon": 28}).json()
    assert len(base["total"]) == 28
    assert set(base["channels"]) == {"bing", "google", "meta"}
    row = base["total"][0]
    assert row["p10"] <= row["p50"] <= row["p90"]
    more = client.post("/api/forecast", json={"horizon": 28, "spend_multipliers": {"meta": 1.5}}).json()
    assert more["planned_spend"]["meta"] == pytest.approx(base["planned_spend"]["meta"] * 1.5, rel=1e-6)
    assert more["horizon_total"]["p50"] > base["horizon_total"]["p50"]


def test_forecast_custom_daily_plan():
    r = client.post("/api/forecast", json={"horizon": 7, "channels": ["google"],
                                           "daily_spend": {"google": [20000] * 7}}).json()
    assert list(r["channels"]) == ["google"]
    assert r["planned_spend"]["google"] == pytest.approx(140000)


def test_forecast_validation():
    assert client.post("/api/forecast", json={"channels": ["tiktok"]}).status_code == 422
    assert client.post("/api/forecast", json={"horizon": 2}).status_code == 422
    assert client.post("/api/forecast", json={"dataset_id": "nope"}).status_code == 404


def test_optimize_and_curves():
    r = client.post("/api/optimize", json={"objective": "p50"}).json()
    total = sum(a["spend"] for a in r["allocation"].values())
    assert total == pytest.approx(r["total_budget"], rel=1e-6)
    assert r["expected_revenue"] >= r["baseline_revenue"]
    tr = client.post("/api/optimize", json={"max_change": 0.2}).json()
    for a in tr["allocation"].values():
        assert a["spend"] <= a["baseline_spend"] * 1.2 + 1
    bad = client.post("/api/optimize", json={"total_budget": 1000, "bounds": {"google": {"min_spend": 5000}}})
    assert bad.status_code == 422
    c = client.get("/api/response-curves").json()
    assert set(c["curves"]) == {"bing", "google", "meta"}


def test_model_details_and_cache():
    m = client.get("/api/models/roas").json()
    assert 0 <= m["meta"]["adstock_decay"] <= 1
    stats = client.get("/api/cache").json()
    assert stats["hits"] >= 1


def test_upload_dataset_and_forecast():
    df = generate_dataset(start="2023-01-01", end="2024-08-31", seed=11)[["date", "channel", "spend", "revenue"]]
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/datasets", files={"file": ("mine.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()["dataset_id"]
    f = client.post("/api/forecast", json={"dataset_id": ds, "horizon": 14}).json()
    assert len(f["total"]) == 14


def test_upload_rejects_bad_csv():
    r = client.post("/api/datasets", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 422
    short = generate_dataset(start="2024-01-01", end="2024-06-30")[["date", "channel", "spend", "revenue"]]
    r = client.post("/api/datasets", files={"file": ("s.csv", io.BytesIO(short.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 422 and "at least" in r.json()["detail"]


def test_backtest_endpoint_small():
    r = client.post("/api/backtest", json={"models": ["roas_quantile"], "folds": 1, "horizon": 14}).json()
    m = r["results"]["roas_quantile"]["metrics"]["total"]
    assert 0 <= m["coverage_p10_p90"] <= 1 and m["mape"] < 0.3
