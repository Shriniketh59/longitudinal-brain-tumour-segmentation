import math

import pytest
from fastapi.testclient import TestClient

from app.longitudinal import LongitudinalRequest, analyse
from app.main import app


def tp(day_str, wt, tc=None, et=None):
    return {"date": day_str, "volumes_cm3": {"whole_tumor": wt, "tumor_core": tc or wt / 2, "enhancing_tumor": et or wt / 4}}


def test_exact_doubling_time_from_two_points():
    out = analyse(LongitudinalRequest(timepoints=[tp("2026-01-01", 10), tp("2026-04-11", 20)]))
    assert out["summary"]["doubling_time_days"] == pytest.approx(100, abs=0.5)
    assert out["summary"]["total_change_pct"] == 100.0
    assert out["summary"]["response_category"] == "progressive_disease"
    assert out["forecast"][0]["low_cm3"] is None          # no uncertainty from 2 points
    assert out["forecast"][0]["probability_progression"] is None


def test_three_points_give_interval_and_probability():
    out = analyse(LongitudinalRequest(timepoints=[tp("2026-01-01", 10), tp("2026-03-01", 13), tp("2026-05-01", 15.5)]))
    f = out["forecast"][1]
    assert f["low_cm3"] < f["volume_cm3"] < f["high_cm3"]
    assert 0 <= f["probability_progression"] <= 1
    assert len(out["curve"]) == 41
    assert out["intervals"][0]["days"] == 59


def test_shrinking_tumor_partial_response_and_halving_time():
    out = analyse(LongitudinalRequest(timepoints=[tp("2026-01-01", 40), tp("2026-06-01", 10)]))
    assert out["summary"]["response_category"] == "partial_response"
    assert out["summary"]["halving_time_days"] is not None
    assert out["summary"]["doubling_time_days"] is None


def test_unsorted_input_is_sorted_and_region_selectable():
    out = analyse(LongitudinalRequest(region="tumor_core", timepoints=[tp("2026-05-01", 30), tp("2026-01-01", 10)]))
    assert out["timepoints"][0]["date"] == "2026-01-01"
    assert out["summary"]["baseline_cm3"] == 5.0


def test_api_rejects_duplicate_dates_and_single_point():
    with TestClient(app) as c:
        assert c.post("/api/longitudinal/analyze", json={"timepoints": [tp("2026-01-01", 1), tp("2026-01-01", 2)]}).status_code == 422
        assert c.post("/api/longitudinal/analyze", json={"timepoints": [tp("2026-01-01", 1)]}).status_code == 422
        r = c.post("/api/longitudinal/analyze", json={"timepoints": [tp("2026-01-01", 1), tp("2026-02-01", 2)]})
        assert r.status_code == 200 and math.isfinite(r.json()["summary"]["growth_rate_pct_per_year"])
