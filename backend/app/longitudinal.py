"""Longitudinal tumor volume analysis: change, growth rate, doubling time,
volumetric response category and a probabilistic growth forecast.

Growth model: exponential, ln V(t) = a + b·t, fitted by least squares on the
timepoints. With three or more timepoints the residual spread gives a Student-t
prediction interval and a probability of progression; with two the fit is exact
and no uncertainty can be estimated.

Response categories follow volumetric conventions derived from RANO: a 25% rise in
the 2D product corresponds to about +40% volume (progression), a 50% fall to about
-65% volume (partial response). They are research aids, not clinical assessments.
"""

import math
from datetime import date

from pydantic import BaseModel, Field, field_validator
from scipy import stats

REGIONS = ("whole_tumor", "tumor_core", "enhancing_tumor")
PROGRESSION = 0.40     # +40% volume vs nadir
PARTIAL = -0.65        # -65% volume vs baseline
COMPLETE_CM3 = 0.1     # below this enhancing volume counts as not measurable
FORECAST_DAYS = (90, 180, 365)
MIN_CM3 = 0.01         # log of zero is undefined; tiny volumes are clamped


class Timepoint(BaseModel):
    date: date
    label: str = Field(default="", max_length=80)
    volumes_cm3: dict[str, float]

    @field_validator("volumes_cm3")
    @classmethod
    def _check(cls, v: dict[str, float]) -> dict[str, float]:
        missing = [r for r in REGIONS if r not in v]
        if missing:
            raise ValueError(f"missing regions: {', '.join(missing)}")
        if any(x < 0 or not math.isfinite(x) for x in v.values()):
            raise ValueError("volumes must be finite and non-negative")
        return v


class LongitudinalRequest(BaseModel):
    timepoints: list[Timepoint] = Field(min_length=2, max_length=50)
    region: str = Field(default="whole_tumor", pattern="^(whole_tumor|tumor_core|enhancing_tumor)$")

    @field_validator("timepoints")
    @classmethod
    def _distinct_dates(cls, v: list[Timepoint]) -> list[Timepoint]:
        if len({t.date for t in v}) != len(v):
            raise ValueError("each timepoint needs a different date")
        return v


def _fit(days: list[float], vols: list[float]) -> dict:
    """Least-squares fit of ln V on days. Returns slope, intercept and residual scale."""
    n = len(days)
    ys = [math.log(max(v, MIN_CM3)) for v in vols]
    mx, my = sum(days) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in days)
    slope = sum((x - mx) * (y - my) for x, y in zip(days, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(days, ys)]
    s = math.sqrt(sum(r * r for r in resid) / (n - 2)) if n > 2 else None
    return {"slope": slope, "intercept": intercept, "s": s, "n": n, "mx": mx, "sxx": sxx}


def _predict(fit: dict, day: float, level: float = 0.95) -> dict:
    mean_log = fit["intercept"] + fit["slope"] * day
    out = {"volume_cm3": round(math.exp(mean_log), 3), "low_cm3": None, "high_cm3": None}
    if fit["s"] is not None and fit["s"] > 0:
        se = fit["s"] * math.sqrt(1 + 1 / fit["n"] + (day - fit["mx"]) ** 2 / fit["sxx"])
        t = stats.t.ppf(0.5 + level / 2, fit["n"] - 2)
        out["low_cm3"] = round(math.exp(mean_log - t * se), 3)
        out["high_cm3"] = round(math.exp(mean_log + t * se), 3)
        out["_se"], out["_mean_log"] = se, mean_log
    return out


def _category(baseline: float, nadir: float, last: float, enhancing_last: float) -> str:
    if nadir > 0 and (last - nadir) / nadir >= PROGRESSION:
        return "progressive_disease"
    if enhancing_last < COMPLETE_CM3 and baseline >= COMPLETE_CM3:
        return "complete_response"
    if baseline > 0 and (last - baseline) / baseline <= PARTIAL:
        return "partial_response"
    return "stable_disease"


def analyse(req: LongitudinalRequest) -> dict:
    tps = sorted(req.timepoints, key=lambda t: t.date)
    t0 = tps[0].date
    days = [float((t.date - t0).days) for t in tps]
    region = req.region
    vols = [t.volumes_cm3[region] for t in tps]

    intervals = []
    for i in range(1, len(tps)):
        dv = vols[i] - vols[i - 1]
        intervals.append({
            "from": tps[i - 1].date.isoformat(),
            "to": tps[i].date.isoformat(),
            "days": int(days[i] - days[i - 1]),
            "change_cm3": round(dv, 3),
            "change_pct": round(dv / vols[i - 1] * 100, 1) if vols[i - 1] > 0 else None,
        })

    fit = _fit(days, vols)
    rate_per_day = fit["slope"]
    doubling = math.log(2) / rate_per_day if rate_per_day > 1e-9 else None
    halving = math.log(2) / -rate_per_day if rate_per_day < -1e-9 else None

    last_day, last_vol = days[-1], vols[-1]
    forecasts = []
    for ahead in FORECAST_DAYS:
        p = _predict(fit, last_day + ahead)
        prob = None
        if "_se" in p:
            # P(volume >= +40% of the last measurement) under the log-normal predictive distribution.
            z = (math.log(max(last_vol, MIN_CM3) * (1 + PROGRESSION)) - p["_mean_log"]) / p["_se"]
            prob = round(float(stats.t.sf(z, fit["n"] - 2)), 4)
        forecasts.append({
            "days_ahead": ahead,
            "date": date.fromordinal(tps[-1].date.toordinal() + ahead).isoformat(),
            "volume_cm3": p["volume_cm3"],
            "low_cm3": p["low_cm3"],
            "high_cm3": p["high_cm3"],
            "probability_progression": prob,
        })

    # Smooth curve for charts: fitted values from the first scan to the furthest forecast.
    horizon = last_day + FORECAST_DAYS[-1]
    curve = []
    for k in range(41):
        d = horizon * k / 40
        p = _predict(fit, d)
        curve.append({"day": round(d, 1), "date": date.fromordinal(t0.toordinal() + round(d)).isoformat(),
                      "volume_cm3": p["volume_cm3"], "low_cm3": p["low_cm3"], "high_cm3": p["high_cm3"]})

    baseline, nadir = vols[0], min(vols)
    enhancing_last = tps[-1].volumes_cm3["enhancing_tumor"]
    return {
        "region": region,
        "timepoints": [
            {"date": t.date.isoformat(), "label": t.label, "day": int(d), "volumes_cm3": t.volumes_cm3}
            for t, d in zip(tps, days)
        ],
        "intervals": intervals,
        "summary": {
            "baseline_cm3": round(baseline, 3),
            "nadir_cm3": round(nadir, 3),
            "latest_cm3": round(last_vol, 3),
            "total_change_pct": round((last_vol - baseline) / baseline * 100, 1) if baseline > 0 else None,
            "change_from_nadir_pct": round((last_vol - nadir) / nadir * 100, 1) if nadir > 0 else None,
            "growth_rate_pct_per_year": round((math.exp(rate_per_day * 365.25) - 1) * 100, 1),
            "doubling_time_days": round(doubling, 1) if doubling else None,
            "halving_time_days": round(halving, 1) if halving else None,
            "response_category": _category(baseline, nadir, last_vol, enhancing_last),
            "fit_quality": "exact (2 timepoints, no uncertainty)" if fit["s"] is None else f"{fit['n']} timepoints",
        },
        "forecast": forecasts,
        "curve": curve,
        "model": "exponential growth, log-linear least squares",
        "thresholds": {"progression_pct": PROGRESSION * 100, "partial_response_pct": PARTIAL * 100},
    }
