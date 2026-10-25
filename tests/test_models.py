"""Tests for lap cleaning, degradation fitting and strategy, on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from pitwall.data import _normalise, mark_clean_laps
from pitwall.degradation import CompoundModel, fit_stint, fuel_corrected
from pitwall.strategy import DriverState, evaluate, one_stop_curve, stint_time


def synthetic_stint(deg=0.08, base=80.0, laps=range(2, 22), fuel=0.06, race_laps=50, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for lap in laps:
        tyre_life = lap - laps[0] + 1
        fuel_cost = fuel * (race_laps - lap)
        rows.append({"driver": "TST", "team": "T", "lap": lap, "stint": 1, "compound": "MEDIUM",
                     "tyre_life": tyre_life, "track_status": "1", "position": 1,
                     "lap_time_s": base + deg * tyre_life + fuel_cost + rng.normal(0, noise),
                     "pit_in_time": None, "pit_out_time": None, "rainfall": False, "deleted": False})
    return _normalise(pd.DataFrame(rows))


def test_fuel_correction_removes_the_fuel_effect():
    laps = pd.DataFrame({"lap": [10, 40], "lap_time_s": [82.4, 80.6]})
    corrected = fuel_corrected(laps, race_laps=50, fuel_effect=0.06)
    assert corrected.tolist() == pytest.approx([80.0, 80.0])


def test_fit_recovers_known_degradation():
    laps = mark_clean_laps(synthetic_stint(deg=0.08))
    fit = fit_stint(laps, race_laps=50)
    assert fit.deg_s_per_lap == pytest.approx(0.08, abs=0.01)
    assert fit.base_s == pytest.approx(80.0, abs=0.15)
    assert fit.r2 > 0.9 and fit.cliff_lap is None


def test_ignoring_fuel_makes_tyres_look_better_than_they_are():
    laps = mark_clean_laps(synthetic_stint(deg=0.08))
    naive = fit_stint(laps, race_laps=50, fuel_effect=0.0)
    assert naive.deg_s_per_lap < 0.04  # fuel burn hides half the wear


def test_cliff_is_detected():
    laps = synthetic_stint(deg=0.05, noise=0.0)
    laps.loc[laps.index[-3:], "lap_time_s"] += 0.8
    fit = fit_stint(mark_clean_laps(laps), race_laps=50)
    assert fit.cliff_lap == int(laps["lap"].iloc[-3])


def test_cleaning_rules_and_reasons():
    laps = synthetic_stint()
    laps.loc[0, "lap"] = 1
    laps.loc[3, "track_status"] = "4"
    laps.loc[5, "is_in_lap"] = True
    laps.loc[7, "lap_time_s"] += 10
    clean = mark_clean_laps(laps)
    assert clean.loc[0, "excluded_reason"] == "start lap"
    assert clean.loc[3, "excluded_reason"] == "yellow / safety car / VSC"
    assert clean.loc[5, "excluded_reason"] == "pit in-lap"
    assert clean.loc[7, "excluded_reason"].startswith("outlier")
    assert clean["clean"].sum() == len(laps) - 4


MODELS = {
    "SOFT": CompoundModel("SOFT", 0.10, -0.6, 5, "data"),
    "MEDIUM": CompoundModel("MEDIUM", 0.06, 0.0, 5, "data"),
    "HARD": CompoundModel("HARD", 0.03, 0.4, 5, "data"),
}


def test_stint_time_matches_lap_by_lap_sum():
    model = MODELS["MEDIUM"]
    slow = sum(80 + model.offset_s + model.deg_s_per_lap * age for age in range(5, 25))
    assert stint_time(20, 5, 80.0, model) == pytest.approx(slow)


def test_one_stop_optimum_matches_brute_force():
    state = DriverState("TST", 10, "MEDIUM", 10, 80.0, {"MEDIUM"})
    curve = one_stop_curve(state, MODELS, race_laps=50, pit_loss_s=22.0, compound="HARD")
    options = evaluate(state, MODELS, race_laps=50, pit_loss_s=22.0, max_stops=1)
    hard = next(o for o in options if o.name == "1 stop -> HARD")
    assert hard.stops[0][0] == min(curve, key=curve.get)
    assert hard.window[0] <= hard.stops[0][0] <= hard.window[1]


def test_two_compound_rule_is_enforced():
    state = DriverState("TST", 10, "MEDIUM", 10, 80.0, {"MEDIUM"})
    options = evaluate(state, MODELS, race_laps=50, pit_loss_s=22.0)
    assert all(o.name != "no more stops" for o in options)
    assert all(o.name != "1 stop -> MEDIUM" for o in options)


def test_safety_car_now_pulls_the_stop_forward():
    state = DriverState("TST", 20, "MEDIUM", 20, 80.0, {"MEDIUM"})
    normal = evaluate(state, MODELS, 50, 22.0, max_stops=1)
    under_sc = evaluate(state, MODELS, 50, 22.0, max_stops=1, safety_car_now=True)
    assert under_sc[0].stops[0][0] == 21
    assert under_sc[0].total_time_s < normal[0].total_time_s


def test_unknown_pace_is_an_error_not_a_silent_nan():
    state = DriverState("TST", 10, "MEDIUM", 10, float("nan"), {"MEDIUM"})
    with pytest.raises(ValueError):
        evaluate(state, MODELS, 50, 22.0)
