import numpy as np
import pandas as pd

from pitwall.corners import corner_metrics
from pitwall.debrief import check, parse


def test_parse_laps_corners_symptom_and_phase():
    d = parse("Rear stepping out on exit of T7 and T9, can't get on the power, laps 19-23 and 25",
              "lec", "me", "d1")
    assert d.driver == "LEC"
    assert d.laps == [19, 20, 21, 22, 23, 25]
    assert d.corners == [7, 9]
    assert d.symptom == "oversteer" and d.phase == "exit"


def test_parse_turn_words_and_lap_to_lap():
    d = parse("Understeer mid-corner in turns 11-12 from lap 30 to 34", "NOR", "me", "d2")
    assert d.corners == [11, 12]
    assert d.laps == [30, 31, 32, 33, 34]
    assert d.symptom == "understeer" and d.phase == "mid"


def test_parse_ignores_numbers_that_are_not_laps_or_corners():
    d = parse("Tyres dropped off after 20 minutes, laps 40-42", "PIA", "me", "d3")
    assert d.laps == [40, 41, 42] and d.corners == []
    assert d.symptom == "tyre_drop"


def _fixture(debrief_speed_change: float):
    laps = pd.DataFrame({"driver": "NOR", "lap": range(2, 22), "stint": 1, "clean": True})
    rng = np.random.default_rng(0)
    speeds = 105 + rng.normal(0, 1.0, 20)
    speeds[-5:] += debrief_speed_change
    metrics = pd.DataFrame({"driver": "NOR", "lap": range(2, 22), "corner": 9, "min_speed_kmh": speeds})
    return laps, metrics


def test_check_supported_when_data_moves_the_predicted_way():
    laps, metrics = _fixture(-4.0)
    d = parse("Understeer mid-corner T9, laps 17-21", "NOR", "me", "d")
    assert check(d, laps, metrics).verdict == "supported"


def test_check_contradicted_when_data_moves_the_other_way():
    laps, metrics = _fixture(+4.0)
    d = parse("Understeer mid-corner T9, laps 17-21", "NOR", "me", "d")
    assert check(d, laps, metrics).verdict == "contradicted"


def test_check_inconclusive_when_nothing_changed():
    laps, metrics = _fixture(0.0)
    d = parse("Understeer mid-corner T9, laps 17-21", "NOR", "me", "d")
    assert check(d, laps, metrics).verdict == "inconclusive"


def test_corner_metrics_on_a_synthetic_lap():
    distance = np.arange(0, 1000, 10.0)
    speed = np.where(np.abs(distance - 500) < 50, 100.0, 250.0)
    speed[np.argmin(np.abs(distance - 500))] = 90.0
    brake = ((distance >= 380) & (distance < 470)).astype(float)
    throttle = np.where(distance >= 560, 100.0, 20.0)
    tel = {"distance": distance, "speed": speed, "brake": brake, "throttle": throttle,
           "acc_y": np.zeros_like(distance)}
    (row,) = corner_metrics(tel, [(1, 500.0)])
    assert row["min_speed_kmh"] == 90.0
    assert row["brake_on_m"] == 120.0       # brake on at 380 m, corner at 500 m
    assert row["full_throttle_m"] == 60.0   # slowest at 500 m, full throttle at 560 m
