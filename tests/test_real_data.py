"""End-to-end checks on the bundled real races."""

import pytest

from pitwall import backtest
from pitwall.core import PitWall


@pytest.fixture(scope="module")
def dutch_2024():
    return PitWall.from_race("data/races/2024_dutch_gp_race.csv")


def test_race_loads_and_knows_the_result(dutch_2024):
    summary = dutch_2024.summary()
    assert summary["laps"] == 72
    assert summary["finishing_order"][:2] == ["NOR", "VER"]
    assert 15 < summary["pit_loss_s"] < 30


def test_norris_one_stop_medium_to_hard(dutch_2024):
    stints = dutch_2024.driver_stints("NOR")
    assert [s["compound"] for s in stints] == ["MEDIUM", "HARD"]
    assert dutch_2024.actual_stops("NOR")[0]["lap"] == 28


def test_degradation_is_positive_for_every_compound(dutch_2024):
    for model in dutch_2024.models.values():
        assert model.deg_s_per_lap >= 0


def test_recommendation_has_a_window(dutch_2024):
    rec = dutch_2024.recommend("NOR", 20)
    first, last = rec["pit_window"]
    assert 20 < first <= last < 72


def test_bundled_debriefs_cover_all_verdicts(dutch_2024):
    verdicts = {d["verdict"] for d in dutch_2024.debriefs()}
    assert {"supported", "contradicted", "inconclusive"} <= verdicts


def test_backtest_never_uses_future_laps():
    pw = PitWall.from_race("data/races/2026_british_gp_race.csv")
    results = backtest.run(pw.laps, pw.race_laps, pw.pit_loss_s)
    assert (results["decision_lap"] < results["actual_first_stop"]).all()
    assert backtest.summarise(results)["cars"] > 10


def test_mixed_conditions_race_still_loads():
    pw = PitWall.from_race("data/races/2026_dutch_gp_race.csv")
    reasons = set(pw.laps["excluded_reason"])
    assert "rain" in reasons and "yellow / safety car / VSC" in reasons
