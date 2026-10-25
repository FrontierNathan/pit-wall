"""Backtest: would the model have called the stop the teams actually made?

For every car that pitted, we go back to a few laps BEFORE its first stop,
rebuild the tyre models using only the laps that had been driven by then
(so the model can't peek at the future), and ask for the best plan with the
same number of stops the team actually made. Then we compare laps.

This is an evaluation, not a scoreboard. Teams know things this model
doesn't (rivals' plans, traffic, tyre temperatures), so disagreement isn't
automatically the model being wrong, but a model that is nowhere near real
decisions shouldn't be trusted with new ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DRY_COMPOUNDS, mark_clean_laps, pit_stops
from .degradation import FUEL_EFFECT_S_PER_LAP, compound_models, fit_all_stints
from .strategy import DriverState, evaluate

LOOKBACK_LAPS = 8
HIT_TOLERANCE_LAPS = 3


@dataclass
class BacktestRow:
    driver: str
    decision_lap: int
    actual_first_stop: int
    actual_stops: int
    recommended_first_stop: int | None
    error_laps: int | None
    hit: bool
    model_preferred_stops: int


def run(laps: pd.DataFrame, race_laps: int, pit_loss_s: float,
        fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> pd.DataFrame:
    laps = laps if "clean" in laps.columns else mark_clean_laps(laps)
    # Stops under a safety car or red flag are opportunism, not planned
    # strategy, so we only test the model on green-flag stops.
    stops = pit_stops(laps)
    stops = stops[~stops["neutralised"]]
    rows = []
    for driver, driver_stops in stops.groupby("driver"):
        first_stop = int(driver_stops["lap"].min())
        n_stops = len(driver_stops)
        decision_lap = max(first_stop - LOOKBACK_LAPS, 5)
        if n_stops > 2 or decision_lap >= first_stop:
            continue

        known = laps[laps["lap"] <= decision_lap]          # only the past
        models = compound_models(fit_all_stints(known, race_laps, fuel_effect))
        mine = known[known["driver"] == driver].sort_values("lap")
        if mine.empty:
            continue
        last = mine.iloc[-1]
        compound = last["compound"] if last["compound"] in DRY_COMPOUNDS else "MEDIUM"
        own_fits = [f for f in fit_all_stints(mine, race_laps, fuel_effect)]
        base = (np.median([f.base_s - models[f.compound].offset_s for f in own_fits])
                if own_fits else float(known[known["clean"]]["lap_time_s"].median()))
        if not np.isfinite(base):
            continue  # no clean laps yet (e.g. a wet start): nothing to learn from
        tyre_age = int(last["tyre_life"]) if pd.notna(last["tyre_life"]) else 1
        state = DriverState(driver, int(last["lap"]), compound, tyre_age,
                            float(base), set(mine["compound"]) & set(DRY_COMPOUNDS))

        options = evaluate(state, models, race_laps, pit_loss_s, max_stops=2)
        if not options:
            continue
        preferred_stops = len(options[0].stops)
        same_count = [o for o in options if len(o.stops) == n_stops]
        recommended = same_count[0].stops[0][0] if same_count else None
        error = None if recommended is None else recommended - first_stop
        rows.append(BacktestRow(driver, decision_lap, first_stop, n_stops, recommended, error,
                                error is not None and abs(error) <= HIT_TOLERANCE_LAPS, preferred_stops))
    return pd.DataFrame([r.__dict__ for r in rows])


def summarise(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"cars": 0}
    errors = results["error_laps"].dropna()
    return {
        "cars": len(results),
        "hit_rate": round(float(results["hit"].mean()), 2),
        "hit_tolerance_laps": HIT_TOLERANCE_LAPS,
        "mean_abs_error_laps": round(float(errors.abs().mean()), 1) if len(errors) else None,
        "mean_error_laps": round(float(errors.mean()), 1) if len(errors) else None,
        "stop_count_agreement": round(float((results["model_preferred_stops"] == results["actual_stops"]).mean()), 2),
    }
