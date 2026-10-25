"""Pit strategy: given where we are now, when should we stop and on which tyre?

The model is the simplest one that is still useful:

    time for a stint = sum over its laps of (base + compound offset + deg * tyre age)
    race time        = time of every stint + pit loss for every stop

We try every possible stop lap (and every pair of laps for two stops), keep the
best plan for each tyre sequence, and report the PIT WINDOW: all laps where
stopping costs less than `window_tolerance_s` more than the best lap.

Things this deliberately ignores (and a real team wouldn't): traffic after the
stop, rivals' strategies (undercut/overcut), tyre warm-up, and tyre sets
available. They are listed in docs/LEARNING_GUIDE.md as next steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

from .degradation import CompoundModel

DRY = ("SOFT", "MEDIUM", "HARD")
SAFETY_CAR_PIT_LOSS_FACTOR = 0.5  # stopping under a safety car costs about half


@dataclass
class DriverState:
    driver: str
    current_lap: int          # last completed lap
    compound: str             # tyre currently fitted
    tyre_age: int             # laps on the current set
    base_s: float             # this car's new-tyre pace on the reference compound
    compounds_used: set[str] = field(default_factory=set)


@dataclass
class StrategyOption:
    name: str
    stops: list[tuple[int, str]]  # (lap of pit stop, compound fitted)
    total_time_s: float
    delta_to_best_s: float = 0.0
    window: tuple[int, int] | None = None  # for one-stop options


def stint_time(laps: int, first_age: int, base_s: float, model: CompoundModel) -> float:
    """Total time for `laps` laps starting at tyre age `first_age`."""
    if laps <= 0:
        return 0.0
    # Same as summing lap by lap, but in one step: the ages form an arithmetic
    # series first_age, first_age+1, ..., whose sum is laps * (first + last) / 2.
    last_age = first_age + laps - 1
    sum_of_ages = laps * (first_age + last_age) / 2
    return laps * (base_s + model.offset_s) + model.deg_s_per_lap * sum_of_ages


def one_stop_curve(state: DriverState, models: dict[str, CompoundModel], race_laps: int,
                   pit_loss_s: float, compound: str, safety_car_now: bool = False) -> dict[int, float]:
    """Total remaining race time for a single stop onto `compound`, for every possible stop lap.

    Plotting this curve shows why there is a WINDOW rather than one magic lap:
    the bottom of the curve is usually flat for several laps.
    """
    current = models[state.compound]
    totals = {}
    for lap in range(state.current_lap + 1, race_laps):
        loss = pit_loss_s
        if safety_car_now and lap == state.current_lap + 1:
            loss = pit_loss_s * SAFETY_CAR_PIT_LOSS_FACTOR
        totals[lap] = (stint_time(lap - state.current_lap, state.tyre_age + 1, state.base_s, current)
                       + loss
                       + stint_time(race_laps - lap, 1, state.base_s, models[compound]))
    return totals


def _uses_two_compounds(state: DriverState, new_compounds: list[str]) -> bool:
    """F1 rule for dry races: at least two different dry compounds must be used."""
    used = set(state.compounds_used) | {state.compound} | set(new_compounds)
    return len(used & set(DRY)) >= 2


def evaluate(state: DriverState, models: dict[str, CompoundModel], race_laps: int,
             pit_loss_s: float, max_stops: int = 2, safety_car_now: bool = False,
             window_tolerance_s: float = 1.0) -> list[StrategyOption]:
    """Rank the best plan for every tyre sequence, fastest first."""
    remaining = race_laps - state.current_lap
    if remaining <= 0:
        return []
    if not math.isfinite(state.base_s):
        raise ValueError(f"no pace estimate for {state.driver} yet (no clean laps to learn from)")
    current = models[state.compound]

    def loss_for_stop(lap: int) -> float:
        # A safety car "now" only helps if we stop on the very next lap.
        if safety_car_now and lap == state.current_lap + 1:
            return pit_loss_s * SAFETY_CAR_PIT_LOSS_FACTOR
        return pit_loss_s

    def first_stint(stop_lap: int) -> float:
        # Laps from now up to and including the in-lap on the current tyres.
        return stint_time(stop_lap - state.current_lap, state.tyre_age + 1, state.base_s, current)

    options: list[StrategyOption] = []

    if _uses_two_compounds(state, []):
        total = stint_time(remaining, state.tyre_age + 1, state.base_s, current)
        options.append(StrategyOption("no more stops", [], round(total, 2)))

    first_lap, last_lap = state.current_lap + 1, race_laps - 1

    if max_stops >= 1:
        for compound in DRY:
            if not _uses_two_compounds(state, [compound]):
                continue
            totals = one_stop_curve(state, models, race_laps, pit_loss_s, compound, safety_car_now)
            if not totals:
                continue
            best_lap = min(totals, key=totals.get)
            best = totals[best_lap]
            window_laps = [lap for lap, t in totals.items() if t <= best + window_tolerance_s]
            options.append(StrategyOption(
                name=f"1 stop -> {compound}",
                stops=[(best_lap, compound)],
                total_time_s=round(best, 2),
                window=(min(window_laps), max(window_laps)),
            ))

    if max_stops >= 2:
        for c1, c2 in product(DRY, DRY):
            if not _uses_two_compounds(state, [c1, c2]):
                continue
            best_total, best_stops = None, None
            for lap1 in range(first_lap, last_lap):
                head = first_stint(lap1) + loss_for_stop(lap1)
                for lap2 in range(lap1 + 2, last_lap + 1):
                    total = (head
                             + stint_time(lap2 - lap1, 1, state.base_s, models[c1])
                             + pit_loss_s
                             + stint_time(race_laps - lap2, 1, state.base_s, models[c2]))
                    if best_total is None or total < best_total:
                        best_total, best_stops = total, [(lap1, c1), (lap2, c2)]
            if best_total is not None:
                options.append(StrategyOption(f"2 stops -> {c1} -> {c2}", best_stops, round(best_total, 2)))

    options.sort(key=lambda o: o.total_time_s)
    if options:
        fastest = options[0].total_time_s
        for option in options:
            option.delta_to_best_s = round(option.total_time_s - fastest, 2)
    return options
