"""Tyre degradation: how much slower does a car get per lap of tyre age?

Two effects change lap time over a stint and they pull in opposite directions:

1. Fuel burns off, so the car gets lighter and FASTER every lap.
2. Tyres wear, so the car gets SLOWER every lap.

If we fit lap time against tyre age without removing the fuel effect, the
tyres look better than they are. So we first "fuel-correct" each lap to the
time it would have done on an empty tank, then fit a straight line:

    corrected_lap_time = base + deg * tyre_age

`deg` (seconds per lap) is the degradation rate; `base` is the pace on new tyres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Assumption: each lap of fuel on board costs ~0.06 s. The usual rule of thumb
# is ~0.03 s per kg of fuel and ~1.5-2 kg burned per lap. It is a parameter, so
# you can test how sensitive the answers are to it.
FUEL_EFFECT_S_PER_LAP = 0.06

# A stint "falls off a cliff" when its last few clean laps are all clearly
# slower than the straight line predicts.
CLIFF_THRESHOLD_S = 0.3
CLIFF_LAPS = 3
MIN_LAPS_FOR_FIT = 5

# Used only when the race has too little data for a compound. Always labelled
# as "prior" in results so nobody mistakes them for measurements.
PRIOR_DEG = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
PRIOR_OFFSET = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.4}  # vs MEDIUM, seconds


@dataclass
class StintFit:
    driver: str
    stint: int
    compound: str
    start_lap: int
    end_lap: int
    n_clean: int
    base_s: float
    deg_s_per_lap: float
    r2: float
    cliff_lap: int | None


@dataclass
class CompoundModel:
    compound: str
    deg_s_per_lap: float
    offset_s: float  # pace on new tyres relative to the reference compound
    n_stints: int
    source: str  # "data" or "prior"


def fuel_corrected(laps: pd.DataFrame, race_laps: int,
                   fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> pd.Series:
    """Lap time as if the tank were empty: remove the cost of the fuel still on board."""
    laps_of_fuel_left = race_laps - laps["lap"]
    return laps["lap_time_s"] - fuel_effect * laps_of_fuel_left


def fit_stint(laps: pd.DataFrame, race_laps: int,
              fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> StintFit | None:
    """Fit a straight line to one stint's clean laps. Returns None if too few laps."""
    clean = laps[laps["clean"]].sort_values("lap")
    if len(clean) < MIN_LAPS_FOR_FIT:
        return None
    x = clean["tyre_life"].to_numpy(dtype=float)
    y = fuel_corrected(clean, race_laps, fuel_effect).to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return None
    deg, base = np.polyfit(x, y, 1)
    predicted = base + deg * x
    residuals = y - predicted
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((residuals ** 2).sum()) / total if total > 0 else 0.0

    cliff_lap = None
    if len(residuals) >= MIN_LAPS_FOR_FIT + CLIFF_LAPS and np.all(residuals[-CLIFF_LAPS:] > CLIFF_THRESHOLD_S):
        cliff_lap = int(clean["lap"].iloc[-CLIFF_LAPS])

    return StintFit(
        driver=str(laps["driver"].iloc[0]),
        stint=int(laps["stint"].iloc[0]),
        compound=str(laps["compound"].mode().iloc[0]),
        start_lap=int(laps["lap"].min()),
        end_lap=int(laps["lap"].max()),
        n_clean=len(clean),
        base_s=round(float(base), 3),
        deg_s_per_lap=round(float(deg), 4),
        r2=round(r2, 3),
        cliff_lap=cliff_lap,
    )


def fit_all_stints(df: pd.DataFrame, race_laps: int,
                   fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> list[StintFit]:
    fits = []
    for _, stint_laps in df.groupby(["driver", "stint"]):
        fit = fit_stint(stint_laps, race_laps, fuel_effect)
        if fit is not None:
            fits.append(fit)
    return fits


def compound_models(fits: list[StintFit], reference: str = "MEDIUM") -> dict[str, CompoundModel]:
    """Summarise degradation per compound across all cars.

    deg: median of the stint slopes (the median ignores odd stints).
    offset: how much faster/slower the compound is on new tyres. Cars differ in
    pace, so we compare compounds WITHIN each driver (their stints on both
    compounds) and take the median of those differences.
    """
    by_compound: dict[str, list[StintFit]] = {}
    for fit in fits:
        by_compound.setdefault(fit.compound, []).append(fit)

    offsets = _paired_offsets(fits, reference)
    models = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        stints = by_compound.get(compound, [])
        # Negative slopes are physically odd (usually track evolution); keep
        # them in the data but don't let a compound look "self-healing".
        slopes = [f.deg_s_per_lap for f in stints]
        if len(stints) >= 2 and compound in offsets:
            models[compound] = CompoundModel(
                compound=compound,
                deg_s_per_lap=round(max(float(np.median(slopes)), 0.0), 4),
                offset_s=offsets[compound],
                n_stints=len(stints),
                source="data",
            )
        else:
            models[compound] = CompoundModel(
                compound=compound,
                deg_s_per_lap=PRIOR_DEG[compound],
                offset_s=PRIOR_OFFSET[compound] - PRIOR_OFFSET[reference],
                n_stints=len(stints),
                source="prior",
            )
    return models


def _paired_offsets(fits: list[StintFit], reference: str) -> dict[str, float]:
    """Median within-driver difference in new-tyre pace versus the reference compound."""
    base_by_driver: dict[str, dict[str, list[float]]] = {}
    for fit in fits:
        base_by_driver.setdefault(fit.driver, {}).setdefault(fit.compound, []).append(fit.base_s)

    diffs: dict[str, list[float]] = {}
    for compounds in base_by_driver.values():
        if reference not in compounds:
            continue
        ref = float(np.median(compounds[reference]))
        for compound, bases in compounds.items():
            diffs.setdefault(compound, []).append(float(np.median(bases)) - ref)

    offsets = {c: round(float(np.median(d)), 3) for c, d in diffs.items() if len(d) >= 2}
    if diffs.get(reference):
        offsets[reference] = 0.0
    return offsets


def fits_table(fits: list[StintFit]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in fits])
