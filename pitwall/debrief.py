"""Driver debriefs: turn what a driver FEELS into something the data can check.

A debrief like "rear stepping out on exit of T7, laps 19-23, can't get on the
power" contains a symptom (oversteer/traction), a corner phase (exit), corners
(7) and laps (19-23). We parse those out, then ask the telemetry whether the
car actually behaved that way on those laps compared with the rest of the stint.

The verdict is one of:
  supported     - the data moved in the direction the symptom predicts, clearly
  contradicted  - the data moved clearly the OTHER way
  inconclusive  - no clear change (or not enough laps to tell)

"Clearly" uses Welch's t-statistic: the difference in averages divided by its
uncertainty. |t| >= 2 is roughly "less than a 5% chance this is just noise".
We also require a minimum size, so a tiny but consistent change doesn't count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SYMPTOM_KEYWORDS = {
    "understeer": ["understeer", "push", "front end", "front washing", "no front", "won't turn",
                   "front sliding", "losing the front", "lost the front"],
    "oversteer": ["oversteer", "rear stepping", "rear sliding", "loose", "snap", "rear unstable",
                  "rear end", "losing the rear", "lost the rear"],
    "traction": ["traction", "wheelspin", "can't get on the power", "cannot get on the power", "power down"],
    "braking": ["lock", "locking", "lock-up", "brake", "braking"],
    "tyre_drop": ["dropped off", "drop off", "no grip", "cliff", "tyres gone", "graining", "falling off"],
}
PHASE_KEYWORDS = {
    "entry": ["entry", "turn-in", "turn in", "on the brakes", "braking into", "into"],
    "mid": ["mid-corner", "mid corner", "apex", "middle"],
    "exit": ["exit", "on the power", "on throttle", "traction", "out of"],
}

# Which telemetry number each symptom should move, and in which direction.
#   +1 = the number should go UP when the symptom is real, -1 = DOWN.
EXPECTED = {
    ("understeer", None): ("min_speed_kmh", -1),
    ("oversteer", "entry"): ("brake_on_m", +1),
    ("oversteer", None): ("full_throttle_m", +1),
    ("traction", None): ("full_throttle_m", +1),
    ("braking", None): ("brake_on_m", +1),
}
MIN_EFFECT = {"min_speed_kmh": 2.0, "full_throttle_m": 10.0, "brake_on_m": 15.0, "lap_residual_s": 0.2}
T_THRESHOLD = 2.0


@dataclass
class Debrief:
    id: str
    driver: str
    text: str
    author: str
    laps: list[int] = field(default_factory=list)
    corners: list[int] = field(default_factory=list)
    symptom: str | None = None
    phase: str | None = None


@dataclass
class Evidence:
    verdict: str            # supported | contradicted | inconclusive | no data
    metric: str | None
    debrief_mean: float | None
    baseline_mean: float | None
    delta: float | None
    t_stat: float | None
    n_debrief: int
    n_baseline: int
    explanation: str


# ---------------------------------------------------------------- parsing

def parse(text: str, driver: str, author: str, debrief_id: str) -> Debrief:
    lowered = text.lower()
    return Debrief(
        id=debrief_id,
        driver=driver.upper(),
        text=text,
        author=author,
        laps=_parse_numbers(lowered, r"\blaps?\s+"),
        corners=_parse_numbers(lowered, r"\b(?:t|turns?\s+)"),
        symptom=_first_match(lowered, SYMPTOM_KEYWORDS),
        phase=_first_match(lowered, PHASE_KEYWORDS),
    )


def _parse_numbers(text: str, prefix: str) -> list[int]:
    """Find '12', '12-15', '12 to 15' and '12, 14' after a prefix like 'laps '."""
    numbers: set[int] = set()
    pattern = prefix + r"(\d+(?:\s*(?:-|–|to|,|and)\s*\d+)*)"
    for match in re.finditer(pattern, text):
        for part in re.split(r"\s*(?:,|and)\s*", match.group(1)):
            span = re.split(r"\s*(?:-|–|to)\s*", part)
            if len(span) == 2:
                start, end = sorted(int(s) for s in span)
                numbers.update(range(start, end + 1))
            elif span[0].isdigit():
                numbers.add(int(span[0]))
    return sorted(numbers)


def _first_match(text: str, keywords: dict[str, list[str]]) -> str | None:
    """The category whose keyword appears earliest in the text."""
    best, best_pos = None, len(text) + 1
    for category, words in keywords.items():
        for word in words:
            pos = text.find(word)
            if pos != -1 and pos < best_pos:
                best, best_pos = category, pos
    return best


# --------------------------------------------------------------- evidence

def expected_metric(debrief: Debrief) -> tuple[str, int] | None:
    if debrief.symptom is None:
        return None
    return EXPECTED.get((debrief.symptom, debrief.phase)) or EXPECTED.get((debrief.symptom, None))


def check(debrief: Debrief, laps: pd.DataFrame, corner_metrics: pd.DataFrame | None,
          lap_residuals: pd.DataFrame | None = None) -> Evidence:
    """Compare the debrief's laps with the rest of the same stint (clean laps only).

    laps: our lap table with `clean` and `stint` columns.
    corner_metrics: rows of driver, lap, corner, <metrics> (may be None).
    lap_residuals: rows of driver, lap, residual_s (lap time above the stint's
                   degradation line) for symptoms with no corner, e.g. tyre_drop.
    """
    driver_laps = laps[(laps["driver"] == debrief.driver) & laps["clean"]]
    if not debrief.laps:
        return _no_data("the debrief doesn't say which laps")
    stints = driver_laps[driver_laps["lap"].isin(debrief.laps)]["stint"].unique()
    if len(stints) == 0:
        return _no_data("none of those laps are clean laps for this driver")
    same_stint = driver_laps[driver_laps["stint"] == stints[0]]
    debrief_laps = set(same_stint["lap"]) & set(debrief.laps)
    baseline_laps = set(same_stint["lap"]) - set(debrief.laps)

    if debrief.symptom == "tyre_drop" or (not debrief.corners and lap_residuals is not None):
        if lap_residuals is None:
            return _no_data("no lap-time model available")
        values = lap_residuals[lap_residuals["driver"] == debrief.driver].set_index("lap")["residual_s"]
        return _compare(values, debrief_laps, baseline_laps, "lap_residual_s", +1)

    target = expected_metric(debrief)
    if target is None:
        return _no_data("couldn't tell which symptom this is")
    if corner_metrics is None or not debrief.corners:
        return _no_data("needs corner telemetry and a corner number (e.g. 'T7')")
    metric, direction = target
    rows = corner_metrics[(corner_metrics["driver"] == debrief.driver)
                          & (corner_metrics["corner"].isin(debrief.corners))]
    if rows.empty:
        return _no_data(f"no telemetry for {debrief.driver} at those corners")
    values = rows.groupby("lap")[metric].mean()  # average over the corners mentioned
    return _compare(values, debrief_laps, baseline_laps, metric, direction)


def _compare(values: pd.Series, debrief_laps: set[int], baseline_laps: set[int],
             metric: str, direction: int) -> Evidence:
    a = values[values.index.isin(debrief_laps)].dropna()
    b = values[values.index.isin(baseline_laps)].dropna()
    if len(a) < 2 or len(b) < 3:
        return Evidence("inconclusive", metric, None, None, None, None, len(a), len(b),
                        "too few laps to compare")
    delta = float(a.mean() - b.mean())
    standard_error = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    t = delta / standard_error if standard_error > 0 else 0.0
    big_enough = abs(delta) >= MIN_EFFECT[metric]
    signed_t = t * direction
    if signed_t >= T_THRESHOLD and big_enough:
        verdict = "supported"
    elif signed_t <= -T_THRESHOLD and big_enough:
        verdict = "contradicted"
    else:
        verdict = "inconclusive"
    explanation = (
        f"{metric} averaged {a.mean():.1f} on the debrief laps vs {b.mean():.1f} on the rest of the stint "
        f"(change {delta:+.1f}, t = {t:+.1f}). The symptom predicts it should go "
        f"{'up' if direction > 0 else 'down'}."
    )
    return Evidence(verdict, metric, round(float(a.mean()), 2), round(float(b.mean()), 2),
                    round(delta, 2), round(t, 2), len(a), len(b), explanation)


def _no_data(reason: str) -> Evidence:
    return Evidence("no data", None, None, None, None, None, 0, 0, reason)
