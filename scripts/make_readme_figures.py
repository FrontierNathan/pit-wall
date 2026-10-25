"""Regenerate the figures in docs/img/ used by the README.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pitwall.core import PitWall  # noqa: E402
from pitwall.strategy import one_stop_curve  # noqa: E402

COLORS = {"SOFT": "#da291c", "MEDIUM": "#e0b000", "HARD": "#7a7a7a"}
OUT = ROOT / "docs" / "img"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def degradation(pw: PitWall, driver: str) -> None:
    laps = pw.corrected_laps(driver)
    fig, ax = plt.subplots(figsize=(8, 4))
    for stint in pw.driver_stints(driver):
        part = laps[laps["stint"] == stint["stint"]]
        color = COLORS.get(stint["compound"], "black")
        ax.scatter(part["lap"], part["corrected_s"], s=14, color=color, alpha=0.8,
                   label=f"{stint['compound'].title()} — {stint['deg_s_per_lap']:.3f} s/lap")
        fit = stint["base_s"] + stint["deg_s_per_lap"] * part["tyre_life"]
        ax.plot(part["lap"], fit, color=color, linewidth=2)
    style(ax, f"{driver}, 2024 Dutch GP: tyre degradation after removing the fuel effect",
          "Lap", "Lap time on an empty tank (s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=160)


def pit_window(pw: PitWall, driver: str, after_lap: int) -> None:
    state = pw.driver_state(driver, after_lap)
    curve = one_stop_curve(state, pw.models, pw.race_laps, pw.pit_loss_s, "HARD")
    best = min(curve.values())
    laps = [lap for lap in curve if curve[lap] - best < 12]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(laps, [curve[lap] - best for lap in laps], marker="o", markersize=3, color=COLORS["HARD"])
    ax.axhspan(0, 1, color="#2e7d32", alpha=0.12, label="within 1 s of best = the window")
    actual = pw.actual_stops(driver)[0]["lap"]
    ax.axvline(actual, color="#da291c", linestyle="--", label=f"actual stop (lap {actual})")
    style(ax, f"{driver}: one stop onto hards, decided after lap {after_lap}",
          "Stop on lap", "Seconds slower than best")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pit_window.png", dpi=160)


def debrief(pw: PitWall, driver: str, corner: int, laps: range, metric: str, label: str) -> None:
    m = pw.corner_metrics
    stint_of = pw.laps[pw.laps["driver"] == driver].set_index("lap")["stint"]
    clean = set(pw.laps[(pw.laps["driver"] == driver) & pw.laps["clean"]]["lap"])
    rows = m[(m["driver"] == driver) & (m["corner"] == corner) & m["lap"].isin(clean)]
    rows = rows[rows["lap"].map(stint_of) == stint_of[laps.start]]
    flagged = rows["lap"].isin(laps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(rows.loc[~flagged, "lap"], rows.loc[~flagged, metric], color="#7a7a7a", s=22, label="rest of stint")
    ax.scatter(rows.loc[flagged, "lap"], rows.loc[flagged, metric], color="#da291c", s=30,
               label=f"debrief laps {laps.start}-{laps.stop - 1}")
    style(ax, f"{driver}, T{corner}: illustrative debrief vs telemetry", "Lap", label)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "debrief_evidence.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    race = PitWall.from_race(ROOT / "data/races/2024_dutch_gp_race.csv")
    degradation(race, "NOR")
    pit_window(race, "NOR", 20)
    debrief(race, "LEC", 7, range(19, 24), "full_throttle_m", "Metres from apex to full throttle")
    print(f"Saved figures to {OUT}")
