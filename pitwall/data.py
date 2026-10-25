"""Load race lap data and decide which laps are "clean" enough to model.

The raw source is TracingInsights' public F1 timing files (built with FastF1),
stored one race per file. We rename the short field names to readable ones and
keep only what the models need.

A lap is CLEAN when it reflects tyre performance rather than something else.
Every excluded lap gets a reason, so a human can see why it was left out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# TracingInsights field name -> our column name
FIELD_MAP = {
    "drv": "driver",
    "team": "team",
    "lap": "lap",
    "time": "lap_time_s",
    "s1": "s1_s",
    "s2": "s2_s",
    "s3": "s3_s",
    "compound": "compound",
    "stint": "stint",
    "life": "tyre_life",
    "pos": "position",
    "status": "track_status",
    "pin": "pit_in_time",
    "pout": "pit_out_time",
    "wR": "rainfall",
    "wTT": "track_temp_c",
    "del": "deleted",
    "vst": "speed_trap_kmh",
    "sesT": "session_time_s",  # when the lap ended, seconds from session start
}
COLUMNS = list(FIELD_MAP.values())
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# A lap more than this much slower than its stint's median is treated as an
# outlier (traffic, a mistake, a blue-flag lap) rather than tyre wear.
OUTLIER_FACTOR = 1.04


def load_race(path: str | Path) -> pd.DataFrame:
    """Load a race from our CSV format or a TracingInsights session_laptimes.json."""
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        df = pd.DataFrame(raw).rename(columns=FIELD_MAP)
        df = df[[c for c in COLUMNS if c in df.columns]]
    else:
        df = pd.read_csv(path)
    return _normalise(df)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({"None": np.nan, "": np.nan})
    numeric = ["lap", "lap_time_s", "s1_s", "s2_s", "s3_s", "stint", "tyre_life", "position",
               "pit_in_time", "pit_out_time", "track_temp_c", "speed_trap_kmh", "session_time_s"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("rainfall", "deleted"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
        else:
            df[col] = False
    # Track status is a string of codes seen during the lap, e.g. "1" or "124".
    df["track_status"] = (
        df["track_status"].fillna("1").astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df["compound"] = df["compound"].fillna("UNKNOWN").str.upper()
    df["is_in_lap"] = df["pit_in_time"].notna()
    df["is_out_lap"] = df["pit_out_time"].notna()
    df = df.dropna(subset=["lap", "stint"])
    df["lap"] = df["lap"].astype(int)
    df["stint"] = df["stint"].astype(int)
    return df.sort_values(["driver", "lap"]).reset_index(drop=True)


def total_laps(df: pd.DataFrame) -> int:
    return int(df["lap"].max())


def mark_clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Add `clean` (bool) and `excluded_reason` (str) columns.

    Rules are checked in order and the first one that applies is recorded.
    """
    df = df.copy()
    reasons = pd.Series("", index=df.index, dtype=object)

    def exclude(mask: pd.Series, reason: str) -> None:
        reasons[(reasons == "") & mask] = reason

    exclude(df["lap_time_s"].isna(), "no lap time")
    exclude(df["lap"] == 1, "start lap")
    exclude(df["is_in_lap"], "pit in-lap")
    exclude(df["is_out_lap"], "pit out-lap")
    exclude(df["track_status"] != "1", "yellow / safety car / VSC")
    exclude(df["rainfall"], "rain")
    exclude(df["deleted"], "deleted by stewards")
    exclude(~df["compound"].isin(DRY_COMPOUNDS), "not a dry compound")
    exclude(df["tyre_life"].isna(), "unknown tyre age")

    # Outliers are judged only against laps that survived the rules above.
    candidate = reasons == ""
    stint_median = (
        df[candidate].groupby(["driver", "stint"])["lap_time_s"].transform("median")
    )
    slow = pd.Series(False, index=df.index)
    slow[candidate] = df.loc[candidate, "lap_time_s"] > stint_median * OUTLIER_FACTOR
    exclude(slow, "outlier (traffic or mistake)")

    df["excluded_reason"] = reasons
    df["clean"] = reasons == ""
    return df


NEUTRALISED_CODES = ("4", "5", "6", "7")  # safety car, red flag, VSC, VSC ending


def pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pit stop: lap of the in-lap and the compounds before/after.

    `neutralised` marks stops made under a safety car, VSC or red flag. Those
    are cheaper (or free, under a red flag) and are not normal strategy calls.
    """
    columns = ["driver", "lap", "from_compound", "to_compound", "neutralised"]
    rows = []
    for driver, laps in df.groupby("driver"):
        laps = laps.sort_values("lap")
        for _, in_lap in laps[laps["is_in_lap"]].iterrows():
            after = laps[laps["lap"] == in_lap["lap"] + 1]
            if after.empty:
                continue  # retired in the pit lane, not a real stop
            status = str(in_lap["track_status"]) + str(after.iloc[0]["track_status"])
            rows.append({
                "driver": driver,
                "lap": int(in_lap["lap"]),
                "from_compound": in_lap["compound"],
                "to_compound": after.iloc[0]["compound"],
                "neutralised": any(code in status for code in NEUTRALISED_CODES),
            })
    return pd.DataFrame(rows, columns=columns)


def estimate_pit_loss(df: pd.DataFrame, default: float = 20.0) -> float:
    """Time lost by stopping, measured from the race itself.

    For each stop: (in-lap + out-lap) minus two "normal" laps for that driver,
    where normal = median clean lap within five laps either side. The median
    over all stops is robust to the odd slow stop. Stops under a safety car or
    red flag are skipped (the clock doesn't behave normally then), and so is
    anything outside 5-60 s, which would be a problem stop or a data error.
    """
    df = df if "clean" in df.columns else mark_clean_laps(df)
    losses = []
    stops = pit_stops(df)
    for _, stop in stops[~stops["neutralised"]].iterrows():
        laps = df[df["driver"] == stop["driver"]]
        in_lap = laps[laps["lap"] == stop["lap"]]["lap_time_s"]
        out_lap = laps[laps["lap"] == stop["lap"] + 1]["lap_time_s"]
        nearby = laps[(laps["clean"]) & ((laps["lap"] - stop["lap"]).abs() <= 5)]
        if in_lap.isna().all() or out_lap.isna().all() or len(nearby) < 3:
            continue
        normal = nearby["lap_time_s"].median()
        loss = float(in_lap.iloc[0] + out_lap.iloc[0] - 2 * normal)
        if 5 <= loss <= 60:
            losses.append(loss)
    return round(float(np.median(losses)), 1) if losses else default
