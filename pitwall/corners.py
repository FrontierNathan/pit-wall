"""Turn raw lap telemetry into a few numbers per corner.

A driver describes a corner in words ("understeer mid-corner in Turn 3").
An engineer checks that against numbers. These are the numbers we use:

* min_speed_kmh      - slowest point through the corner. Understeer usually
                       forces a lower minimum speed.
* brake_on_m         - how many metres before the corner the brake goes on.
                       A driver who doesn't trust the front brakes earlier.
* full_throttle_m    - metres from the slowest point until the throttle is
                       fully open. Poor traction / oversteer on exit delays this.
* peak_lat_g         - highest sideways acceleration near the corner.

Caveat worth saying out loud in an interview: public telemetry is sampled at
roughly 4 Hz, so at 250 km/h one sample is ~17 m apart. Brake and throttle
points are only accurate to about that distance, and lateral g here is derived
from GPS position, not measured. Teams have hundreds of channels at 100+ Hz.
"""

from __future__ import annotations

import numpy as np

MIN_SPEED_WINDOW_M = 80     # look +/- this far around the corner for minimum speed
BRAKE_SEARCH_M = 300        # look this far before the corner for the brake point
THROTTLE_SEARCH_M = 400     # look this far after the slowest point for full throttle
FULL_THROTTLE_PCT = 99
MAX_GAP_BRAKE_TO_APEX_M = 150  # braking that ended further back belongs to another corner


def corner_metrics(tel: dict, corners: list[tuple[int, float]]) -> list[dict]:
    """Compute metrics for every corner on one lap.

    tel: dict of equal-length lists with keys distance, speed, throttle, brake, acc_y
    corners: list of (corner_number, distance_from_start_line_m)
    """
    distance = np.asarray(tel["distance"], dtype=float)
    speed = np.asarray(tel["speed"], dtype=float)
    throttle = np.asarray(tel["throttle"], dtype=float)
    brake = np.asarray(tel["brake"], dtype=float)
    lat_acc = np.asarray(tel.get("acc_y", [np.nan] * len(distance)), dtype=float)

    rows = []
    previous_corner_m = 0.0
    for number, corner_m in sorted(corners, key=lambda c: c[1]):
        near = np.abs(distance - corner_m) <= MIN_SPEED_WINDOW_M
        if not near.any():
            continue
        near_idx = np.flatnonzero(near)
        slowest = near_idx[np.argmin(speed[near_idx])]

        brake_on_m = _brake_point(distance, brake, slowest, corner_m,
                                  search_from=max(corner_m - BRAKE_SEARCH_M, previous_corner_m))
        previous_corner_m = corner_m

        after = (distance >= distance[slowest]) & (distance <= distance[slowest] + THROTTLE_SEARCH_M)
        throttle_idx = np.flatnonzero(after & (throttle >= FULL_THROTTLE_PCT))
        full_throttle_m = float(distance[throttle_idx[0]] - distance[slowest]) if throttle_idx.size else None

        lat = np.abs(lat_acc[near_idx])
        peak_lat_g = float(np.nanmax(lat) / 9.81) if np.isfinite(lat).any() else None

        rows.append({
            "corner": int(number),
            "min_speed_kmh": round(float(speed[slowest]), 1),
            "brake_on_m": None if brake_on_m is None else round(brake_on_m, 1),
            "full_throttle_m": None if full_throttle_m is None else round(full_throttle_m, 1),
            "peak_lat_g": None if peak_lat_g is None else round(peak_lat_g, 2),
        })
    return rows


def _brake_point(distance: np.ndarray, brake: np.ndarray, slowest: int,
                 corner_m: float, search_from: float) -> float | None:
    """Metres before the corner where the braking for THIS corner starts.

    We take the braking run that ends closest to the slowest point and walk
    back to where it began. If that run ended long before the slowest point,
    it was braking for an earlier corner, so this corner is taken flat.
    """
    window = np.flatnonzero((distance >= search_from) & (distance <= distance[slowest]) & (brake > 0))
    if window.size == 0:
        return None
    end = window[-1]
    if distance[slowest] - distance[end] > MAX_GAP_BRAKE_TO_APEX_M:
        return None
    start = end
    while start > 0 and brake[start - 1] > 0 and distance[start - 1] >= search_from:
        start -= 1
    return float(corner_m - distance[start])
