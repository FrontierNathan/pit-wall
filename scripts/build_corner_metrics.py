"""Download lap telemetry and reduce it to per-corner metrics (see pitwall/corners.py).

Usage:
    python scripts/build_corner_metrics.py --year 2024 --event "Dutch Grand Prix" \
        --drivers NOR VER LEC PIA --out data/telemetry/2024_dutch_gp_corner_metrics.csv

One telemetry file per lap is ~100 kB, so four drivers over a 72-lap race is
~30 MB of downloads that become a ~200 kB CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitwall.corners import corner_metrics  # noqa: E402
from pitwall.sources import download, race_url, telemetry_url  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--session", default="Race")
    parser.add_argument("--drivers", nargs="+", required=True)
    parser.add_argument("--laps", type=int, default=80, help="highest lap number to try")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corners_raw = json.loads(download(race_url(args.year, args.event, args.session, "corners.json")))
    corners = list(zip(corners_raw["CornerNumber"], corners_raw["Distance"]))
    print(f"{len(corners)} corners")

    def one_lap(job: tuple[str, int]) -> list[dict]:
        driver, lap = job
        try:
            tel = json.loads(download(telemetry_url(args.year, args.event, args.session, driver, lap)))["tel"]
        except HTTPError:
            return []  # lap doesn't exist (retired, or race was shorter)
        return [{"driver": driver, "lap": lap, **row} for row in corner_metrics(tel, corners)]

    jobs = [(d, lap) for d in args.drivers for lap in range(1, args.laps + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [row for result in pool.map(one_lap, jobs) for row in result]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["driver", "lap", "corner"]).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} corner rows to {args.out}")


if __name__ == "__main__":
    main()
