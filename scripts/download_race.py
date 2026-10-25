"""Download a race's lap times from TracingInsights and save them in our CSV format.

Usage:
    python scripts/download_race.py --year 2026 --event "Italian Grand Prix"
    python scripts/download_race.py --year 2024 --event "Dutch Grand Prix" --out data/races/2024_dutch_gp_race.csv

Data: https://github.com/TracingInsights (Apache-2.0), built with FastF1.
Unofficial; not associated with the Formula 1 companies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitwall.data import COLUMNS, load_race  # noqa: E402
from pitwall.sources import download, race_url  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True, help='e.g. "Dutch Grand Prix"')
    parser.add_argument("--session", default="Race")
    parser.add_argument("--out", type=Path, help="CSV path (default: data/races/<year>_<event>.csv)")
    args = parser.parse_args()

    out = args.out or Path("data/races") / f"{args.year}_{args.event.lower().replace(' ', '_')}.csv"
    raw_path = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)

    url = race_url(args.year, args.event, args.session, "session_laptimes.json")
    print(f"Downloading {url}")
    raw_path.write_bytes(download(url))

    df = load_race(raw_path)
    df[[c for c in COLUMNS if c in df.columns]].to_csv(out, index=False)
    raw_path.unlink()
    print(f"Saved {len(df)} laps for {df['driver'].nunique()} drivers to {out}")


if __name__ == "__main__":
    main()
