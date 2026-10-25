"""Command line for Pit Wall.

Examples:
    python -m pitwall summary
    python -m pitwall stints NOR
    python -m pitwall strategy NOR --after-lap 20
    python -m pitwall strategy NOR --after-lap 20 --safety-car
    python -m pitwall debriefs
    python -m pitwall backtest
    python -m pitwall ask "When should Norris have stopped, and why?"   # needs ANTHROPIC_API_KEY

Use --race to pick another race file in data/races/.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .backtest import run as run_backtest
from .backtest import summarise
from .core import PitWall

DEFAULT_RACE = "data/races/2024_dutch_gp_race.csv"


def main() -> None:
    parser = argparse.ArgumentParser(prog="pitwall", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default=DEFAULT_RACE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    stints = sub.add_parser("stints")
    stints.add_argument("driver")
    strategy = sub.add_parser("strategy")
    strategy.add_argument("driver")
    strategy.add_argument("--after-lap", type=int, required=True)
    strategy.add_argument("--safety-car", action="store_true")
    sub.add_parser("debriefs")
    sub.add_parser("backtest")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    args = parser.parse_args()

    pw = PitWall.from_race(args.race)
    pd.set_option("display.width", 140)

    if args.command == "summary":
        print(json.dumps(pw.summary(), indent=2))
    elif args.command == "stints":
        print(pd.DataFrame(pw.driver_stints(args.driver)).to_string(index=False))
    elif args.command == "strategy":
        result = pw.recommend(args.driver, args.after_lap, safety_car_now=args.safety_car)
        print(json.dumps(result, indent=2))
        print("\nWhat actually happened:", pw.actual_stops(args.driver))
    elif args.command == "debriefs":
        for d in pw.debriefs():
            print(f"[{d['verdict'].upper():12}] {d['driver']}: {d['text']}")
            print(f"               {d['evidence']['explanation']}\n")
    elif args.command == "backtest":
        results = run_backtest(pw.laps, pw.race_laps, pw.pit_loss_s)
        print(results.to_string(index=False))
        print(json.dumps(summarise(results), indent=2))
    elif args.command == "ask":
        from .engineer import RaceEngineer

        reply = RaceEngineer(pw).ask(args.question)
        for call in reply.tool_calls:
            print(f"  tool: {call.name}({call.input})")
        print("\n" + reply.text)


if __name__ == "__main__":
    main()
