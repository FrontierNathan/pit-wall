"""PitWall: one object that loads a race, builds the ontology and answers questions.

The Streamlit app, the command line and the AI race engineer all go through
this class, so every interface gives the same answers.

Ontology (see docs/ARCHITECTURE.md for the diagram):

  Driver --driver_stints--> Stint --stint_laps--> Lap
  Driver --driver_pit_stops--> PitStop
  Driver --driver_debriefs--> Debrief --debrief_laps--> Lap
  Driver --driver_plans--> StrategyPlan

Actions (proposed by anyone, approved only by a named human):
  set_pit_plan  - commit a pit strategy for a driver
  log_debrief   - add a driver debrief and check it against telemetry
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import debrief as debriefs
from .data import DRY_COMPOUNDS, estimate_pit_loss, load_race, mark_clean_laps, pit_stops, total_laps
from .degradation import (FUEL_EFFECT_S_PER_LAP, CompoundModel, StintFit, compound_models,
                          fit_all_stints, fuel_corrected)
from .ontology import ActionType, LinkType, ObjectType, Ontology
from .strategy import DriverState, StrategyOption, evaluate


class PitWall:
    def __init__(self, laps: pd.DataFrame, corner_metrics: pd.DataFrame | None = None,
                 race_name: str = "race", fuel_effect: float = FUEL_EFFECT_S_PER_LAP):
        self.race_name = race_name
        self.fuel_effect = fuel_effect
        self.laps = mark_clean_laps(laps)
        self.race_laps = total_laps(self.laps)
        self.corner_metrics = corner_metrics
        self.pit_loss_s = estimate_pit_loss(self.laps)
        self.fits: list[StintFit] = fit_all_stints(self.laps, self.race_laps, fuel_effect)
        self.models: dict[str, CompoundModel] = compound_models(self.fits)
        self.ontology = Ontology()
        self._define_schema()
        self._load_objects()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_race(cls, race_csv: str | Path) -> "PitWall":
        """Load a race plus its telemetry and debriefs if they exist next to it.

        data/races/2024_dutch_gp_race.csv pairs with
        data/telemetry/2024_dutch_gp_corner_metrics.csv and
        data/debriefs/2024_dutch_gp_debriefs.json
        """
        race_csv = Path(race_csv)
        stem = race_csv.stem.removesuffix("_race")
        data_dir = race_csv.parent.parent
        return cls.from_files(race_csv,
                              data_dir / "telemetry" / f"{stem}_corner_metrics.csv",
                              data_dir / "debriefs" / f"{stem}_debriefs.json")

    @classmethod
    def from_files(cls, race_csv: str | Path, corner_csv: str | Path | None = None,
                   debriefs_json: str | Path | None = None) -> "PitWall":
        corner_metrics = pd.read_csv(corner_csv) if corner_csv and Path(corner_csv).exists() else None
        pitwall = cls(load_race(race_csv), corner_metrics, race_name=Path(race_csv).stem)
        if debriefs_json and Path(debriefs_json).exists():
            for item in json.loads(Path(debriefs_json).read_text()):
                pitwall.add_debrief(item["driver"], item["text"], item["author"])
        return pitwall

    def _define_schema(self) -> None:
        o = self.ontology
        o.define_object(ObjectType("Driver", "code", ("code", "team", "finish_position"),
                                   "A car and its driver in this race"))
        o.define_object(ObjectType("Stint", "id", ("id", "driver", "stint", "compound", "start_lap", "end_lap",
                                                   "n_clean_laps", "base_s", "deg_s_per_lap", "r2", "cliff_lap"),
                                   "A run of laps on one set of tyres"))
        o.define_object(ObjectType("Lap", "id", ("id", "driver", "lap", "lap_time_s", "compound", "tyre_life",
                                                 "clean", "excluded_reason"),
                                   "One lap by one car"))
        o.define_object(ObjectType("PitStop", "id", ("id", "driver", "lap", "from_compound", "to_compound"),
                                   "A tyre change"))
        o.define_object(ObjectType("Debrief", "id", ("id", "driver", "text", "author", "laps", "corners",
                                                     "symptom", "phase", "verdict", "evidence"),
                                   "What the driver felt, and whether telemetry agrees"))
        o.define_object(ObjectType("StrategyPlan", "id", ("id", "driver", "decision_lap", "stops",
                                                          "expected_time_s", "status", "approved_by"),
                                   "A committed pit plan"))
        for link in (LinkType("driver_stints", "Driver", "Stint"),
                     LinkType("stint_laps", "Stint", "Lap"),
                     LinkType("driver_pit_stops", "Driver", "PitStop"),
                     LinkType("driver_debriefs", "Driver", "Debrief"),
                     LinkType("debrief_laps", "Debrief", "Lap"),
                     LinkType("driver_plans", "Driver", "StrategyPlan")):
            o.define_link(link)
        o.define_action(ActionType(
            "set_pit_plan", "Commit a pit strategy for a driver",
            ("driver", "decision_lap", "stops"),
            validate=self._validate_pit_plan, apply=self._apply_pit_plan))
        o.define_action(ActionType(
            "log_debrief", "Add a driver debrief and check it against the data",
            ("driver", "text", "author"),
            validate=self._validate_debrief, apply=self._apply_debrief))

    def _load_objects(self) -> None:
        o, laps = self.ontology, self.laps
        final = laps.sort_values("lap").groupby("driver").tail(1)
        for _, row in final.iterrows():
            position = None if pd.isna(row["position"]) else int(row["position"])
            o.add("Driver", code=row["driver"], team=row["team"], finish_position=position)

        fit_by_key = {(f.driver, f.stint): f for f in self.fits}
        for (driver, stint), stint_laps in laps.groupby(["driver", "stint"]):
            stint_id = f"{driver}-S{stint}"
            fit = fit_by_key.get((driver, stint))
            o.add("Stint", id=stint_id, driver=driver, stint=int(stint),
                  compound=stint_laps["compound"].mode().iloc[0],
                  start_lap=int(stint_laps["lap"].min()), end_lap=int(stint_laps["lap"].max()),
                  n_clean_laps=int(stint_laps["clean"].sum()),
                  base_s=fit.base_s if fit else None, deg_s_per_lap=fit.deg_s_per_lap if fit else None,
                  r2=fit.r2 if fit else None, cliff_lap=fit.cliff_lap if fit else None)
            o.link("driver_stints", driver, stint_id)
            for _, lap in stint_laps.iterrows():
                lap_id = f"{driver}-L{lap['lap']}"
                o.add("Lap", id=lap_id, driver=driver, lap=int(lap["lap"]),
                      lap_time_s=None if pd.isna(lap["lap_time_s"]) else float(lap["lap_time_s"]),
                      compound=lap["compound"],
                      tyre_life=None if pd.isna(lap["tyre_life"]) else int(lap["tyre_life"]),
                      clean=bool(lap["clean"]), excluded_reason=lap["excluded_reason"])
                o.link("stint_laps", stint_id, lap_id)

        for i, stop in pit_stops(laps).iterrows():
            stop_id = f"{stop['driver']}-P{i}"
            o.add("PitStop", id=stop_id, driver=stop["driver"], lap=int(stop["lap"]),
                  from_compound=stop["from_compound"], to_compound=stop["to_compound"])
            o.link("driver_pit_stops", stop["driver"], stop_id)

    # ----------------------------------------------------------------- queries
    def drivers(self) -> list[str]:
        ordered = sorted(self.ontology.all("Driver"), key=lambda d: d["finish_position"] or 99)
        return [d["code"] for d in ordered]

    def summary(self) -> dict:
        stops = pit_stops(self.laps)
        return {
            "race": self.race_name,
            "laps": self.race_laps,
            "drivers": len(self.drivers()),
            "finishing_order": self.drivers()[:10],
            "pit_loss_s": self.pit_loss_s,
            "pit_stops": len(stops),
            "clean_laps_pct": round(100 * float(self.laps["clean"].mean()), 1),
            "compound_models": {c: asdict(m) for c, m in self.models.items()},
            "fuel_effect_s_per_lap": self.fuel_effect,
        }

    def driver_stints(self, driver: str) -> list[dict]:
        return self.ontology.linked("driver_stints", driver.upper())

    def degradation_table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.fits])

    def corrected_laps(self, driver: str) -> pd.DataFrame:
        """Clean laps for one driver with their fuel-corrected time (for charts)."""
        laps = self.laps[(self.laps["driver"] == driver.upper()) & self.laps["clean"]].copy()
        laps["corrected_s"] = fuel_corrected(laps, self.race_laps, self.fuel_effect)
        return laps

    def lap_residuals(self) -> pd.DataFrame:
        """How much slower each clean lap was than its stint's degradation line."""
        rows = []
        fits = {(f.driver, f.stint): f for f in self.fits}
        clean = self.laps[self.laps["clean"]]
        corrected = fuel_corrected(clean, self.race_laps, self.fuel_effect)
        for idx, lap in clean.iterrows():
            fit = fits.get((lap["driver"], lap["stint"]))
            if fit:
                expected = fit.base_s + fit.deg_s_per_lap * lap["tyre_life"]
                rows.append({"driver": lap["driver"], "lap": int(lap["lap"]),
                             "residual_s": float(corrected[idx] - expected)})
        return pd.DataFrame(rows, columns=["driver", "lap", "residual_s"])

    def driver_state(self, driver: str, at_lap: int) -> DriverState:
        """Where the car stands after `at_lap`: tyre, age, compounds used, pace."""
        driver = driver.upper()
        laps = self.laps[(self.laps["driver"] == driver) & (self.laps["lap"] <= at_lap)].sort_values("lap")
        if laps.empty:
            raise ValueError(f"no laps for {driver} up to lap {at_lap}")
        last = laps.iloc[-1]
        compound = last["compound"] if last["compound"] in DRY_COMPOUNDS else "MEDIUM"
        used = set(laps["compound"]) & set(DRY_COMPOUNDS)
        tyre_age = int(last["tyre_life"]) if pd.notna(last["tyre_life"]) else 1
        return DriverState(driver, int(last["lap"]), compound, tyre_age,
                           self._reference_pace(driver, at_lap), used)

    def _reference_pace(self, driver: str, at_lap: int) -> float:
        """This car's new-tyre pace, expressed on the reference compound.

        Uses only stints that started before `at_lap` (no peeking at the future).
        Falls back to the field median if the car has no fitted stint yet.
        """
        own = [f for f in self.fits if f.driver == driver and f.start_lap <= at_lap]
        candidates = own or [f for f in self.fits if f.start_lap <= at_lap] or self.fits
        paces = [f.base_s - self.models[f.compound].offset_s for f in candidates if f.compound in self.models]
        return float(np.median(paces))

    def recommend(self, driver: str, at_lap: int, safety_car_now: bool = False,
                  max_stops: int = 2) -> dict:
        state = self.driver_state(driver, at_lap)
        options = evaluate(state, self.models, self.race_laps, self.pit_loss_s,
                           max_stops=max_stops, safety_car_now=safety_car_now)
        best_one_stop = next((o for o in options if o.name.startswith("1 stop")), None)
        # The tyre model alone often prefers more stops than teams make, because
        # it doesn't know that passing cars costs time. The break-even says how
        # much time you could lose in traffic before the extra stop stops paying.
        break_even = None
        if options and best_one_stop and options[0].name.startswith("2 stops"):
            break_even = round(best_one_stop.total_time_s - options[0].total_time_s, 1)
        return {
            "driver": state.driver,
            "after_lap": state.current_lap,
            "current_tyre": f"{state.compound} ({state.tyre_age} laps old)",
            "compounds_used": sorted(state.compounds_used),
            "safety_car_now": safety_car_now,
            "pit_loss_s": self.pit_loss_s * (0.5 if safety_car_now else 1),
            "options": [_option_dict(o) for o in options[:6]],
            "pit_window": list(best_one_stop.window) if best_one_stop and best_one_stop.window else None,
            "two_stop_break_even_traffic_s": break_even,
            "assumptions": {
                "fuel_effect_s_per_lap": self.fuel_effect,
                "compound_models": {c: f"{m.deg_s_per_lap} s/lap, {m.offset_s:+.2f} s, from {m.source}"
                                    for c, m in self.models.items()},
                "ignored": "traffic, rivals (undercut/overcut), tyre warm-up, available tyre sets",
            },
        }

    def actual_stops(self, driver: str) -> list[dict]:
        return self.ontology.linked("driver_pit_stops", driver.upper())

    def debriefs(self, driver: str | None = None) -> list[dict]:
        if driver:
            return self.ontology.linked("driver_debriefs", driver.upper())
        return self.ontology.all("Debrief")

    def plans(self, driver: str | None = None) -> list[dict]:
        if driver:
            return self.ontology.linked("driver_plans", driver.upper())
        return self.ontology.all("StrategyPlan")

    # ------------------------------------------------------------- debriefs
    def add_debrief(self, driver: str, text: str, author: str) -> dict:
        """Add a debrief straight away (used when loading saved debriefs)."""
        params = {"driver": driver.upper(), "text": text, "author": author}
        problems = self._validate_debrief(self.ontology, params)
        if problems:
            raise ValueError("; ".join(problems))
        return self._apply_debrief(self.ontology, params)

    def preview_debrief(self, driver: str, text: str) -> tuple[debriefs.Debrief, debriefs.Evidence]:
        parsed = debriefs.parse(text, driver, "preview", "preview")
        return parsed, debriefs.check(parsed, self.laps, self.corner_metrics, self.lap_residuals())

    # --------------------------------------------------------- action rules
    def _validate_pit_plan(self, o: Ontology, params: dict) -> list[str]:
        problems = []
        driver = str(params.get("driver", "")).upper()
        if o.get("Driver", driver) is None:
            return [f"unknown driver '{driver}'"]
        decision_lap = int(params.get("decision_lap", 0))
        stops = params.get("stops") or []
        previous = decision_lap
        for stop in stops:
            lap, compound = int(stop[0]), str(stop[1]).upper()
            if compound not in DRY_COMPOUNDS:
                problems.append(f"'{compound}' is not a dry compound")
            if lap <= previous:
                problems.append(f"stop on lap {lap} must come after lap {previous}")
            if lap >= self.race_laps:
                problems.append(f"stop on lap {lap} is not before the final lap ({self.race_laps})")
            previous = lap
        if not problems:
            state = self.driver_state(driver, decision_lap)
            used = state.compounds_used | {state.compound} | {str(s[1]).upper() for s in stops}
            if len(used & set(DRY_COMPOUNDS)) < 2:
                problems.append("a dry race needs at least two different compounds")
        return problems

    def _apply_pit_plan(self, o: Ontology, params: dict) -> None:
        driver = params["driver"].upper()
        for plan in self.plans(driver):
            if plan["status"] == "active":
                o.update("StrategyPlan", plan["id"], status="superseded")
        plan_id = f"{driver}-plan-{len(self.plans(driver)) + 1}"
        o.add("StrategyPlan", id=plan_id, driver=driver, decision_lap=int(params["decision_lap"]),
              stops=[[int(lap), str(c).upper()] for lap, c in params["stops"]],
              expected_time_s=params.get("expected_time_s"), status="active",
              approved_by=params.get("approved_by"))
        o.link("driver_plans", driver, plan_id)

    def _validate_debrief(self, o: Ontology, params: dict) -> list[str]:
        problems = []
        if o.get("Driver", str(params.get("driver", "")).upper()) is None:
            problems.append(f"unknown driver '{params.get('driver')}'")
        if not str(params.get("text", "")).strip():
            problems.append("the debrief is empty")
        return problems

    def _apply_debrief(self, o: Ontology, params: dict) -> dict:
        driver = params["driver"].upper()
        debrief_id = f"{driver}-D{len(self.debriefs(driver)) + 1}"
        parsed = debriefs.parse(params["text"], driver, params["author"], debrief_id)
        evidence = debriefs.check(parsed, self.laps, self.corner_metrics, self.lap_residuals())
        obj = o.add("Debrief", id=debrief_id, driver=driver, text=parsed.text, author=parsed.author,
                    laps=parsed.laps, corners=parsed.corners, symptom=parsed.symptom, phase=parsed.phase,
                    verdict=evidence.verdict, evidence=asdict(evidence))
        o.link("driver_debriefs", driver, debrief_id)
        for lap in parsed.laps:
            if o.get("Lap", f"{driver}-L{lap}"):
                o.link("debrief_laps", debrief_id, f"{driver}-L{lap}")
        return obj


def _option_dict(option: StrategyOption) -> dict:
    return {
        "plan": option.name,
        "stops": [{"lap": lap, "compound": c} for lap, c in option.stops],
        "total_time_s": option.total_time_s,
        "slower_than_best_by_s": option.delta_to_best_s,
        "window": list(option.window) if option.window else None,
    }
