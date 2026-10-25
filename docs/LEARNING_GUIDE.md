# Learning guide

This guide is for one goal: being able to explain, defend and extend every part of Pit Wall in an interview. Work through one section per study session. At the end of each section, close the file and explain the module out loud in two minutes. If you can't, read it again.

Keep a Python shell open at the repo root and run the "Try it" snippets as you go.

---

## 1. `ontology.py` — the backbone

**What it is.** A minimal version of an ontology: *object types* (Driver, Stint…), *link types* (Driver → Stint), and *action types* (decisions with rules).

**Read for:**
- `add` / `get` / `linked` / `linked_from`: how objects and relationships are stored (dicts keyed by primary key, links as `(source, target)` pairs).
- `propose` → `approve` / `reject`: the life of a decision.
- `_require_human`: the one line that stops an AI from approving anything.
- `approve` validates **again** before applying. Why? (Answer in section 10.)

**Try it:**
```python
from pitwall.core import PitWall
pw = PitWall.from_race("data/races/2024_dutch_gp_race.csv")
pw.ontology.linked("driver_stints", "NOR")
pw.ontology.linked_from("driver_stints", "NOR-S2")
r = pw.ontology.propose("set_pit_plan", {"driver": "NOR", "decision_lap": 20, "stops": [[28, "HARD"]]}, "ai:test")
pw.ontology.approve(r.id, "ai:test")      # raises: AI can't approve
pw.ontology.approve(r.id, "Nathan"); pw.plans("NOR")
```

## 2. `data.py` — trusting the data

**Read for:**
- `FIELD_MAP`: renaming terse source fields to readable ones. Readable names make every later module easier to read.
- `mark_clean_laps`: rules applied in order, *first reason wins*. Why record a reason instead of dropping rows? (Auditability: a human can disagree with a rule.)
- The outlier rule only looks at laps that survived the other rules. If it used all laps, safety-car laps would drag the median up and hide real outliers.
- `estimate_pit_loss`: (in-lap + out-lap) − 2 × a normal lap. Neutralised stops are skipped because a safety car changes what "normal" means.

**Try it:** `pw.laps["excluded_reason"].value_counts()`, then look at one in-lap and out-lap pair for NOR around lap 28.

## 3. `degradation.py` — the physics in numbers

**Core idea:** two effects pull lap time in opposite directions over a stint. Fuel burn makes the car faster; tyre wear makes it slower. Correct for fuel first:

```
corrected = lap_time − fuel_effect × laps_of_fuel_left
corrected ≈ base + deg × tyre_age        (straight-line fit, np.polyfit)
```

**Read for:**
- `fuel_corrected`: check the sign. Early laps carry more fuel, so more is subtracted from them.
- `fit_stint`: slope = degradation (s/lap), intercept = pace on new tyres, R² = how well a straight line explains the stint.
- Cliff detection: the last 3 clean laps all more than 0.3 s above the line.
- `_paired_offsets`: compound pace is compared *within each driver*. A Red Bull on hards vs a Williams on mediums tells you about the cars, not the tyres.
- Priors: used when a compound has too little data, and always labelled `source="prior"`.

**Try it:** run the tests in `tests/test_models.py` and read `test_ignoring_fuel_makes_tyres_look_better_than_they_are`. Then change `fuel_effect` to 0.04 and 0.08 and watch `pw.models` change. That's a sensitivity analysis, and worth mentioning in an interview.

## 4. `strategy.py` — turning degradation into a decision

**Race-time model:** sum each stint's lap times and add pit loss per stop. `stint_time` uses the arithmetic-series formula (sum of ages = n × (first + last) / 2) rather than a loop.

**Read for:**
- `one_stop_curve`: total time for every possible stop lap. Its minimum is the best lap. The **window** is every lap within 1 s of the minimum. A flat-bottomed curve is a wide window.
- The two-compound rule (`_uses_two_compounds`).
- Safety car: pit loss halves, but only for a stop on the very next lap.
- `two_stop_break_even_traffic_s` (in `core.py`): the model's way of saying "I don't know about traffic. You decide if it costs more than X seconds."

**Try it:** `pw.recommend("NOR", 20)` vs `pw.recommend("NOR", 20, safety_car_now=True)`. Explain why the plan changes.

## 5. `corners.py` + `debrief.py` — words to evidence

This is the part only someone who has driven the car would think to build. Practise explaining it best.

**corners.py** reduces ~580 telemetry samples per lap to four numbers per corner: minimum speed, brake point, apex-to-full-throttle distance, and peak lateral g.

**debrief.py:**
1. `parse` uses keyword lists and regular expressions to pull out laps, corners, symptom and phase.
2. `EXPECTED` maps symptom + phase to the signal that should move, and in which direction:
   - understeer → min speed **down**
   - exit oversteer / traction → full throttle **later**
   - braking problems → brake point **earlier**
3. `_compare` uses Welch's t-statistic: difference in means ÷ its standard error. |t| ≥ 2 with a minimum practical size gives a verdict.

**Why "contradicted" matters.** NOR's T7 debrief says he's losing the front, but his minimum speed actually went *up* by 8.8 km/h, probably from lower fuel and a rubbered-in track. A tool that only ever agreed with the driver would be useless. Disagreement is where an engineer starts asking better questions.

**Try it:** `pw.preview_debrief("VER", "understeer mid-corner T10, laps 68-72")`, then invent your own and see what the data says.

## 6. `backtest.py` — honest evaluation

**Key idea: no lookahead.** At the decision lap, the model is rebuilt from `laps[laps["lap"] <= decision_lap]`. Using the full race would leak future information, such as how the hard tyre performed later, and inflate the score.

**Read for:** the comparison is like-for-like. If a team made one stop, we compare with the model's best *one-stop* plan.

**What the results mean.** See the README table. Zandvoort 2024 is +9.4 laps late on average, because track position and the undercut matter more than tyre deg there. Silverstone 2026 is within 3 laps for 57% of cars.

## 7. `core.py` — one entry point

`PitWall` loads data, fits models, builds the ontology and defines the action rules. The app, the CLI and the AI all call it. That's why they can't disagree.

`_reference_pace` only uses stints that started before the decision lap, the same no-peeking rule as the backtest.

## 8. `engineer.py` — the AI loop

```
send question + tool definitions
loop:
    model replies
    if stop_reason != "tool_use": return text
    for each tool_use block: run tool → tool_result (same id)
    send results back
```

**Read for:**
- `TOOLS`: each has a name, a description (the model reads this to decide when to call it) and a JSON schema for inputs.
- `run_tool`: errors become `is_error` results, so the model can recover instead of crashing.
- `propose_pit_plan` uses `AGENT_NAME = "ai:race-engineer"`, so the ontology's human-only rule applies automatically.
- `tests/test_engineer.py` tests the loop with a fake client. No API key is needed, and the tests are fast and deterministic.

## 9. `app.py` — the interface

Streamlit reruns the whole script on every click. State that must survive a rerun (the PitWall with its pending approvals) lives in `st.session_state`. `tests/test_app.py` runs the app headless with `AppTest` and clicks *Propose* then *Approve*.

---

## 10. Interview questions to rehearse

1. **Why fuel-correct before fitting?** Fuel burn (~0.06 s/lap) and tyre wear pull in opposite directions. Without correcting, you underestimate wear. At Zandvoort 2024 it hides it completely.
2. **Why re-validate at approval time?** The race moves on. A plan proposed at lap 20 to stop on lap 22 is invalid at lap 25. Validation at proposal only checks the world as it was.
3. **Why can't the AI approve?** Authority should be enforced by the system, not by instructions to the model. A prompt can be ignored or manipulated; a check in `approve` can't.
4. **Your backtest says 10% at Zandvoort. Isn't the model bad?** It's a tyre-only baseline. The error is biased in one direction (late), which tells you what's missing: rival-aware undercut logic. A biased error you can explain is more useful than a vague "it's sometimes off".
5. **What is lookahead bias and how do you avoid it?** Using information that wasn't available at decision time. Rebuild every model from data up to the decision lap only.
6. **How does the debrief checker decide?** Welch t-statistic on the symptom's predicted signal, debrief laps vs the rest of the stint, plus a minimum effect size. It gives three verdicts, including "contradicted".
7. **What's the weakest assumption?** Linear degradation and no traffic. Next would be the undercut model, then a traffic cost learned from real position changes.
8. **How would this scale to a real team?** Replace the CSVs with live timing feeds, the in-memory ontology with a database, and the approval queue with role-based permissions (strategist vs race engineer). The object/action structure stays the same.
9. **Why an ontology instead of just DataFrames?** DataFrames answer analysts' questions. An ontology gives operators shared objects, decisions with rules, and a record of who decided what. That's the difference between analysis and operations.
10. **What did you build vs what did AI build?** Answer honestly. You designed the domain model, chose what to measure and why, and validated the results against what you know from driving. The code was pair-programmed with Claude. Then show something you changed yourself: the exercises below.

## 11. Exercises: make it yours

Do at least two before an interview, so you can say "here's the part I extended."

1. **Easy — four-eyes rule.** In `Ontology.approve`, refuse if `decided_by` equals the proposer. Add a test.
2. **Easy — corner names.** Add a Zandvoort corner-name map (T1 Tarzan…) and show names in the debrief view.
3. **Medium — sensitivity table.** Add a CLI command that re-runs the backtest for fuel effects 0.03-0.09 and prints how the hit rate changes.
4. **Medium — traffic cost.** Add a `traffic_cost_per_stop_s` parameter to `evaluate` and re-run the Zandvoort backtest. Does stop-count agreement improve?
5. **Hard — undercut.** Use lap-by-lap gaps to the car ahead and behind (compare `session_time_s`, when each car finished the lap), and model "if I stop now and they stop next lap, do I come out ahead?" Re-run the backtest.
6. **Hard — your own data.** Export telemetry from your sim sessions, write an adapter to the same corner-metrics format, and test debriefs written about laps *you* drove.

## 12. Three-minute demo script

1. **Race tab (20 s):** "Real 2024 Dutch GP data. Every excluded lap has a reason."
2. **Tyres (40 s):** NOR's chart. "Fuel-corrected, so the slope is pure tyre wear. Without correction this flatlines."
3. **Strategy (50 s):** slider at lap 20, window 34-41, actual stop lap 28. "The tyre-only model says later. The backtest shows this bias at Zandvoort: it doesn't know about the undercut. That's my next feature." Toggle the safety car and the plan changes.
4. **Debriefs (40 s):** LEC supported, NOR contradicted. "The point isn't to agree with the driver; it's to know when to trust the feel."
5. **Approvals (30 s):** propose, then show the AI can't approve. "Anyone can propose. Only a named human decides, and it's all in the audit log."
