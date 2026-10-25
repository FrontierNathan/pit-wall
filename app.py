"""Pit Wall — Streamlit app.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pitwall.core import PitWall
from pitwall.ontology import ActionValidationError, OntologyError
from pitwall.strategy import one_stop_curve

st.set_page_config(page_title="Pit Wall", page_icon="🏁", layout="wide")

COMPOUND_COLORS = alt.Scale(domain=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                            range=["#da291c", "#f2c300", "#8c8c8c", "#43b02a", "#0067ad"])
VERDICT_ICON = {"supported": "🟢", "contradicted": "🔴", "inconclusive": "🟡", "no data": "⚪"}
RACES = sorted(Path("data/races").glob("*.csv"))


def race_label(path: Path) -> str:
    """data/races/2024_dutch_gp_race.csv -> '2024 Dutch GP'"""
    words = path.stem.removesuffix("_race").split("_")
    return " ".join(w.upper() if w == "gp" else w.capitalize() for w in words)


# ------------------------------------------------------------------ session
def pitwall_for(race: Path):
    """One PitWall per race per browser session (approvals and debriefs live here)."""
    store = st.session_state.setdefault("pitwalls", {})
    if str(race) not in store:
        store[str(race)] = PitWall.from_race(race)
    return store[str(race)]


with st.sidebar:
    st.title("🏁 Pit Wall")
    race = st.selectbox("Race", RACES, format_func=race_label)
    pw = pitwall_for(race)
    driver = st.selectbox("Driver", pw.drivers())
    user = st.text_input("Your name (signs approvals)", value="Nathan")
    st.caption(f"{pw.race_laps} laps · pit loss ≈ {pw.pit_loss_s} s · "
               f"{pw.summary()['clean_laps_pct']}% clean laps")
    st.caption("Data: TracingInsights / FastF1 (unofficial, not associated with Formula 1).")

overview, tyres, strategy, debrief_tab, approvals, engineer_tab = st.tabs(
    ["Race", "Tyres", "Strategy", "Debriefs", "Approvals", "AI engineer"])

# --------------------------------------------------------------------- race
with overview:
    st.subheader("Stints by car")
    stints = pd.DataFrame(pw.ontology.all("Stint"))
    order = pw.drivers()
    chart = alt.Chart(stints).mark_bar(height=12).encode(
        x=alt.X("start_lap:Q", title="Lap"), x2="end_lap:Q",
        y=alt.Y("driver:N", sort=order, title=None),
        color=alt.Color("compound:N", scale=COMPOUND_COLORS),
        tooltip=["driver", "compound", "start_lap", "end_lap", "deg_s_per_lap", "n_clean_laps"],
    ).properties(height=22 * len(order))
    st.altair_chart(chart, width="stretch")
    excluded = pw.laps[~pw.laps["clean"]]["excluded_reason"].value_counts().rename("laps")
    st.caption("Laps left out of the tyre model, and why:")
    st.dataframe(excluded)

# -------------------------------------------------------------------- tyres
with tyres:
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{driver}: fuel-corrected lap times")
        laps = pw.corrected_laps(driver)
        points = alt.Chart(laps).mark_circle(size=40).encode(
            x=alt.X("lap:Q", title="Lap"),
            y=alt.Y("corrected_s:Q", title="Lap time on an empty tank (s)", scale=alt.Scale(zero=False)),
            color=alt.Color("compound:N", scale=COMPOUND_COLORS),
            tooltip=["lap", "lap_time_s", "corrected_s", "tyre_life", "compound"],
        )
        fit_rows = []
        for s in pw.driver_stints(driver):
            if s["deg_s_per_lap"] is None:
                continue
            stint_laps = laps[laps["stint"] == s["stint"]]
            for _, lap in stint_laps.iterrows():
                fit_rows.append({"lap": lap["lap"], "stint": s["stint"], "compound": s["compound"],
                                 "fit": s["base_s"] + s["deg_s_per_lap"] * lap["tyre_life"]})
        lines = alt.Chart(pd.DataFrame(fit_rows)).mark_line().encode(
            x="lap:Q", y="fit:Q", detail="stint:N", color=alt.Color("compound:N", scale=COMPOUND_COLORS))
        st.altair_chart(points + lines, width="stretch")
        st.caption("Each dot is a clean lap with the fuel effect removed. The slope of each line is "
                   "the tyre degradation for that stint.")
    with right:
        st.subheader("Degradation by compound")
        models = pd.DataFrame([m.__dict__ for m in pw.models.values()])
        st.dataframe(models, hide_index=True, width="stretch")
        st.caption("offset_s = pace on new tyres vs MEDIUM. 'prior' means the race didn't have "
                   "enough stints on that compound, so a default assumption is used.")
        st.subheader(f"{driver}'s stints")
        st.dataframe(pd.DataFrame(pw.driver_stints(driver)).drop(columns=["id", "driver"]),
                     hide_index=True, width="stretch")

# ----------------------------------------------------------------- strategy
def strategy_tab() -> None:
    stops = pw.actual_stops(driver)
    default_lap = max(min(s["lap"] for s in stops) - 8, 2) if stops else 15
    c1, c2 = st.columns([2, 1])
    after_lap = c1.slider("Decide after lap", 2, pw.race_laps - 2, default_lap)
    safety_car = c2.toggle("Safety car out right now", help="Halves the pit loss for a stop next lap")
    last_lap = int(pw.laps[pw.laps["driver"] == driver]["lap"].max())
    if last_lap < after_lap:
        st.warning(f"{driver}'s last lap was lap {last_lap} (retired or lapped), so this shows the "
                   f"situation after lap {last_lap}.")
    try:
        rec = pw.recommend(driver, after_lap, safety_car_now=safety_car)
    except ValueError as error:
        st.warning(str(error))
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Current tyre", rec["current_tyre"])
    m2.metric("Best 1-stop window", "–" if not rec["pit_window"] else f"laps {rec['pit_window'][0]}–{rec['pit_window'][1]}")
    m3.metric("Actual stop(s)", ", ".join(f"L{s['lap']} → {s['to_compound']}" for s in stops) or "none")
    if rec["two_stop_break_even_traffic_s"]:
        st.info(f"The tyre model prefers two stops, by {rec['two_stop_break_even_traffic_s']} s. "
                "That only pays off if the extra stop costs less than that in traffic — at tracks "
                "where overtaking is hard, it usually doesn't.")

    options = pd.DataFrame([{**o, "stops": ", ".join(f"L{s['lap']} {s['compound']}" for s in o["stops"])}
                            for o in rec["options"]])
    st.dataframe(options, hide_index=True, width="stretch")

    state = pw.driver_state(driver, after_lap)
    curves = []
    for compound in ("SOFT", "MEDIUM", "HARD"):
        if len((state.compounds_used | {state.compound, compound}) & {"SOFT", "MEDIUM", "HARD"}) < 2:
            continue
        for lap, total in one_stop_curve(state, pw.models, pw.race_laps, pw.pit_loss_s, compound, safety_car).items():
            curves.append({"stop_lap": lap, "compound": compound, "total_s": total})
    if curves:
        curve_df = pd.DataFrame(curves)
        curve_df["vs_best_s"] = curve_df["total_s"] - curve_df["total_s"].min()
        st.altair_chart(alt.Chart(curve_df[curve_df["vs_best_s"] < 15]).mark_line(point=True).encode(
            x=alt.X("stop_lap:Q", title="Stop on lap"),
            y=alt.Y("vs_best_s:Q", title="Seconds slower than the best one-stop"),
            color=alt.Color("compound:N", scale=COMPOUND_COLORS, title="New tyre"),
        ), width="stretch")
        st.caption("A flat bottom = a wide window. That flexibility is what lets you react to rivals.")

    best = rec["options"][0] if rec["options"] else None
    if best and best["stops"] and st.button(f"Propose plan: {best['plan']} ({', '.join('L' + str(s['lap']) for s in best['stops'])})"):
        try:
            record = pw.ontology.propose(
                "set_pit_plan",
                {"driver": driver, "decision_lap": after_lap,
                 "stops": [[s["lap"], s["compound"]] for s in best["stops"]],
                 "expected_time_s": best["total_time_s"]},
                proposed_by=f"human:{user}", rationale="best option from the tyre model")
            st.success(f"Proposal {record.id} sent to Approvals.")
        except ActionValidationError as error:
            st.error(f"Not allowed: {error}")

    plans = pw.plans(driver)
    if plans:
        st.caption("Plans for this driver")
        plans_df = pd.DataFrame(plans)
        plans_df["stops"] = plans_df["stops"].map(lambda stops: ", ".join(f"L{lap} {c}" for lap, c in stops))
        st.dataframe(plans_df, hide_index=True, width="stretch")


with strategy:
    strategy_tab()

# ----------------------------------------------------------------- debriefs
with debrief_tab:
    st.subheader("Does the data agree with what the driver felt?")
    all_debriefs = pw.debriefs()
    if pw.corner_metrics is None:
        st.info("This race has no corner telemetry bundled, so only lap-time checks are possible. "
                "Build it with scripts/build_corner_metrics.py.")
    for d in all_debriefs:
        with st.expander(f"{VERDICT_ICON[d['verdict']]} {d['driver']} — {d['text']}"):
            ev = d["evidence"]
            st.write(f"**Verdict: {d['verdict']}** — {ev['explanation']}")
            st.caption(f"Parsed as: symptom={d['symptom']}, phase={d['phase']}, corners={d['corners']}, "
                       f"laps={d['laps']} · author: {d['author']}")
            if ev["metric"] and ev["metric"] != "lap_residual_s" and pw.corner_metrics is not None:
                rows = pw.corner_metrics[(pw.corner_metrics["driver"] == d["driver"])
                                         & (pw.corner_metrics["corner"].isin(d["corners"]))]
                stint_of = pw.laps[pw.laps["driver"] == d["driver"]].set_index("lap")["stint"]
                rows = rows.assign(stint=rows["lap"].map(stint_of),
                                   debrief_laps=rows["lap"].isin(d["laps"]))
                same_stint = rows[rows["stint"] == rows[rows["debrief_laps"]]["stint"].min()]
                st.altair_chart(alt.Chart(same_stint).mark_circle(size=60).encode(
                    x=alt.X("lap:Q"), y=alt.Y(f"{ev['metric']}:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("debrief_laps:N", title="Debrief laps",
                                    scale=alt.Scale(domain=[True, False], range=["#da291c", "#8c8c8c"])),
                    tooltip=["lap", "corner", ev["metric"]],
                ), width="stretch")

    st.divider()
    st.subheader("Log a debrief")
    st.caption('Write it the way a driver would say it, e.g. "rear sliding on exit of T11, laps 40-44".')
    text = st.text_area("Debrief", key="debrief_text")
    if text.strip():
        parsed, evidence = pw.preview_debrief(driver, text)
        st.write(f"Parsed: symptom **{parsed.symptom}**, phase **{parsed.phase}**, "
                 f"corners **{parsed.corners}**, laps **{parsed.laps}**")
        st.write(f"{VERDICT_ICON[evidence.verdict]} Preview: **{evidence.verdict}** — {evidence.explanation}")
        if st.button("Save debrief"):
            record = pw.ontology.propose("log_debrief", {"driver": driver, "text": text, "author": user},
                                         proposed_by=f"human:{user}")
            pw.ontology.approve(record.id, user, note="logged by the person who wrote it")
            st.success("Saved.")
            st.rerun()

# ---------------------------------------------------------------- approvals
with approvals:
    pending = pw.ontology.pending()
    st.subheader(f"Waiting for a decision ({len(pending)})")
    if not pending:
        st.caption("Nothing to approve. Proposals from you or the AI engineer appear here.")
    for record in pending:
        with st.container(border=True):
            st.write(f"**{record.action}** · proposed by `{record.proposed_by}` · {record.created_at}")
            st.json(record.params)
            if record.rationale:
                st.caption(f"Rationale: {record.rationale}")
            note = st.text_input("Note (required to reject)", key=f"note-{record.id}")
            a, r = st.columns(2)
            if a.button("Approve", key=f"approve-{record.id}", type="primary"):
                result = pw.ontology.approve(record.id, user, note)
                (st.success if result.status == "approved" else st.error)(f"{result.status}: {result.decision_note}")
                st.rerun()
            if r.button("Reject", key=f"reject-{record.id}"):
                try:
                    pw.ontology.reject(record.id, user, note)
                    st.rerun()
                except OntologyError as error:
                    st.error(str(error))

    st.subheader("Audit log")
    log = pw.ontology.audit_log()
    if log:
        log_df = pd.DataFrame(log)
        log_df["params"] = log_df["params"].map(json.dumps)  # nested data -> readable text
        st.dataframe(log_df, hide_index=True, width="stretch")
        st.download_button("Download audit log (JSON)", json.dumps(log, indent=2), "audit_log.json")

# ------------------------------------------------------------- AI engineer
with engineer_tab:
    st.subheader("Ask the race engineer")
    st.caption("An LLM that can only read this race through the same tools you see above, and can only "
               "*propose* plans — you approve them. Each answer shows the tool calls it used.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("Set the ANTHROPIC_API_KEY environment variable and restart to enable this tab. "
                "Everything else works without it.")
    else:
        from pitwall.engineer import RaceEngineer

        chat_key = f"chat-{race}"
        history = st.session_state.setdefault(chat_key, {"messages": [], "display": []})
        for role, content in history["display"]:
            st.chat_message(role).markdown(content)
        question = st.chat_input(f"e.g. When should {driver} have pitted, and was the actual stop close?")
        if question:
            st.chat_message("user").markdown(question)
            with st.spinner("Checking the data…"):
                reply = RaceEngineer(pw).ask(question, history["messages"])
            trace = "\n".join(f"- `{c.name}({json.dumps(c.input)})`" + (" ⚠️" if c.is_error else "")
                              for c in reply.tool_calls)
            answer = reply.text + (f"\n\n<details><summary>Tools used</summary>\n\n{trace}\n</details>" if trace else "")
            st.chat_message("assistant").markdown(answer, unsafe_allow_html=True)
            history["messages"] = reply.messages
            history["display"] += [("user", question), ("assistant", answer)]
