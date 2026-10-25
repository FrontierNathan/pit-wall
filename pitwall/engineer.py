"""The AI race engineer: an LLM that can READ the ontology and PROPOSE actions.

This mirrors how Palantir's AIP connects language models to an ontology:

* The model never sees the raw database. It can only call the tools below,
  each of which goes through PitWall (the same code the app and CLI use).
* It can propose a pit plan, but the proposal lands in the approval queue
  under the name "ai:race-engineer". The ontology refuses any approval from
  an "ai:" actor, so a human always makes the call.
* Every tool call is recorded in a trace, so you can see exactly which
  numbers an answer was built on.

The loop is the standard tool-use pattern of the Claude Messages API:
send the question with tool definitions -> if the model asks for a tool, run
it and send back the result -> repeat until the model gives a final answer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .core import PitWall
from .ontology import ActionValidationError

AGENT_NAME = "ai:race-engineer"
DEFAULT_MODEL = os.environ.get("PITWALL_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a race engineer's assistant on the pit wall.
Answer questions about this race using ONLY the tools provided; never guess numbers.
Be concise: lead with the answer, then the 2-3 numbers that support it.
Always state the main assumption behind a strategy call (the model ignores traffic,
rivals and tyre warm-up). If the data can't answer the question, say so.
You may propose a pit plan with propose_pit_plan when the user asks for one, but you
cannot approve it: a human engineer approves or rejects every proposal."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "race_summary",
        "description": "Race length, finishing order, pit loss, and tyre degradation per compound.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "driver_stints",
        "description": "A driver's stints: compound, laps, degradation rate and fit quality.",
        "input_schema": {
            "type": "object",
            "properties": {"driver": {"type": "string", "description": "Three-letter code, e.g. NOR"}},
            "required": ["driver"],
        },
    },
    {
        "name": "recommend_pit_window",
        "description": ("Best pit plans for a driver given the situation after a lap, using only "
                        "the tyre model. Optionally assume a safety car is out right now."),
        "input_schema": {
            "type": "object",
            "properties": {
                "driver": {"type": "string"},
                "after_lap": {"type": "integer", "description": "Last completed lap"},
                "safety_car_now": {"type": "boolean", "default": False},
            },
            "required": ["driver", "after_lap"],
        },
    },
    {
        "name": "actual_pit_stops",
        "description": "The pit stops the driver actually made in this race.",
        "input_schema": {
            "type": "object",
            "properties": {"driver": {"type": "string"}},
            "required": ["driver"],
        },
    },
    {
        "name": "driver_debriefs",
        "description": "Driver debriefs and whether telemetry supports each one.",
        "input_schema": {
            "type": "object",
            "properties": {"driver": {"type": "string", "description": "Optional; omit for all"}},
        },
    },
    {
        "name": "propose_pit_plan",
        "description": ("Put a pit plan in the approval queue for a human engineer. Does NOT "
                        "commit anything."),
        "input_schema": {
            "type": "object",
            "properties": {
                "driver": {"type": "string"},
                "decision_lap": {"type": "integer"},
                "stops": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lap": {"type": "integer"},
                            "compound": {"type": "string", "enum": ["SOFT", "MEDIUM", "HARD"]},
                        },
                        "required": ["lap", "compound"],
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["driver", "decision_lap", "stops", "rationale"],
        },
    },
]


@dataclass
class ToolCall:
    name: str
    input: dict
    output: Any
    is_error: bool = False


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full history, for follow-ups


class RaceEngineer:
    def __init__(self, pitwall: PitWall, client: Any = None, model: str = DEFAULT_MODEL,
                 max_steps: int = 8):
        self.pitwall = pitwall
        self.model = model
        self.max_steps = max_steps
        if client is None:
            import anthropic  # imported here so the rest of the project works without it

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client

    def ask(self, question: str, history: list[dict] | None = None) -> Reply:
        messages = list(history or []) + [{"role": "user", "content": question}]
        calls: list[ToolCall] = []
        for _ in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model, max_tokens=1500, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                text = "".join(block.text for block in response.content if block.type == "text")
                return Reply(text.strip(), calls, messages)

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = self.run_tool(block.name, dict(block.input))
                calls.append(call)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(call.output, default=str),
                    "is_error": call.is_error,
                })
            messages.append({"role": "user", "content": results})
        return Reply("I couldn't finish within the step limit; try a narrower question.", calls, messages)

    def run_tool(self, name: str, args: dict) -> ToolCall:
        """Run one tool. Errors go back to the model as data so it can recover."""
        pw = self.pitwall
        try:
            if name == "race_summary":
                output = pw.summary()
            elif name == "driver_stints":
                output = pw.driver_stints(args["driver"])
            elif name == "recommend_pit_window":
                output = pw.recommend(args["driver"], int(args["after_lap"]),
                                      safety_car_now=bool(args.get("safety_car_now", False)))
            elif name == "actual_pit_stops":
                output = pw.actual_stops(args["driver"])
            elif name == "driver_debriefs":
                output = pw.debriefs(args.get("driver"))
            elif name == "propose_pit_plan":
                params = {
                    "driver": args["driver"].upper(),
                    "decision_lap": int(args["decision_lap"]),
                    "stops": [[int(s["lap"]), s["compound"].upper()] for s in args["stops"]],
                }
                record = pw.ontology.propose("set_pit_plan", params, AGENT_NAME, args.get("rationale", ""))
                output = {"proposal_id": record.id, "status": record.status,
                          "note": "Waiting for a human engineer to approve or reject."}
            else:
                return ToolCall(name, args, {"error": f"unknown tool {name}"}, is_error=True)
            return ToolCall(name, args, output)
        except ActionValidationError as error:
            return ToolCall(name, args, {"error": "proposal rejected by validation", "problems": error.problems},
                            is_error=True)
        except (KeyError, ValueError) as error:
            return ToolCall(name, args, {"error": str(error)}, is_error=True)
