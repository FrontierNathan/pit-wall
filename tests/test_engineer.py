"""The AI engineer, tested with a fake LLM client so no API key or network is needed."""

from types import SimpleNamespace

import pytest

from pitwall.core import PitWall
from pitwall.engineer import AGENT_NAME, RaceEngineer
from pitwall.ontology import OntologyError

RACE = "data/races/2024_dutch_gp_race.csv"


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def text(value):
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Replays a script of responses and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        content, stop = self.responses.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop)


@pytest.fixture(scope="module")
def pitwall():
    return PitWall.from_race(RACE)


def test_tool_loop_runs_tools_and_returns_final_answer(pitwall):
    client = FakeClient([
        ([tool_use("t1", "recommend_pit_window", {"driver": "NOR", "after_lap": 20})], "tool_use"),
        ([text("Window is laps 34-41.")], "end_turn"),
    ])
    reply = RaceEngineer(pitwall, client=client).ask("When should NOR stop?")
    assert reply.text == "Window is laps 34-41."
    assert [c.name for c in reply.tool_calls] == ["recommend_pit_window"]
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert '"pit_window"' in tool_result["content"]


def test_agent_can_propose_but_never_approve(pitwall):
    stops = [{"lap": 30, "compound": "HARD"}]
    client = FakeClient([
        ([tool_use("t1", "propose_pit_plan",
                   {"driver": "VER", "decision_lap": 20, "stops": stops, "rationale": "test"})], "tool_use"),
        ([text("Proposed.")], "end_turn"),
    ])
    reply = RaceEngineer(pitwall, client=client).ask("Propose a one-stop for VER")
    proposal_id = reply.tool_calls[0].output["proposal_id"]
    record = next(r for r in pitwall.ontology.pending() if r.id == proposal_id)
    assert record.proposed_by == AGENT_NAME
    with pytest.raises(OntologyError):
        pitwall.ontology.approve(proposal_id, AGENT_NAME)
    pitwall.ontology.approve(proposal_id, "Nathan")
    assert pitwall.plans("VER")[-1]["approved_by"] == "Nathan"


def test_invalid_proposal_comes_back_to_the_model_as_an_error(pitwall):
    bad = [{"lap": 5, "compound": "MEDIUM"}]  # before the decision lap, and breaks the two-compound rule
    client = FakeClient([
        ([tool_use("t1", "propose_pit_plan",
                   {"driver": "LEC", "decision_lap": 20, "stops": bad, "rationale": "x"})], "tool_use"),
        ([text("That plan isn't allowed.")], "end_turn"),
    ])
    reply = RaceEngineer(pitwall, client=client).ask("Propose something silly")
    assert reply.tool_calls[0].is_error
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
