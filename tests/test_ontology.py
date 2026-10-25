import pytest

from pitwall.ontology import (ActionType, ActionValidationError, LinkType, ObjectType, Ontology,
                              OntologyError)


def make_ontology() -> Ontology:
    o = Ontology()
    o.define_object(ObjectType("Driver", "code", ("code", "team")))
    o.define_object(ObjectType("Plan", "id", ("id", "driver", "lap", "approved_by")))
    o.define_link(LinkType("driver_plans", "Driver", "Plan"))

    def validate(ont, params):
        problems = []
        if ont.get("Driver", params["driver"]) is None:
            problems.append("unknown driver")
        if params["lap"] < 1:
            problems.append("lap must be positive")
        return problems

    def apply(ont, params):
        plan = ont.add("Plan", id=f"{params['driver']}-1", driver=params["driver"], lap=params["lap"],
                       approved_by=params["approved_by"])
        ont.link("driver_plans", params["driver"], plan["id"])

    o.define_action(ActionType("set_plan", "", ("driver", "lap"), validate, apply))
    o.add("Driver", code="NOR", team="McLaren")
    return o


def test_objects_and_links():
    o = make_ontology()
    o.add("Plan", id="p1", driver="NOR", lap=20, approved_by=None)
    o.link("driver_plans", "NOR", "p1")
    assert o.linked("driver_plans", "NOR")[0]["lap"] == 20
    assert o.linked_from("driver_plans", "p1")[0]["code"] == "NOR"
    assert o.find("Plan", lap=20)[0]["id"] == "p1"


def test_unknown_property_is_rejected():
    o = make_ontology()
    with pytest.raises(OntologyError):
        o.add("Driver", code="VER", colour="blue")


def test_propose_then_approve_writes_back_with_approver():
    o = make_ontology()
    record = o.propose("set_plan", {"driver": "NOR", "lap": 28}, proposed_by="ai:race-engineer")
    assert record.status == "proposed" and o.all("Plan") == []
    o.approve(record.id, "Nathan")
    plan = o.linked("driver_plans", "NOR")[0]
    assert plan["lap"] == 28 and plan["approved_by"] == "Nathan"
    assert o.audit_log()[0]["status"] == "approved"


def test_invalid_proposal_never_enters_the_queue():
    o = make_ontology()
    with pytest.raises(ActionValidationError) as error:
        o.propose("set_plan", {"driver": "XXX", "lap": 0}, proposed_by="Nathan")
    assert len(error.value.problems) == 2
    assert o.pending() == []


def test_ai_agents_cannot_approve_or_reject():
    o = make_ontology()
    record = o.propose("set_plan", {"driver": "NOR", "lap": 28}, proposed_by="ai:race-engineer")
    with pytest.raises(OntologyError):
        o.approve(record.id, "ai:race-engineer")
    with pytest.raises(OntologyError):
        o.reject(record.id, "ai:other-agent", "no")


def test_rejection_needs_a_reason():
    o = make_ontology()
    record = o.propose("set_plan", {"driver": "NOR", "lap": 28}, proposed_by="ai:race-engineer")
    with pytest.raises(OntologyError):
        o.reject(record.id, "Nathan", "  ")
    o.reject(record.id, "Nathan", "track position matters more here")
    assert record.status == "rejected"


def test_approval_rechecks_rules_in_case_the_world_changed():
    o = make_ontology()
    record = o.propose("set_plan", {"driver": "NOR", "lap": 28}, proposed_by="Nathan")
    del o._objects["Driver"]["NOR"]  # e.g. the car retired after the proposal
    o.approve(record.id, "Nathan")
    assert record.status == "failed" and o.all("Plan") == []


def test_a_decision_cannot_be_made_twice():
    o = make_ontology()
    record = o.propose("set_plan", {"driver": "NOR", "lap": 28}, proposed_by="Nathan")
    o.approve(record.id, "Nathan")
    with pytest.raises(OntologyError):
        o.approve(record.id, "Nathan")
