"""A small ontology layer: objects, links, and actions that need human approval.

Palantir's Ontology sits between raw data and the people who make decisions.
It turns rows in tables into the things a team actually talks about (a driver,
a stint, a pit stop) and the decisions it takes on them (actions).

This module is a deliberately small version of that idea:

* ObjectType  - a kind of thing, e.g. "Stint", with a primary key and properties.
* LinkType    - a named relationship, e.g. Driver --driver_stints--> Stint.
* ActionType  - a decision that changes the data, with validation rules.
* ActionRecord- one proposed decision and what happened to it (the audit trail).

The rule that matters most: anyone (a person or an AI agent) may PROPOSE an
action, but only a named human may APPROVE it. Approval re-checks validation,
because the world may have changed since the proposal was made.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

AI_PREFIX = "ai:"  # actors whose name starts with this are AI agents


class OntologyError(Exception):
    """Raised when something does not fit the ontology's definitions."""


class ActionValidationError(OntologyError):
    """Raised when a proposed action breaks one of its rules."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class ObjectType:
    name: str
    primary_key: str
    properties: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class LinkType:
    name: str
    source: str
    target: str
    description: str = ""


@dataclass(frozen=True)
class ActionType:
    name: str
    description: str
    parameters: tuple[str, ...]
    # validate returns a list of human-readable problems; empty list means OK.
    validate: Callable[["Ontology", dict], list[str]]
    # apply writes the decision back into the ontology.
    apply: Callable[["Ontology", dict], None]


@dataclass
class ActionRecord:
    id: str
    action: str
    params: dict
    proposed_by: str
    rationale: str
    status: str = "proposed"  # proposed | approved | rejected | failed
    created_at: str = field(default_factory=lambda: _now())
    decided_by: str | None = None
    decided_at: str | None = None
    decision_note: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ontology:
    def __init__(self) -> None:
        self.object_types: dict[str, ObjectType] = {}
        self.link_types: dict[str, LinkType] = {}
        self.action_types: dict[str, ActionType] = {}
        self._objects: dict[str, dict[str, dict]] = {}
        self._links: dict[str, set[tuple[str, str]]] = {}
        self.actions: list[ActionRecord] = []

    # ------------------------------------------------------------------ schema
    def define_object(self, object_type: ObjectType) -> None:
        if object_type.primary_key not in object_type.properties:
            raise OntologyError(
                f"{object_type.name}: primary key '{object_type.primary_key}' "
                "must be one of its properties"
            )
        self.object_types[object_type.name] = object_type
        self._objects.setdefault(object_type.name, {})

    def define_link(self, link_type: LinkType) -> None:
        for end in (link_type.source, link_type.target):
            if end not in self.object_types:
                raise OntologyError(f"{link_type.name}: unknown object type '{end}'")
        self.link_types[link_type.name] = link_type
        self._links.setdefault(link_type.name, set())

    def define_action(self, action_type: ActionType) -> None:
        self.action_types[action_type.name] = action_type

    # ----------------------------------------------------------------- objects
    def add(self, type_name: str, **props: Any) -> dict:
        object_type = self._object_type(type_name)
        unknown = set(props) - set(object_type.properties)
        if unknown:
            raise OntologyError(f"{type_name}: unknown properties {sorted(unknown)}")
        key = props.get(object_type.primary_key)
        if key is None:
            raise OntologyError(f"{type_name}: missing primary key '{object_type.primary_key}'")
        obj = {name: props.get(name) for name in object_type.properties}
        self._objects[type_name][str(key)] = obj
        return obj

    def get(self, type_name: str, key: Any) -> dict | None:
        self._object_type(type_name)
        return self._objects[type_name].get(str(key))

    def all(self, type_name: str) -> list[dict]:
        self._object_type(type_name)
        return list(self._objects[type_name].values())

    def find(self, type_name: str, **equals: Any) -> list[dict]:
        """All objects of a type whose properties equal the given values."""
        return [
            obj for obj in self.all(type_name)
            if all(obj.get(name) == value for name, value in equals.items())
        ]

    def update(self, type_name: str, key: Any, **changes: Any) -> dict:
        obj = self.get(type_name, key)
        if obj is None:
            raise OntologyError(f"{type_name} '{key}' does not exist")
        unknown = set(changes) - set(self.object_types[type_name].properties)
        if unknown:
            raise OntologyError(f"{type_name}: unknown properties {sorted(unknown)}")
        obj.update(changes)
        return obj

    # ------------------------------------------------------------------- links
    def link(self, link_name: str, source_key: Any, target_key: Any) -> None:
        link_type = self._link_type(link_name)
        if self.get(link_type.source, source_key) is None:
            raise OntologyError(f"{link_name}: no {link_type.source} '{source_key}'")
        if self.get(link_type.target, target_key) is None:
            raise OntologyError(f"{link_name}: no {link_type.target} '{target_key}'")
        self._links[link_name].add((str(source_key), str(target_key)))

    def linked(self, link_name: str, source_key: Any) -> list[dict]:
        """Follow a link forwards: e.g. a Driver's Stints."""
        link_type = self._link_type(link_name)
        keys = sorted(t for s, t in self._links[link_name] if s == str(source_key))
        return [self._objects[link_type.target][k] for k in keys]

    def linked_from(self, link_name: str, target_key: Any) -> list[dict]:
        """Follow a link backwards: e.g. which Driver owns this Stint."""
        link_type = self._link_type(link_name)
        keys = sorted(s for s, t in self._links[link_name] if t == str(target_key))
        return [self._objects[link_type.source][k] for k in keys]

    # ----------------------------------------------------------------- actions
    def propose(self, action_name: str, params: dict, proposed_by: str, rationale: str = "") -> ActionRecord:
        action_type = self._action_type(action_name)
        missing = [p for p in action_type.parameters if p not in params]
        problems = [f"missing parameter '{p}'" for p in missing]
        if not problems:
            problems = action_type.validate(self, params)
        if problems:
            raise ActionValidationError(problems)
        record = ActionRecord(
            id=uuid.uuid4().hex[:8],
            action=action_name,
            params=params,
            proposed_by=proposed_by,
            rationale=rationale,
        )
        self.actions.append(record)
        return record

    def approve(self, action_id: str, decided_by: str, note: str = "") -> ActionRecord:
        record = self._pending(action_id)
        _require_human(decided_by)
        action_type = self._action_type(record.action)
        # Re-check: the data may have changed since the proposal was made.
        problems = action_type.validate(self, record.params)
        record.decided_by, record.decided_at = decided_by, _now()
        if problems:
            record.status = "failed"
            record.decision_note = "no longer valid: " + "; ".join(problems)
            return record
        # The approver's name travels with the write-back, so the data itself
        # records who took the decision.
        action_type.apply(self, {**record.params, "approved_by": decided_by})
        record.status, record.decision_note = "approved", note
        return record

    def reject(self, action_id: str, decided_by: str, note: str) -> ActionRecord:
        record = self._pending(action_id)
        _require_human(decided_by)
        if not note.strip():
            raise OntologyError("a rejection needs a reason, so the proposer can learn from it")
        record.status, record.decided_by = "rejected", decided_by
        record.decided_at, record.decision_note = _now(), note
        return record

    def pending(self) -> list[ActionRecord]:
        return [r for r in self.actions if r.status == "proposed"]

    def audit_log(self) -> list[dict]:
        return [asdict(r) for r in self.actions]

    def save_audit_log(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.audit_log(), indent=2))

    # ----------------------------------------------------------------- helpers
    def _object_type(self, name: str) -> ObjectType:
        if name not in self.object_types:
            raise OntologyError(f"unknown object type '{name}'")
        return self.object_types[name]

    def _link_type(self, name: str) -> LinkType:
        if name not in self.link_types:
            raise OntologyError(f"unknown link type '{name}'")
        return self.link_types[name]

    def _action_type(self, name: str) -> ActionType:
        if name not in self.action_types:
            raise OntologyError(f"unknown action type '{name}'")
        return self.action_types[name]

    def _pending(self, action_id: str) -> ActionRecord:
        for record in self.actions:
            if record.id == action_id:
                if record.status != "proposed":
                    raise OntologyError(f"action {action_id} is already {record.status}")
                return record
        raise OntologyError(f"no action with id {action_id}")


def _require_human(actor: str) -> None:
    if not actor or not actor.strip():
        raise OntologyError("decisions must be signed by a named person")
    if actor.startswith(AI_PREFIX):
        raise OntologyError("AI agents can propose actions but never approve or reject them")
