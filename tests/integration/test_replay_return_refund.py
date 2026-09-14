"""Integration tests replaying the recorded ``return_refund`` session end to end.

The session mentions the order id only in its first message: every test checks
that the whole recorded history flows through the store, the LLM, the tool
layer and the telemetry pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from mardik.agent import Agent, Reply
from mardik.runner import load_session, replay
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
SESSION_NAME = "return_refund"


class SpyLLM:
    """Wraps an LLM and records the messages it receives."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[list[dict[str, Any]]] = []

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self.calls.append(messages)
        return self._inner.invoke(messages)


@pytest.fixture(autouse=True)
def _sessions_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolve recordings independently of the directory pytest is launched from.
    monkeypatch.setenv("MARDIK_SESSIONS_DIR", str(SESSIONS_DIR))


@pytest.fixture
def session() -> dict[str, Any]:
    return load_session(SESSION_NAME)


@pytest.fixture
def spy_llm(fake_llm) -> SpyLLM:
    return SpyLLM(fake_llm)


def _agent(llm, telemetry) -> Agent:
    return Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)


# --- Recording contract -----------------------------------------------------


@pytest.mark.parametrize("path", sorted(SESSIONS_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_recorded_session_is_well_formed(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["session_id"]
    roles = [m["role"] for m in data["messages"]]
    assert roles, "a session needs at least one message"
    assert roles[-1] == "user", "replay plays the last message as the new user turn"
    assert all(a != b for a, b in zip(roles, roles[1:])), "roles must alternate"


def test_recorded_session_ids_are_unique():
    ids = [
        json.loads(p.read_text(encoding="utf-8"))["session_id"] for p in SESSIONS_DIR.glob("*.json")
    ]
    assert len(ids) == len(set(ids))


# --- Functional behaviour ---------------------------------------------------


def test_reply_uses_order_id_from_first_turn(session, fake_llm, telemetry):
    result = replay(session, _agent(fake_llm, telemetry), SessionStore())

    assert result.session_id == session["session_id"]
    assert session["expected"]["reply_contains"] in result.reply
    assert f"#{session['expected']['order_id']}" in result.reply


def test_llm_receives_full_recorded_history_in_order(session, spy_llm, telemetry):
    replay(session, _agent(spy_llm, telemetry), SessionStore())

    assert len(spy_llm.calls) == 1
    assert spy_llm.calls[0] == session["messages"]


def test_store_contains_history_plus_new_exchange(session, fake_llm, telemetry):
    store = SessionStore()
    result = replay(session, _agent(fake_llm, telemetry), store)

    history = store.history(session["session_id"])
    assert history[: len(session["messages"])] == session["messages"]
    assert history[-1] == {"role": "assistant", "content": result.reply}
    assert len(history) == len(session["messages"]) + 1
    assert store.turns(session["session_id"]) == 1


def test_replay_does_not_leak_context_between_sessions(session, fake_llm, telemetry):
    store = SessionStore()
    agent = _agent(fake_llm, telemetry)

    delivery = replay(load_session("replay_delivery"), agent, store)
    refund = replay(session, agent, store)

    assert "#1042" in delivery.reply
    assert "#3157" in refund.reply
    assert "#1042" not in " ".join(m["content"] for m in store.history(session["session_id"]))


def test_follow_up_turn_keeps_session_context(session, fake_llm, telemetry):
    store = SessionStore()
    agent = _agent(fake_llm, telemetry)
    replay(session, agent, store)

    follow_up = agent.run_turn(store, session["session_id"], "Merci, et sous combien de jours ?")

    assert session["expected"]["reply_contains"] in follow_up.reply
    assert store.turns(session["session_id"]) == 2


# --- Observability ----------------------------------------------------------


def test_replay_produces_a_single_well_nested_trace(session, fake_llm, telemetry, span_exporter):
    replay(session, _agent(fake_llm, telemetry), SessionStore())

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert set(spans) == {"agent.turn", "llm.invoke", "tool.call"}

    root = spans["agent.turn"]
    assert root.parent is None
    for child in ("llm.invoke", "tool.call"):
        assert spans[child].context.trace_id == root.context.trace_id
        assert spans[child].parent.span_id == root.context.span_id


def test_tool_span_names_the_expected_tool(session, fake_llm, telemetry, span_exporter):
    replay(session, _agent(fake_llm, telemetry), SessionStore())

    tool_spans = [s for s in span_exporter.get_finished_spans() if s.name == "tool.call"]
    assert [s.attributes["tool.name"] for s in tool_spans] == [session["expected"]["tool"]]


def test_latency_metric_is_tagged_with_outcome_not_session_id(session, fake_llm, telemetry, metric_reader):
    replay(session, _agent(fake_llm, telemetry), SessionStore())

    points = [
        point
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "latency_ms"
        for point in metric.data.data_points
    ]
    assert len(points) == 1
    # session_id stays on spans and logs: as a metric label it would create one series per session.
    assert points[0].attributes == {"outcome": "ok"}
    assert points[0].count == 1
    assert points[0].sum > 0


def test_turn_completed_log_carries_session_id(session, fake_llm, telemetry):
    with capture_logs() as logs:
        replay(session, _agent(fake_llm, telemetry), SessionStore())

    completed = [entry for entry in logs if entry.get("event") == "turn.completed"]
    assert len(completed) == 1
    assert completed[0]["session_id"] == session["session_id"]
    assert completed[0]["latency_ms"] >= 0
