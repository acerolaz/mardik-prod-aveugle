"""Integration tests replaying the recorded ``size_exchange`` session end to end.

A size exchange needs two tools in a single turn: the order status (can the item
still be swapped?) and the help-center exchange procedure. The order id only
appears in the first message, the exchange request only in the last one.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Callable

import pytest
from structlog.testing import capture_logs

from mardik.agent import Agent, Reply
from mardik.runner import load_session, replay
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS, knowledge_base, lookup_order

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
SESSION_NAME = "size_exchange"
EXCHANGE_TOPIC = "échange de taille"


class SizeExchangeFakeLLM:
    """Scripted LLM for size-exchange conversations.

    Order id anywhere in the conversation + exchange intent in the last user
    message -> look the order up, then fetch the exchange procedure.
    """

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        text = " ".join(str(m.get("content", "")) for m in messages)
        last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        match = re.search(r"#(\d+)", text)
        if match is None:
            return Reply(content="Pouvez-vous indiquer votre numéro de commande ?", tool_calls=[])
        calls: list[dict[str, Any]] = [
            {"name": "lookup_order", "args": {"order_id": match.group(1)}}
        ]
        if re.search(r"taille|échang", last_user, re.IGNORECASE):
            calls.append({"name": "knowledge_base", "args": {"topic": EXCHANGE_TOPIC}})
        return Reply(content="", tool_calls=calls)


class RecordingTools(dict[str, Callable[..., str]]):
    """Tool registry that records every call the agent dispatches."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        super().__init__({name: self._wrap(name, fn) for name, fn in DEFAULT_TOOLS.items()})

    def _wrap(self, name: str, fn: Callable[..., str]) -> Callable[..., str]:
        def tool(**kwargs: Any) -> str:
            self.calls.append((name, kwargs))
            return fn(**kwargs)

        return tool


@pytest.fixture(autouse=True)
def _sessions_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolve recordings independently of the directory pytest is launched from.
    monkeypatch.setenv("MARDIK_SESSIONS_DIR", str(SESSIONS_DIR))


@pytest.fixture
def session() -> dict[str, Any]:
    return load_session(SESSION_NAME)


@pytest.fixture
def exchange_llm() -> SizeExchangeFakeLLM:
    return SizeExchangeFakeLLM()


def _agent(llm, telemetry, tools=None) -> Agent:
    return Agent(llm=llm, tools=tools if tools is not None else DEFAULT_TOOLS, telemetry=telemetry)


def _with_order_id(session: dict[str, Any], order_id: str) -> dict[str, Any]:
    data = copy.deepcopy(session)
    original = f"#{session['expected']['order_id']}"
    for message in data["messages"]:
        message["content"] = message["content"].replace(original, f"#{order_id}")
    return data


# --- Functional behaviour ---------------------------------------------------


def test_tools_are_called_in_order_with_arguments_from_history(session, exchange_llm, telemetry):
    tools = RecordingTools()
    replay(session, _agent(exchange_llm, telemetry, tools), SessionStore())

    assert tools.calls == [
        ("lookup_order", {"order_id": session["expected"]["order_id"]}),
        ("knowledge_base", {"topic": session["expected"]["kb_topic"]}),
    ]


def test_reply_points_to_exchange_procedure(session, exchange_llm, telemetry):
    result = replay(session, _agent(exchange_llm, telemetry), SessionStore())

    assert result.session_id == session["session_id"]
    assert result.reply == knowledge_base(session["expected"]["kb_topic"])


@pytest.mark.xfail(
    strict=True,
    reason="Agent.run_turn overwrites the reply with each tool output: only the last one is kept",
)
def test_reply_combines_order_status_and_exchange_procedure(session, exchange_llm, telemetry):
    result = replay(session, _agent(exchange_llm, telemetry), SessionStore())

    assert session["expected"]["order_status"] in result.reply
    assert session["expected"]["kb_topic"] in result.reply


def test_order_status_is_found_from_first_turn(session, fake_llm, telemetry):
    result = replay(session, _agent(fake_llm, telemetry), SessionStore())

    assert result.reply == lookup_order(session["expected"]["order_id"])
    assert session["expected"]["order_status"] in result.reply


@pytest.mark.parametrize(
    ("order_id", "status"),
    [
        ("2098", "en préparation"),
        ("1042", "expédiée, livraison prévue demain"),
        ("3157", "retournée, remboursement en cours"),
        ("9999", "introuvable"),
    ],
)
def test_exchange_request_reports_actual_order_status(
    session, fake_llm, telemetry, order_id, status
):
    result = replay(_with_order_id(session, order_id), _agent(fake_llm, telemetry), SessionStore())

    assert result.reply == f"Commande #{order_id} : {status}."


def test_exchange_request_without_order_id_asks_for_it_then_resolves(
    exchange_llm, telemetry, span_exporter
):
    store = SessionStore()
    agent = _agent(exchange_llm, telemetry)

    first = agent.run_turn(
        store, "size-exchange-no-id", "Je voudrais changer la taille de mon pull."
    )
    assert "numéro de commande" in first.reply
    assert not [s for s in span_exporter.get_finished_spans() if s.name == "tool.call"]

    second = agent.run_turn(
        store, "size-exchange-no-id", "C'est la #2098, je veux échanger la taille."
    )
    assert second.reply == knowledge_base(EXCHANGE_TOPIC)
    assert store.turns("size-exchange-no-id") == 2
    assert len(store.history("size-exchange-no-id")) == 4


def test_store_contains_history_plus_new_exchange(session, exchange_llm, telemetry):
    store = SessionStore()
    result = replay(session, _agent(exchange_llm, telemetry), store)

    history = store.history(session["session_id"])
    assert history[:-1] == session["messages"]
    assert history[-1] == {"role": "assistant", "content": result.reply}
    assert store.turns(session["session_id"]) == 1


def test_size_exchange_does_not_leak_into_refund_session(session, exchange_llm, telemetry):
    store = SessionStore()
    tools = RecordingTools()
    agent = _agent(exchange_llm, telemetry, tools)

    replay(session, agent, store)
    replay(load_session("return_refund"), agent, store)

    assert tools.calls[-1] == ("lookup_order", {"order_id": "3157"})
    refund_history = " ".join(m["content"] for m in store.history("return-refund-001"))
    assert "#2098" not in refund_history


# --- Observability ----------------------------------------------------------


def test_both_tool_calls_are_traced_under_the_turn(session, exchange_llm, telemetry, span_exporter):
    replay(session, _agent(exchange_llm, telemetry), SessionStore())

    spans = span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "agent.turn")
    tool_spans = [s for s in spans if s.name == "tool.call"]

    assert sorted(s.name for s in spans) == ["agent.turn", "llm.invoke", "tool.call", "tool.call"]
    assert [s.attributes["tool.name"] for s in tool_spans] == session["expected"]["tools"]
    for span in spans:
        assert span.context.trace_id == root.context.trace_id
        if span is not root:
            assert span.parent.span_id == root.context.span_id


def test_latency_and_log_are_tagged_with_session_id(
    session, exchange_llm, telemetry, metric_reader
):
    with capture_logs() as logs:
        replay(session, _agent(exchange_llm, telemetry), SessionStore())

    points = [
        point
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "latency_ms"
        for point in metric.data.data_points
    ]
    assert len(points) == 1
    assert points[0].attributes == {"session_id": session["session_id"]}

    completed = [entry for entry in logs if entry.get("event") == "turn.completed"]
    assert [entry["session_id"] for entry in completed] == [session["session_id"]]
