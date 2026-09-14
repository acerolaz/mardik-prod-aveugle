"""Integration regression tests for two recurring production incidents.

Each scenario replays a recorded session end to end (runner -> store -> LLM ->
tools -> telemetry) and pins the behaviour restored by the fix.

* Incident 1 - partial answers: a turn calling several tools only returned the
  output of the last one (the order status was dropped on size exchanges).
* Incident 2 - corrupted session memory: an LLM timeout left an orphan user
  message in the store, so the client retry duplicated it and counted two turns.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import StatusCode
from structlog.testing import capture_logs

from mardik.agent import Agent, Reply
from mardik.errors import LLMTimeoutError
from mardik.runner import load_session, replay
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS, knowledge_base, lookup_order

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
EXCHANGE_TOPIC = "échange de taille"
PREAMBLE = "Je vérifie votre commande."


class MultiToolLLM:
    """Order id in history -> order lookup, then the exchange procedure.

    With ``preamble`` set, the model also writes some text alongside its tool calls.
    """

    def __init__(self, preamble: str = "") -> None:
        self.preamble = preamble

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        text = " ".join(str(m.get("content", "")) for m in messages)
        order_id = re.search(r"#(\d+)", text).group(1)  # type: ignore[union-attr]
        return Reply(
            content=self.preamble,
            tool_calls=[
                {"name": "lookup_order", "args": {"order_id": order_id}},
                {"name": "knowledge_base", "args": {"topic": EXCHANGE_TOPIC}},
            ],
        )


class TimeoutThenAnswerLLM:
    """Times out on the first call, then delegates (simulates a client retry)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[list[dict[str, Any]]] = []

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise TimeoutError("upstream deadline exceeded")
        return self._inner.invoke(messages)


class HangingLLM:
    """Never answers in time: blocks until released (or 2 s, so a regression fails, not hangs)."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self.release.wait(timeout=2)
        return Reply(content="trop tard", tool_calls=[])


@pytest.fixture(autouse=True)
def _sessions_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolve recordings independently of the directory pytest is launched from.
    monkeypatch.setenv("MARDIK_SESSIONS_DIR", str(SESSIONS_DIR))


def _agent(llm: Any, telemetry: Any) -> Agent:
    return Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)


def _last_user_turn(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    *previous, last = data["messages"]
    return previous, last


# --- Incident 1: every tool output reaches the customer ---------------------


def test_multi_tool_replay_returns_every_tool_output_in_call_order(telemetry):
    data = load_session("size_exchange")
    order_id = data["expected"]["order_id"]

    result = replay(data, _agent(MultiToolLLM(), telemetry), SessionStore())

    assert result.reply == "\n".join([lookup_order(order_id), knowledge_base(EXCHANGE_TOPIC)])


def test_multi_tool_replay_keeps_llm_text_before_tool_outputs(telemetry):
    data = load_session("size_exchange")
    order_id = data["expected"]["order_id"]

    result = replay(data, _agent(MultiToolLLM(preamble=PREAMBLE), telemetry), SessionStore())

    assert result.reply.split("\n") == [
        PREAMBLE,
        lookup_order(order_id),
        knowledge_base(EXCHANGE_TOPIC),
    ]


def test_multi_tool_reply_is_stored_whole_and_fed_to_the_next_turn(telemetry):
    data = load_session("size_exchange")
    store = SessionStore()
    agent = _agent(MultiToolLLM(), telemetry)

    first = replay(data, agent, store)
    agent.run_turn(store, data["session_id"], "Merci, et pour le L ?")

    assistant_replies = [
        m["content"] for m in store.history(data["session_id"]) if m["role"] == "assistant"
    ]
    assert first.reply in assistant_replies
    assert data["expected"]["order_status"] in assistant_replies[-2]
    assert store.turns(data["session_id"]) == 2


@pytest.mark.parametrize("name", ["replay_delivery", "return_refund"])
def test_single_tool_replays_are_unchanged(name, fake_llm, telemetry):
    data = load_session(name)
    order_id = re.search(r"#(\d+)", data["messages"][0]["content"]).group(1)  # type: ignore[union-attr]

    result = replay(data, _agent(fake_llm, telemetry), SessionStore())

    assert result.reply == lookup_order(order_id)


# --- Incident 2: a failed turn leaves no trace in the session ---------------


def test_timeout_incident_retry_yields_clean_history(fake_llm, telemetry):
    data = load_session("incident_timeout")
    previous, last = _last_user_turn(data)
    session_id = data["session_id"]
    store = SessionStore()
    llm = TimeoutThenAnswerLLM(fake_llm)
    agent = _agent(llm, telemetry)

    with pytest.raises(LLMTimeoutError):
        replay(data, agent, store)
    retry = agent.run_turn(store, session_id, last["content"])

    assert store.history(session_id) == [
        *previous,
        last,
        {"role": "assistant", "content": retry.reply},
    ]
    assert store.turns(session_id) == 1


def test_timeout_incident_retry_sends_the_question_to_the_llm_once(fake_llm, telemetry):
    data = load_session("incident_timeout")
    previous, last = _last_user_turn(data)
    llm = TimeoutThenAnswerLLM(fake_llm)
    agent = _agent(llm, telemetry)
    store = SessionStore()

    with pytest.raises(LLMTimeoutError):
        replay(data, agent, store)
    agent.run_turn(store, data["session_id"], last["content"])

    assert llm.calls[0] == llm.calls[1] == [*previous, last]


def test_timeout_incident_retry_answers_from_recorded_context(fake_llm, telemetry):
    data = load_session("incident_timeout")
    _, last = _last_user_turn(data)
    agent = _agent(TimeoutThenAnswerLLM(fake_llm), telemetry)
    store = SessionStore()

    with pytest.raises(LLMTimeoutError):
        replay(data, agent, store)
    retry = agent.run_turn(store, data["session_id"], last["content"])

    assert retry.reply == lookup_order("2098")


def test_repeated_timeouts_never_grow_the_session(timeout_llm, telemetry):
    data = load_session("incident_timeout")
    previous, last = _last_user_turn(data)
    store = SessionStore()
    agent = _agent(timeout_llm, telemetry)

    with pytest.raises(LLMTimeoutError):
        replay(data, agent, store)
    for _ in range(3):
        with pytest.raises(LLMTimeoutError):
            agent.run_turn(store, data["session_id"], last["content"])

    assert store.history(data["session_id"]) == previous
    assert store.turns(data["session_id"]) == 0


def test_timeout_incident_does_not_leak_into_other_sessions(fake_llm, telemetry):
    incident = load_session("incident_timeout")
    delivery = load_session("replay_delivery")
    store = SessionStore()

    with pytest.raises(LLMTimeoutError):
        replay(incident, _agent(TimeoutThenAnswerLLM(fake_llm), telemetry), store)
    result = replay(delivery, _agent(fake_llm, telemetry), store)

    assert result.reply == lookup_order("1042")
    assert store.history(delivery["session_id"])[:-1] == delivery["messages"]
    assert store.turns(incident["session_id"]) == 0


def test_hanging_llm_is_cut_off_by_the_deadline(telemetry, span_exporter):
    data = load_session("incident_timeout")
    previous, _ = _last_user_turn(data)
    llm = HangingLLM()
    agent = Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry, llm_timeout_s=0.05)
    store = SessionStore()

    start = time.perf_counter()
    try:
        with pytest.raises(LLMTimeoutError):
            replay(data, agent, store)
        elapsed = time.perf_counter() - start
    finally:
        llm.release.set()

    assert elapsed < 1.0
    assert store.history(data["session_id"]) == previous
    assert store.turns(data["session_id"]) == 0
    [turn] = [s for s in span_exporter.get_finished_spans() if s.name == "agent.turn"]
    assert turn.status.status_code == StatusCode.ERROR


def test_timeout_incident_is_observable_as_a_failed_turn(fake_llm, telemetry, span_exporter):
    data = load_session("incident_timeout")
    _, last = _last_user_turn(data)
    agent = _agent(TimeoutThenAnswerLLM(fake_llm), telemetry)
    store = SessionStore()

    with capture_logs() as logs:
        with pytest.raises(LLMTimeoutError):
            replay(data, agent, store)
        agent.run_turn(store, data["session_id"], last["content"])

    turns = [s for s in span_exporter.get_finished_spans() if s.name == "agent.turn"]
    assert [s.status.status_code for s in turns] == [StatusCode.ERROR, StatusCode.UNSET]
    completed = [e for e in logs if e.get("event") == "turn.completed"]
    assert len(completed) == 1
