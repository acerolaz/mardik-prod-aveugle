"""Integration regression test: concurrent turns landing on the *same* session.

A customer double-submitting, or chatting from two tabs, sends several turns for
one session at once. They all go through the single shared ``SessionStore``,
whose turn counter is a read-then-write: without the lock, concurrent commits
lose turns. The load scenarios in ``test_observability`` give every turn its own
session id, so they cannot see this; this test replays one recorded session and
fires every turn at that same id.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from mardik.agent import Agent, Reply
from mardik.runner import load_session
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
CONCURRENT_TURNS = 24


class RendezvousLLM:
    """Holds every call until ``parties`` calls are in flight, so commits overlap."""

    def __init__(self, inner: Any, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self._barrier.wait()
        return self._inner.invoke(messages)


@pytest.fixture(autouse=True)
def _sessions_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARDIK_SESSIONS_DIR", str(SESSIONS_DIR))


def _latency_ok_count(metric_reader: InMemoryMetricReader) -> int:
    return sum(
        point.count
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "latency_ms"
        for point in metric.data.data_points
        if point.attributes["outcome"] == "ok"
    )


def test_concurrent_turns_on_one_session_are_all_counted_and_stored(
    fake_llm, telemetry, span_exporter, metric_reader
):
    data = load_session("return_refund")
    session_id = data["session_id"]
    *previous, last = data["messages"]
    store = SessionStore()
    store.load_history(session_id, previous)
    agent = Agent(
        llm=RendezvousLLM(fake_llm, threading.Barrier(CONCURRENT_TURNS, timeout=10)),
        tools=DEFAULT_TOOLS,
        telemetry=telemetry,
    )

    with ThreadPoolExecutor(max_workers=CONCURRENT_TURNS) as pool:
        results = list(
            pool.map(
                lambda _: agent.run_turn(store, session_id, last["content"]),
                range(CONCURRENT_TURNS),
            )
        )

    # Every turn is counted: none lost to the read-then-write race.
    assert store.turns(session_id) == CONCURRENT_TURNS

    # Every turn is stored as an adjacent (user, assistant) pair after the recording.
    history = store.history(session_id)
    assert history[: len(previous)] == previous
    appended = history[len(previous) :]
    assert len(appended) == 2 * CONCURRENT_TURNS
    pairs = list(zip(appended[::2], appended[1::2]))
    assert all(user == last and assistant["role"] == "assistant" for user, assistant in pairs)
    assert sorted(a["content"] for _, a in pairs) == sorted(r.reply for r in results)

    # The signals agree with the store.
    turns = [s for s in span_exporter.get_finished_spans() if s.name == "agent.turn"]
    assert len(turns) == CONCURRENT_TURNS
    assert {s.attributes["session.id"] for s in turns} == {session_id}
    assert _latency_ok_count(metric_reader) == CONCURRENT_TURNS
