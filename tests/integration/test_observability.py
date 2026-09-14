"""End-to-end observability of replayed sessions.

Each scenario replays recorded sessions through the full stack (runner -> store ->
LLM -> tools -> telemetry) with the real ``Telemetry`` bundle wired to in-memory
exporters, then checks the three signals together, as an operator would read them:

* traces: one well-formed trace per turn, ERROR status on failures;
* metrics: ``latency_ms{outcome}``, ``errors_total{error.type}``,
  ``tool_calls_total{tool.name, outcome}`` aggregate correctly;
* logs: the JSON lines actually written to stdout join their trace via ``trace_id``.

The load scenarios cover the README's known gap: observability under concurrent turns.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mardik.agent import Agent, Reply
from mardik.errors import LLMTimeoutError
from mardik.runner import load_session, replay
from mardik.session import SessionStore
from mardik.telemetry import Telemetry, build_resource, build_telemetry
from mardik.tools import DEFAULT_TOOLS, lookup_order

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
ANSWERED_SESSIONS = ["replay_delivery", "return_refund", "size_exchange"]
CONCURRENT_TURNS = 24


class UnknownToolLLM:
    """The model asks for a tool that is not registered (hallucinated tool name)."""

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        return Reply(content="", tool_calls=[{"name": "cancel_order", "args": {"order_id": "1"}}])


class UsageReportingLLM:
    """Delegates, and reports token usage the way a real chat client would."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        reply = self._inner.invoke(messages)
        reply.usage = {"input_tokens": 12, "output_tokens": 3}
        return reply


class RendezvousLLM:
    """Holds every call until ``parties`` calls are in flight, then delegates.

    Guarantees the turns really overlap inside their ``llm.invoke`` spans, so any
    context leaking between threads would show up in the traces.
    """

    def __init__(self, inner: Any, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self._barrier.wait()
        return self._inner.invoke(messages)


@pytest.fixture(autouse=True)
def _sessions_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolve recordings independently of the directory pytest is launched from.
    monkeypatch.setenv("MARDIK_SESSIONS_DIR", str(SESSIONS_DIR))


def _agent(llm: Any, telemetry: Any) -> Agent:
    return Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)


def _json_logs(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """Parse the JSON lines the production log pipeline wrote to stdout."""
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def _points(metric_reader: InMemoryMetricReader, name: str) -> list[Any]:
    return [
        point
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _counter(metric_reader: InMemoryMetricReader, name: str) -> dict[tuple[Any, ...], int]:
    return {tuple(sorted(p.attributes.items())): p.value for p in _points(metric_reader, name)}


def _latency_counts(metric_reader: InMemoryMetricReader) -> dict[str, int]:
    return {p.attributes["outcome"]: p.count for p in _points(metric_reader, "latency_ms")}


def _hex_ids(span: ReadableSpan) -> tuple[str, str]:
    return format(span.context.trace_id, "032x"), format(span.context.span_id, "016x")


def _traces(spans: list[ReadableSpan]) -> dict[int, list[ReadableSpan]]:
    by_trace: dict[int, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        by_trace[span.context.trace_id].append(span)
    return by_trace


def _session(name: str, suffix: str) -> dict[str, Any]:
    """A recorded session under a unique id, so concurrent replays share one store safely."""
    data = load_session(name)
    return {**data, "session_id": f"{data['session_id']}-{suffix}"}


# --- A single replayed turn, read through all three signals ------------------


def test_json_log_line_joins_the_turn_trace(fake_llm, telemetry, span_exporter, capsys):
    data = load_session("replay_delivery")
    replay(data, _agent(fake_llm, telemetry), SessionStore())

    turn = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.turn")
    [completed] = [e for e in _json_logs(capsys) if e["event"] == "turn.completed"]
    assert completed["session_id"] == data["session_id"] == turn.attributes["session.id"]
    assert (completed["trace_id"], completed["span_id"]) == _hex_ids(turn)
    assert completed["level"] == "info"
    assert completed["latency_ms"] >= 0
    assert "timestamp" in completed


def test_successful_turn_is_consistent_across_signals(
    fake_llm, telemetry, span_exporter, metric_reader
):
    replay(load_session("replay_delivery"), _agent(fake_llm, telemetry), SessionStore())

    spans = span_exporter.get_finished_spans()
    assert {s.status.status_code for s in spans} == {StatusCode.UNSET}
    tool = next(s for s in spans if s.name == "tool.call")
    assert dict(tool.attributes) == {
        "tool.name": "lookup_order",
        "tool.input": '{"order_id": "1042"}',
        "tool.output": lookup_order("1042"),
        "tool.outcome": "ok",
    }

    assert _latency_counts(metric_reader) == {"ok": 1}
    assert _counter(metric_reader, "tool_calls_total") == {
        (("outcome", "ok"), ("tool.name", "lookup_order")): 1
    }
    assert _points(metric_reader, "errors_total") == []


def test_llm_span_carries_token_usage(fake_llm, telemetry, span_exporter, metric_reader):
    llm = UsageReportingLLM(fake_llm)
    replay(load_session("replay_delivery"), _agent(llm, telemetry), SessionStore())

    [llm_span] = [s for s in span_exporter.get_finished_spans() if s.name == "llm.invoke"]
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 3


def test_large_tool_output_is_truncated_on_the_span(telemetry, span_exporter):
    class BigToolLLM:
        def invoke(self, messages: list[dict[str, Any]]) -> Reply:
            return Reply(content="", tool_calls=[{"name": "dump", "args": {}}])

    agent = Agent(llm=BigToolLLM(), tools={"dump": lambda: "x" * 2000}, telemetry=telemetry)
    result = replay(load_session("replay_delivery"), agent, SessionStore())

    [tool] = [s for s in span_exporter.get_finished_spans() if s.name == "tool.call"]
    assert len(tool.attributes["tool.output"]) == 512
    # Truncation is for the trace only: the customer still gets the whole answer.
    assert len(result.reply) == 2000


def test_tool_call_without_args_runs_with_the_args_it_traces(telemetry, span_exporter):
    class NoArgsLLM:
        def invoke(self, messages: list[dict[str, Any]]) -> Reply:
            # Some models omit "args" entirely for a tool that takes no parameters.
            return Reply(content="", tool_calls=[{"name": "ping"}])

    agent = Agent(llm=NoArgsLLM(), tools={"ping": lambda: "pong"}, telemetry=telemetry)
    result = replay(load_session("replay_delivery"), agent, SessionStore())

    assert result.reply == "pong"
    [tool] = [s for s in span_exporter.get_finished_spans() if s.name == "tool.call"]
    assert tool.attributes["tool.input"] == "{}"
    assert tool.attributes["tool.outcome"] == "ok"


def test_spans_and_metrics_share_the_service_identity(fake_llm, span_exporter, metric_reader):
    telemetry = build_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        resource=build_resource("mardik-it"),
    )
    replay(load_session("replay_delivery"), _agent(fake_llm, telemetry), SessionStore())

    span_services = {
        s.resource.attributes["service.name"] for s in span_exporter.get_finished_spans()
    }
    metric_services = {
        rm.resource.attributes["service.name"]
        for rm in metric_reader.get_metrics_data().resource_metrics
    }
    assert span_services == metric_services == {"mardik-it"}


# --- Failed turns ------------------------------------------------------------


def test_llm_timeout_incident_is_observable_on_every_signal(
    timeout_llm, telemetry, span_exporter, metric_reader, capsys
):
    data = load_session("incident_timeout")
    with pytest.raises(LLMTimeoutError):
        replay(data, _agent(timeout_llm, telemetry), SessionStore())

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert set(spans) == {"agent.turn", "llm.invoke"}
    assert spans["agent.turn"].status.status_code == StatusCode.ERROR
    assert spans["llm.invoke"].status.status_code == StatusCode.ERROR
    assert spans["llm.invoke"].parent.span_id == spans["agent.turn"].context.span_id
    [exception] = [e for e in spans["llm.invoke"].events if e.name == "exception"]
    assert exception.attributes["exception.type"] == "TimeoutError"

    assert _counter(metric_reader, "errors_total") == {(("error.type", "LLMTimeoutError"),): 1}
    assert _latency_counts(metric_reader) == {"error": 1}
    assert _points(metric_reader, "tool_calls_total") == []

    [failed] = [e for e in _json_logs(capsys) if e["event"] == "turn.failed"]
    assert failed["level"] == "error"
    assert failed["session_id"] == data["session_id"]
    assert failed["error_type"] == "LLMTimeoutError"
    assert (failed["trace_id"], failed["span_id"]) == _hex_ids(spans["agent.turn"])


def test_failing_tool_call_is_observable_on_every_signal(
    telemetry, span_exporter, metric_reader, capsys
):
    data = load_session("replay_delivery")
    store = SessionStore()
    with pytest.raises(KeyError):
        replay(data, _agent(UnknownToolLLM(), telemetry), store)

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert set(spans) == {"agent.turn", "llm.invoke", "tool.call"}
    tool = spans["tool.call"]
    assert dict(tool.attributes) == {
        "tool.name": "cancel_order",
        "tool.input": '{"order_id": "1"}',
        "tool.outcome": "error",
    }
    assert tool.status.status_code == StatusCode.ERROR
    assert spans["llm.invoke"].status.status_code == StatusCode.UNSET
    assert spans["agent.turn"].status.status_code == StatusCode.ERROR

    assert _counter(metric_reader, "tool_calls_total") == {
        (("outcome", "error"), ("tool.name", "cancel_order")): 1
    }
    assert _counter(metric_reader, "errors_total") == {(("error.type", "KeyError"),): 1}
    assert _latency_counts(metric_reader) == {"error": 1}

    [failed] = [e for e in _json_logs(capsys) if e["event"] == "turn.failed"]
    assert failed["trace_id"] == _hex_ids(spans["agent.turn"])[0]
    # The failed turn is observable, but it leaves the session untouched.
    assert store.history(data["session_id"]) == data["messages"][:-1]


# --- Under load --------------------------------------------------------------


def _replay_concurrently(
    jobs: list[tuple[dict[str, Any], Any]], telemetry: Telemetry
) -> list[BaseException | None]:
    """Replay every ``(session, llm)`` job at the same time on one shared store."""
    store = SessionStore()

    def run(job: tuple[dict[str, Any], Any]) -> BaseException | None:
        data, llm = job
        try:
            replay(data, _agent(llm, telemetry), store)
        except (LLMTimeoutError, KeyError) as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(run, jobs))


def test_concurrent_turns_each_get_an_isolated_well_formed_trace(
    fake_llm, telemetry, span_exporter
):
    barrier = threading.Barrier(CONCURRENT_TURNS, timeout=10)
    llm = RendezvousLLM(fake_llm, barrier)
    jobs = [
        (_session(ANSWERED_SESSIONS[i % len(ANSWERED_SESSIONS)], str(i)), llm)
        for i in range(CONCURRENT_TURNS)
    ]
    assert _replay_concurrently(jobs, telemetry) == [None] * CONCURRENT_TURNS

    traces = _traces(span_exporter.get_finished_spans())
    assert len(traces) == CONCURRENT_TURNS
    seen_sessions = set()
    for spans in traces.values():
        assert sorted(s.name for s in spans) == ["agent.turn", "llm.invoke", "tool.call"]
        root = next(s for s in spans if s.name == "agent.turn")
        assert root.parent is None
        for child in spans:
            if child is not root:
                assert child.parent.span_id == root.context.span_id
        seen_sessions.add(root.attributes["session.id"])
    assert seen_sessions == {data["session_id"] for data, _ in jobs}


def test_concurrent_turn_logs_point_to_their_own_trace(fake_llm, telemetry, span_exporter, capsys):
    barrier = threading.Barrier(CONCURRENT_TURNS, timeout=10)
    llm = RendezvousLLM(fake_llm, barrier)
    jobs = [(_session("replay_delivery", str(i)), llm) for i in range(CONCURRENT_TURNS)]
    _replay_concurrently(jobs, telemetry)

    turn_ids = {
        s.attributes["session.id"]: _hex_ids(s)
        for s in span_exporter.get_finished_spans()
        if s.name == "agent.turn"
    }
    completed = [e for e in _json_logs(capsys) if e["event"] == "turn.completed"]
    assert len(completed) == CONCURRENT_TURNS
    assert {e["session_id"]: (e["trace_id"], e["span_id"]) for e in completed} == turn_ids


def test_metrics_aggregate_a_mixed_load_without_session_labels(
    fake_llm, timeout_llm, telemetry, span_exporter, metric_reader
):
    barrier = threading.Barrier(CONCURRENT_TURNS, timeout=10)
    # One turn in three times out, one in three calls an unknown tool.
    llms = [
        RendezvousLLM(fake_llm, barrier),
        RendezvousLLM(timeout_llm, barrier),
        RendezvousLLM(UnknownToolLLM(), barrier),
    ]
    jobs = [(_session("replay_delivery", str(i)), llms[i % 3]) for i in range(CONCURRENT_TURNS)]
    outcomes = _replay_concurrently(jobs, telemetry)

    per_kind = CONCURRENT_TURNS // 3
    assert sum(exc is None for exc in outcomes) == per_kind
    assert _latency_counts(metric_reader) == {"ok": per_kind, "error": 2 * per_kind}
    assert _counter(metric_reader, "errors_total") == {
        (("error.type", "LLMTimeoutError"),): per_kind,
        (("error.type", "KeyError"),): per_kind,
    }
    assert _counter(metric_reader, "tool_calls_total") == {
        (("outcome", "ok"), ("tool.name", "lookup_order")): per_kind,
        (("outcome", "error"), ("tool.name", "cancel_order")): per_kind,
    }
    for name in ("latency_ms", "errors_total", "tool_calls_total"):
        for point in _points(metric_reader, name):
            assert not {"session_id", "session.id"} & set(point.attributes)

    turns = [s for s in span_exporter.get_finished_spans() if s.name == "agent.turn"]
    statuses = [s.status.status_code for s in turns]
    assert statuses.count(StatusCode.ERROR) == 2 * per_kind
    assert statuses.count(StatusCode.UNSET) == per_kind


def test_batched_export_flushes_every_concurrent_span_on_shutdown(fake_llm, metric_reader):
    exporter = InMemorySpanExporter()
    telemetry = build_telemetry(span_exporter=exporter, metric_reader=metric_reader, batch=True)
    barrier = threading.Barrier(CONCURRENT_TURNS, timeout=10)
    llm = RendezvousLLM(fake_llm, barrier)
    jobs = [(_session("return_refund", str(i)), llm) for i in range(CONCURRENT_TURNS)]
    _replay_concurrently(jobs, telemetry)

    telemetry.shutdown()

    spans = exporter.get_finished_spans()
    assert len(spans) == 3 * CONCURRENT_TURNS
    assert len(_traces(spans)) == CONCURRENT_TURNS
