import pytest
import structlog
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from structlog.testing import LogCapture

from mardik.agent import Agent
from mardik.config import Settings
from mardik.session import SessionStore
from mardik.telemetry import (
    NoOpTelemetry,
    add_trace_context,
    build_default_telemetry,
    build_resource,
    build_telemetry,
)
from mardik.tools import DEFAULT_TOOLS


def _metric_points(metric_reader, name):
    return [
        point
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _run(llm, telemetry):
    agent = Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
    return agent.run_turn(SessionStore(), "obs", "Statut de ma commande #1042 ?")


@pytest.fixture
def log_capture():
    previous = structlog.get_config()
    capture = LogCapture()
    structlog.configure(processors=[add_trace_context, capture])
    yield capture
    structlog.configure(**previous)


def test_logs_carry_trace_id_of_the_turn(fake_llm, telemetry, span_exporter, log_capture):
    _run(fake_llm, telemetry)

    turn = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.turn")
    completed = next(e for e in log_capture.entries if e["event"] == "turn.completed")
    assert completed["trace_id"] == format(turn.context.trace_id, "032x")
    assert len(completed["span_id"]) == 16


def test_logs_outside_a_span_have_no_trace_id(telemetry, log_capture):
    telemetry.logger.info("idle")
    assert "trace_id" not in log_capture.entries[0]


def test_tool_calls_are_counted(fake_llm, telemetry, metric_reader):
    _run(fake_llm, telemetry)

    points = _metric_points(metric_reader, "tool_calls_total")
    assert [(p.attributes, p.value) for p in points] == [
        ({"tool.name": "lookup_order", "outcome": "ok"}, 1)
    ]


def test_no_metric_is_labelled_with_session_id(fake_llm, telemetry, metric_reader):
    _run(fake_llm, telemetry)

    for name in ("latency_ms", "tool_calls_total", "errors_total"):
        for point in _metric_points(metric_reader, name):
            assert "session_id" not in point.attributes


def test_spans_carry_service_identity(fake_llm, span_exporter, metric_reader):
    telemetry = build_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        resource=build_resource("mardik-test"),
    )
    _run(fake_llm, telemetry)

    attributes = span_exporter.get_finished_spans()[0].resource.attributes
    assert attributes["service.name"] == "mardik-test"
    assert "service.version" in attributes


def test_shutdown_flushes_batched_spans_and_is_idempotent(fake_llm, metric_reader):
    exporter = InMemorySpanExporter()
    telemetry = build_telemetry(span_exporter=exporter, metric_reader=metric_reader, batch=True)
    _run(fake_llm, telemetry)

    telemetry.shutdown()
    telemetry.shutdown()

    assert {s.name for s in exporter.get_finished_spans()} >= {"agent.turn", "tool.call"}


def test_default_telemetry_builds_without_network():
    settings = Settings(
        azure_endpoint="",
        azure_api_key="",
        azure_model="m",
        otel_endpoint="http://localhost:4317",
        service_name="mardik",
        log_level="INFO",
        metrics_exporter="otlp",
    )
    telemetry = build_default_telemetry(settings)
    telemetry.shutdown()


def test_noop_telemetry_is_substitutable(fake_llm):
    result = _run(fake_llm, NoOpTelemetry())
    assert "expédiée" in result.reply
