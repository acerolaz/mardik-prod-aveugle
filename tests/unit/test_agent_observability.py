from structlog.testing import capture_logs

from mardik.agent import Agent
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS


def _run(fake_llm, telemetry):
    agent = Agent(llm=fake_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
    return agent.run_turn(SessionStore(), "obs", "Statut de ma commande #1042 ?")


def test_tool_call_is_traced(fake_llm, telemetry, span_exporter):
    _run(fake_llm, telemetry)
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "tool.call" in names


def test_latency_metric_emitted(fake_llm, telemetry, metric_reader):
    _run(fake_llm, telemetry)
    data = metric_reader.get_metrics_data()
    points = [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "latency_ms"
        for point in metric.data.data_points
    ]
    assert points, "expected at least one latency_ms measurement"


def test_turn_completion_is_logged_structured(fake_llm, telemetry):
    with capture_logs() as logs:
        _run(fake_llm, telemetry)
    events = [entry.get("event") for entry in logs]
    assert "turn.completed" in events


def test_trace_context_propagated_across_threads(fake_llm, telemetry, span_exporter):
    _run(fake_llm, telemetry)
    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert "agent.turn" in spans
    assert "llm.invoke" in spans
    assert spans["llm.invoke"].context.trace_id == spans["agent.turn"].context.trace_id
