import pytest
from opentelemetry.trace import StatusCode
from structlog.testing import capture_logs

from mardik.agent import Agent
from mardik.errors import LLMTimeoutError
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS


def test_llm_timeout_surfaces_as_domain_error(timeout_llm, telemetry):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
    with pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")


class _BrokenLLM:
    def invoke(self, messages):
        raise RuntimeError("unexpected SDK failure")


def _error_points(metric_reader):
    return [
        point
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "errors_total"
        for point in metric.data.data_points
    ]


def test_timeout_is_observable(timeout_llm, telemetry, span_exporter, metric_reader):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
    with capture_logs() as logs, pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert spans["agent.turn"].status.status_code is StatusCode.ERROR
    assert spans["llm.invoke"].status.status_code is StatusCode.ERROR
    assert [(p.attributes, p.value) for p in _error_points(metric_reader)] == [
        ({"error.type": "LLMTimeoutError"}, 1)
    ]
    failed = [entry for entry in logs if entry["event"] == "turn.failed"]
    assert failed and failed[0]["session_id"] == "err"


def test_unexpected_llm_error_keeps_its_type(telemetry):
    agent = Agent(llm=_BrokenLLM(), tools=DEFAULT_TOOLS, telemetry=telemetry)
    with pytest.raises(RuntimeError, match="unexpected SDK failure"):
        agent.run_turn(SessionStore(), "err", "Bonjour")
