from typing import Any

import pytest
from opentelemetry.trace import StatusCode
from structlog.testing import capture_logs

from mardik.agent import Agent, Reply
from mardik.errors import LLMTimeoutError
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS


class FlakyLLM:
    """Times out on the first call, answers afterwards (a client retry)."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise TimeoutError("upstream deadline exceeded")
        return Reply(content="Je vérifie votre commande.", tool_calls=[])


def test_llm_timeout_surfaces_as_domain_error(timeout_llm, telemetry):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
    with pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")


def test_failed_turn_leaves_session_untouched(timeout_llm, telemetry):
    store = SessionStore()
    store.load_history("err", [{"role": "user", "content": "Bonjour"}])
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)

    with pytest.raises(LLMTimeoutError):
        agent.run_turn(store, "err", "Où est ma commande ?")

    assert store.history("err") == [{"role": "user", "content": "Bonjour"}]
    assert store.turns("err") == 0


def test_retry_after_timeout_does_not_duplicate_user_message(telemetry):
    store = SessionStore()
    llm = FlakyLLM()
    agent = Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)

    with pytest.raises(LLMTimeoutError):
        agent.run_turn(store, "retry", "Où est ma commande ?")
    agent.run_turn(store, "retry", "Où est ma commande ?")

    assert llm.calls[1] == [{"role": "user", "content": "Où est ma commande ?"}]
    assert store.history("retry") == [
        {"role": "user", "content": "Où est ma commande ?"},
        {"role": "assistant", "content": "Je vérifie votre commande."},
    ]
    assert store.turns("retry") == 1


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
