"""Shared fixtures: in-memory telemetry exporters and scripted LLMs."""

from __future__ import annotations

import re
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mardik.agent import Reply
from mardik.telemetry import Telemetry, build_telemetry


class ContextAwareFakeLLM:
    """Looks across the whole conversation for an order id (#1234).

    Found -> asks for an order lookup; otherwise -> asks the user to clarify.
    """

    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        text = " ".join(str(m.get("content", "")) for m in messages)
        match = re.search(r"#(\d+)", text)
        if match:
            return Reply(
                content="",
                tool_calls=[{"name": "lookup_order", "args": {"order_id": match.group(1)}}],
            )
        return Reply(
            content="Pouvez-vous indiquer votre numéro de commande ?",
            tool_calls=[],
        )


class TimeoutFakeLLM:
    def invoke(self, messages: list[dict[str, Any]]) -> Reply:
        raise TimeoutError("upstream deadline exceeded")


@pytest.fixture
def fake_llm() -> ContextAwareFakeLLM:
    return ContextAwareFakeLLM()


@pytest.fixture
def timeout_llm() -> TimeoutFakeLLM:
    return TimeoutFakeLLM()


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def telemetry(
    span_exporter: InMemorySpanExporter, metric_reader: InMemoryMetricReader
) -> Telemetry:
    return build_telemetry(span_exporter=span_exporter, metric_reader=metric_reader, level="INFO")
