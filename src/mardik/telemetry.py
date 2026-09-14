"""Observability primitives: structured logging, tracing and metrics.

The :class:`Telemetry` object bundles a tracer, a logger and the metric
instruments the agent emits. It is injectable so that tests can wire in-memory
exporters and inspect what was recorded; production wiring lives in
:func:`build_default_telemetry`.

Signals:

- traces: spans ``agent.turn`` > ``llm.invoke`` / ``tool.call`` (OTLP -> Jaeger);
- metrics: ``latency_ms{outcome}``, ``errors_total{error.type}``,
  ``tool_calls_total{tool.name, outcome}`` — never ``session_id`` (cardinality);
- logs: JSON lines carrying ``trace_id`` / ``span_id`` when a span is active.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import structlog
from opentelemetry import trace
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Tracer

from . import __version__

if TYPE_CHECKING:
    from .config import Settings


def add_trace_context(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: attach the active span's ids so logs join their trace."""
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit structured JSON lines."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_trace_context,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=False,
    )


def build_resource(service_name: str = "mardik") -> Resource:
    """Service identity attached to every span and metric."""
    return Resource.create({"service.name": service_name, "service.version": __version__})


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class Telemetry:
    """Bundle of tracer + logger + metric instruments used by the agent."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self.tracer = tracer
        self.logger = structlog.get_logger("mardik")
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider
        self._is_shut_down = False
        self.latency_ms = meter.create_histogram(
            "latency_ms",
            unit="ms",
            description="End-to-end latency of an agent turn.",
        )
        self.errors = meter.create_counter(
            "errors_total",
            description="Count of agent turns that ended in an error.",
        )
        self.tool_calls = meter.create_counter(
            "tool_calls_total",
            description="Count of tool invocations, by tool and outcome.",
        )

    @contextmanager
    def track_turn(self, session_id: str) -> Iterator[None]:
        """Measure one agent turn: latency, error counter and completion/failure log."""
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            # Observation only: the exception is always re-raised, never swallowed.
            elapsed_ms = _elapsed_ms(start)
            error_type = type(exc).__name__
            self.latency_ms.record(elapsed_ms, attributes={"outcome": "error"})
            self.errors.add(1, attributes={"error.type": error_type})
            self.logger.error(
                "turn.failed",
                session_id=session_id,
                latency_ms=round(elapsed_ms, 1),
                error_type=error_type,
            )
            raise
        elapsed_ms = _elapsed_ms(start)
        self.latency_ms.record(elapsed_ms, attributes={"outcome": "ok"})
        self.logger.info("turn.completed", session_id=session_id, latency_ms=round(elapsed_ms, 1))

    def shutdown(self) -> None:
        """Flush pending spans/metrics and release exporters. Safe to call twice."""
        if self._is_shut_down:
            return
        self._is_shut_down = True
        if self._tracer_provider is not None:
            self._tracer_provider.force_flush()
            self._tracer_provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.force_flush()
            self._meter_provider.shutdown()


def build_telemetry(
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    level: str = "INFO",
    resource: Resource | None = None,
    batch: bool = False,
) -> Telemetry:
    """Build a self-contained Telemetry bundle.

    Defaults to console exporters; tests pass in-memory exporters/readers.
    ``batch=True`` exports spans in the background (production).
    """
    configure_logging(level)
    resource = resource or build_resource()

    exporter = span_exporter or ConsoleSpanExporter()
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(processor)

    reader = metric_reader or PeriodicExportingMetricReader(ConsoleMetricExporter())
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])

    return Telemetry(
        tracer=tracer_provider.get_tracer("mardik"),
        meter=meter_provider.get_meter("mardik"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


def build_default_telemetry(settings: Settings) -> Telemetry:
    """Production wiring: OTLP/gRPC spans to the collector + periodic metrics export."""
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    metric_exporter: MetricExporter
    if settings.metrics_exporter == "otlp":
        metric_exporter = OTLPMetricExporter(endpoint=settings.otel_endpoint)
    else:
        metric_exporter = ConsoleMetricExporter()

    return build_telemetry(
        span_exporter=OTLPSpanExporter(endpoint=settings.otel_endpoint),
        metric_reader=PeriodicExportingMetricReader(metric_exporter),
        level=settings.log_level,
        resource=build_resource(settings.service_name),
        batch=True,
    )


class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        return False

    def set_attribute(self, *args: object, **kwargs: object) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, *args: object, **kwargs: object) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpInstrument:
    def record(self, *args: object, **kwargs: object) -> None:
        pass

    def add(self, *args: object, **kwargs: object) -> None:
        pass


class NoOpTelemetry:
    """Telemetry that records nothing — used when observability is not wired."""

    def __init__(self) -> None:
        self.tracer = _NoOpTracer()
        self.logger = structlog.get_logger("mardik")
        self.latency_ms = _NoOpInstrument()
        self.errors = _NoOpInstrument()
        self.tool_calls = _NoOpInstrument()

    @contextmanager
    def track_turn(self, session_id: str) -> Iterator[None]:
        yield

    def shutdown(self) -> None:
        pass
