"""Observability primitives: structured logging, tracing and metrics.

The :class:`Telemetry` object bundles a tracer, a logger and the metric
instruments the agent emits. It is injectable so that tests can wire in-memory
exporters and inspect what was recorded; production wiring lives in
:func:`build_default_telemetry`.
"""
from __future__ import annotations

import logging

import structlog
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Tracer


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit structured JSON lines."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        cache_logger_on_first_use=False,
    )


class Telemetry:
    """Bundle of tracer + logger + metric instruments used by the agent."""

    def __init__(self, tracer: Tracer, meter: Meter) -> None:
        self.tracer = tracer
        self.logger = structlog.get_logger("mardik")
        self.latency_ms = meter.create_histogram(
            "latency_ms",
            unit="ms",
            description="End-to-end latency of an agent turn.",
        )
        self.errors = meter.create_counter(
            "errors_total",
            description="Count of agent turns that ended in an error.",
        )

    def record_latency(self, value_ms: float, **attributes: str) -> None:
        self.latency_ms.record(value_ms, attributes=attributes)


def build_telemetry(
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    level: str = "INFO",
) -> Telemetry:
    """Build a self-contained Telemetry bundle.

    Defaults to console exporters; tests pass in-memory exporters/readers.
    """
    configure_logging(level)

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter or ConsoleSpanExporter())
    )

    reader = metric_reader or PeriodicExportingMetricReader(ConsoleMetricExporter())
    meter_provider = MeterProvider(metric_readers=[reader])

    return Telemetry(
        tracer=tracer_provider.get_tracer("mardik"),
        meter=meter_provider.get_meter("mardik"),
    )


def build_default_telemetry(level: str = "INFO") -> Telemetry:
    """Production wiring: OTLP/gRPC span export to the collector + periodic metrics."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return build_telemetry(
        span_exporter=OTLPSpanExporter(),
        metric_reader=PeriodicExportingMetricReader(ConsoleMetricExporter()),
        level=level,
    )


class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc: object) -> bool:
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

    def record_latency(self, value_ms: float, **attributes: str) -> None:
        pass
