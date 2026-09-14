# Observability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Mardik's existing traces, metrics and structured logs reliable and usable end to end (spec option A), without adding any backend.

**Architecture:** Keep the injectable `Telemetry` bundle. Configuration gains three settings; `telemetry.py` gains a `Resource`, a selectable metric exporter, batch span export, an idempotent `shutdown()`, a trace-context log processor and a `track_turn()` context manager that reports latency/errors/logs. `agent.py` uses a `ThreadPoolExecutor` so LLM exceptions keep their type, sets span attributes, and raises a typed `ToolNotFoundError`.

**Tech Stack:** Python 3.11, `uv`, OpenTelemetry SDK 1.44 (`opentelemetry-exporter-otlp-proto-grpc`), structlog 24.4, pytest 8.

**Spec:** `docs/superpowers/specs/2026-09-14-observability-hardening-design.md`

## Global Constraints

- Python `>=3.11,<3.12`; **no new dependency** in `pyproject.toml`.
- Run everything through `uv run` from the repository (worktree) root. `uv.lock` is not tracked by git: if it is missing, copy it from the main checkout before `uv sync`.
- `uv run ruff check .` and `uv run ruff format --check .` clean; line length 100.
- `uv run mypy src` clean (this plan also fixes the one pre-existing error on `_NoOpSpan.__exit__`).
- Never write `except Exception` / bare `except`; catch specific types only.
- `session_id` never appears in metric attributes (spec D5). It stays on spans (`session.id`) and logs (`session_id`).
- No message content and no tool arguments in span attributes (spec D6).
- Metric names: `latency_ms`, `errors_total`, `tool_calls_total`. Attribute values for `outcome`: `"ok"` | `"error"`.
- Log events: `turn.completed` (info), `turn.failed` (error, with key `error.type`).
- Baseline before starting: `uv run pytest -q` → `36 passed, 1 xfailed`. The strict `xfail` in `tests/integration/test_replay_size_exchange.py` must stay xfailed.

## File Map

| File | Responsibility | Tasks |
|---|---|---|
| `src/mardik/config.py` | Settings + env validation | 1 |
| `src/mardik/telemetry.py` | Resource, exporters, providers lifecycle, instruments, log processor, turn tracker, no-op | 2, 3, 5, 6 |
| `src/mardik/agent.py` | Turn orchestration, LLM worker, tool dispatch, span attributes | 4, 5, 6 |
| `src/mardik/errors.py` | `ToolNotFoundError` | 6 |
| `src/mardik/app.py` | Wiring from settings, shutdown on exit | 2 |
| `tests/conftest.py` | `settings` and `metric_points` fixtures | 1, 5 |
| `tests/unit/test_config.py` (new) | Settings parsing | 1 |
| `tests/unit/test_telemetry.py` (new) | Resource, exporter selection, shutdown | 2 |
| `tests/unit/test_wiring.py` | `build_agent` / `main` wiring | 2 |
| `tests/unit/test_log_correlation.py` (new) | `trace_id` in logs | 3 |
| `tests/unit/test_agent_errors.py` | Error paths | 4, 5, 6 |
| `tests/unit/test_agent_observability.py` | Happy-path signals | 5, 6 |
| `tests/integration/test_replay*.py` | Replay suites (cardinality update, error metric) | 5 |
| `.env.example`, `README.md` | Documentation | 1, 7 |

---

### Task 1: Settings for environment and metric export

**Files:**
- Modify: `src/mardik/config.py` (whole file)
- Modify: `tests/conftest.py` (add `settings` fixture)
- Modify: `.env.example`
- Create: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Settings` gains `environment: str = "development"`, `metrics_exporter: str = "console"`, `metrics_export_interval_ms: int = 60000` (defaults keep existing constructors valid).
  - `METRICS_EXPORTERS: tuple[str, ...] = ("console", "otlp", "none")` in `mardik.config`.
  - `load_settings()` raises `ValueError` on an unknown exporter or a non-integer interval.
  - pytest fixture `settings -> Settings` (service `mardik-test`, environment `test`, exporter `none`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config.py`:

```python
import pytest

from mardik.config import load_settings

_VARS = ("APP_ENV", "OTEL_METRICS_EXPORTER", "OTEL_METRIC_EXPORT_INTERVAL")


def test_observability_settings_defaults(monkeypatch):
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.environment == "development"
    assert settings.metrics_exporter == "console"
    assert settings.metrics_export_interval_ms == 60000


def test_observability_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", " OTLP ")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_INTERVAL", "5000")

    settings = load_settings()

    assert settings.environment == "production"
    assert settings.metrics_exporter == "otlp"
    assert settings.metrics_export_interval_ms == 5000


def test_unknown_metrics_exporter_is_rejected(monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "prometheus")

    with pytest.raises(ValueError, match="OTEL_METRICS_EXPORTER"):
        load_settings()


def test_non_integer_export_interval_is_rejected(monkeypatch):
    monkeypatch.delenv("OTEL_METRICS_EXPORTER", raising=False)
    monkeypatch.setenv("OTEL_METRIC_EXPORT_INTERVAL", "soon")

    with pytest.raises(ValueError, match="OTEL_METRIC_EXPORT_INTERVAL"):
        load_settings()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'environment'` and `DID NOT RAISE <class 'ValueError'>`.

- [ ] **Step 3: Implement**

Replace `src/mardik/config.py` with:

```python
"""Runtime configuration loaded from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass

METRICS_EXPORTERS: tuple[str, ...] = ("console", "otlp", "none")


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str
    azure_api_key: str
    azure_model: str
    otel_endpoint: str
    service_name: str
    log_level: str
    environment: str = "development"
    metrics_exporter: str = "console"
    metrics_export_interval_ms: int = 60000


def _metrics_exporter() -> str:
    value = os.environ.get("OTEL_METRICS_EXPORTER", "console").strip().lower()
    if value not in METRICS_EXPORTERS:
        allowed = ", ".join(METRICS_EXPORTERS)
        raise ValueError(f"OTEL_METRICS_EXPORTER must be one of {allowed}, got {value!r}")
    return value


def _metrics_export_interval_ms() -> int:
    raw = os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "60000")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"OTEL_METRIC_EXPORT_INTERVAL must be an integer, got {raw!r}") from exc


def load_settings() -> Settings:
    return Settings(
        azure_endpoint=os.environ.get("AZURE_AI_ENDPOINT", ""),
        azure_api_key=os.environ.get("AZURE_AI_API_KEY", ""),
        azure_model=os.environ.get("AZURE_AI_MODEL", "Kimi-K2.6"),
        otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        service_name=os.environ.get("OTEL_SERVICE_NAME", "mardik"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        environment=os.environ.get("APP_ENV", "development"),
        metrics_exporter=_metrics_exporter(),
        metrics_export_interval_ms=_metrics_export_interval_ms(),
    )
```

Add to `tests/conftest.py` — import next to the other `mardik` imports:

```python
from mardik.config import Settings
```

and the fixture after `metric_reader`:

```python
@pytest.fixture
def settings() -> Settings:
    return Settings(
        azure_endpoint="",
        azure_api_key="",
        azure_model="Kimi-K2.6",
        otel_endpoint="http://localhost:4317",
        service_name="mardik-test",
        log_level="INFO",
        environment="test",
        metrics_exporter="none",
    )
```

In `.env.example`, replace the OpenTelemetry block with:

```
# OpenTelemetry — OTLP/gRPC collector (Jaeger all-in-one exposes 4317, traces only)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=mardik
# Metrics exporter: console | otlp | none (Jaeger does not store metrics)
OTEL_METRICS_EXPORTER=console
OTEL_METRIC_EXPORT_INTERVAL=60000
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -v && uv run pytest -q`
Expected: 4 passed; full suite `40 passed, 1 xfailed`.

- [ ] **Step 5: Commit**

```bash
git add src/mardik/config.py tests/conftest.py tests/unit/test_config.py .env.example
git commit -m "feat(config): add environment and metric export settings"
```

---

### Task 2: Resource, metric exporter selection, batch export and shutdown

**Files:**
- Modify: `src/mardik/telemetry.py` (imports, `Telemetry.__init__`, `build_telemetry`, `build_default_telemetry`; add `build_resource`, `build_metric_exporter`, `_make_telemetry`, `Telemetry.shutdown`, `NoOpTelemetry.shutdown`)
- Modify: `src/mardik/app.py` (`build_agent`, `main`)
- Modify: `tests/unit/test_wiring.py` (whole file)
- Create: `tests/unit/test_telemetry.py`

**Interfaces:**
- Consumes: `Settings` (fields from Task 1), fixture `settings`.
- Produces:
  - `build_resource(settings: Settings) -> Resource`
  - `build_metric_exporter(settings: Settings) -> MetricExporter | None` — spec §4.1 names this selection `build_metric_reader`; it is isolated one level lower so tests never start a reader thread.
  - `build_telemetry(span_exporter=None, metric_reader=None, level="INFO", resource: Resource | None = None, batch: bool = False) -> Telemetry`
  - `build_default_telemetry(settings: Settings) -> Telemetry` (**signature change**: was `level: str`)
  - `Telemetry(tracer, meter, tracer_provider: TracerProvider | None = None, meter_provider: MeterProvider | None = None)` with public attributes `tracer_provider`, `meter_provider`
  - `Telemetry.shutdown() -> None`, `NoOpTelemetry.shutdown() -> None` — idempotent.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_telemetry.py`:

```python
from dataclasses import replace

import pytest
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

from mardik import __version__
from mardik.telemetry import build_metric_exporter, build_resource, build_telemetry


def test_spans_carry_service_resource(settings, span_exporter, metric_reader):
    telemetry = build_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        resource=build_resource(settings),
    )

    with telemetry.tracer.start_as_current_span("probe"):
        pass

    (span,) = span_exporter.get_finished_spans()
    assert span.resource.attributes["service.name"] == "mardik-test"
    assert span.resource.attributes["service.version"] == __version__
    assert span.resource.attributes["deployment.environment"] == "test"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("console", ConsoleMetricExporter), ("otlp", OTLPMetricExporter)],
)
def test_metric_exporter_follows_settings(settings, name, expected):
    exporter = build_metric_exporter(replace(settings, metrics_exporter=name))

    assert isinstance(exporter, expected)


def test_metric_export_can_be_disabled(settings):
    assert build_metric_exporter(replace(settings, metrics_exporter="none")) is None


def test_shutdown_flushes_batched_spans_and_is_idempotent(span_exporter, metric_reader):
    telemetry = build_telemetry(
        span_exporter=span_exporter, metric_reader=metric_reader, batch=True
    )
    with telemetry.tracer.start_as_current_span("probe"):
        pass

    telemetry.shutdown()
    telemetry.shutdown()

    assert [span.name for span in span_exporter.get_finished_spans()] == ["probe"]
```

Replace `tests/unit/test_wiring.py` with:

```python
import pytest

from mardik import app
from mardik.agent import Agent
from mardik.errors import LLMTimeoutError
from mardik.telemetry import Telemetry, build_telemetry
from mardik.tools import DEFAULT_TOOLS


def test_build_agent_wires_telemetry_from_settings(fake_llm, settings):
    agent = app.build_agent(llm=fake_llm, settings=settings)
    try:
        assert isinstance(agent.telemetry, Telemetry)
        resource = agent.telemetry.tracer_provider.resource
        assert resource.attributes["service.name"] == settings.service_name
    finally:
        agent.telemetry.shutdown()


def test_main_shuts_telemetry_down_even_when_the_turn_fails(
    monkeypatch, timeout_llm, settings, span_exporter, metric_reader
):
    telemetry = build_telemetry(span_exporter=span_exporter, metric_reader=metric_reader)
    calls: list[str] = []
    monkeypatch.setattr(telemetry, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(app, "load_settings", lambda: settings)
    monkeypatch.setattr(
        app,
        "build_agent",
        lambda settings: Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry),
    )

    with pytest.raises(LLMTimeoutError):
        app.main()

    assert calls == ["shutdown"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_telemetry.py tests/unit/test_wiring.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_metric_exporter'` for `test_telemetry.py`; `AttributeError: 'Telemetry' object has no attribute 'tracer_provider'` and `assert [] == ['shutdown']` for `test_wiring.py`.

- [ ] **Step 3: Implement `telemetry.py`**

Replace the import block of `src/mardik/telemetry.py` (everything between the module docstring and `def configure_logging`) with:

```python
from __future__ import annotations

import logging

import structlog
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Tracer

from . import __version__
from .config import Settings
```

Replace `Telemetry.__init__` signature and add the provider attributes and `shutdown` (instruments unchanged for now):

```python
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
        self.tracer_provider = tracer_provider
        self.meter_provider = meter_provider
        self._shut_down = False
        self.logger = structlog.get_logger("mardik")
        # ... existing latency_ms / errors instruments and record_latency stay as they are ...

    def shutdown(self) -> None:
        """Flush pending spans and metrics, then release exporters. Safe to call twice."""
        if self._shut_down:
            return
        self._shut_down = True
        if self.tracer_provider is not None:
            self.tracer_provider.force_flush()
            self.tracer_provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.force_flush()
            self.meter_provider.shutdown()
```

Replace `build_telemetry` and `build_default_telemetry` with:

```python
def build_resource(settings: Settings) -> Resource:
    """Identify the emitting service on every span and metric."""
    return Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": __version__,
            "deployment.environment": settings.environment,
        }
    )


def build_metric_exporter(settings: Settings) -> MetricExporter | None:
    """Pick the metric exporter named by ``settings.metrics_exporter``."""
    if settings.metrics_exporter == "none":
        return None
    if settings.metrics_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter(endpoint=settings.otel_endpoint)
    return ConsoleMetricExporter()


def _make_telemetry(
    span_processor: SpanProcessor,
    metric_readers: list[MetricReader],
    resource: Resource | None,
    level: str,
) -> Telemetry:
    configure_logging(level)
    resource = resource or Resource.create()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)
    meter_provider = MeterProvider(metric_readers=metric_readers, resource=resource)

    return Telemetry(
        tracer=tracer_provider.get_tracer("mardik"),
        meter=meter_provider.get_meter("mardik"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


def build_telemetry(
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    level: str = "INFO",
    resource: Resource | None = None,
    batch: bool = False,
) -> Telemetry:
    """Build a self-contained Telemetry bundle.

    Defaults to console exporters; tests pass in-memory exporters/readers.
    """
    exporter = span_exporter or ConsoleSpanExporter()
    processor: SpanProcessor = (
        BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    )
    reader = metric_reader or PeriodicExportingMetricReader(ConsoleMetricExporter())
    return _make_telemetry(processor, [reader], resource, level)


def build_default_telemetry(settings: Settings) -> Telemetry:
    """Production wiring: batched OTLP/gRPC spans + the configured metric exporter."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    readers: list[MetricReader] = []
    metric_exporter = build_metric_exporter(settings)
    if metric_exporter is not None:
        readers.append(
            PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=settings.metrics_export_interval_ms,
            )
        )
    return _make_telemetry(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)),
        readers,
        build_resource(settings),
        settings.log_level,
    )
```

Add to `NoOpTelemetry`:

```python
    def shutdown(self) -> None:
        pass
```

- [ ] **Step 4: Implement `app.py`**

In `build_agent`, replace `telemetry = build_default_telemetry(settings.log_level)` with:

```python
        telemetry = build_default_telemetry(settings)
```

Replace `main` with:

```python
def main() -> None:
    settings = load_settings()
    agent = build_agent(settings=settings)
    try:
        result = agent.run_turn(SessionStore(), "cli", "Bonjour")
        print(result.reply)
    finally:
        agent.telemetry.shutdown()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_telemetry.py tests/unit/test_wiring.py -v && uv run pytest -q`
Expected: 7 passed in the two files; full suite `46 passed, 1 xfailed`.

- [ ] **Step 6: Commit**

```bash
git add src/mardik/telemetry.py src/mardik/app.py tests/unit/test_telemetry.py tests/unit/test_wiring.py
git commit -m "feat(telemetry): add service resource, selectable metric exporter and clean shutdown"
```

---

### Task 3: Correlate logs with traces

**Files:**
- Modify: `src/mardik/telemetry.py` (`configure_logging`; add `add_trace_context`)
- Create: `tests/unit/test_log_correlation.py`

**Interfaces:**
- Consumes: `build_telemetry` fixture `telemetry`, `span_exporter`, `fake_llm`.
- Produces: `add_trace_context(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict`, included in `configure_logging`'s processor chain before `JSONRenderer`.

> Note: `structlog.testing.capture_logs()` replaces the whole processor chain, so it cannot observe `trace_id`. The correlation test configures `[add_trace_context, LogCapture()]` explicitly, and a separate test pins that `configure_logging` includes the processor.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_log_correlation.py`:

```python
import structlog
from structlog.testing import LogCapture

from mardik.agent import Agent
from mardik.session import SessionStore
from mardik.telemetry import add_trace_context, configure_logging
from mardik.tools import DEFAULT_TOOLS


def test_logging_pipeline_adds_trace_context():
    configure_logging("INFO")

    assert add_trace_context in structlog.get_config()["processors"]


def test_turn_log_carries_the_trace_id_of_its_turn(fake_llm, telemetry, span_exporter):
    capture = LogCapture()
    structlog.configure(processors=[add_trace_context, capture])
    try:
        agent = Agent(llm=fake_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)
        agent.run_turn(SessionStore(), "corr", "Statut de ma commande #1042 ?")
    finally:
        configure_logging("INFO")

    turn = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.turn")
    completed = next(e for e in capture.entries if e["event"] == "turn.completed")
    assert completed["trace_id"] == format(turn.context.trace_id, "032x")
    assert len(completed["span_id"]) == 16


def test_no_trace_context_outside_a_span():
    assert add_trace_context(None, "info", {"event": "idle"}) == {"event": "idle"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_log_correlation.py -v`
Expected: FAIL — `ImportError: cannot import name 'add_trace_context'`.

- [ ] **Step 3: Implement**

In `src/mardik/telemetry.py`, add to the imports:

```python
from opentelemetry import trace
from structlog.typing import EventDict, WrappedLogger
```

Add above `configure_logging`:

```python
def add_trace_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Stamp the active span's ids on a log entry so it can be joined to its trace."""
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict
```

In `configure_logging`, change the processor list to:

```python
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            add_trace_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_log_correlation.py -v && uv run pytest -q`
Expected: 3 passed; full suite `49 passed, 1 xfailed`.

- [ ] **Step 5: Commit**

```bash
git add src/mardik/telemetry.py tests/unit/test_log_correlation.py
git commit -m "feat(logging): add trace_id and span_id to structured logs"
```

---

### Task 4: Propagate LLM exceptions with their original type

**Files:**
- Modify: `src/mardik/agent.py` (imports, `_invoke_llm`)
- Modify: `tests/unit/test_agent_errors.py`

**Interfaces:**
- Consumes: fixtures `timeout_llm`, `telemetry`, `span_exporter`.
- Produces: `Agent._invoke_llm(messages) -> Reply` raising `LLMTimeoutError` for `TimeoutError` and the original exception otherwise. No public signature change.

- [ ] **Step 1: Write the tests**

Append to `tests/unit/test_agent_errors.py` (add `from opentelemetry.trace import StatusCode` to the imports):

```python
class ExplodingLLM:
    def invoke(self, messages):
        raise RuntimeError("model gateway exploded")


def test_unexpected_llm_error_keeps_its_type(telemetry, span_exporter):
    agent = Agent(llm=ExplodingLLM(), tools=DEFAULT_TOOLS, telemetry=telemetry)

    with pytest.raises(RuntimeError, match="model gateway exploded"):
        agent.run_turn(SessionStore(), "err", "Bonjour")

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert spans["llm.invoke"].status.status_code is StatusCode.ERROR


def test_llm_timeout_marks_llm_and_turn_spans_as_error(timeout_llm, telemetry, span_exporter):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)

    with pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert spans["llm.invoke"].status.status_code is StatusCode.ERROR
    assert spans["agent.turn"].status.status_code is StatusCode.ERROR
```

- [ ] **Step 2: Run tests to verify the new failure**

Run: `uv run pytest tests/unit/test_agent_errors.py -v`
Expected: `test_unexpected_llm_error_keeps_its_type` FAILS with `KeyError: 'reply'`. `test_llm_timeout_marks_llm_and_turn_spans_as_error` already PASSES — it is a regression guard for the executor refactor (OTel records the exception by default).

- [ ] **Step 3: Implement**

In `src/mardik/agent.py`, replace `import threading` with:

```python
from concurrent.futures import ThreadPoolExecutor
```

Replace `_invoke_llm` with:

```python
    def _invoke_llm(self, messages: list[dict[str, Any]]) -> Reply:
        # The Azure SDK call is blocking, so run it on a worker thread, in a copy
        # of the current context so the worker's span joins the trace.
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(ctx.run, self._invoke_llm_sync, messages)
        try:
            # result() re-raises the worker's exception with its original type.
            return future.result()
        except TimeoutError as exc:
            raise LLMTimeoutError("LLM invocation timed out") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_agent_errors.py tests/unit/test_agent_observability.py -v && uv run pytest -q`
Expected: all pass, including `test_trace_context_propagated_across_threads`; full suite `51 passed, 1 xfailed`.

- [ ] **Step 5: Commit**

```bash
git add src/mardik/agent.py tests/unit/test_agent_errors.py
git commit -m "fix(agent): propagate unexpected LLM errors with their original type"
```

---

### Task 5: Turn tracking — latency outcome, error counter, failure log, span attributes

**Files:**
- Modify: `src/mardik/telemetry.py` (`Telemetry`: remove `record_latency`, add `record_error`, `track_turn`; add `_TurnTracker`; `NoOpTelemetry` and `_NoOpSpan`)
- Modify: `src/mardik/agent.py` (`run_turn`, imports)
- Modify: `tests/conftest.py` (add `metric_points` fixture)
- Modify: `tests/unit/test_agent_observability.py`, `tests/unit/test_agent_errors.py`
- Modify: `tests/integration/test_replay.py`, `tests/integration/test_replay_return_refund.py`, `tests/integration/test_replay_size_exchange.py`

**Interfaces:**
- Consumes: `Telemetry` from Task 2, `_invoke_llm` from Task 4.
- Produces:
  - `Telemetry.record_error(error_type: str) -> None` → `errors_total{error.type}` += 1
  - `Telemetry.track_turn(session_id: str) -> _TurnTracker` (context manager, never suppresses exceptions)
  - `NoOpTelemetry.record_error`, `NoOpTelemetry.track_turn` with the same shapes
  - `Telemetry.record_latency` **removed**
  - Span `agent.turn` attributes: `session.id: str`, `mardik.history.length: int`, `mardik.tool_calls.count: int`
  - pytest fixture `metric_points -> Callable[[str], list[DataPoint]]`

- [ ] **Step 1: Add the `metric_points` fixture**

In `tests/conftest.py`, add after the `metric_reader` fixture:

```python
@pytest.fixture
def metric_points(metric_reader: InMemoryMetricReader) -> Any:
    """Return a function collecting every data point recorded for a metric name."""

    def collect(name: str) -> list[Any]:
        data = metric_reader.get_metrics_data()
        if data is None:
            return []
        return [
            point
            for rm in data.resource_metrics
            for sm in rm.scope_metrics
            for metric in sm.metrics
            if metric.name == name
            for point in metric.data.data_points
        ]

    return collect
```

- [ ] **Step 2: Write the failing unit tests**

Append to `tests/unit/test_agent_observability.py`:

```python
def test_latency_metric_is_tagged_with_outcome_only(fake_llm, telemetry, metric_points):
    _run(fake_llm, telemetry)

    assert [dict(point.attributes) for point in metric_points("latency_ms")] == [
        {"outcome": "ok"}
    ]


def test_no_metric_carries_session_id(fake_llm, telemetry, metric_points):
    _run(fake_llm, telemetry)

    for name in ("latency_ms", "errors_total", "tool_calls_total"):
        for point in metric_points(name):
            assert "session_id" not in point.attributes


def test_turn_span_describes_the_session(fake_llm, telemetry, span_exporter):
    _run(fake_llm, telemetry)

    turn = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.turn")
    assert turn.attributes["session.id"] == "obs"
    assert turn.attributes["mardik.history.length"] == 1
    assert turn.attributes["mardik.tool_calls.count"] == 1


def test_agent_without_telemetry_still_answers(fake_llm):
    agent = Agent(llm=fake_llm, tools=DEFAULT_TOOLS)

    result = agent.run_turn(SessionStore(), "noop", "Statut de ma commande #1042 ?")

    assert "expédiée" in result.reply
```

Append to `tests/unit/test_agent_errors.py` (add `from structlog.testing import capture_logs` to the imports):

```python
def test_llm_timeout_is_counted_and_logged(timeout_llm, telemetry, metric_points):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS, telemetry=telemetry)

    with capture_logs() as logs, pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")

    errors = [(dict(p.attributes), p.value) for p in metric_points("errors_total")]
    assert errors == [({"error.type": "LLMTimeoutError"}, 1)]
    assert [dict(p.attributes) for p in metric_points("latency_ms")] == [{"outcome": "error"}]
    failed = [entry for entry in logs if entry["event"] == "turn.failed"]
    assert len(failed) == 1
    assert failed[0]["log_level"] == "error"
    assert failed[0]["session_id"] == "err"
    assert failed[0]["error.type"] == "LLMTimeoutError"


def test_agent_without_telemetry_still_surfaces_timeout(timeout_llm):
    agent = Agent(llm=timeout_llm, tools=DEFAULT_TOOLS)

    with pytest.raises(LLMTimeoutError):
        agent.run_turn(SessionStore(), "err", "Bonjour")
```

- [ ] **Step 3: Update the integration tests (spec D5 contradicts their current assertions)**

In `tests/integration/test_replay_return_refund.py`, rename `test_latency_metric_is_tagged_with_session_id` to `test_latency_metric_is_tagged_with_outcome_not_session_id` and replace

```python
    assert points[0].attributes == {"session_id": session["session_id"]}
```

with

```python
    assert points[0].attributes == {"outcome": "ok"}
```

(keep the `count == 1` and `sum > 0` assertions).

In `tests/integration/test_replay_size_exchange.py`, rename `test_latency_and_log_are_tagged_with_session_id` to `test_latency_is_tagged_with_outcome_and_log_with_session_id` and make the same one-line replacement (the log assertion on `session_id` stays).

Append to `tests/integration/test_replay.py`:

```python
def test_replay_timeout_incident_is_counted(timeout_llm, telemetry, metric_points):
    data = load_session("incident_timeout")

    with pytest.raises(LLMTimeoutError):
        replay(data, _agent(timeout_llm, telemetry), SessionStore())

    assert [point.value for point in metric_points("errors_total")] == [1]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest -q`
Expected: FAIL — latency attributes are `{'session_id': ...}`, `errors_total` has no points, `turn.failed` missing, `KeyError: 'session.id'`.

- [ ] **Step 5: Implement `telemetry.py`**

Add to the imports:

```python
import time
from types import TracebackType
```

In `Telemetry`, delete `record_latency` and add:

```python
    def record_error(self, error_type: str) -> None:
        self.errors.add(1, attributes={"error.type": error_type})

    def track_turn(self, session_id: str) -> _TurnTracker:
        """Time one agent turn and report its outcome (metrics + log)."""
        return _TurnTracker(self, session_id)
```

Add after the `Telemetry` class:

```python
class _TurnTracker:
    """Context manager reporting a turn's latency, errors and completion log.

    It observes the exception leaving the block but never suppresses it.
    """

    def __init__(self, telemetry: Telemetry, session_id: str) -> None:
        self._telemetry = telemetry
        self._session_id = session_id
        self._start = 0.0

    def __enter__(self) -> _TurnTracker:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        telemetry = self._telemetry
        if exc_type is None:
            telemetry.latency_ms.record(elapsed_ms, attributes={"outcome": "ok"})
            telemetry.logger.info(
                "turn.completed", session_id=self._session_id, latency_ms=round(elapsed_ms, 1)
            )
            return
        error_type = exc_type.__name__
        telemetry.latency_ms.record(elapsed_ms, attributes={"outcome": "error"})
        telemetry.record_error(error_type)
        telemetry.logger.error(
            "turn.failed",
            session_id=self._session_id,
            latency_ms=round(elapsed_ms, 1),
            **{"error.type": error_type},
        )
```

Change `_NoOpSpan.__exit__` (fixes the pre-existing mypy error) to:

```python
    def __exit__(self, *exc: object) -> None:
        return None
```

Add before `NoOpTelemetry`:

```python
class _NoOpTurn:
    def __enter__(self) -> _NoOpTurn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None
```

In `NoOpTelemetry`, delete `record_latency` and add:

```python
    def record_error(self, error_type: str) -> None:
        pass

    def track_turn(self, session_id: str) -> _NoOpTurn:
        return _NoOpTurn()
```

Also remove the `-> "_NoOpSpan"` string quotes if ruff flags them (`from __future__ import annotations` is active).

- [ ] **Step 6: Implement `agent.py`**

Remove `import time`. Replace `run_turn` with:

```python
    def run_turn(
        self, store: SessionStore, session_id: str, user_message: str
    ) -> TurnResult:
        with self.telemetry.tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("session.id", session_id)
            # Inside the span so turn logs carry this turn's trace_id.
            with self.telemetry.track_turn(session_id):
                store.append(session_id, {"role": "user", "content": user_message})
                store.record_turn(session_id)

                history = store.history(session_id)
                span.set_attribute("mardik.history.length", len(history))
                reply = self._invoke_llm(history)
                span.set_attribute("mardik.tool_calls.count", len(reply.tool_calls))

                text = reply.content
                for call in reply.tool_calls:
                    text = self._dispatch_tool(call)

                store.append(session_id, {"role": "assistant", "content": text})
                return TurnResult(session_id=session_id, reply=text)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest -q && uv run mypy src`
Expected: `58 passed, 1 xfailed`; mypy `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add src/mardik/telemetry.py src/mardik/agent.py tests/conftest.py tests/unit tests/integration
git commit -m "feat(telemetry): report turn outcome, count errors and drop session_id from metrics"
```

---

### Task 6: Typed tool errors and tool call metrics

**Files:**
- Modify: `src/mardik/errors.py`
- Modify: `src/mardik/telemetry.py` (`Telemetry.__init__` instruments, `NoOpTelemetry.__init__`)
- Modify: `src/mardik/agent.py` (`_dispatch_tool`, imports)
- Modify: `tests/unit/test_agent_errors.py`, `tests/unit/test_agent_observability.py`

**Interfaces:**
- Consumes: `metric_points` fixture (Task 5), `Telemetry.record_error` via `track_turn` (Task 5).
- Produces:
  - `ToolNotFoundError(MardikError)` in `mardik.errors`
  - `Telemetry.tool_calls` counter `tool_calls_total{tool.name, outcome}`; `NoOpTelemetry.tool_calls` no-op
  - Span `tool.call` attribute `tool.outcome: "ok" | "error"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_agent_errors.py` (add `from mardik.agent import Reply` and `from mardik.errors import ToolNotFoundError` to the imports):

```python
class UnknownToolLLM:
    def invoke(self, messages):
        return Reply(content="", tool_calls=[{"name": "cancel_order", "args": {}}])


def test_unknown_tool_is_a_traced_domain_error(telemetry, span_exporter, metric_points):
    agent = Agent(llm=UnknownToolLLM(), tools=DEFAULT_TOOLS, telemetry=telemetry)

    with pytest.raises(ToolNotFoundError, match="cancel_order"):
        agent.run_turn(SessionStore(), "err", "Annulez ma commande")

    tool_span = next(s for s in span_exporter.get_finished_spans() if s.name == "tool.call")
    assert tool_span.status.status_code is StatusCode.ERROR
    assert tool_span.attributes["tool.outcome"] == "error"
    calls = [(dict(p.attributes), p.value) for p in metric_points("tool_calls_total")]
    assert calls == [({"tool.name": "cancel_order", "outcome": "error"}, 1)]
    errors = [dict(p.attributes) for p in metric_points("errors_total")]
    assert errors == [{"error.type": "ToolNotFoundError"}]
```

Append to `tests/unit/test_agent_observability.py`:

```python
def test_successful_tool_call_is_counted(fake_llm, telemetry, span_exporter, metric_points):
    _run(fake_llm, telemetry)

    calls = [(dict(p.attributes), p.value) for p in metric_points("tool_calls_total")]
    assert calls == [({"tool.name": "lookup_order", "outcome": "ok"}, 1)]
    tool_span = next(s for s in span_exporter.get_finished_spans() if s.name == "tool.call")
    assert tool_span.attributes["tool.outcome"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_agent_errors.py tests/unit/test_agent_observability.py -v`
Expected: FAIL — `ImportError: cannot import name 'ToolNotFoundError'` (fix the import to see the next failure: `KeyError: 'cancel_order'`, empty `tool_calls_total`).

- [ ] **Step 3: Implement**

Append to `src/mardik/errors.py`:

```python
class ToolNotFoundError(MardikError):
    """Raised when the LLM asks for a tool the agent does not provide."""
```

In `Telemetry.__init__`, after the `errors` counter:

```python
        self.tool_calls = meter.create_counter(
            "tool_calls_total",
            description="Count of tool invocations, by tool and outcome.",
        )
```

In `NoOpTelemetry.__init__`, after `self.errors = _NoOpInstrument()`:

```python
        self.tool_calls = _NoOpInstrument()
```

In `src/mardik/agent.py`, change the errors import to:

```python
from .errors import LLMTimeoutError, ToolNotFoundError
```

Replace `_dispatch_tool` with:

```python
    def _dispatch_tool(self, call: dict[str, Any]) -> str:
        name = call["name"]
        with self.telemetry.tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", name)
            outcome = "error"
            try:
                tool = self._tools.get(name)
                if tool is None:
                    raise ToolNotFoundError(f"Unknown tool: {name}")
                result = tool(**call["args"])
                outcome = "ok"
                return result
            finally:
                span.set_attribute("tool.outcome", outcome)
                self.telemetry.tool_calls.add(
                    1, attributes={"tool.name": name, "outcome": outcome}
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: `60 passed, 1 xfailed` (includes `test_agent_without_telemetry_still_answers`, which exercises `NoOpTelemetry.tool_calls`).

- [ ] **Step 5: Commit**

```bash
git add src/mardik/errors.py src/mardik/telemetry.py src/mardik/agent.py tests/unit
git commit -m "feat(agent): raise ToolNotFoundError and count tool calls by outcome"
```

---

### Task 7: Documentation and end-to-end verification

**Files:**
- Modify: `README.md` (replace the `## Known issues` section)

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing documentation of the emitted signals.

- [ ] **Step 1: Document the signals**

In `README.md`, replace the whole `## Known issues` section (heading + paragraph) with:

````markdown
## Observabilité

| Signal | Détail |
|---|---|
| Spans | `agent.turn` (`session.id`, `mardik.history.length`, `mardik.tool_calls.count`) → `llm.invoke` et `tool.call` (`tool.name`, `tool.outcome`). Les exceptions passent le span en ERROR. |
| Métriques | `latency_ms{outcome}`, `errors_total{error.type}`, `tool_calls_total{tool.name, outcome}`. Jamais de `session_id` en attribut (cardinalité). |
| Logs | JSON sur stdout : `turn.completed` / `turn.failed` avec `session_id`, `latency_ms`, `trace_id`, `span_id`. |
| Resource | `service.name`, `service.version`, `deployment.environment` (`APP_ENV`). |

Export :

- Traces : OTLP/gRPC par lots vers `OTEL_EXPORTER_OTLP_ENDPOINT`, vidées à l'arrêt du process.
- Métriques : `OTEL_METRICS_EXPORTER=console|otlp|none` (défaut `console`), période
  `OTEL_METRIC_EXPORT_INTERVAL` en ms. **Jaeger ne stocke que les traces** : en local, lire les
  métriques sur la console ; utiliser `otlp` derrière un collecteur OpenTelemetry en production.

Pour retrouver la trace d'un log : copier son `trace_id` dans la recherche de Jaeger.
````

- [ ] **Step 2: Run the full verification**

Run:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -q
```

Expected: format and lint clean, `Success: no issues found`, `60 passed, 1 xfailed`. If `ruff format --check` reports files, run `uv run ruff format .`, re-run the tests, and include the formatting in the commit.

- [ ] **Step 3: Manual end-to-end check (needs Azure credentials in `.env`; skip and say so if unavailable)**

```bash
make up
set -a && . ./.env && set +a && uv run python -m mardik.app
```

Expected: one JSON log `turn.completed` carrying a `trace_id`; in http://localhost:16686, service `mardik` shows a trace `agent.turn` whose id equals that `trace_id`, even though the process exited right after the turn. Then `make down`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document emitted observability signals and metric export"
```
