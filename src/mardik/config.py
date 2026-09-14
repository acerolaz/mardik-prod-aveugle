"""Runtime configuration loaded from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str
    azure_api_key: str
    azure_model: str
    otel_endpoint: str
    service_name: str
    log_level: str
    metrics_exporter: str = "console"


def load_settings() -> Settings:
    return Settings(
        azure_endpoint=os.environ.get("AZURE_AI_ENDPOINT", ""),
        azure_api_key=os.environ.get("AZURE_AI_API_KEY", ""),
        azure_model=os.environ.get("AZURE_AI_MODEL", "Kimi-K2.6"),
        otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        service_name=os.environ.get("OTEL_SERVICE_NAME", "mardik"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        metrics_exporter=os.environ.get("OTEL_METRICS_EXPORTER", "console").lower(),
    )
