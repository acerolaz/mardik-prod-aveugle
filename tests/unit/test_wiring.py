from mardik.app import build_agent
from mardik.config import Settings
from mardik.telemetry import Telemetry


def test_build_agent_wires_telemetry(fake_llm):
    settings = Settings(
        azure_endpoint="",
        azure_api_key="",
        azure_model="m",
        otel_endpoint="http://localhost:4317",
        service_name="mardik",
        log_level="INFO",
    )
    agent = build_agent(llm=fake_llm, settings=settings)
    assert isinstance(agent.telemetry, Telemetry)
    agent.telemetry.shutdown()
