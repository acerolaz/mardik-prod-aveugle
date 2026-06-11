from mardik.app import build_agent
from mardik.telemetry import Telemetry


def test_build_agent_wires_telemetry(fake_llm):
    agent = build_agent(llm=fake_llm)
    assert isinstance(agent.telemetry, Telemetry)
