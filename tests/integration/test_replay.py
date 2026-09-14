import pytest

from mardik.agent import Agent
from mardik.errors import LLMTimeoutError
from mardik.runner import load_session, replay
from mardik.session import SessionStore
from mardik.tools import DEFAULT_TOOLS


def _agent(llm, telemetry):
    return Agent(llm=llm, tools=DEFAULT_TOOLS, telemetry=telemetry)


def test_replay_preserves_session_context(fake_llm, telemetry):
    data = load_session("replay_delivery")
    result = replay(data, _agent(fake_llm, telemetry), SessionStore())
    assert "expédiée" in result.reply


def test_replay_smoke(fake_llm, telemetry):
    data = load_session("replay_delivery")
    result = replay(data, _agent(fake_llm, telemetry), SessionStore())
    assert result.reply is not None


def test_replay_timeout_incident(timeout_llm, telemetry):
    data = load_session("incident_timeout")
    with pytest.raises(LLMTimeoutError):
        replay(data, _agent(timeout_llm, telemetry), SessionStore())


def test_replay_timeout_incident_keeps_recorded_history(timeout_llm, telemetry):
    data = load_session("incident_timeout")
    store = SessionStore()
    with pytest.raises(LLMTimeoutError):
        replay(data, _agent(timeout_llm, telemetry), store)
    assert store.history(data["session_id"]) == data["messages"][:-1]
    assert store.turns(data["session_id"]) == 0
