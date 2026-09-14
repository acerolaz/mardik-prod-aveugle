"""The Mardik agent: turns a user message into a reply, calling tools as needed."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .errors import LLMTimeoutError
from .session import SessionStore
from .telemetry import NoOpTelemetry


@dataclass
class Reply:
    content: str
    tool_calls: list[dict[str, Any]]


@dataclass
class TurnResult:
    session_id: str
    reply: str


class LLM(Protocol):
    def invoke(self, messages: list[dict[str, Any]]) -> Reply: ...


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: dict[str, Callable[..., str]],
        telemetry: Any | None = None,
    ) -> None:
        self.llm = llm
        self._tools = tools
        self.telemetry = telemetry if telemetry is not None else NoOpTelemetry()

    def _invoke_llm_sync(self, messages: list[dict[str, Any]]) -> Reply:
        with self.telemetry.tracer.start_as_current_span("llm.invoke"):
            return self.llm.invoke(messages)

    def _invoke_llm(self, messages: list[dict[str, Any]]) -> Reply:
        # The Azure SDK call is blocking, so run it on a worker thread. It runs in a
        # copy of the current context so the worker's span joins the trace.
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(ctx.run, self._invoke_llm_sync, messages)
            try:
                return future.result()
            except TimeoutError as exc:
                raise LLMTimeoutError("LLM invocation timed out") from exc

    def _dispatch_tool(self, call: dict[str, Any]) -> str:
        name = call["name"]
        with self.telemetry.tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", name)
            succeeded = False
            try:
                result = self._tools[name](**call["args"])
                succeeded = True
                return result
            finally:
                outcome = "ok" if succeeded else "error"
                span.set_attribute("tool.outcome", outcome)
                self.telemetry.tool_calls.add(1, attributes={"tool.name": name, "outcome": outcome})

    def run_turn(self, store: SessionStore, session_id: str, user_message: str) -> TurnResult:
        with self.telemetry.tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("session.id", session_id)
            with self.telemetry.track_turn(session_id):
                store.append(session_id, {"role": "user", "content": user_message})
                store.record_turn(session_id)

                reply = self._invoke_llm(store.history(session_id))

                text = reply.content
                for call in reply.tool_calls:
                    text = self._dispatch_tool(call)

                store.append(session_id, {"role": "assistant", "content": text})
                return TurnResult(session_id=session_id, reply=text)
