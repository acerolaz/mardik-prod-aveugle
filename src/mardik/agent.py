"""The Mardik agent: turns a user message into a reply, calling tools as needed."""
from __future__ import annotations

import contextvars
import threading
import time
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
        # The Azure SDK call is blocking, so run it on a worker thread.
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                box["reply"] = self._invoke_llm_sync(messages)
            except TimeoutError as exc:
                # Exceptions do not cross thread boundaries: hand it back.
                box["error"] = exc

        # Run in a copy of the current context so the worker's span joins the trace.
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(worker,))
        thread.start()
        thread.join()
        if "error" in box:
            raise LLMTimeoutError("LLM invocation timed out") from box["error"]
        return box["reply"]

    def _dispatch_tool(self, call: dict[str, Any]) -> str:
        with self.telemetry.tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", call["name"])
            tool = self._tools[call["name"]]
            return tool(**call["args"])

    def run_turn(
        self, store: SessionStore, session_id: str, user_message: str
    ) -> TurnResult:
        with self.telemetry.tracer.start_as_current_span("agent.turn"):
            start = time.perf_counter()
            store.append(session_id, {"role": "user", "content": user_message})
            store.record_turn(session_id)

            reply = self._invoke_llm(store.history(session_id))

            text = reply.content
            for call in reply.tool_calls:
                text = self._dispatch_tool(call)

            store.append(session_id, {"role": "assistant", "content": text})
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.telemetry.record_latency(elapsed_ms, session_id=session_id)
            self.telemetry.logger.info(
                "turn.completed", session_id=session_id, latency_ms=round(elapsed_ms, 1)
            )
            return TurnResult(session_id=session_id, reply=text)
