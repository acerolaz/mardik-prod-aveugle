"""The Mardik agent: turns a user message into a reply, calling tools as needed."""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .errors import LLMTimeoutError
from .session import SessionStore
from .telemetry import NoOpTelemetry

# Tool payloads go on spans for debugging; cap them so one big result can't bloat a trace.
MAX_SPAN_PAYLOAD_CHARS = 512


def _truncate(text: str) -> str:
    return text[:MAX_SPAN_PAYLOAD_CHARS]


@dataclass
class Reply:
    content: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int] | None = None


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
        llm_timeout_s: float | None = 30.0,
    ) -> None:
        self.llm = llm
        self._tools = tools
        self.telemetry = telemetry if telemetry is not None else NoOpTelemetry()
        self.llm_timeout_s = llm_timeout_s

    def _invoke_llm_sync(self, messages: list[dict[str, Any]]) -> Reply:
        with self.telemetry.tracer.start_as_current_span("llm.invoke") as span:
            reply = self.llm.invoke(messages)
            if reply.usage:
                for key in ("input_tokens", "output_tokens"):
                    if key in reply.usage:
                        span.set_attribute(f"gen_ai.usage.{key}", reply.usage[key])
            return reply

    def _invoke_llm(self, messages: list[dict[str, Any]]) -> Reply:
        # The Azure SDK call is blocking, so run it on a worker thread. It runs in a
        # copy of the current context so the worker's span joins the trace.
        ctx = contextvars.copy_context()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(ctx.run, self._invoke_llm_sync, messages)
        try:
            # Catches both the deadline below and a TimeoutError raised by the SDK.
            return future.result(timeout=self.llm_timeout_s)
        except TimeoutError as exc:
            raise LLMTimeoutError("LLM invocation timed out") from exc
        finally:
            # Never wait for a hung call: the turn fails now, the worker is abandoned.
            pool.shutdown(wait=False, cancel_futures=True)

    def _dispatch_tool(self, call: dict[str, Any]) -> str:
        name = call["name"]
        with self.telemetry.tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", name)
            span.set_attribute(
                "tool.input", _truncate(json.dumps(call["args"], ensure_ascii=False))
            )
            succeeded = False
            try:
                result = self._tools[name](**call["args"])
                span.set_attribute("tool.output", _truncate(result))
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
                user_entry = {"role": "user", "content": user_message}

                # Nothing is written to the store until the turn succeeds: a failed
                # turn (e.g. LLM timeout) must not leave an orphan user message that
                # a client retry would then duplicate.
                reply = self._invoke_llm([*store.history(session_id), user_entry])

                # Keep every tool output: overwriting dropped all but the last one.
                parts = [reply.content] if reply.content else []
                parts.extend(self._dispatch_tool(call) for call in reply.tool_calls)
                text = "\n".join(parts)

                store.commit_turn(session_id, user_entry, {"role": "assistant", "content": text})
                return TurnResult(session_id=session_id, reply=text)
