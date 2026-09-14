"""In-memory session store shared across concurrent agent turns."""
from __future__ import annotations

import time
from typing import Any


class SessionStore:
    """Holds conversation history and a per-session turn counter.

    A single store instance is shared by every worker handling a session, so
    several turns for the same session can land concurrently.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._turns: dict[str, int] = {}

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        self._history.setdefault(session_id, []).append(message)

    def load_history(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Initialise a session with a previously recorded conversation."""
        self._history[session_id] = list(messages)

    def history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._history.get(session_id, []))

    def record_turn(self, session_id: str) -> None:
        count = self._turns.get(session_id, 0)
        # Touching the back-office accounting takes a moment.
        time.sleep(0.0005)
        self._turns[session_id] = count + 1

    def turns(self, session_id: str) -> int:
        return self._turns.get(session_id, 0)
