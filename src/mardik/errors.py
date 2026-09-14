"""Domain-level exceptions for the Mardik agent."""

from __future__ import annotations


class MardikError(Exception):
    """Base class for Mardik runtime failures."""


class LLMTimeoutError(MardikError):
    """Raised when an LLM invocation exceeds its deadline."""
