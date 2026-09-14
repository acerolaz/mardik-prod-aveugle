"""Tools the agent can call to answer support questions."""

from __future__ import annotations

from typing import Callable

# Canned back-office data. In production these would hit the order service.
_ORDER_STATUS = {
    "1042": "expédiée, livraison prévue demain",
    "2098": "en préparation",
    "3157": "retournée, remboursement en cours",
}


def lookup_order(order_id: str) -> str:
    """Return a human-readable delivery status for an order id."""
    status = _ORDER_STATUS.get(str(order_id), "introuvable")
    return f"Commande #{order_id} : {status}."


def knowledge_base(topic: str) -> str:
    """Return a short canned help-center answer for a topic."""
    return f"Voir le centre d'aide pour : {topic}."


DEFAULT_TOOLS: dict[str, Callable[..., str]] = {
    "lookup_order": lookup_order,
    "knowledge_base": knowledge_base,
}
