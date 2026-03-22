from __future__ import annotations

from enum import StrEnum
from typing import Any

from blockchain_reader.symbols import sanitize_symbol

PROTOCOL_DERIVED_PROTOCOLS: frozenset[str] = frozenset(
    {
        "aura",
        "balancer",
        "beefy",
        "curve",
        "liquid_staking",
    }
)


class ValuationRoute(StrEnum):
    DIRECT = "DIRECT"
    AAVE = "AAVE"
    PROTOCOL_DERIVED = "PROTOCOL_DERIVED"


def build_symbol_protocol_map(token_metadata: dict[str, dict[str, Any]]) -> dict[str, str]:
    """
    Builds a symbol -> protocol map from token metadata.

    args:
        token_metadata: Address-keyed token metadata.

    returns:
        Symbol-keyed protocol map with lowercase protocol names.
    """
    mapping: dict[str, str] = {}
    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue

        symbol = sanitize_symbol(meta.get("symbol"))
        if not symbol or symbol in mapping:
            continue

        protocol = sanitize_symbol(meta.get("protocol")).lower()
        mapping[symbol] = protocol
    return mapping


def classify_valuation_route(
    *,
    symbol: str,
    symbol_protocol: dict[str, str] | None = None,
    protocol_derived_symbols: set[str] | None = None,
) -> ValuationRoute:
    """
    Classifies where valuation should be sourced for a symbol.

    args:
        symbol: Raw symbol.
        symbol_protocol: Optional symbol -> protocol mapping.
        protocol_derived_symbols: Optional set of symbols with protocol-underlying rows.

    returns:
        Valuation ownership route for the symbol.
    """
    normalized = sanitize_symbol(symbol)
    if not normalized:
        return ValuationRoute.DIRECT

    protocol_map = symbol_protocol or {}
    protocol = sanitize_symbol(protocol_map.get(normalized)).lower()
    if protocol == "aave" or normalized.lower().startswith("variabledebt"):
        return ValuationRoute.AAVE

    protocol_symbols = protocol_derived_symbols or set()
    if normalized in protocol_symbols:
        return ValuationRoute.PROTOCOL_DERIVED
    if protocol in PROTOCOL_DERIVED_PROTOCOLS:
        return ValuationRoute.PROTOCOL_DERIVED

    return ValuationRoute.DIRECT
