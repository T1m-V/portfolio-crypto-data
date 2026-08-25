from __future__ import annotations

from typing import Any

from portfolio_crypto_data.symbols import price_proxy_symbol, sanitize_symbol

AAVE_DEBT_PREFIXES = ("variabledebt", "stabledebt")


def is_aave_debt_symbol(symbol: str, meta: dict[str, Any] | None = None) -> bool:
    if meta:
        position_type = sanitize_symbol(
            meta.get("position_type") or meta.get("aave_position")
        ).lower()
        if position_type:
            return position_type in {"debt", "borrowed"}
    normalized = sanitize_symbol(symbol).lower()
    return normalized.startswith(AAVE_DEBT_PREFIXES)


def aave_base_symbol(symbol: str, meta: dict[str, Any] | None = None) -> str:
    explicit = ""
    if meta:
        explicit = sanitize_symbol(meta.get("price_source")) or sanitize_symbol(
            meta.get("family")
        )
    base = explicit or sanitize_symbol(symbol)
    return price_proxy_symbol(base) or base
