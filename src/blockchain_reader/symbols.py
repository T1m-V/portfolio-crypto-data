import re
from typing import Any

_SYMBOL_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def sanitize_symbol(symbol: str | None) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""

    ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
    return _SYMBOL_SANITIZE_PATTERN.sub("", ascii_text).strip()


def build_symbol_family_map(token_metadata: dict[str, dict[str, Any]]) -> dict[str, str]:
    symbol_family: dict[str, str] = {}
    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue

        symbol = sanitize_symbol(meta.get("symbol"))
        if not symbol:
            continue

        family = sanitize_symbol(meta.get("family")) or symbol
        symbol_family[symbol] = family

    return symbol_family


def canonicalize_symbol(symbol: str | None, symbol_family: dict[str, str]) -> str:
    normalized = sanitize_symbol(symbol)
    if not normalized:
        return ""
    return symbol_family.get(normalized, normalized)


def build_address_symbol_map(
    token_metadata: dict[str, dict[str, Any]], symbol_family: dict[str, str]
) -> dict[str, str]:
    by_address: dict[str, str] = {}
    for address, meta in token_metadata.items():
        if not isinstance(meta, dict):
            continue

        canonical = canonicalize_symbol(meta.get("symbol"), symbol_family=symbol_family)
        if canonical:
            by_address[str(address).lower()] = canonical
    return by_address


def build_known_canonical_symbols(
    token_metadata: dict[str, dict[str, Any]], symbol_family: dict[str, str]
) -> set[str]:
    known: set[str] = set()
    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue

        symbol = sanitize_symbol(meta.get("symbol"))
        family = sanitize_symbol(meta.get("family"))
        if symbol:
            known.add(symbol)
            known.add(symbol_family.get(symbol, symbol))
        if family:
            known.add(family)

    return known
