import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from portfolio_core import active_context

from portfolio_crypto_data.symbols import sanitize_symbol

ERC20_ABI = [
    {
        "name": name,
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": output_type}],
    }
    for name, output_type in (("symbol", "string"), ("decimals", "uint8"))
]


def load_token_metadata(
    *,
    chain: str,
    tokens_folder: Path | None = None,
) -> dict[str, dict[str, Any]]:
    root = tokens_folder or active_context().paths.tokens
    token_path = root / f"{chain}_tokens.json"
    if not token_path.exists():
        return {}

    with open(token_path, "r") as f:
        raw = json.load(f)
    return {str(addr).lower(): meta for addr, meta in raw.items() if isinstance(meta, dict)}


def metadata_by_symbol(
    token_metadata: dict[str, dict[str, Any]],
    key: Callable[[object], str] = sanitize_symbol,
) -> dict[str, dict[str, Any]]:
    """Index token metadata by symbol; the first address wins for duplicate symbols."""
    by_symbol: dict[str, dict[str, Any]] = {}
    for meta in token_metadata.values():
        symbol = key(meta.get("symbol"))
        if symbol:
            by_symbol.setdefault(symbol, meta)
    return by_symbol
