import json
from pathlib import Path
from typing import Any

from portfolio_core import active_context


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
