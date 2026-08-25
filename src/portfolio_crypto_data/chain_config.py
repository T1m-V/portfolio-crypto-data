from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portfolio_core import active_context

from portfolio_crypto_data.symbols import sanitize_symbol

DEFAULT_PROTOCOLS = ("beefy", "balancer", "curve", "aave", "liquid_staking")


@dataclass(frozen=True, slots=True)
class EvmChainConfig:
    chain: str
    chain_id: str
    wallet_address: str
    rpc_url: str
    explorer_api_url: str
    explorer_api_key: str
    native_symbol: str
    native_decimals: int
    protocols: tuple[str, ...]
    explorer_include_chain_id: bool


def _required_text(*, raw: dict[str, Any], key: str, chain: str) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise ValueError(f"Chain '{chain}' is missing required config field '{key}'.")
    return value


def _parse_protocols(*, raw: dict[str, Any], chain: str) -> tuple[str, ...]:
    configured = raw.get("protocols", DEFAULT_PROTOCOLS)
    if not isinstance(configured, (list, tuple)):
        raise ValueError(f"Chain '{chain}' config field 'protocols' must be a list.")

    protocols: list[str] = []
    for value in configured:
        protocol = sanitize_symbol(value).lower()
        if not protocol:
            raise ValueError(f"Chain '{chain}' has an invalid protocol name.")
        if protocol not in protocols:
            protocols.append(protocol)
    return tuple(protocols)


def _optional_bool(*, raw: dict[str, Any], key: str, chain: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Chain '{chain}' config field '{key}' must be a boolean.")
    return value


def parse_evm_chain_config(*, chain: str, raw: dict[str, Any]) -> EvmChainConfig:
    native_symbol = sanitize_symbol(raw.get("native_symbol"))
    if not native_symbol:
        raise ValueError(
            f"Chain '{chain}' is missing required config field 'native_symbol'."
        )

    try:
        native_decimals = int(raw.get("native_decimals", 18))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Chain '{chain}' config field 'native_decimals' must be an integer."
        ) from exc
    if native_decimals < 0:
        raise ValueError(
            f"Chain '{chain}' config field 'native_decimals' cannot be negative."
        )

    rpc_url = str(raw.get("rpc_url") or raw.get("alchemy_url") or "").strip()
    if not rpc_url:
        raise ValueError(
            f"Chain '{chain}' is missing required config field 'rpc_url'."
        )

    return EvmChainConfig(
        chain=chain,
        chain_id=_required_text(raw=raw, key="chain_id", chain=chain),
        wallet_address=_required_text(raw=raw, key="my_address", chain=chain).lower(),
        rpc_url=rpc_url,
        explorer_api_url=_required_text(raw=raw, key="api_url", chain=chain),
        explorer_api_key=str(raw.get("api_key") or "").strip(),
        native_symbol=native_symbol,
        native_decimals=native_decimals,
        protocols=_parse_protocols(raw=raw, chain=chain),
        explorer_include_chain_id=_optional_bool(
            raw=raw,
            key="explorer_include_chain_id",
            chain=chain,
            default=True,
        ),
    )


def load_evm_chain_configs(config_path: Path | None = None) -> tuple[EvmChainConfig, ...]:
    path = config_path or active_context().paths.chain_config
    if not path.exists():
        raise FileNotFoundError(f"Config '{path}' not found.")

    with open(path, encoding="utf-8") as file:
        config_data = json.load(file)
    if not isinstance(config_data, dict):
        raise ValueError("Chain configuration must be a JSON object.")

    configs: list[EvmChainConfig] = []
    for raw_chain in sorted(config_data):
        raw = config_data[raw_chain]
        if not isinstance(raw, dict):
            raise ValueError(f"Chain '{raw_chain}' configuration must be an object.")
        enabled = _optional_bool(raw=raw, key="enabled", chain=raw_chain, default=True)
        if not enabled:
            continue
        chain = sanitize_symbol(raw_chain).lower()
        if not chain:
            raise ValueError("Chain configuration contains an invalid chain name.")
        if any(config.chain == chain for config in configs):
            raise ValueError(f"Chain '{chain}' is configured more than once.")
        configs.append(parse_evm_chain_config(chain=chain, raw=raw))

    if not configs:
        raise ValueError("No enabled EVM chains are configured.")
    return tuple(configs)


def load_evm_chain_config(chain: str) -> EvmChainConfig:
    normalized = sanitize_symbol(chain).lower()
    for config in load_evm_chain_configs():
        if config.chain == normalized:
            return config
    raise ValueError(f"Chain '{normalized}' is not enabled in config.")
