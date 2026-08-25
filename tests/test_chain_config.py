from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_crypto_data.chain_config import DEFAULT_PROTOCOLS, load_evm_chain_configs
from portfolio_crypto_data.extraction.token_manager import TokenManager


def _base_config(*, native_symbol: str) -> dict[str, object]:
    return {
        "chain_id": "25",
        "my_address": "0xABC",
        "rpc_url": "https://rpc.test",
        "api_url": "https://explorer.test/api",
        "api_key": "test-key",
        "native_symbol": native_symbol,
        "native_decimals": 18,
        "protocols": [],
    }


def test_chain_configs_are_typed_sorted_and_capability_driven(tmp_path: Path) -> None:
    path = tmp_path / "chains.json"
    path.write_text(
        json.dumps(
            {
                "cronos": {
                    **_base_config(native_symbol="CRO"),
                    "explorer_include_chain_id": False,
                },
                "arbitrum": {
                    **_base_config(native_symbol="ETH"),
                    "chain_id": "42161",
                    "protocols": ["aave", "beefy", "aave"],
                },
                "disabled": {
                    **_base_config(native_symbol="TEST"),
                    "enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    configs = load_evm_chain_configs(config_path=path)

    assert [config.chain for config in configs] == ["arbitrum", "cronos"]
    assert configs[0].protocols == ("aave", "beefy")
    assert configs[1].native_symbol == "CRO"
    assert not configs[1].explorer_include_chain_id


def test_chain_config_requires_an_explicit_native_symbol(tmp_path: Path) -> None:
    path = tmp_path / "chains.json"
    raw = _base_config(native_symbol="CRO")
    raw.pop("native_symbol")
    path.write_text(json.dumps({"cronos": raw}), encoding="utf-8")

    with pytest.raises(ValueError, match="native_symbol"):
        load_evm_chain_configs(config_path=path)


def test_chain_config_preserves_existing_protocol_defaults(tmp_path: Path) -> None:
    path = tmp_path / "chains.json"
    raw = _base_config(native_symbol="ETH")
    raw.pop("protocols")
    path.write_text(json.dumps({"existing_chain": raw}), encoding="utf-8")

    (config,) = load_evm_chain_configs(config_path=path)

    assert config.protocols == DEFAULT_PROTOCOLS


def test_token_manager_uses_chain_native_metadata(tmp_path: Path) -> None:
    token_path = tmp_path / "cronos_tokens.json"
    token_path.write_text(
        json.dumps({"native": {"symbol": "ETH", "decimals": 18}}),
        encoding="utf-8",
    )

    manager = TokenManager(
        token_path=token_path,
        w3=None,
        native_symbol="CRO",
        native_decimals=18,
    )

    assert manager.cache["native"] == {
        "symbol": "CRO",
        "decimals": 18,
        "resolved": True,
    }
