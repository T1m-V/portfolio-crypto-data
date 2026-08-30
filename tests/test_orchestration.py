from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from portfolio_crypto_data import cli, update
from portfolio_crypto_data.chain_config import EvmChainConfig


def _chain_config(
    chain: str,
    *,
    native_symbol: str,
    protocols: tuple[str, ...] = (),
) -> EvmChainConfig:
    return EvmChainConfig(
        chain=chain,
        chain_id="1",
        wallet_address="0xwallet",
        rpc_url="https://rpc.test",
        explorer_api_url="https://explorer.test/api",
        explorer_api_key="",
        native_symbol=native_symbol,
        native_decimals=18,
        protocols=protocols,
        explorer_include_chain_id=True,
    )


def test_update_runs_each_configured_chain_with_its_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    arbitrum = _chain_config("arbitrum", native_symbol="ETH", protocols=("one",))
    cronos = _chain_config("cronos", native_symbol="CRO")

    async def retrieve(*, config: EvmChainConfig) -> None:
        calls.append(f"transactions:{config.chain}:{config.native_symbol}")

    monkeypatch.setattr(update, "retrieve_transactions", retrieve)
    monkeypatch.setattr(
        update,
        "_build_snapshots",
        lambda *, config: calls.append(f"snapshots:{config.chain}"),
    )
    monkeypatch.setattr(
        update,
        "PROTOCOL_PROCESSORS",
        {"one": lambda **kwargs: calls.append(f"protocol:{kwargs['chain']}:one")},
    )
    monkeypatch.setattr(
        update,
        "generate_protocol_lp_price_files",
        lambda **kwargs: calls.append(f"lp_prices:{kwargs['chain']}"),
    )
    monkeypatch.setattr(
        update,
        "build_accounting_artifacts",
        lambda **kwargs: calls.append(f"accounting:{kwargs['chain']}"),
    )
    monkeypatch.setattr(
        update,
        "build_chain_dashboard_artifacts",
        lambda **kwargs: calls.append(f"dashboard:{kwargs['chain']}"),
    )

    update.update_onchain(chain_configs=(arbitrum, cronos))

    assert calls == [
        "transactions:arbitrum:ETH",
        "snapshots:arbitrum",
        "protocol:arbitrum:one",
        "lp_prices:arbitrum",
        "accounting:arbitrum",
        "dashboard:arbitrum",
        "transactions:cronos:CRO",
        "snapshots:cronos",
        "accounting:cronos",
        "dashboard:cronos",
    ]


def test_rebuild_is_local_for_every_configured_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    configs = (
        _chain_config("arbitrum", native_symbol="ETH"),
        _chain_config("cronos", native_symbol="CRO"),
    )
    monkeypatch.setattr(
        update,
        "_build_snapshots",
        lambda *, config: calls.append(f"snapshots:{config.chain}"),
    )
    monkeypatch.setattr(
        update,
        "build_accounting_artifacts",
        lambda **kwargs: calls.append((kwargs["chain"], kwargs["as_of_date"])),
    )
    monkeypatch.setattr(
        update,
        "build_chain_dashboard_artifacts",
        lambda **kwargs: calls.append(f"dashboard:{kwargs['chain']}"),
    )

    update.rebuild_derived(
        as_of_date=date(2026, 8, 8),
        chain_configs=configs,
    )

    assert calls == [
        "snapshots:arbitrum",
        ("arbitrum", date(2026, 8, 8)),
        "dashboard:arbitrum",
        "snapshots:cronos",
        ("cronos", date(2026, 8, 8)),
        "dashboard:cronos",
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("update", ["update", "nexo", "crypto_com_app"]),
        ("rebuild", ["rebuild"]),
    ],
)
def test_cli_preserves_dashboard_command_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    expected: list[str],
) -> None:
    calls: list[str] = []
    (tmp_path / "portfolio.toml").write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(update, "update_onchain", lambda: calls.append("update"))
    monkeypatch.setattr(update, "rebuild_derived", lambda: calls.append("rebuild"))
    monkeypatch.setattr(cli, "_refresh_nexo", lambda **_: calls.append("nexo"))
    monkeypatch.setattr(
        cli,
        "_refresh_crypto_com_app",
        lambda **_: calls.append("crypto_com_app"),
    )

    result = cli.main(["--data-dir", str(tmp_path), command])

    assert result == 0
    assert calls == expected
    assert not (tmp_path / "runtime" / "mutation.lock").exists()


def test_rebuild_requires_transactions() -> None:
    config = _chain_config("cronos", native_symbol="CRO")
    with pytest.raises(FileNotFoundError, match="missing transaction file"):
        update.rebuild_derived(chain_configs=(config,))


def test_update_rejects_unsupported_protocol_before_running_pipeline() -> None:
    config = _chain_config(
        "cronos",
        native_symbol="CRO",
        protocols=("unknown",),
    )

    with pytest.raises(ValueError, match="unsupported protocol.*unknown"):
        update.update_onchain(chain_configs=(config,))
