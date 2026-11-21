from __future__ import annotations

import os
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


def test_build_snapshots_fails_fast_when_direct_token_lacks_price(
    monkeypatch: pytest.MonkeyPatch,
    isolate_portfolio_io,
) -> None:
    import pandas as pd

    config = _chain_config("arbitrum", native_symbol="ETH", protocols=("beefy",))
    tx_path = isolate_portfolio_io.paths.crypto_transactions / "arbitrum_transactions.csv"
    tx_path.write_text("Date,Action,Coin,Quantity\n", encoding="utf-8")

    def fake_generate_raw_snapshots(*, output_csv: Path, **_) -> None:
        pd.DataFrame(
            {
                "Date": ["2025-10-06 00:00:00"],
                "Coin": ["MAGIC"],
                "Quantity": [1.0],
                "Principal Invested": [0.0],
            }
        ).to_csv(output_csv, index=False)

    monkeypatch.setattr(update, "generate_raw_snapshots", fake_generate_raw_snapshots)

    with pytest.raises(
        ValueError,
        match="Missing EUR price for direct token 'MAGIC' on 2025-10-06",
    ):
        update._build_snapshots(config=config)


def test_update_chain_rebuilds_snapshots_after_refreshing_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    config = _chain_config("cronos", native_symbol="CRO", protocols=("liquid_staking",))

    async def fake_retrieve(*, config: EvmChainConfig) -> None:
        calls.append(f"tx:{config.chain}")

    monkeypatch.setattr(update, "retrieve_transactions", fake_retrieve)
    monkeypatch.setattr(
        update, "_build_snapshots", lambda *, config: calls.append(f"snap:{config.chain}")
    )
    monkeypatch.setattr(
        update, "sync_block_map", lambda *, chain, config: calls.append(f"bmap:{chain}")
    )
    monkeypatch.setattr(
        update, "_refresh_protocols", lambda *, config, logger: calls.append(f"prot:{config.chain}")
    )
    monkeypatch.setattr(
        update,
        "build_accounting_artifacts",
        lambda *, chain, **_: calls.append(f"acct:{chain}"),
    )
    monkeypatch.setattr(
        update,
        "build_chain_dashboard_artifacts",
        lambda *, chain: calls.append(f"dash:{chain}"),
    )

    update._update_chain(config=config, logger=update.PipelineLogger())

    assert calls == [
        "tx:cronos",
        "snap:cronos",
        "bmap:cronos",
        "prot:cronos",
        "snap:cronos",
        "acct:cronos",
        "dash:cronos",
    ]


def test_rebuild_derived_incremental_skips_up_to_date_snapshots_and_forwards_from_date(
    monkeypatch: pytest.MonkeyPatch,
    isolate_portfolio_io,
) -> None:
    config = _chain_config("arbitrum", native_symbol="ETH")
    paths = isolate_portfolio_io.paths
    tx_path = paths.crypto_transactions / "arbitrum_transactions.csv"
    snap_path = paths.crypto_snapshots / "arbitrum_raw_snapshots.csv"
    acct_dir = paths.accounting / "arbitrum"
    acct_dir.mkdir(parents=True, exist_ok=True)
    events_path = acct_dir / "principal_events.csv"
    daily_path = acct_dir / "principal_daily.csv"

    tx_path.write_text("Date\n", encoding="utf-8")
    os.utime(tx_path, (1_000_000, 1_000_000))
    for output_path in (snap_path, events_path, daily_path):
        output_path.write_text("Date\n", encoding="utf-8")
        os.utime(output_path, (1_000_010, 1_000_010))

    snapshot_calls: list[str] = []
    accounting_calls: list[dict[str, object]] = []
    dashboard_calls: list[str] = []

    monkeypatch.setattr(
        update, "_build_snapshots", lambda *, config: snapshot_calls.append(config.chain)
    )
    monkeypatch.setattr(
        update, "build_accounting_artifacts", lambda **kwargs: accounting_calls.append(kwargs)
    )
    monkeypatch.setattr(
        update,
        "build_chain_dashboard_artifacts",
        lambda *, chain: dashboard_calls.append(chain),
    )

    update.rebuild_derived(
        as_of_date=date(2025, 1, 3),
        chain_configs=(config,),
        incremental=True,
        from_date="2025-01-02",
    )

    assert snapshot_calls == []
    assert accounting_calls == [
        {
            "chain": "arbitrum",
            "as_of_date": date(2025, 1, 3),
            "incremental": True,
            "from_date": "2025-01-02",
        }
    ]
    assert dashboard_calls == ["arbitrum"]

    os.utime(tx_path, (1_000_020, 1_000_020))
    update.rebuild_derived(
        as_of_date=date(2025, 1, 3),
        chain_configs=(config,),
        incremental=True,
    )
    assert snapshot_calls == ["arbitrum"]


def test_cli_rebuild_forwards_incremental_and_from_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "portfolio.toml").write_text("schema_version = 1\n", encoding="utf-8")
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        update,
        "rebuild_derived",
        lambda **kwargs: captured.append(kwargs),
    )

    exit_code = cli.main(
        ["--data-dir", str(tmp_path), "rebuild", "--incremental", "--from-date", "2025-01-02"]
    )

    assert exit_code == 0
    assert captured == [
        {
            "chain_configs": None,
            "incremental": True,
            "from_date": date(2025, 1, 2),
        }
    ]
