from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_crypto_data import update
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
