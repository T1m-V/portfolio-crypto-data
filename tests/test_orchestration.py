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
