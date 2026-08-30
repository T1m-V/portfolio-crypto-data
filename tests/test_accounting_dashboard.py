from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from portfolio_core import active_context

from portfolio_crypto_data import accounting, dashboard_artifacts

CHAIN = "arbitrum"


def _write_workspace(
    *,
    metadata: dict[str, dict[str, object]],
    snapshots: list[dict[str, object]],
    principal: list[dict[str, object]],
) -> None:
    paths = active_context().paths
    (paths.tokens / f"{CHAIN}_tokens.json").write_text(json.dumps(metadata))
    pd.DataFrame(snapshots).to_csv(
        paths.crypto_snapshots / f"{CHAIN}_raw_snapshots.csv", index=False
    )
    accounting_root = paths.accounting / CHAIN
    accounting_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        principal,
        columns=["Date", "Coin", "PrincipalInvestedEUR"],
    ).to_csv(accounting_root / "principal_daily.csv", index=False)


def _price(symbol: str, price: float, *dates: str) -> None:
    values = dates or ("2025-01-01",)
    pd.DataFrame([{"Date": day, "Price": price} for day in values]).to_csv(
        active_context().paths.prices / f"{symbol}.csv", index=False
    )


def _protocol(protocol: str, symbol: str, rows: list[dict[str, object]]) -> Path:
    folder = active_context().paths.protocol_underlying_tokens / protocol
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{CHAIN}_{symbol}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_accounting_collapses_btc_wrappers_and_allocates_principal() -> None:
    _write_workspace(
        metadata={
            "btc": {"symbol": "BTC"},
            "wbtc": {"symbol": "WBTC", "family": "BTC"},
            "tbtc": {"symbol": "tBTC", "family": "BTC"},
            "renbtc": {"symbol": "renBTC", "family": "BTC"},
        },
        snapshots=[
            {"Date": "2025-01-01", "Coin": coin, "Quantity": 1, "Principal Invested": 0}
            for coin in ("WBTC", "tBTC", "renBTC")
        ],
        principal=[
            {"Date": "2025-01-01", "Coin": "BTC", "PrincipalInvestedEUR": 120_000}
        ],
    )
    _price("BTC", 40_000)

    result = accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-01-01")
    row = pd.read_csv(result.paths.base_daily).iloc[0]

    assert row[["Coin", "Quantity", "MarketValueEUR", "PrincipalInvestedEUR"]].tolist() == [
        "BTC",
        3.0,
        120_000.0,
        120_000.0,
    ]


def test_accounting_fails_immediately_for_material_missing_price() -> None:
    _write_workspace(
        metadata={"aaa": {"symbol": "AAA"}},
        snapshots=[
            {"Date": "2025-01-01", "Coin": "AAA", "Quantity": 2, "Principal Invested": 0}
        ],
        principal=[],
    )

    with pytest.raises(ValueError, match="Missing EUR price for AAA"):
        accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-01-01")


def test_accounting_requires_protocol_state_for_aave_positions() -> None:
    _write_workspace(
        metadata={
            "debt": {
                "symbol": "variableDebtArbLINK",
                "protocol": "aave",
                "price_source": "LINK",
                "position_type": "debt",
            }
        },
        snapshots=[
            {
                "Date": "2025-01-01",
                "Coin": "variableDebtArbLINK",
                "Quantity": 1,
                "Principal Invested": 0,
            }
        ],
        principal=[],
    )

    with pytest.raises(ValueError, match="Aave positions require"):
        accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-01-01")


def test_accounting_expands_aave_liquid_staking_to_eth() -> None:
    _write_workspace(
        metadata={
            "eth": {"symbol": "ETH"},
            "wsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
            "aave": {
                "symbol": "aArbwstETH",
                "protocol": "aave",
                "price_source": "wstETH",
            },
        },
        snapshots=[
            {
                "Date": "2025-03-15",
                "Coin": "aArbwstETH",
                "Quantity": 1,
                "Principal Invested": 0,
            }
        ],
        principal=[
            {"Date": "2025-03-15", "Coin": "ETH", "PrincipalInvestedEUR": 1_000}
        ],
    )
    _price("ETH", 1_000, "2025-03-15")
    _protocol("liquid_staking", "wstETH", [{"date": "2025-03-15", "asset_ETH": 1}])
    _protocol("aave", "aave_daily_exposure", [{"date": "2025-03-15", "net_wstETH": 1}])

    result = accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-03-15")
    eth = pd.read_csv(result.paths.base_daily).set_index("Coin").loc["ETH"]

    columns = ["Quantity", "MarketValueEUR", "PrincipalInvestedEUR", "ProfitLossEUR"]
    assert eth[columns].tolist() == [
        1.0,
        1_000.0,
        1_000.0,
        0.0,
    ]
    assert bool(eth["HasAaveExposure"])


def test_dashboard_builder_preserves_six_file_read_contract() -> None:
    _write_workspace(
        metadata={
            "lp": {"symbol": "mooFishUSDT-USDC", "protocol": "beefy"},
            "usdc": {"symbol": "USDC"},
            "usdt": {"symbol": "USDT"},
        },
        snapshots=[
            {
                "Date": "2025-01-01",
                "Coin": "mooFishUSDT-USDC",
                "Quantity": 1,
                "Principal Invested": 0,
            }
        ],
        principal=[
            {"Date": "2025-01-01", "Coin": "USDC", "PrincipalInvestedEUR": 2_000},
            {"Date": "2025-01-01", "Coin": "USDT", "PrincipalInvestedEUR": 2_000},
        ],
    )
    _protocol(
        "beefy",
        "mooFishUSDT-USDC",
        [{"date": "2025-01-01", "asset_USDC": 2_000, "asset_USDT": 2_000}],
    )
    pd.DataFrame(
        [
            {
                "TX Hash": "hash",
                "Date": "01/01/2025 10:00:00",
                "Qty in": "1",
                "Token in": "mooFishUSDT-USDC",
                "Qty out": "4000",
                "Token out": "USDC",
                "Type": "Swap",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(active_context().paths.crypto_transactions / f"{CHAIN}_transactions.csv", index=False)
    accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-01-01")

    paths = dashboard_artifacts.build_chain_dashboard_artifacts(chain=CHAIN)

    contracts = {
        paths.asset_daily: dashboard_artifacts.ASSET_DAILY_COLUMNS,
        paths.timeseries_daily: dashboard_artifacts.TIMESERIES_DAILY_COLUMNS,
        paths.composition_daily: dashboard_artifacts.COMPOSITION_DAILY_COLUMNS,
        paths.source_daily: dashboard_artifacts.SOURCE_DAILY_COLUMNS,
        paths.transactions_dashboard: dashboard_artifacts.TRANSACTIONS_DASHBOARD_COLUMNS,
        paths.assets: dashboard_artifacts.ASSETS_COLUMNS,
    }
    assert all(pd.read_csv(path).columns.tolist() == columns for path, columns in contracts.items())

    timeseries = pd.read_csv(paths.timeseries_daily).set_index("Selection")
    assert timeseries.loc["ALL", ["MarketValueEUR", "PrincipalInvestedEUR"]].tolist() == [
        4_000.0,
        4_000.0,
    ]
    assert "mooFishUSDT-USDC" not in timeseries.index
    assert pd.read_csv(paths.assets)["Value"].tolist() == ["USDC", "USDT"]


def test_dashboard_timeseries_carries_realized_pnl_after_close() -> None:
    asset_daily = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2025-01-01"),
                "Selection": "BTC",
                "MarketValueEUR": 120,
                "PrincipalInvestedEUR": 100,
                "Quantity": 1,
            },
            {
                "Date": pd.Timestamp("2025-01-03"),
                "Selection": "ETH",
                "MarketValueEUR": 50,
                "PrincipalInvestedEUR": 40,
                "Quantity": 2,
            },
        ]
    )
    transactions = pd.DataFrame([{"Date": "2025-01-02 10:00:00", "AssetKeys": "BTC"}])

    result = dashboard_artifacts._build_timeseries_daily(
        asset_daily=asset_daily,
        transactions=transactions,
    )
    btc = result[result["Selection"] == "BTC"].sort_values("Date")

    assert btc["MarketValueEUR"].tolist() == [120.0, 0.0, 0.0]
    assert btc["PrincipalInvestedEUR"].tolist() == [100.0, 0.0, 0.0]
    assert btc["ProfitLossEUR"].tolist() == [20.0, 20.0, 20.0]
    assert btc["TxCount"].tolist() == [0, 1, 0]


def test_dashboard_debt_is_negative_exposure() -> None:
    _write_workspace(
        metadata={
            "link": {"symbol": "LINK"},
            "debt": {
                "symbol": "variableDebtArbLINK",
                "protocol": "aave",
                "price_source": "LINK",
                "position_type": "debt",
            },
        },
        snapshots=[
            {
                "Date": "2025-01-01",
                "Coin": "variableDebtArbLINK",
                "Quantity": 12.5,
                "Principal Invested": 0,
            }
        ],
        principal=[
            {"Date": "2025-01-01", "Coin": "LINK", "PrincipalInvestedEUR": -200}
        ],
    )
    _price("LINK", 16)
    _protocol(
        "aave",
        "aave_daily_exposure",
        [{"date": "2025-01-01", "net_LINK": -12.5}],
    )
    transaction_columns = [
        "TX Hash",
        "Date",
        "Qty in",
        "Token in",
        "Qty out",
        "Token out",
        "Type",
        "Fee",
        "Fee Token",
    ]
    pd.DataFrame(columns=transaction_columns).to_csv(
        active_context().paths.crypto_transactions / f"{CHAIN}_transactions.csv", index=False
    )
    accounting.build_accounting_artifacts(chain=CHAIN, as_of_date="2025-01-01")

    paths = dashboard_artifacts.build_chain_dashboard_artifacts(chain=CHAIN)
    link = pd.read_csv(paths.timeseries_daily).set_index("Selection").loc["LINK"]

    assert link[["MarketValueEUR", "PrincipalInvestedEUR", "ProfitLossEUR"]].tolist() == [
        -200.0,
        -200.0,
        0.0,
    ]
