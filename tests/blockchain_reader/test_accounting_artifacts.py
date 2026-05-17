from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import blockchain_reader.accounting as accounting
import blockchain_reader.shared.prices as shared_prices


def _patch_accounting_paths(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "accounting": tmp_path / "accounting",
        "snapshots": tmp_path / "snapshots",
        "protocol": tmp_path / "protocol",
        "tokens": tmp_path / "tokens",
        "prices": tmp_path / "prices",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(accounting, "BLOCKCHAIN_ACCOUNTING_FOLDER", paths["accounting"])
    monkeypatch.setattr(accounting, "BLOCKCHAIN_SNAPSHOT_FOLDER", paths["snapshots"])
    monkeypatch.setattr(accounting, "PROTOCOL_UNDERLYING_TOKEN_FOLDER", paths["protocol"])
    monkeypatch.setattr(accounting, "TOKENS_FOLDER", paths["tokens"])
    monkeypatch.setattr(accounting, "PRICES_FOLDER", paths["prices"])
    return paths


def _write_principal_daily(paths: dict[str, Path], rows: list[dict[str, object]]) -> None:
    root = paths["accounting"] / "arbitrum"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Date", "Coin", "PrincipalInvestedEUR"]).to_csv(
        root / "principal_daily.csv",
        index=False,
    )


def test_accounting_canonicalizes_btc_wrappers_to_base_asset(monkeypatch, tmp_path) -> None:
    paths = _patch_accounting_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)
    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xbtc": {"symbol": "BTC"},
                "0xwbtc": {"symbol": "WBTC", "family": "BTC"},
                "0xtbtc": {"symbol": "tBTC", "family": "BTC"},
                "0xrenbtc": {"symbol": "renBTC", "family": "BTC"},
            }
        )
    )
    pd.DataFrame([{"Date": "2025-01-01", "Price": 40000.0}]).to_csv(
        paths["prices"] / "BTC.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "Coin": coin,
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            }
            for coin in ("WBTC", "tBTC", "renBTC")
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [{"Date": "2025-01-01", "Coin": "BTC", "PrincipalInvestedEUR": 120000.0}],
    )

    result = accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-01-01")

    assert result.errors == []
    base = pd.read_csv(result.paths.base_daily)
    assert base["Coin"].tolist() == ["BTC"]
    assert base.iloc[0]["Quantity"] == 3.0
    assert base.iloc[0]["MarketValueEUR"] == 120000.0
    assert base.iloc[0]["PrincipalInvestedEUR"] == 120000.0


def test_accounting_missing_material_price_creates_blocking_issue(monkeypatch, tmp_path) -> None:
    paths = _patch_accounting_paths(monkeypatch, tmp_path)
    (paths["tokens"] / "arbitrum_tokens.json").write_text(json.dumps({"0xaaa": {"symbol": "AAA"}}))
    pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "Coin": "AAA",
                "Quantity": 2.0,
                "Principal Invested": 10.0,
            }
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)

    result = accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-01-01")

    assert len(result.errors) == 1
    issues = pd.read_csv(result.paths.issues)
    assert issues.iloc[0]["Reason"] == "known_symbol_missing_price"


def test_accounting_combines_aave_wsteth_quantity_with_eth_principal_proxy(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_accounting_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)
    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xeth": {"symbol": "ETH"},
                "0xwsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
                "0xaavewsteth": {"symbol": "aArbwstETH", "protocol": "aave"},
            }
        )
    )
    pd.DataFrame([{"Date": "2025-03-15", "Price": 1000.0}]).to_csv(
        paths["prices"] / "ETH.csv",
        index=False,
    )
    liquid_staking = paths["protocol"] / "liquid_staking"
    liquid_staking.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2025-03-15", "asset_ETH": 1.0}]).to_csv(
        liquid_staking / "arbitrum_wstETH.csv",
        index=False,
    )
    aave = paths["protocol"] / "aave"
    aave.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2025-03-15", "net_wstETH": 1.0}]).to_csv(
        aave / "arbitrum_aave_daily_exposure.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Date": "2025-03-15",
                "Coin": "ETH",
                "Quantity": 0.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2025-03-15",
                "Coin": "aArbwstETH",
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            },
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [{"Date": "2025-03-15", "Coin": "ETH", "PrincipalInvestedEUR": 1000.0}],
    )

    result = accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-03-15")

    assert result.errors == []
    base = pd.read_csv(result.paths.base_daily)
    eth = base[base["Coin"] == "ETH"].iloc[0]
    assert eth["Quantity"] == 1.0
    assert eth["MarketValueEUR"] == 1000.0
    assert eth["PrincipalInvestedEUR"] == 1000.0
    assert eth["ProfitLossEUR"] == 0.0
    assert bool(eth["HasAaveExposure"])

    source = pd.read_csv(result.paths.source_base_daily)
    aave_eth = source[(source["Source"] == "Aave") & (source["BaseCoin"] == "ETH")].iloc[0]
    assert aave_eth["MarketValueEUR"] == 1000.0
    assert aave_eth["PrincipalInvestedEUR"] == 1000.0


def test_accounting_uses_future_single_family_composition_for_base_identity(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_accounting_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)
    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xeth": {"symbol": "ETH"},
                "0xwsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
            }
        )
    )
    pd.DataFrame([{"Date": "2025-01-01", "Price": 1000.0}]).to_csv(
        paths["prices"] / "ETH.csv",
        index=False,
    )
    liquid_staking = paths["protocol"] / "liquid_staking"
    liquid_staking.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": "2025-01-02", "asset_ETH": 1.1}]).to_csv(
        liquid_staking / "arbitrum_wstETH.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "Coin": "wstETH",
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            },
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [{"Date": "2025-01-01", "Coin": "ETH", "PrincipalInvestedEUR": 1100.0}],
    )

    result = accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-01-01")

    assert result.errors == []
    base = pd.read_csv(result.paths.base_daily)
    assert base["Coin"].tolist() == ["ETH"]
    assert base.iloc[0]["Quantity"] == 1.1
    assert base.iloc[0]["MarketValueEUR"] == 1100.0
    assert base.iloc[0]["PrincipalInvestedEUR"] == 1100.0


def test_accounting_nested_eth_wrapper_to_aave_does_not_create_pnl_cliff(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_accounting_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)
    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xeth": {"symbol": "ETH"},
                "0xweth": {"symbol": "WETH", "family": "ETH"},
                "0xwsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
                "0xbpt": {
                    "symbol": "wstETH-WETH-BPT",
                    "family": "ETH",
                    "protocol": "balancer",
                },
                "0xmoo": {
                    "symbol": "mooBalancerArbwstETH-ETHV3",
                    "family": "ETH",
                    "protocol": "beefy",
                },
                "0xaavewsteth": {"symbol": "aArbwstETH", "protocol": "aave"},
            }
        )
    )
    pd.DataFrame(
        [
            {"Date": "2024-12-17", "Price": 1000.0},
            {"Date": "2024-12-18", "Price": 1000.0},
        ]
    ).to_csv(paths["prices"] / "ETH.csv", index=False)
    liquid_staking = paths["protocol"] / "liquid_staking"
    liquid_staking.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-12-17", "asset_ETH": 1.0},
            {"date": "2024-12-18", "asset_ETH": 1.0},
        ]
    ).to_csv(liquid_staking / "arbitrum_wstETH.csv", index=False)
    balancer = paths["protocol"] / "balancer"
    balancer.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-12-17", "asset_WETH": 0.5, "asset_wstETH": 0.5},
            {"date": "2024-12-18", "asset_WETH": 0.5, "asset_wstETH": 0.5},
        ]
    ).to_csv(balancer / "arbitrum_wstETH-WETH-BPT.csv", index=False)
    beefy = paths["protocol"] / "beefy"
    beefy.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-12-17", "asset_wstETH-WETH-BPT": 1.0},
            {"date": "2024-12-18", "asset_wstETH-WETH-BPT": 1.0},
        ]
    ).to_csv(beefy / "arbitrum_mooBalancerArbwstETH-ETHV3.csv", index=False)
    aave = paths["protocol"] / "aave"
    aave.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-12-17", "net_wstETH": 0.0},
            {"date": "2024-12-18", "net_wstETH": 2.0},
        ]
    ).to_csv(aave / "arbitrum_aave_daily_exposure.csv", index=False)
    pd.DataFrame(
        [
            {
                "Date": "2024-12-17",
                "Coin": "ETH",
                "Quantity": 0.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2024-12-17",
                "Coin": "mooBalancerArbwstETH-ETHV3",
                "Quantity": 2.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2024-12-18",
                "Coin": "ETH",
                "Quantity": 0.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2024-12-18",
                "Coin": "mooBalancerArbwstETH-ETHV3",
                "Quantity": 0.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2024-12-18",
                "Coin": "aArbwstETH",
                "Quantity": 2.0,
                "Principal Invested": 0.0,
            },
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [
            {"Date": "2024-12-17", "Coin": "ETH", "PrincipalInvestedEUR": 2000.0},
            {"Date": "2024-12-18", "Coin": "ETH", "PrincipalInvestedEUR": 2000.0},
        ],
    )

    result = accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2024-12-18")

    assert result.errors == []
    base = pd.read_csv(result.paths.base_daily)
    eth = base[base["Coin"] == "ETH"].sort_values("Date")
    assert eth["MarketValueEUR"].tolist() == [2000.0, 2000.0]
    assert eth["PrincipalInvestedEUR"].tolist() == [2000.0, 2000.0]
    assert eth["ProfitLossEUR"].tolist() == [0.0, 0.0]
