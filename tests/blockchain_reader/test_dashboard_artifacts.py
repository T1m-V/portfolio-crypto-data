import json

import pandas as pd

import blockchain_reader.accounting as accounting
import blockchain_reader.dashboard_artifacts as dashboard_artifacts
import blockchain_reader.shared.prices as shared_prices


def _patch_artifact_paths(monkeypatch, tmp_path) -> dict[str, object]:
    paths = {
        "dashboard": tmp_path / "dashboard",
        "accounting": tmp_path / "accounting",
        "protocol": tmp_path / "protocol",
        "snapshots": tmp_path / "snapshots",
        "transactions": tmp_path / "transactions",
        "tokens": tmp_path / "tokens",
        "prices": tmp_path / "prices",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        dashboard_artifacts,
        "BLOCKCHAIN_DASHBOARD_FOLDER",
        paths["dashboard"],
    )
    monkeypatch.setattr(
        dashboard_artifacts,
        "BLOCKCHAIN_TRANSACTIONS_FOLDER",
        paths["transactions"],
    )
    monkeypatch.setattr(dashboard_artifacts, "TOKENS_FOLDER", paths["tokens"])
    monkeypatch.setattr(accounting, "BLOCKCHAIN_ACCOUNTING_FOLDER", paths["accounting"])
    monkeypatch.setattr(accounting, "PROTOCOL_UNDERLYING_TOKEN_FOLDER", paths["protocol"])
    monkeypatch.setattr(accounting, "BLOCKCHAIN_SNAPSHOT_FOLDER", paths["snapshots"])
    monkeypatch.setattr(accounting, "TOKENS_FOLDER", paths["tokens"])
    monkeypatch.setattr(accounting, "PRICES_FOLDER", paths["prices"])
    return paths


def _write_principal_daily(paths: dict[str, object], rows: list[dict[str, object]]) -> None:
    root = paths["accounting"] / "arbitrum"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Date", "Coin", "PrincipalInvestedEUR"]).to_csv(
        root / "principal_daily.csv",
        index=False,
    )


def test_build_arbitrum_dashboard_artifacts_writes_schema_and_splits_lp_principal(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_artifact_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)

    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xmoo": {"symbol": "mooFishUSDT-USDC", "protocol": "beefy"},
                "0xusdc": {"symbol": "USDC"},
                "0xusdt": {"symbol": "USDT"},
            }
        )
    )
    pd.DataFrame(
        [
            {
                "date": "2022-12-01",
                "asset_USDC": 2000.0,
                "asset_USDT": 2000.0,
            }
        ]
    ).to_csv(paths["protocol"] / "arbitrum_mooFishUSDT-USDC.csv", index=False)
    pd.DataFrame(
        [
            {
                "Date": "2022-12-01",
                "Coin": "mooFishUSDT-USDC",
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            }
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [
            {"Date": "2022-12-01", "Coin": "USDC", "PrincipalInvestedEUR": 2000.0},
            {"Date": "2022-12-01", "Coin": "USDT", "PrincipalInvestedEUR": 2000.0},
        ],
    )
    pd.DataFrame(
        [
            {
                "TX Hash": "h1",
                "Date": "01/12/2022 10:00:00",
                "Qty in": "1",
                "Token in": "mooFishUSDT-USDC",
                "Qty out": "4000",
                "Token out": "USDC",
                "Type": "Swap",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)
    stale_data_quality = paths["dashboard"] / "arbitrum" / "data_quality.csv"
    stale_data_quality.parent.mkdir(parents=True, exist_ok=True)
    stale_data_quality.write_text("old,data\n")

    accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2022-12-01")
    artifact_paths = dashboard_artifacts.build_arbitrum_dashboard_artifacts(chain="arbitrum")

    expected_files = {
        artifact_paths.asset_daily: dashboard_artifacts.ASSET_DAILY_COLUMNS,
        artifact_paths.timeseries_daily: dashboard_artifacts.TIMESERIES_DAILY_COLUMNS,
        artifact_paths.composition_daily: dashboard_artifacts.COMPOSITION_DAILY_COLUMNS,
        artifact_paths.source_daily: dashboard_artifacts.SOURCE_DAILY_COLUMNS,
        artifact_paths.transactions_dashboard: dashboard_artifacts.TRANSACTIONS_DASHBOARD_COLUMNS,
        artifact_paths.assets: dashboard_artifacts.ASSETS_COLUMNS,
    }
    for path, columns in expected_files.items():
        assert path.exists()
        assert pd.read_csv(path).columns.tolist() == columns
    assert not stale_data_quality.exists()

    timeseries = pd.read_csv(artifact_paths.timeseries_daily)
    usdc = timeseries[timeseries["Selection"] == "USDC"].iloc[0]
    usdt = timeseries[timeseries["Selection"] == "USDT"].iloc[0]
    full = timeseries[timeseries["Selection"] == "ALL"].iloc[0]

    assert usdc["PrincipalInvestedEUR"] == 2000.0
    assert usdt["PrincipalInvestedEUR"] == 2000.0
    assert full["MarketValueEUR"] == 4000.0
    assert full["PrincipalInvestedEUR"] == 4000.0
    assert timeseries[timeseries["Selection"] == "mooFishUSDT-USDC"].empty

    assets = pd.read_csv(artifact_paths.assets)
    assert assets["Value"].tolist() == ["USDC", "USDT"]

    sources = pd.read_csv(artifact_paths.source_daily)
    moo_sources = sources[sources["Source"] == "mooFishUSDT-USDC"]
    assert set(moo_sources["Selection"]) == {"USDC", "USDT"}
    assert moo_sources.set_index("Selection")["PrincipalInvestedEUR"].to_dict() == {
        "USDC": 2000.0,
        "USDT": 2000.0,
    }

    composition = pd.read_csv(artifact_paths.composition_daily)
    route_items = composition[
        (composition["Selection"] == "ALL") & (composition["CompositionMode"] == "route")
    ]
    assert route_items.set_index("Label")["ValueEUR"].to_dict() == {"DIRECT": 4000.0}

    transactions = pd.read_csv(artifact_paths.transactions_dashboard)
    assert "MOOFISHUSDT-USDC" in transactions.iloc[0]["AssetKeys"]
    assert "USDC" in transactions.iloc[0]["AssetKeys"]
    assert "USDT" in transactions.iloc[0]["AssetKeys"]


def test_build_arbitrum_dashboard_artifacts_treats_variable_debt_as_negative_principal(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_artifact_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)

    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xlink": {"symbol": "LINK"},
                "0xdebtlink": {"symbol": "variableDebtArbLINK", "protocol": "aave"},
            }
        )
    )
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)
    pd.DataFrame([{"Date": "2024-06-08", "Price": 14.64}]).to_csv(
        paths["prices"] / "LINK.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "Date": "2024-06-08",
                "Coin": "variableDebtArbLINK",
                "Quantity": 12.5,
                "Principal Invested": 0.0,
            }
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [{"Date": "2024-06-08", "Coin": "LINK", "PrincipalInvestedEUR": -183.0}],
    )
    pd.DataFrame(
        [
            {
                "TX Hash": "borrow",
                "Date": "08/06/2024 07:53:00",
                "Qty in": "12.5, 12.5",
                "Token in": "variableDebtArbLINK, LINK",
                "Qty out": "",
                "Token out": "",
                "Type": "Receive",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)

    accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2024-06-08")
    artifact_paths = dashboard_artifacts.build_arbitrum_dashboard_artifacts(chain="arbitrum")

    timeseries = pd.read_csv(artifact_paths.timeseries_daily)
    link = timeseries[timeseries["Selection"] == "LINK"].iloc[0]
    assert link["MarketValueEUR"] == -183.0
    assert link["PrincipalInvestedEUR"] == -183.0
    assert link["ProfitLossEUR"] == 0.0

    sources = pd.read_csv(artifact_paths.source_daily)
    link_source = sources[sources["Selection"] == "LINK"].iloc[0]
    assert link_source["Source"] == "variableDebtArbLINK"
    assert link_source["Quantity"] == -12.5
    assert link_source["MarketValueEUR"] == -183.0
    assert link_source["PrincipalInvestedEUR"] == -183.0
    assert link_source["ProfitLossEUR"] == 0.0

    transactions = pd.read_csv(artifact_paths.transactions_dashboard)
    assert "LINK" in transactions.iloc[0]["AssetKeys"]


def test_build_arbitrum_dashboard_artifacts_uses_eth_principal_proxy_for_aave_wsteth(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_artifact_paths(monkeypatch, tmp_path)
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
    pd.DataFrame(
        [
            {
                "TX Hash": "unwrap",
                "Date": "15/03/2025 12:00:00",
                "Qty in": "1",
                "Token in": "aArbwstETH",
                "Qty out": "1",
                "Token out": "ETH",
                "Type": "Swap",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)

    accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-03-15")
    artifact_paths = dashboard_artifacts.build_arbitrum_dashboard_artifacts(chain="arbitrum")

    timeseries = pd.read_csv(artifact_paths.timeseries_daily)
    eth = timeseries[timeseries["Selection"] == "ETH"].iloc[0]
    assert eth["MarketValueEUR"] == 1000.0
    assert eth["PrincipalInvestedEUR"] == 1000.0
    assert eth["ProfitLossEUR"] == 0.0

    asset_daily = pd.read_csv(artifact_paths.asset_daily)
    eth_asset = asset_daily[
        (asset_daily["Selection"] == "ETH") & (asset_daily["Coin"] == "ETH")
    ].iloc[0]
    assert eth_asset["HasAaveExposure"]
    assert eth_asset["PrincipalInvestedEUR"] == 1000.0


def test_build_arbitrum_dashboard_artifacts_keeps_nested_eth_wrapper_pnl_stable(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_artifact_paths(monkeypatch, tmp_path)
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
    pd.DataFrame(
        [
            {
                "TX Hash": "wrap",
                "Date": "18/12/2024 07:30:00",
                "Qty in": "2",
                "Token in": "wstETH",
                "Qty out": "2",
                "Token out": "mooBalancerArbwstETH-ETHV3",
                "Type": "Swap",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)

    accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2024-12-18")
    artifact_paths = dashboard_artifacts.build_arbitrum_dashboard_artifacts(chain="arbitrum")

    timeseries = pd.read_csv(artifact_paths.timeseries_daily)
    eth = timeseries[timeseries["Selection"] == "ETH"].sort_values("Date")
    assert eth["MarketValueEUR"].tolist() == [2000.0, 2000.0]
    assert eth["PrincipalInvestedEUR"].tolist() == [2000.0, 2000.0]
    assert eth["ProfitLossEUR"].tolist() == [0.0, 0.0]

    full = timeseries[timeseries["Selection"] == "ALL"].sort_values("Date")
    assert full["ProfitLossEUR"].tolist() == [0.0, 0.0]


def test_build_timeseries_daily_extends_closed_assets_with_zero_value_and_rolled_pnl() -> None:
    asset_daily = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2025-01-01"),
                "Selection": "BTC",
                "MarketValueEUR": 120.0,
                "PrincipalInvestedEUR": 100.0,
                "Quantity": 1.0,
            },
            {
                "Date": pd.Timestamp("2025-01-03"),
                "Selection": "ETH",
                "MarketValueEUR": 50.0,
                "PrincipalInvestedEUR": 40.0,
                "Quantity": 2.0,
            },
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "Date": "2025-01-02 10:00:00",
                "AssetKeys": "BTC",
            }
        ]
    )

    result = dashboard_artifacts._build_timeseries_daily(
        asset_daily=asset_daily,
        transactions=transactions,
    )

    btc = result[result["Selection"] == "BTC"].sort_values("Date")
    assert btc["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ]
    assert btc["MarketValueEUR"].tolist() == [120.0, 0.0, 0.0]
    assert btc["PrincipalInvestedEUR"].tolist() == [100.0, 0.0, 0.0]
    assert btc["Quantity"].tolist() == [1.0, 0.0, 0.0]
    assert btc["ProfitLossEUR"].tolist() == [20.0, 20.0, 20.0]
    assert btc["TxCount"].tolist() == [0, 1, 0]


def test_protocol_source_closure_reallocates_base_principal_to_active_source(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_artifact_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(shared_prices, "get_forex_rate", lambda **_: 1.0)

    (paths["tokens"] / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xold": {"symbol": "OLD-ETH", "protocol": "beefy"},
                "0xnew": {"symbol": "NEW-ETH", "protocol": "beefy"},
                "0xeth": {"symbol": "ETH"},
            }
        )
    )
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Price": 1000.0},
            {"Date": "2025-01-02", "Price": 1000.0},
        ]
    ).to_csv(paths["prices"] / "ETH.csv", index=False)
    for symbol in ("OLD-ETH", "NEW-ETH"):
        pd.DataFrame(
            [
                {"date": "2025-01-01", "asset_ETH": 1.0},
                {"date": "2025-01-02", "asset_ETH": 1.0},
            ]
        ).to_csv(paths["protocol"] / f"arbitrum_{symbol}.csv", index=False)
    pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "Coin": "OLD-ETH",
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2025-01-02",
                "Coin": "OLD-ETH",
                "Quantity": 0.0,
                "Principal Invested": 0.0,
            },
            {
                "Date": "2025-01-02",
                "Coin": "NEW-ETH",
                "Quantity": 1.0,
                "Principal Invested": 0.0,
            },
        ]
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)
    _write_principal_daily(
        paths,
        [
            {"Date": "2025-01-01", "Coin": "ETH", "PrincipalInvestedEUR": 1000.0},
            {"Date": "2025-01-02", "Coin": "ETH", "PrincipalInvestedEUR": 1000.0},
        ],
    )
    pd.DataFrame(
        [
            {
                "TX Hash": "h1",
                "Date": "02/01/2025 12:00:00",
                "Qty in": "1",
                "Token in": "NEW-ETH",
                "Qty out": "1",
                "Token out": "OLD-ETH",
                "Type": "Swap",
                "Fee": "0",
                "Fee Token": "ETH",
            }
        ]
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)

    accounting.build_accounting_artifacts(chain="arbitrum", as_of_date="2025-01-02")
    artifact_paths = dashboard_artifacts.build_arbitrum_dashboard_artifacts(chain="arbitrum")

    timeseries = pd.read_csv(artifact_paths.timeseries_daily)
    eth = timeseries[timeseries["Selection"] == "ETH"].sort_values("Date")
    assert eth["PrincipalInvestedEUR"].tolist() == [1000.0, 1000.0]
    assert eth["ProfitLossEUR"].tolist() == [0.0, 0.0]

    sources = pd.read_csv(artifact_paths.source_daily)
    old_rows = sources[
        (sources["Date"] == "2025-01-02")
        & (sources["Selection"] == "ETH")
        & (sources["Source"] == "OLD-ETH")
    ]
    assert old_rows.empty
    new_row = sources[
        (sources["Date"] == "2025-01-02")
        & (sources["Selection"] == "ETH")
        & (sources["Source"] == "NEW-ETH")
    ].iloc[0]
    assert new_row["Quantity"] == 1.0
    assert new_row["PrincipalInvestedEUR"] == 1000.0
