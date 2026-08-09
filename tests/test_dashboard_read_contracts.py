from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from portfolio_core import PortfolioContext, active_context

from portfolio_crypto_data.nexo_dashboard import (
    get_nexo_start_date,
    list_nexo_coins,
    load_and_process_nexo_data,
    load_recent_nexo_transactions,
)


def _workspace_manifest(root: Path) -> dict[Path, tuple[int, int, bytes]]:
    return {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_nexo_dashboard_read_contract() -> None:
    paths = active_context().paths
    snapshot = paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Coin": "BTC", "Quantity": 1, "Principal Invested": 90},
            {"Date": "2025-01-02", "Coin": "BTC", "Quantity": 1.5, "Principal Invested": 120},
        ]
    ).to_csv(snapshot, index=False)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Price": 100},
            {"Date": "2025-01-02", "Price": 110},
        ]
    ).to_csv(paths.prices / "BTC.csv", index=False)

    transaction_folder = paths.crypto_transactions / "cex" / "nexo"
    transaction_folder.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Date / Time (UTC)": "02/01/2025 10:00",
                "Type": "Interest",
                "Input Currency": "BTC",
                "Output Currency": "-",
                "Details": "interest",
            },
            {
                "Date / Time (UTC)": "01/01/2025 09:00",
                "Type": "Locking Term Deposit",
                "Input Currency": "BTC",
                "Output Currency": "BTC",
                "Details": "internal",
            },
        ]
    ).to_csv(transaction_folder / "export.csv", index=False)

    assert list_nexo_coins() == ["BTC"]
    assert get_nexo_start_date(["BTC"]) == "2025-01-01"
    daily = load_and_process_nexo_data("2025-01-02", ["BTC"])
    assert daily[["Quantity", "Price", "Market Value"]].iloc[-1].tolist() == [
        1.5,
        110.0,
        165.0,
    ]
    recent = load_recent_nexo_transactions(end_date_str="2025-01-02 23:59", coins=["BTC"])
    assert recent["Type"].tolist() == ["Interest"]


def test_nexo_dashboard_loads_currency_metadata_once_per_projection(
    isolate_portfolio_io: PortfolioContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = isolate_portfolio_io
    paths = context.paths
    snapshot = paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Coin": "BTC", "Quantity": 1, "Principal Invested": 90},
            {
                "Date": "2025-01-03",
                "Coin": "BTC",
                "Quantity": 1.5,
                "Principal Invested": 120,
            },
            {"Date": "2025-01-02", "Coin": "ETH", "Quantity": 2, "Principal Invested": 300},
            {
                "Date": "2025-01-01",
                "Coin": "USDC",
                "Quantity": 100,
                "Principal Invested": 100,
            },
        ]
    ).to_csv(snapshot, index=False)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Price": 100},
            {"Date": "2025-01-03", "Price": 110},
        ]
    ).to_csv(paths.direct_price("BTC"), index=False)
    pd.DataFrame(
        [
            {"Date": "2025-01-02", "Price": 200},
            {"Date": "2025-01-03", "Price": 210},
        ]
    ).to_csv(paths.direct_price("ETH"), index=False)

    paths.currency_metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.currency_metadata.write_text(
        json.dumps(
            {
                "BTC": {"name": "Bitcoin", "group": "Layer 1", "currency": "EUR"},
                "USDC": {"name": "USD Coin", "group": "Stablecoin", "currency": "EUR"},
            }
        ),
        encoding="utf-8",
    )

    original_currency_metadata = PortfolioContext.currency_metadata
    metadata_calls = 0

    def tracked_currency_metadata(self: PortfolioContext) -> dict[str, dict[str, object]]:
        nonlocal metadata_calls
        metadata_calls += 1
        return original_currency_metadata(self)

    monkeypatch.setattr(PortfolioContext, "currency_metadata", tracked_currency_metadata)
    before = _workspace_manifest(paths.root)

    all_assets = load_and_process_nexo_data("2025-01-03")
    assert metadata_calls == 1
    single_coin = load_and_process_nexo_data("2025-01-03", ["BTC"])
    assert metadata_calls == 2

    assert _workspace_manifest(paths.root) == before
    assert all_assets["Coin"].value_counts().to_dict() == {"BTC": 3, "USDC": 3, "ETH": 2}
    asset_details = all_assets.drop_duplicates("Coin").set_index("Coin")
    assert asset_details.loc["BTC", ["Asset Name", "Asset Group", "Currency"]].tolist() == [
        "Bitcoin",
        "Layer 1",
        "EUR",
    ]
    assert asset_details.loc["ETH", ["Asset Name", "Asset Group", "Currency"]].tolist() == [
        "ETH",
        "Unknown",
        "USD",
    ]
    assert asset_details.loc["USDC", ["Asset Name", "Asset Group", "Currency"]].tolist() == [
        "USD Coin",
        "Stablecoin",
        "USD",
    ]
    assert all_assets.loc[all_assets["Coin"] == "ETH", "Market Value"].iloc[-1] == 420.0
    pd.testing.assert_frame_equal(
        single_coin.reset_index(drop=True),
        all_assets[all_assets["Coin"] == "BTC"].reset_index(drop=True),
    )


def test_nexo_dashboard_is_empty_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_currency_metadata = PortfolioContext.currency_metadata
    metadata_calls = 0

    def tracked_currency_metadata(self: PortfolioContext) -> dict[str, dict[str, object]]:
        nonlocal metadata_calls
        metadata_calls += 1
        return original_currency_metadata(self)

    monkeypatch.setattr(PortfolioContext, "currency_metadata", tracked_currency_metadata)
    assert list_nexo_coins() == []
    assert get_nexo_start_date() is None
    assert load_and_process_nexo_data("2025-01-01").empty
    assert load_recent_nexo_transactions(end_date_str="2025-01-01").empty
    assert metadata_calls == 0
