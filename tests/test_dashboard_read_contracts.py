from __future__ import annotations

import pandas as pd
from portfolio_core import active_context

from portfolio_crypto_data.nexo_dashboard import (
    get_nexo_start_date,
    list_nexo_coins,
    load_and_process_nexo_data,
    load_recent_nexo_transactions,
)


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


def test_nexo_dashboard_is_empty_without_snapshot() -> None:
    assert list_nexo_coins() == []
    assert get_nexo_start_date() is None
    assert load_and_process_nexo_data("2025-01-01").empty
    assert load_recent_nexo_transactions(end_date_str="2025-01-01").empty
