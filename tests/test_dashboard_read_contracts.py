from __future__ import annotations

import pandas as pd
import pytest
from portfolio_core import active_context

from portfolio_crypto_data.crypto_com_app_dashboard import (
    get_crypto_com_app_start_date,
    list_crypto_com_app_coins,
    load_and_process_crypto_com_app_data,
    load_recent_crypto_com_app_transactions,
)
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
                "Input Amount": "0.1",
                "Output Currency": "-",
                "Output Amount": "0",
                "Details": "interest",
            },
            {
                "Date / Time (UTC)": "01/01/2025 09:00",
                "Type": "Locking Term Deposit",
                "Input Currency": "BTC",
                "Input Amount": "1",
                "Output Currency": "BTC",
                "Output Amount": "1",
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


def test_crypto_com_app_dashboard_read_contract() -> None:
    paths = active_context().paths
    snapshot = (
        paths.crypto_snapshots
        / "cex"
        / "crypto_com_app"
        / "crypto_com_app_raw_snapshots.csv"
    )
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Coin": "CRO", "Quantity": 2, "Principal Invested": 1},
            {"Date": "2025-01-03", "Coin": "CRO", "Quantity": 3, "Principal Invested": 2},
        ]
    ).to_csv(snapshot, index=False)
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Price": 0.10},
            {"Date": "2025-01-03", "Price": 0.20},
        ]
    ).to_csv(paths.direct_price("CRO"), index=False)

    transaction_folder = paths.crypto_transactions / "cex" / "crypto_com_app"
    transaction_folder.mkdir(parents=True)
    base = {
        "Transaction Description": "Synthetic transaction",
        "Currency": "CRO",
        "Amount": "1",
        "To Currency": "",
        "To Amount": "",
        "Native Currency": "EUR",
        "Native Amount": "0.1",
        "Native Amount (in USD)": "0.11",
        "Transaction Hash": "",
    }
    pd.DataFrame(
        [
            {
                **base,
                "Timestamp (UTC)": "2025-01-01 09:00:00",
                "Transaction Kind": "finance.lockup.dpos_lock.crypto_wallet",
                "Amount": "-1",
            },
            {
                **base,
                "Timestamp (UTC)": "2025-01-02 09:00:00",
                "Transaction Kind": "crypto_earn_program_created",
                "Amount": "-1",
            },
            {
                **base,
                "Timestamp (UTC)": "2025-01-03 09:00:00",
                "Transaction Kind": "crypto_earn_program_withdrawn",
            },
            {
                **base,
                "Timestamp (UTC)": "2025-01-02 10:00:00",
                "Transaction Description": "USDC > CRO",
                "Transaction Kind": "crypto_exchange",
                "Currency": "USDC",
                "Amount": "-1",
                "To Currency": "CRO",
                "To Amount": "2",
            },
            {
                **base,
                "Timestamp (UTC)": "2025-01-03 11:00:00",
                "Transaction Description": "Convert Dust",
                "Transaction Kind": "dust_conversion_debited",
                "Currency": "USDC",
                "Amount": "-0.01",
            },
            {
                **base,
                "Timestamp (UTC)": "2025-01-03 11:00:00",
                "Transaction Description": "Convert Dust",
                "Transaction Kind": "dust_conversion_credited",
                "Currency": "CRO",
                "Amount": "0.1",
            },
        ]
    ).to_csv(transaction_folder / "export.csv", index=False)

    assert list_crypto_com_app_coins() == ["CRO"]
    assert get_crypto_com_app_start_date(["CRO"]) == "2025-01-01"
    daily = load_and_process_crypto_com_app_data("2025-01-03", ["CRO"])
    assert daily[["Quantity", "Price", "Market Value"]].iloc[-1].tolist() == pytest.approx(
        [3.0, 0.20, 0.60]
    )

    recent = load_recent_crypto_com_app_transactions(
        end_date_str="2025-01-03 23:59",
        coins=["CRO"],
        limit=None,
    )
    assert recent["Transaction Kind"].tolist() == ["dust_conversion", "crypto_exchange"]
    dust = recent.iloc[0]
    assert (dust["Currency"], dust["Amount"], dust["To Currency"], dust["To Amount"]) == (
        "USDC",
        "-0.01",
        "CRO",
        "0.1",
    )
