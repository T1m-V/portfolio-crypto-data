"""Read-only Crypto.com App dashboard projections owned by the crypto package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from portfolio_core import active_context

from portfolio_crypto_data.cex.crypto_com_app_snapshots import (
    APP_TIMESTAMP_FORMAT,
    SKIPPED_KINDS,
    _load_crypto_com_app_exports,
)
from portfolio_crypto_data.cex.dashboard_projection import (
    get_cex_snapshot_start_date,
    list_cex_snapshot_coins,
    load_and_process_cex_data,
)
from portfolio_crypto_data.symbols import sanitize_symbol


def _snapshot_path() -> Path:
    return (
        active_context().paths.crypto_snapshots
        / "cex"
        / "crypto_com_app"
        / "crypto_com_app_raw_snapshots.csv"
    )


def _transaction_folder() -> Path:
    return active_context().paths.crypto_transactions / "cex" / "crypto_com_app"


def _load_transactions() -> pd.DataFrame:
    folder = _transaction_folder()
    if not folder.exists() or not any(folder.glob("*.csv")):
        return pd.DataFrame()
    return _load_crypto_com_app_exports(input_csv=folder)


def _parse_transaction_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(
        frame["Timestamp (UTC)"],
        format=APP_TIMESTAMP_FORMAT,
        errors="coerce",
    )
    return frame.dropna(subset=["Date"])


def _filter_supported_activity(frame: pd.DataFrame) -> pd.DataFrame:
    kinds = frame["Transaction Kind"].fillna("").str.strip().str.lower()
    return frame[~kinds.isin(SKIPPED_KINDS)]


def _filter_coins(frame: pd.DataFrame, coins: list[str] | None) -> pd.DataFrame:
    if not coins:
        return frame
    selected = {sanitize_symbol(coin) for coin in coins}
    selected.discard("")
    primary = frame["Currency"].map(sanitize_symbol)
    received = frame["To Currency"].map(sanitize_symbol)
    return frame[primary.isin(selected) | received.isin(selected)]


def list_crypto_com_app_coins() -> list[str]:
    return list_cex_snapshot_coins(snapshot_path=_snapshot_path())


def get_crypto_com_app_start_date(coins: list[str] | None = None) -> str | None:
    if coins == []:
        return None

    dates: list[pd.Timestamp] = []
    snapshot_date = get_cex_snapshot_start_date(
        snapshot_path=_snapshot_path(),
        coins=coins,
    )
    if snapshot_date is not None:
        dates.append(snapshot_date)

    transactions = _load_transactions()
    if not transactions.empty:
        transactions = _filter_coins(
            _filter_supported_activity(_parse_transaction_dates(transactions)),
            coins,
        )
        if not transactions.empty:
            dates.append(transactions["Date"].min())

    return min(dates).strftime("%Y-%m-%d") if dates else None


def load_and_process_crypto_com_app_data(
    end_date_str: str,
    coins: list[str] | None = None,
) -> pd.DataFrame:
    return load_and_process_cex_data(
        snapshot_path=_snapshot_path(),
        end_date_str=end_date_str,
        coins=coins,
        provider_name="Crypto.com App",
    )


def _consolidate_dust_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    kinds = frame["Transaction Kind"].str.strip().str.lower()
    dust = frame[kinds.isin({"dust_conversion_credited", "dust_conversion_debited"})]
    if dust.empty:
        return frame

    replacement_rows: list[pd.Series] = []
    consumed: set[int] = set()
    for _, group in dust.groupby(["Date", "Transaction Description"], sort=False):
        group_kinds = group["Transaction Kind"].str.strip().str.lower()
        credited = group[group_kinds == "dust_conversion_credited"]
        debited = group[group_kinds == "dust_conversion_debited"]
        if len(group) != 2 or len(credited) != 1 or len(debited) != 1:
            continue
        row = debited.iloc[0].copy()
        row["Transaction Kind"] = "dust_conversion"
        row["To Currency"] = credited.iloc[0]["Currency"]
        row["To Amount"] = credited.iloc[0]["Amount"]
        replacement_rows.append(row)
        consumed.update(int(index) for index in group.index)

    if not consumed:
        return frame
    remaining = frame.drop(index=list(consumed))
    return pd.concat(
        [remaining, pd.DataFrame(replacement_rows)],
        ignore_index=True,
        sort=False,
    )


def load_recent_crypto_com_app_transactions(
    *,
    end_date_str: str,
    coins: list[str] | None = None,
    limit: int | None = 5,
) -> pd.DataFrame:
    frame = _load_transactions()
    if frame.empty:
        return frame
    frame = _filter_supported_activity(_parse_transaction_dates(frame))
    frame = frame[frame["Date"] <= pd.to_datetime(end_date_str)]
    frame = _filter_coins(_consolidate_dust_for_display(frame), coins)
    frame = frame.sort_values("Date", ascending=False).copy()
    if limit is not None:
        frame = frame.head(limit)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d %H:%M")
    internal_columns = [column for column in frame.columns if column.startswith("__")]
    return frame.drop(columns=internal_columns)
