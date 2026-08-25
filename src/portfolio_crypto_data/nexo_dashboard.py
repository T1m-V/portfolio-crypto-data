"""Read-only NEXO dashboard projections owned by the crypto package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from portfolio_core import active_context

from portfolio_crypto_data.cex.dashboard_projection import (
    get_cex_snapshot_start_date,
    list_cex_snapshot_coins,
    load_and_process_cex_data,
)
from portfolio_crypto_data.cex.nexo_snapshots import _load_nexo_transaction_exports
from portfolio_crypto_data.symbols import sanitize_symbol

IGNORED_NEXO_TYPES = {"locking term deposit", "unlocking term deposit"}


def _snapshot_path() -> Path:
    return (
        active_context().paths.crypto_snapshots
        / "cex"
        / "nexo"
        / "nexo_raw_snapshots.csv"
    )


def _transaction_folder() -> Path:
    return active_context().paths.crypto_transactions / "cex" / "nexo"


def _canonicalize_nexo_coin(value: object) -> str:
    coin = sanitize_symbol(value)
    if not coin or coin == "-":
        return ""
    if coin.upper() in {"USD", "USDX", "XUSD"}:
        return "USD"
    return coin


def _load_transactions() -> pd.DataFrame:
    folder = _transaction_folder()
    if not folder.exists() or not any(folder.glob("*.csv")):
        return pd.DataFrame()
    return _load_nexo_transaction_exports(input_csv=folder)


def _prepare_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(
        frame["Date / Time (UTC)"],
        dayfirst=True,
        errors="coerce",
    )
    frame = frame.dropna(subset=["Date"])
    type_series = frame["Type"].fillna("").str.strip().str.lower()
    details_series = frame["Details"].fillna("").str.strip().str.lower()
    is_internal_wallet_hop = details_series.str.contains(
        r"transfer from .*wallet to .*wallet",
        regex=True,
        na=False,
    )
    return frame[~(type_series.isin(IGNORED_NEXO_TYPES) | is_internal_wallet_hop)]


def _filter_coins(frame: pd.DataFrame, coins: list[str] | None) -> pd.DataFrame:
    if not coins:
        return frame
    canonical_coins = {_canonicalize_nexo_coin(coin) for coin in coins}
    canonical_coins.discard("")
    input_coins = frame["Input Currency"].map(_canonicalize_nexo_coin)
    output_coins = frame["Output Currency"].map(_canonicalize_nexo_coin)
    return frame[input_coins.isin(canonical_coins) | output_coins.isin(canonical_coins)]


def list_nexo_coins() -> list[str]:
    return list_cex_snapshot_coins(snapshot_path=_snapshot_path())


def get_nexo_start_date(coins: list[str] | None = None) -> str | None:
    """Return the first relevant NEXO snapshot or transaction date."""
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
        transactions = _filter_coins(_prepare_transactions(transactions), coins)
        if not transactions.empty:
            dates.append(transactions["Date"].min())

    return min(dates).strftime("%Y-%m-%d") if dates else None


def load_and_process_nexo_data(
    end_date_str: str,
    coins: list[str] | None = None,
) -> pd.DataFrame:
    return load_and_process_cex_data(
        snapshot_path=_snapshot_path(),
        end_date_str=end_date_str,
        coins=coins,
        provider_name="NEXO",
    )


def load_recent_nexo_transactions(
    *,
    end_date_str: str,
    coins: list[str] | None = None,
    limit: int | None = 5,
) -> pd.DataFrame:
    """Load the latest relevant NEXO transactions up to an as-of date."""
    frame = _load_transactions()
    if frame.empty:
        return frame

    frame = _filter_coins(_prepare_transactions(frame), coins)
    frame = frame[frame["Date"] <= pd.to_datetime(end_date_str)]
    frame = frame.sort_values("Date", ascending=False).copy()
    if limit is not None:
        frame = frame.head(limit)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d %H:%M")
    internal_columns = [column for column in frame.columns if column.startswith("__")]
    return frame.drop(columns=internal_columns)
