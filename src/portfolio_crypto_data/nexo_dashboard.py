"""Read-only Nexo dashboard projections owned by the crypto package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from portfolio_core import active_context

from portfolio_crypto_data.symbols import sanitize_symbol

USD_STABLES = {"USD", "USDX", "xUSD", "USDC", "USDT", "DAI"}
EUR_STABLES = {"EUR", "EURX"}

COLS_TO_FILL = ["Quantity", "Principal Invested"]
IGNORED_NEXO_TYPES = {"locking term deposit", "unlocking term deposit"}
SNAPSHOT_COLUMNS = ["Date", "Coin", *COLS_TO_FILL]


def _empty_snapshot_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def _canonicalize_nexo_coin(value: object) -> str:
    coin = sanitize_symbol(value)
    if not coin or coin == "-":
        return ""
    if coin.upper() in {"USD", "USDX", "XUSD"}:
        return "USD"
    return coin


def _load_nexo_snapshot(end_dt: pd.Timestamp, coins: list[str] | None) -> pd.DataFrame:
    snapshot_path = (
        active_context().paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
    )
    if not snapshot_path.exists():
        return _empty_snapshot_frame()

    snapshots = pd.read_csv(snapshot_path)
    snapshots["Date"] = pd.to_datetime(snapshots["Date"], errors="coerce")
    snapshots = snapshots.dropna(subset=["Date"])
    snapshots = snapshots[snapshots["Date"] <= end_dt]
    if coins:
        snapshots = snapshots[snapshots["Coin"].isin(coins)]
    return snapshots


def _load_nexo_transaction_exports(transaction_folder: Path) -> pd.DataFrame:
    csv_paths = sorted(path for path in transaction_folder.glob("*.csv") if path.is_file())
    if not csv_paths:
        return pd.DataFrame()

    frames = [pd.read_csv(path, dtype=str) for path in csv_paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_usd_eur(end_dt: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(active_context().paths.direct_price("USD_EUR"))
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Price"])
    frame = frame[frame["Date"] <= end_dt]
    frame = frame.sort_values("Date")
    return frame[["Date", "Price"]]


def _resolve_currency(coin: str) -> str:
    if coin in EUR_STABLES:
        return "EUR"
    if coin in USD_STABLES:
        return "USD"
    return str(active_context().currency_metadata().get(coin, {}).get("currency", "USD"))


def _build_price_frame(
    *,
    coin: str,
    coin_start: pd.Timestamp,
    end_dt: pd.Timestamp,
    usd_eur: pd.DataFrame,
) -> pd.DataFrame:
    currency = _resolve_currency(coin=coin)
    full_dates = pd.date_range(start=coin_start, end=end_dt, freq="D")

    price_path = active_context().paths.direct_price(coin)
    if price_path.exists():
        prices = pd.read_csv(price_path)
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        prices["Price"] = pd.to_numeric(prices["Price"], errors="coerce")
        prices = prices.dropna(subset=["Date", "Price"])
        prices = prices[prices["Date"] <= end_dt]
        prices = prices.sort_values("Date")
        prices = prices.drop_duplicates(subset=["Date"], keep="last")

        if not prices.empty:
            prices = (
                prices.set_index("Date")
                .reindex(full_dates)
                .ffill()
                .bfill()
                .reset_index()
                .rename(columns={"index": "Date"})
            )
            if currency != "EUR":
                prices = prices.merge(
                    usd_eur.rename(columns={"Price": "FX"}),
                    on="Date",
                    how="left",
                )
                prices["FX"] = prices["FX"].ffill().bfill().fillna(1.0)
                prices["Price"] = prices["Price"] * prices["FX"]
                prices = prices.drop(columns=["FX"])
            prices["Coin"] = coin
            return prices[["Date", "Coin", "Price"]]

    fallback = pd.DataFrame({"Date": full_dates})
    if coin in EUR_STABLES:
        fallback["Price"] = 1.0
    elif currency != "EUR":
        fallback = fallback.merge(
            usd_eur.rename(columns={"Price": "FX"}),
            on="Date",
            how="left",
        )
        fallback["FX"] = fallback["FX"].ffill().bfill().fillna(1.0)
        fallback["Price"] = fallback["FX"]
        fallback = fallback.drop(columns=["FX"])
    else:
        fallback["Price"] = 0.0

    fallback["Coin"] = coin
    return fallback[["Date", "Coin", "Price"]]


def list_nexo_coins() -> list[str]:
    snapshot_path = (
        active_context().paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
    )
    if not snapshot_path.exists():
        return []

    snapshots = pd.read_csv(snapshot_path, usecols=["Coin"])
    coins = [coin for coin in snapshots["Coin"].dropna().unique().tolist() if str(coin).strip()]
    return sorted(coins)


def get_nexo_start_date(coins: list[str] | None = None) -> str | None:
    """
    Finds the first NEXO snapshot or relevant transaction date for a dashboard selection.

    args:
        coins: Optional coin filter. None means all NEXO assets.

    returns:
        First date as YYYY-MM-DD, or None when unavailable.
    """
    if coins == []:
        return None

    date_parts: list[pd.Series] = []
    paths = active_context().paths
    snapshot_path = paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
    transactions_folder = paths.crypto_transactions / "cex" / "nexo"
    if snapshot_path.exists():
        snapshots = pd.read_csv(snapshot_path, usecols=["Date", "Coin"])
        snapshots["Date"] = pd.to_datetime(snapshots["Date"], errors="coerce")
        snapshots = snapshots.dropna(subset=["Date"])
        if coins:
            snapshots = snapshots[snapshots["Coin"].isin(coins)]
        if not snapshots.empty:
            date_parts.append(snapshots["Date"])

    transactions = _load_nexo_transaction_exports(transactions_folder)
    if not transactions.empty and "Date / Time (UTC)" in transactions.columns:
        transactions = transactions.copy()
        transactions["Date"] = pd.to_datetime(
            transactions["Date / Time (UTC)"],
            dayfirst=True,
            errors="coerce",
        )
        transactions = transactions.dropna(subset=["Date"])
        type_series = transactions["Type"].fillna("").str.strip().str.lower()
        details_series = transactions["Details"].fillna("").str.strip().str.lower()
        is_internal_wallet_hop = details_series.str.contains(
            r"transfer from .*wallet to .*wallet",
            regex=True,
            na=False,
        )
        transactions = transactions[
            ~(type_series.isin(IGNORED_NEXO_TYPES) | is_internal_wallet_hop)
        ]

        if coins:
            canonical_coins = {_canonicalize_nexo_coin(coin) for coin in coins}
            canonical_coins.discard("")
            input_coins = transactions["Input Currency"].map(_canonicalize_nexo_coin)
            output_coins = transactions["Output Currency"].map(_canonicalize_nexo_coin)
            transactions = transactions[
                input_coins.isin(canonical_coins) | output_coins.isin(canonical_coins)
            ]

        if not transactions.empty:
            date_parts.append(transactions["Date"])

    if not date_parts:
        return None
    return pd.concat(date_parts).min().strftime("%Y-%m-%d")


def load_and_process_nexo_data(end_date_str: str, coins: list[str] | None = None) -> pd.DataFrame:
    end_dt = pd.to_datetime(end_date_str)
    snapshots = _load_nexo_snapshot(end_dt=end_dt, coins=coins)
    if snapshots.empty:
        return pd.DataFrame()

    usd_eur = _load_usd_eur(end_dt=end_dt)
    if usd_eur.empty:
        raise ValueError("USD_EUR history is required for NEXO valuation.")

    price_frames: list[pd.DataFrame] = []
    for coin in sorted(snapshots["Coin"].unique().tolist()):
        coin_start = snapshots.loc[snapshots["Coin"] == coin, "Date"].min()
        price_frames.append(
            _build_price_frame(
                coin=coin,
                coin_start=coin_start,
                end_dt=end_dt,
                usd_eur=usd_eur,
            )
        )

    prices = pd.concat(price_frames, ignore_index=True)
    merged = pd.merge(prices, snapshots, on=["Date", "Coin"], how="left")
    merged = merged.sort_values(["Coin", "Date"])
    merged[COLS_TO_FILL] = merged.groupby("Coin")[COLS_TO_FILL].ffill().fillna(0)
    merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce").fillna(0)

    merged["Asset Name"] = merged["Coin"].map(
        lambda coin: active_context().currency_metadata().get(coin, {}).get("name", coin)
    )
    merged["Asset Group"] = merged["Coin"].map(
        lambda coin: active_context().currency_metadata().get(coin, {}).get("group", "Unknown")
    )
    merged["Currency"] = merged["Coin"].map(_resolve_currency)
    merged["Market Value"] = merged["Quantity"] * merged["Price"]
    merged["Cumulative Fees"] = 0.0
    merged["Cumulative Taxes"] = 0.0
    merged["Gross Dividends"] = 0.0
    return merged


def load_recent_nexo_transactions(
    *,
    end_date_str: str,
    coins: list[str] | None = None,
    limit: int | None = 5,
) -> pd.DataFrame:
    """
    Loads and filters latest NEXO transactions up to an as-of date.

    args:
        end_date_str: Selected dashboard date.
        coins: Optional symbol filter.
        limit: Max number of rows.

    returns:
        Latest filtered transaction rows.
    """
    frame = _load_nexo_transaction_exports(
        active_context().paths.crypto_transactions / "cex" / "nexo"
    )
    if frame.empty:
        return frame

    frame["Date"] = pd.to_datetime(frame["Date / Time (UTC)"], dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame = frame[frame["Date"] <= pd.to_datetime(end_date_str)]
    type_series = frame["Type"].fillna("").str.strip().str.lower()
    details_series = frame["Details"].fillna("").str.strip().str.lower()
    is_internal_wallet_hop = details_series.str.contains(
        r"transfer from .*wallet to .*wallet",
        regex=True,
        na=False,
    )
    frame = frame[~(type_series.isin(IGNORED_NEXO_TYPES) | is_internal_wallet_hop)]

    if coins:
        canonical_coins = {_canonicalize_nexo_coin(coin) for coin in coins}
        canonical_coins.discard("")
        input_coins = frame["Input Currency"].map(_canonicalize_nexo_coin)
        output_coins = frame["Output Currency"].map(_canonicalize_nexo_coin)
        frame = frame[input_coins.isin(canonical_coins) | output_coins.isin(canonical_coins)]

    frame = frame.sort_values("Date", ascending=False).copy()
    if limit is not None:
        frame = frame.head(limit)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d %H:%M")
    return frame
