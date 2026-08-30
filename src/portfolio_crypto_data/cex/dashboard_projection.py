from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from portfolio_core import active_context

from portfolio_crypto_data.cex.snapshot_io import SNAPSHOT_COLUMNS

USD_STABLES = {"USD", "USDX", "xUSD", "USDC", "USDT", "DAI"}
EUR_STABLES = {"EUR", "EURX"}
COLS_TO_FILL = ["Quantity", "Principal Invested"]


def load_cex_snapshot(
    *,
    snapshot_path: Path,
    end_dt: pd.Timestamp,
    coins: list[str] | None,
) -> pd.DataFrame:
    if not snapshot_path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    snapshots = pd.read_csv(snapshot_path)
    snapshots["Date"] = pd.to_datetime(snapshots["Date"], errors="coerce")
    snapshots = snapshots.dropna(subset=["Date"])
    snapshots = snapshots[snapshots["Date"] <= end_dt]
    if coins:
        snapshots = snapshots[snapshots["Coin"].isin(coins)]
    return snapshots


def list_cex_snapshot_coins(*, snapshot_path: Path) -> list[str]:
    if not snapshot_path.exists():
        return []
    snapshots = pd.read_csv(snapshot_path, usecols=["Coin"])
    coins = [coin for coin in snapshots["Coin"].dropna().unique() if str(coin).strip()]
    return sorted(coins)


def get_cex_snapshot_start_date(
    *,
    snapshot_path: Path,
    coins: list[str] | None,
) -> pd.Timestamp | None:
    if not snapshot_path.exists():
        return None
    snapshots = pd.read_csv(snapshot_path, usecols=["Date", "Coin"])
    snapshots["Date"] = pd.to_datetime(snapshots["Date"], errors="coerce")
    snapshots = snapshots.dropna(subset=["Date"])
    if coins:
        snapshots = snapshots[snapshots["Coin"].isin(coins)]
    if snapshots.empty:
        return None
    return snapshots["Date"].min()


def _load_usd_eur(*, end_dt: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(active_context().paths.direct_price("USD_EUR"))
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Price"])
    frame = frame[frame["Date"] <= end_dt]
    return frame.sort_values("Date")[["Date", "Price"]]


def _resolve_currency(*, coin: str, metadata: dict[str, dict[str, Any]]) -> str:
    if coin in EUR_STABLES:
        return "EUR"
    if coin in USD_STABLES:
        return "USD"
    return str(metadata.get(coin, {}).get("currency", "USD"))


def _build_price_frame(
    *,
    coin: str,
    coin_start: pd.Timestamp,
    end_dt: pd.Timestamp,
    usd_eur: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    currency = _resolve_currency(coin=coin, metadata=metadata)
    full_dates = pd.date_range(start=coin_start, end=end_dt, freq="D")
    price_path = active_context().paths.direct_price(coin)
    if price_path.exists():
        prices = pd.read_csv(price_path)
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
        prices["Price"] = pd.to_numeric(prices["Price"], errors="coerce")
        prices = prices.dropna(subset=["Date", "Price"])
        prices = prices[prices["Date"] <= end_dt]
        prices = prices.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
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


def load_and_process_cex_data(
    *,
    snapshot_path: Path,
    end_date_str: str,
    coins: list[str] | None,
    provider_name: str,
) -> pd.DataFrame:
    end_dt = pd.to_datetime(end_date_str)
    snapshots = load_cex_snapshot(
        snapshot_path=snapshot_path,
        end_dt=end_dt,
        coins=coins,
    )
    if snapshots.empty:
        return pd.DataFrame()

    usd_eur = _load_usd_eur(end_dt=end_dt)
    if usd_eur.empty:
        raise ValueError(f"USD_EUR history is required for {provider_name} valuation.")
    metadata = active_context().currency_metadata()
    price_frames = [
        _build_price_frame(
            coin=coin,
            coin_start=snapshots.loc[snapshots["Coin"] == coin, "Date"].min(),
            end_dt=end_dt,
            usd_eur=usd_eur,
            metadata=metadata,
        )
        for coin in sorted(snapshots["Coin"].unique())
    ]
    prices = pd.concat(price_frames, ignore_index=True)
    merged = pd.merge(prices, snapshots, on=["Date", "Coin"], how="left")
    merged = merged.sort_values(["Coin", "Date"])
    merged[COLS_TO_FILL] = merged.groupby("Coin")[COLS_TO_FILL].ffill().fillna(0)
    merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce").fillna(0)
    merged["Asset Name"] = merged["Coin"].map(
        lambda coin: metadata.get(coin, {}).get("name", coin)
    )
    merged["Asset Group"] = merged["Coin"].map(
        lambda coin: metadata.get(coin, {}).get("group", "Unknown")
    )
    merged["Currency"] = merged["Coin"].map(
        lambda coin: _resolve_currency(coin=coin, metadata=metadata)
    )
    merged["Market Value"] = merged["Quantity"] * merged["Price"]
    merged["Cumulative Fees"] = 0.0
    merged["Cumulative Taxes"] = 0.0
    merged["Gross Dividends"] = 0.0
    return merged
