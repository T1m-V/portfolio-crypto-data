import functools
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from portfolio_core import active_context

STABLE_PRICE_SYMBOLS: dict[str, Decimal] = {
    "EUR": Decimal("1"),
    "EURX": Decimal("1"),
    "USD": Decimal("1"),
    "USDC": Decimal("1"),
    "DAI": Decimal("1"),
    "USDT": Decimal("1"),
    "BUSD": Decimal("1"),
    "USDX": Decimal("1"),
    "xUSD": Decimal("1"),
}
PRICE_CURRENCY_OVERRIDES: dict[str, str] = {
    "EUR": "EUR",
    "EURX": "EUR",
    "USD": "USD",
    "USDC": "USD",
    "USDT": "USD",
    "DAI": "USD",
    "BUSD": "USD",
    "USDX": "USD",
    "xUSD": "USD",
}


def _normalize_date(value: str | pd.Timestamp | date) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


@functools.lru_cache(maxsize=None)
def _load_price_history_cached(
    symbol: str,
    file_path: str,
) -> pd.DataFrame | None:
    stable_price = STABLE_PRICE_SYMBOLS.get(symbol)
    if stable_price is not None:
        return pd.DataFrame(
            {"Date": [pd.to_datetime("2000-01-01").date()], "Price": [stable_price]}
        )

    resolved_path = Path(file_path)
    if not resolved_path.exists():
        return None

    df = pd.read_csv(resolved_path)
    if "Date" not in df.columns or "Price" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    if df.empty:
        return None
    return df[["Date", "Price"]]


@functools.lru_cache(maxsize=None)
def _load_currency_metadata_cached(file_path: str) -> dict[str, dict[str, Any]]:
    resolved_path = Path(file_path)
    if not resolved_path.exists():
        return {}
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _resolve_price_file_path(
    *,
    symbol: str,
    prices_folder: Path | None,
) -> Path:
    root = prices_folder or active_context().paths.prices
    return root / f"{symbol}.csv"


def clear_price_cache() -> None:
    _load_price_history_cached.cache_clear()
    _load_currency_metadata_cached.cache_clear()


def get_price_on_or_before(
    *,
    symbol: str,
    as_of_date: str | pd.Timestamp | date,
    prices_folder: Path | None = None,
    fallback_to_oldest: bool = False,
) -> Decimal | None:
    file_path = _resolve_price_file_path(
        symbol=symbol,
        prices_folder=prices_folder,
    )
    history = _load_price_history_cached(
        symbol=symbol,
        file_path=str(file_path),
    )
    if history is None:
        return None

    target_date = _normalize_date(as_of_date)
    eligible = history[history["Date"] <= target_date]
    if eligible.empty:
        if not fallback_to_oldest:
            return None
        return Decimal(str(history.iloc[0]["Price"]))
    return Decimal(str(eligible.iloc[-1]["Price"]))


def get_price_eur_on_or_before(
    *,
    symbol: str,
    as_of_date: str | pd.Timestamp | date,
    prices_folder: Path | None = None,
    currency_metadata: dict[str, dict[str, Any]] | None = None,
    fallback_to_oldest: bool = False,
) -> Decimal | None:
    price = get_price_on_or_before(
        symbol=symbol,
        as_of_date=as_of_date,
        prices_folder=prices_folder,
        fallback_to_oldest=fallback_to_oldest,
    )
    if price is None:
        return None

    context = active_context()
    resolved_prices_folder = prices_folder or context.paths.prices
    metadata = (
        currency_metadata
        if currency_metadata is not None
        else _load_currency_metadata_cached(str(context.paths.currency_metadata))
    )
    currency = PRICE_CURRENCY_OVERRIDES.get(
        symbol,
        str(metadata.get(symbol, {}).get("currency", "USD")),
    )
    if currency == "EUR":
        return price

    fx_rate = get_price_on_or_before(
        symbol=f"{currency}_EUR",
        as_of_date=as_of_date,
        prices_folder=resolved_prices_folder,
        fallback_to_oldest=True,
    )
    if fx_rate is None:
        fx_path = resolved_prices_folder / f"{currency}_EUR.csv"
        raise FileNotFoundError(f"No forex data for {currency}: {fx_path}")
    return price * fx_rate
