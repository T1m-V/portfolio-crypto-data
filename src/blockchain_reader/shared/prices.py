import functools
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from file_paths import (
    CURRENCY_METADATA,
    get_direct_price_file_path,
    get_lp_price_file_path,
)
from historical_transactions.portfolio_snapshots import get_forex_rate

STABLE_PRICE_SYMBOLS: dict[str, Decimal] = {
    "EUR": Decimal("1"),
    "EURX": Decimal("1"),
    "USD": Decimal("1"),
    "USDC": Decimal("1"),
    "DAI": Decimal("1"),
    "USDT": Decimal("1"),
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


def _resolve_price_file_path(
    *,
    symbol: str,
    prices_folder: Path | None,
    chain: str | None,
    use_lp_prices: bool,
) -> Path:
    if use_lp_prices and not chain:
        raise ValueError("LP price lookup requires `chain` from context.")

    if use_lp_prices:
        return get_lp_price_file_path(
            chain=str(chain),
            symbol=symbol,
            prices_folder=prices_folder,
        )
    return get_direct_price_file_path(symbol=symbol, prices_folder=prices_folder)


def clear_price_cache() -> None:
    _load_price_history_cached.cache_clear()


def get_price_on_or_before(
    *,
    symbol: str,
    as_of_date: str | pd.Timestamp | date,
    prices_folder: Path | None = None,
    chain: str | None = None,
    use_lp_prices: bool = False,
    fallback_to_oldest: bool = False,
) -> Decimal | None:
    file_path = _resolve_price_file_path(
        symbol=symbol,
        prices_folder=prices_folder,
        chain=chain,
        use_lp_prices=use_lp_prices,
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
    chain: str | None = None,
    use_lp_prices: bool = False,
    currency_metadata: dict[str, dict[str, Any]] | None = None,
    fallback_to_oldest: bool = False,
) -> Decimal | None:
    price = get_price_on_or_before(
        symbol=symbol,
        as_of_date=as_of_date,
        prices_folder=prices_folder,
        chain=chain,
        use_lp_prices=use_lp_prices,
        fallback_to_oldest=fallback_to_oldest,
    )
    if price is None:
        return None

    metadata = currency_metadata or CURRENCY_METADATA
    currency = PRICE_CURRENCY_OVERRIDES.get(
        symbol,
        str(metadata.get(symbol, {}).get("currency", "USD")),
    )
    fx_rate = Decimal(str(get_forex_rate(currency=currency, date=str(_normalize_date(as_of_date)))))
    return price * fx_rate
