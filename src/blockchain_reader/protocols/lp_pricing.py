from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_reader.symbols import sanitize_symbol
from file_paths import PRICES_FOLDER, PROTOCOL_UNDERLYING_TOKEN_FOLDER, TOKENS_FOLDER
from price_history.price_data_utils import load_price_csv, merge_price_frames, save_price_csv

DUST = Decimal("0.000000000001")
MAX_RECURSION_DEPTH = 10


@dataclass(frozen=True)
class SymbolMetadata:
    price_source: str
    family: str


@dataclass
class PricingContext:
    symbol_metadata: dict[str, SymbolMetadata]
    protocol_rows: dict[str, pd.DataFrame]
    price_cache: dict[str, pd.DataFrame]


def _parse_protocol_date(raw_value: object) -> date | None:
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value

    value = str(raw_value or "").strip()
    if not value:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _load_token_metadata(chain: str) -> dict[str, dict[str, Any]]:
    token_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    if not token_path.exists():
        return {}

    with open(token_path, "r") as f:
        raw = json.load(f)
    return {str(addr).lower(): meta for addr, meta in raw.items() if isinstance(meta, dict)}


def _build_symbol_metadata(token_metadata: dict[str, dict[str, Any]]) -> dict[str, SymbolMetadata]:
    merged: dict[str, dict[str, str]] = {}

    for meta in token_metadata.values():
        symbol = sanitize_symbol(meta.get("symbol"))
        if not symbol:
            continue

        price_source = sanitize_symbol(meta.get("price_source"))
        family = sanitize_symbol(meta.get("family"))
        current = merged.get(symbol, {"price_source": "", "family": ""})

        if not current["price_source"] and price_source:
            current["price_source"] = price_source
        if not current["family"] and family:
            current["family"] = family
        merged[symbol] = current

    return {
        symbol: SymbolMetadata(
            price_source=meta["price_source"],
            family=meta["family"],
        )
        for symbol, meta in merged.items()
    }


def _load_protocol_rows(chain: str) -> dict[str, pd.DataFrame]:
    rows: dict[str, pd.DataFrame] = {}
    root = PROTOCOL_UNDERLYING_TOKEN_FOLDER
    if not root.exists():
        return rows

    for csv_path in root.rglob(f"{chain}_*.csv"):
        if csv_path.parent.name == "aave":
            continue
        if csv_path.name == f"{chain}_base_ingredients.csv":
            continue

        symbol = sanitize_symbol(csv_path.stem[len(chain) + 1 :])
        if not symbol:
            continue

        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            continue

        df["date"] = df["date"].map(_parse_protocol_date)
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            continue

        rows[symbol] = df

    return rows


def _load_price_history(symbol: str, ctx: PricingContext) -> pd.DataFrame:
    if symbol in ctx.price_cache:
        return ctx.price_cache[symbol]

    frame = load_price_csv(file_path=PRICES_FOLDER / f"{symbol}.csv")
    if frame.empty:
        ctx.price_cache[symbol] = frame
        return frame

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Price"])[["Date", "Price"]].sort_values("Date")
    ctx.price_cache[symbol] = frame
    return frame


def _price_from_history(symbol: str, target_date: date, ctx: PricingContext) -> Decimal | None:
    history = _load_price_history(symbol=symbol, ctx=ctx)
    if history.empty:
        return None

    eligible = history[history["Date"] <= target_date]
    if eligible.empty:
        return None

    return Decimal(str(eligible.iloc[-1]["Price"]))


def _find_protocol_row(symbol: str, target_date: date, ctx: PricingContext) -> pd.Series | None:
    df = ctx.protocol_rows.get(symbol)
    if df is None or df.empty:
        return None

    eligible = df[df["date"] <= target_date]
    if eligible.empty:
        return None

    return eligible.iloc[-1]


def _resolve_from_protocol(
    symbol: str,
    target_date: date,
    ctx: PricingContext,
    visited: set[str],
    depth: int,
) -> Decimal | None:
    if depth > MAX_RECURSION_DEPTH or symbol in visited:
        return None

    row = _find_protocol_row(symbol=symbol, target_date=target_date, ctx=ctx)
    if row is None:
        return None

    asset_columns = [col for col in row.index if isinstance(col, str) and col.startswith("asset_")]
    if not asset_columns:
        return None

    total = Decimal(0)
    next_visited = set(visited)
    next_visited.add(symbol)

    for column in asset_columns:
        quantity_raw = row[column]
        if pd.isna(quantity_raw):
            return None

        quantity = Decimal(str(quantity_raw))
        if abs(quantity) <= DUST:
            continue

        component_symbol = sanitize_symbol(column.replace("asset_", "", 1))
        if not component_symbol:
            return None

        component_price = resolve_symbol_price(
            symbol=component_symbol,
            target_date=target_date,
            ctx=ctx,
            visited=next_visited,
            depth=depth + 1,
        )
        if component_price is None:
            return None

        total += quantity * component_price

    return total


def resolve_symbol_price(
    symbol: str,
    target_date: date,
    ctx: PricingContext,
    visited: set[str] | None = None,
    depth: int = 0,
) -> Decimal | None:
    normalized = sanitize_symbol(symbol)
    if not normalized:
        return None
    if depth > MAX_RECURSION_DEPTH:
        return None

    direct = _price_from_history(symbol=normalized, target_date=target_date, ctx=ctx)
    if direct is not None:
        return direct

    metadata = ctx.symbol_metadata.get(normalized)
    if metadata and metadata.price_source and metadata.price_source != normalized:
        source_price = _price_from_history(
            symbol=metadata.price_source,
            target_date=target_date,
            ctx=ctx,
        )
        if source_price is not None:
            return source_price

    if metadata and metadata.family and metadata.family not in {normalized, metadata.price_source}:
        family_price = _price_from_history(
            symbol=metadata.family,
            target_date=target_date,
            ctx=ctx,
        )
        if family_price is not None:
            return family_price

    return _resolve_from_protocol(
        symbol=normalized,
        target_date=target_date,
        ctx=ctx,
        visited=visited or set(),
        depth=depth,
    )


def _build_incoming_prices(symbol: str, df: pd.DataFrame, ctx: PricingContext) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.sort_values("date").iterrows():
        target_date = row["date"]
        if not isinstance(target_date, date):
            continue

        price = resolve_symbol_price(symbol=symbol, target_date=target_date, ctx=ctx)
        if price is None:
            continue

        rows.append({"Date": target_date, "Price": float(price)})

    if not rows:
        return pd.DataFrame(columns=["Date", "Price"])

    return pd.DataFrame(rows, columns=["Date", "Price"])


def generate_protocol_lp_price_files(chain: str) -> list[Path]:
    """
    Builds protocol token price files from non-AAVE protocol-underlying exports.

    args:
        chain: Chain identifier used for protocol-underlying file discovery.

    returns:
        List of updated price CSV paths in data/prices.
    """
    token_metadata = _load_token_metadata(chain=chain)
    ctx = PricingContext(
        symbol_metadata=_build_symbol_metadata(token_metadata=token_metadata),
        protocol_rows=_load_protocol_rows(chain=chain),
        price_cache={},
    )

    updated_files: list[Path] = []
    for symbol, df in sorted(ctx.protocol_rows.items(), key=lambda item: item[0]):
        incoming = _build_incoming_prices(symbol=symbol, df=df, ctx=ctx)
        if incoming.empty:
            continue

        output_path = PRICES_FOLDER / f"{symbol}.csv"
        existing = load_price_csv(file_path=output_path)
        merged = merge_price_frames(existing=existing, incoming=incoming)
        save_price_csv(file_path=output_path, frame=merged)
        updated_files.append(output_path)

    return updated_files
