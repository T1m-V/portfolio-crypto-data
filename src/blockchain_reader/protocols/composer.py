import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from blockchain_reader.symbols import (
    build_known_canonical_symbols,
    build_symbol_family_map,
    canonicalize_symbol,
    sanitize_symbol,
)
from file_paths import BLOCKCHAIN_SNAPSHOT_FOLDER, PROTOCOL_UNDERLYING_TOKEN_FOLDER, TOKENS_FOLDER

DUST = Decimal("0.000000000001")


@dataclass(frozen=True)
class ExpansionContext:
    chain: str
    protocol_rows: dict[str, pd.DataFrame]
    symbol_family: dict[str, str]
    aave_overlay: pd.DataFrame | None
    aave_wrapper_symbols: set[str]
    known_symbols: set[str]


def _load_token_metadata(chain: str) -> dict[str, dict[str, object]]:
    token_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    with open(token_path, "r") as f:
        raw = json.load(f)
    return {str(addr).lower(): meta for addr, meta in raw.items() if isinstance(meta, dict)}


def _load_aave_wrapper_symbols(token_metadata: dict[str, dict[str, object]]) -> set[str]:
    wrappers: set[str] = set()
    for meta in token_metadata.values():
        if meta.get("protocol") != "aave":
            continue
        symbol = sanitize_symbol(str(meta.get("symbol", "")))
        if symbol:
            wrappers.add(symbol)
    return wrappers


def _load_protocol_rows(chain: str) -> dict[str, pd.DataFrame]:
    protocol_rows: dict[str, pd.DataFrame] = {}
    root = PROTOCOL_UNDERLYING_TOKEN_FOLDER
    if not root.exists():
        return protocol_rows

    for csv_path in root.rglob(f"{chain}_*.csv"):
        if csv_path.parent.name == "aave":
            continue
        if csv_path.name in (
            f"{chain}_aave_daily_exposure.csv",
            f"{chain}_base_ingredients.csv",
        ):
            continue
        symbol = csv_path.stem[len(chain) + 1 :]
        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        protocol_rows[symbol] = df.sort_values("date")
    return protocol_rows


def _load_aave_overlay(chain: str) -> pd.DataFrame | None:
    overlay_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / "aave" / f"{chain}_aave_daily_exposure.csv"
    if not overlay_path.exists():
        return None

    df = pd.read_csv(overlay_path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date")


def _find_row_for_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["date"] <= date.date()]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _expand_symbol(
    symbol: str,
    quantity: Decimal,
    date: pd.Timestamp,
    ctx: ExpansionContext,
    out: dict[str, Decimal],
    depth: int = 0,
) -> None:
    normalized_symbol = sanitize_symbol(symbol)
    terminal_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)
    terminal_symbol = terminal_symbol or normalized_symbol

    if depth > 8:
        if terminal_symbol:
            out[terminal_symbol] += quantity
        return

    df = ctx.protocol_rows.get(normalized_symbol)
    if df is None:
        if terminal_symbol:
            out[terminal_symbol] += quantity
        return

    row = _find_row_for_date(df=df, date=date)
    if row is None:
        if terminal_symbol:
            out[terminal_symbol] += quantity
        return

    asset_columns = [c for c in row.index if isinstance(c, str) and c.startswith("asset_")]
    if not asset_columns:
        if terminal_symbol:
            out[terminal_symbol] += quantity
        return

    for column in asset_columns:
        base_symbol = column.replace("asset_", "", 1)
        per_unit = Decimal(str(row[column]))
        _expand_symbol(
            symbol=base_symbol,
            quantity=quantity * per_unit,
            date=date,
            ctx=ctx,
            out=out,
            depth=depth + 1,
        )


def _apply_aave_overlay(
    out: dict[str, Decimal], date: pd.Timestamp, ctx: ExpansionContext
) -> tuple[int, int]:
    if ctx.aave_overlay is None:
        return 0, 0

    row = _find_row_for_date(df=ctx.aave_overlay, date=date)
    if row is None:
        return 0, 0

    unknown_symbol_count = 0
    dust_value_count = 0

    for column in row.index:
        if not isinstance(column, str) or not column.startswith("net_"):
            continue
        value = row[column]
        if pd.isna(value):
            continue
        numeric_value = Decimal(str(value))
        if abs(numeric_value) <= DUST:
            dust_value_count += 1
            continue
        raw_symbol = column.replace("net_", "", 1)
        symbol = canonicalize_symbol(raw_symbol, symbol_family=ctx.symbol_family)
        if not symbol or (ctx.known_symbols and symbol not in ctx.known_symbols):
            unknown_symbol_count += 1
            continue
        out[symbol] += numeric_value

    return unknown_symbol_count, dust_value_count


def compose_base_ingredients(chain: str) -> Path:
    snapshots_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_snapshots.csv"
    df = pd.read_csv(snapshots_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    token_metadata = _load_token_metadata(chain=chain)
    symbol_family = build_symbol_family_map(token_metadata=token_metadata)

    ctx = ExpansionContext(
        chain=chain,
        protocol_rows=_load_protocol_rows(chain=chain),
        symbol_family=symbol_family,
        aave_overlay=_load_aave_overlay(chain=chain),
        aave_wrapper_symbols=_load_aave_wrapper_symbols(token_metadata=token_metadata),
        known_symbols=build_known_canonical_symbols(
            token_metadata=token_metadata, symbol_family=symbol_family
        ),
    )

    rows: list[dict[str, object]] = []
    unknown_overlay_symbols = 0
    dust_overlay_values = 0
    grouped = df.groupby("Date")
    for date, group in grouped:
        out: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        unknown_count, dust_count = _apply_aave_overlay(out=out, date=date, ctx=ctx)
        unknown_overlay_symbols += unknown_count
        dust_overlay_values += dust_count

        for _, snap in group.iterrows():
            symbol = sanitize_symbol(str(snap["Coin"]))
            if symbol in ctx.aave_wrapper_symbols:
                continue
            quantity = Decimal(str(snap["Quantity"]))
            if abs(quantity) <= DUST:
                continue
            _expand_symbol(symbol=symbol, quantity=quantity, date=date, ctx=ctx, out=out)

        for symbol, qty in sorted(out.items()):
            if abs(qty) <= DUST:
                continue
            rows.append({"Date": date.date(), "Coin": symbol, "Quantity": float(qty)})

    out_df = pd.DataFrame(rows)
    output_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / f"{chain}_base_ingredients.csv"
    out_df.to_csv(output_path, index=False)
    print(f"[compose] Saved to {output_path}")
    if ctx.aave_overlay is not None:
        print(
            "[compose] Aave overlay skips: "
            f"unknown_symbols={unknown_overlay_symbols}, dust_values={dust_overlay_values}"
        )
    return output_path
