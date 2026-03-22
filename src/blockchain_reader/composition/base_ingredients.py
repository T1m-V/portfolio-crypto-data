from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from blockchain_reader.shared.prices import clear_price_cache, get_price_eur_on_or_before
from blockchain_reader.shared.token_metadata import load_token_metadata
from blockchain_reader.symbols import (
    build_known_canonical_symbols,
    build_symbol_family_map,
    canonicalize_symbol,
    sanitize_symbol,
)
from file_paths import (
    BLOCKCHAIN_SNAPSHOT_FOLDER,
    PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
)

DUST = Decimal("0.000000000001")
VALUE_DUST_EUR = Decimal("0.01")


@dataclass(frozen=True)
class ExpansionContext:
    chain: str
    protocol_rows: dict[str, pd.DataFrame]
    symbol_family: dict[str, str]
    aave_overlay: pd.DataFrame | None
    aave_wrapper_symbols: set[str]
    known_symbols: set[str]


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


def _estimate_value_eur(
    *,
    symbol: str,
    quantity: Decimal,
    date: pd.Timestamp,
    symbol_family: dict[str, str],
) -> Decimal | None:
    normalized_symbol = sanitize_symbol(symbol)
    canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=symbol_family)

    candidates: list[str] = []
    for candidate in (normalized_symbol, canonical_symbol):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        price = get_price_eur_on_or_before(
            symbol=candidate,
            as_of_date=date,
            prices_folder=PRICES_FOLDER,
            fallback_to_oldest=False,
        )
        if price is not None:
            return abs(quantity) * price

    return None


def _build_exception_row(
    *,
    date: pd.Timestamp,
    symbol: str,
    quantity: Decimal,
    reason: str,
    action: str,
    estimated_value_eur: Decimal | None,
) -> dict[str, object]:
    estimated = float(estimated_value_eur) if estimated_value_eur is not None else ""
    return {
        "Date": date.date(),
        "Coin": symbol,
        "Quantity": float(quantity),
        "Reason": reason,
        "EstimatedValueEUR": estimated,
        "Action": action,
    }


def _filter_composed_quantities(
    *,
    out: dict[str, Decimal],
    date: pd.Timestamp,
    ctx: ExpansionContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []

    for symbol, qty in sorted(out.items()):
        if abs(qty) <= DUST:
            continue

        normalized_symbol = sanitize_symbol(symbol)
        canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)
        output_symbol = canonical_symbol or normalized_symbol
        if not output_symbol:
            continue

        estimated_value_eur = _estimate_value_eur(
            symbol=output_symbol,
            quantity=qty,
            date=date,
            symbol_family=ctx.symbol_family,
        )
        is_known = not ctx.known_symbols or (
            normalized_symbol in ctx.known_symbols or canonical_symbol in ctx.known_symbols
        )

        if not is_known:
            if estimated_value_eur is not None and estimated_value_eur < VALUE_DUST_EUR:
                continue
            exceptions.append(
                _build_exception_row(
                    date=date,
                    symbol=output_symbol,
                    quantity=qty,
                    reason="unknown_symbol_material",
                    action="add token metadata or protocol mapping",
                    estimated_value_eur=estimated_value_eur,
                )
            )
            continue

        if estimated_value_eur is not None:
            if estimated_value_eur < VALUE_DUST_EUR:
                continue
        else:
            if abs(qty) <= DUST:
                continue
            exceptions.append(
                _build_exception_row(
                    date=date,
                    symbol=output_symbol,
                    quantity=qty,
                    reason="known_symbol_missing_price",
                    action="add price file or price_source mapping",
                    estimated_value_eur=estimated_value_eur,
                )
            )

        rows.append({"Date": date.date(), "Coin": output_symbol, "Quantity": float(qty)})

    return rows, exceptions


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
    out: dict[str, Decimal],
    date: pd.Timestamp,
    ctx: ExpansionContext,
    exceptions: list[dict[str, object]],
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
        normalized_symbol = sanitize_symbol(raw_symbol)
        canonical_symbol = canonicalize_symbol(raw_symbol, symbol_family=ctx.symbol_family)
        if not normalized_symbol:
            unknown_symbol_count += 1
            if abs(numeric_value) > DUST:
                exceptions.append(
                    _build_exception_row(
                        date=date,
                        symbol=raw_symbol or "<empty>",
                        quantity=numeric_value,
                        reason="unknown_aave_overlay_symbol",
                        action="fix aave overlay header or token metadata",
                        estimated_value_eur=None,
                    )
                )
            continue
        if ctx.known_symbols and (
            normalized_symbol not in ctx.known_symbols and canonical_symbol not in ctx.known_symbols
        ):
            unknown_symbol_count += 1
            estimated_value_eur = _estimate_value_eur(
                symbol=normalized_symbol,
                quantity=numeric_value,
                date=date,
                symbol_family=ctx.symbol_family,
            )
            if estimated_value_eur is None or estimated_value_eur >= VALUE_DUST_EUR:
                exceptions.append(
                    _build_exception_row(
                        date=date,
                        symbol=normalized_symbol,
                        quantity=numeric_value,
                        reason="unknown_aave_overlay_symbol",
                        action="fix aave overlay header or token metadata",
                        estimated_value_eur=estimated_value_eur,
                    )
                )
            continue
        _expand_symbol(
            symbol=normalized_symbol,
            quantity=numeric_value,
            date=date,
            ctx=ctx,
            out=out,
        )

    return unknown_symbol_count, dust_value_count


def compose_base_ingredients(chain: str) -> Path:
    clear_price_cache()

    snapshots_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv"
    df = pd.read_csv(snapshots_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    token_metadata = load_token_metadata(
        chain=chain,
        tokens_folder=TOKENS_FOLDER,
    )
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
    exceptions: list[dict[str, object]] = []
    unknown_overlay_symbols = 0
    dust_overlay_values = 0
    grouped = df.groupby("Date")
    for date, group in grouped:
        out: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        unknown_count, dust_count = _apply_aave_overlay(
            out=out,
            date=date,
            ctx=ctx,
            exceptions=exceptions,
        )
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

        kept_rows, row_exceptions = _filter_composed_quantities(out=out, date=date, ctx=ctx)
        rows.extend(kept_rows)
        exceptions.extend(row_exceptions)

    out_df = pd.DataFrame(rows)
    output_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / f"{chain}_base_ingredients.csv"
    out_df.to_csv(output_path, index=False)

    exceptions_df = pd.DataFrame(
        exceptions,
        columns=["Date", "Coin", "Quantity", "Reason", "EstimatedValueEUR", "Action"],
    )
    exceptions_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / f"{chain}_base_ingredients_exceptions.csv"
    exceptions_df.to_csv(exceptions_path, index=False)
    print(f"[compose] Saved to {output_path}")
    print(f"[compose] Exceptions saved to {exceptions_path} ({len(exceptions)} rows)")
    if ctx.aave_overlay is not None:
        print(
            "[compose] Aave overlay skips: "
            f"unknown_symbols={unknown_overlay_symbols}, dust_values={dust_overlay_values}"
        )
    return output_path
