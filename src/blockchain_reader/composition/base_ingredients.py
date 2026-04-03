from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from blockchain_reader.datetime_utils import format_daily_datetime, parse_daily_datetime
from blockchain_reader.shared.prices import clear_price_cache, get_price_eur_on_or_before
from blockchain_reader.shared.token_metadata import load_token_metadata
from blockchain_reader.shared.valuation_routes import (
    ValuationRoute,
    build_symbol_protocol_map,
    classify_valuation_route,
)
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
AAVE_SYMBOL_ALIASES: dict[str, str] = {"USD0": "USDT", "USDT0": "USDT", "USDT": "USDT"}


@dataclass(frozen=True)
class ExpansionContext:
    chain: str
    protocol_rows: dict[str, pd.DataFrame]
    symbol_protocol: dict[str, str]
    protocol_derived_symbols: set[str]
    symbol_family: dict[str, str]
    aave_overlay: pd.DataFrame | None
    aave_wrapper_symbols: set[str]
    known_symbols: set[str]


def _normalize_aave_symbol(symbol: str) -> str:
    normalized = sanitize_symbol(symbol)
    if not normalized:
        return ""
    return AAVE_SYMBOL_ALIASES.get(normalized.upper(), normalized)


def _resolve_route(symbol: str, ctx: ExpansionContext) -> ValuationRoute:
    return classify_valuation_route(
        symbol=symbol,
        symbol_protocol=ctx.symbol_protocol,
        protocol_derived_symbols=ctx.protocol_derived_symbols,
    )


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
        symbol = sanitize_symbol(csv_path.stem[len(chain) + 1 :])
        if not symbol:
            continue
        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"].map(parse_daily_datetime), errors="coerce")
        df = df.dropna(subset=["date"])
        protocol_rows[symbol] = df.sort_values("date")
    return protocol_rows


def _load_aave_overlay(chain: str) -> pd.DataFrame | None:
    overlay_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / "aave" / f"{chain}_aave_daily_exposure.csv"
    if not overlay_path.exists():
        return None

    df = pd.read_csv(overlay_path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"].map(parse_daily_datetime), errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date")


def _find_row_for_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    target = pd.to_datetime(date).normalize()
    eligible = df[df["date"] <= target]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _estimate_value_eur(
    *,
    symbol: str,
    quantity: Decimal,
    date: pd.Timestamp,
    chain: str,
    symbol_family: dict[str, str],
    route: ValuationRoute,
) -> Decimal | None:
    normalized_symbol = sanitize_symbol(symbol)
    canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=symbol_family)

    candidates = [normalized_symbol]
    if (
        route == ValuationRoute.DIRECT
        and canonical_symbol
        and canonical_symbol != normalized_symbol
    ):
        candidates.append(canonical_symbol)

    for candidate in candidates:
        price = get_price_eur_on_or_before(
            symbol=candidate,
            as_of_date=date,
            prices_folder=PRICES_FOLDER,
            chain=chain,
            use_lp_prices=route == ValuationRoute.PROTOCOL_DERIVED,
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
        "Date": format_daily_datetime(date),
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
        route = _resolve_route(symbol=normalized_symbol, ctx=ctx)
        canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)
        output_symbol = normalized_symbol
        if route == ValuationRoute.DIRECT:
            output_symbol = canonical_symbol or normalized_symbol
        if not output_symbol:
            continue

        estimated_value_eur = _estimate_value_eur(
            symbol=output_symbol,
            quantity=qty,
            date=date,
            chain=ctx.chain,
            symbol_family=ctx.symbol_family,
            route=route,
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
            reason = "known_symbol_missing_price"
            action = "add direct price file or direct symbol metadata"
            if route == ValuationRoute.PROTOCOL_DERIVED:
                reason = "protocol_symbol_missing_price"
                action = "run protocol pipeline or add protocol adapter"
            if route == ValuationRoute.AAVE:
                reason = "aave_symbol_missing_overlay_price"
                action = "fix aave overlay mapping, then rerun aave/composer"
            exceptions.append(
                _build_exception_row(
                    date=date,
                    symbol=output_symbol,
                    quantity=qty,
                    reason=reason,
                    action=action,
                    estimated_value_eur=estimated_value_eur,
                )
            )

        rows.append(
            {
                "Date": format_daily_datetime(date),
                "Coin": output_symbol,
                "Quantity": float(qty),
            }
        )

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
    route = _resolve_route(symbol=normalized_symbol, ctx=ctx)
    terminal_symbol = normalized_symbol
    if route == ValuationRoute.DIRECT:
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
        normalized_symbol = _normalize_aave_symbol(raw_symbol)
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
                chain=ctx.chain,
                symbol_family=ctx.symbol_family,
                route=ValuationRoute.DIRECT,
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


def _normalize_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["Date"] = pd.to_datetime(
        normalized["Date"].map(parse_daily_datetime),
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["Date"])
    normalized["Date"] = normalized["Date"].dt.normalize()
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce").fillna(0)
    return normalized.sort_values(["Date", "Coin"])


def _collect_composition_dates(
    *,
    snapshots: pd.DataFrame,
    ctx: ExpansionContext,
) -> list[pd.Timestamp]:
    if snapshots.empty:
        return []

    min_date = snapshots["Date"].min()
    max_date = snapshots["Date"].max()
    dates = set(pd.to_datetime(snapshots["Date"]).dt.normalize())
    for protocol_df in ctx.protocol_rows.values():
        if protocol_df.empty:
            continue
        for date in pd.to_datetime(protocol_df["date"]).dropna().dt.normalize():
            if min_date <= date <= max_date:
                dates.add(date)

    if ctx.aave_overlay is not None and not ctx.aave_overlay.empty:
        for date in pd.to_datetime(ctx.aave_overlay["date"]).dropna().dt.normalize():
            if min_date <= date <= max_date:
                dates.add(date)

    return sorted(dates)


def _build_snapshot_groups(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    grouped: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, group in df.groupby("Date"):
        grouped[pd.Timestamp(date).normalize()] = group
    return grouped


def _should_carry_protocol_position(symbol: str, ctx: ExpansionContext) -> bool:
    normalized_symbol = sanitize_symbol(symbol)
    if not normalized_symbol:
        return False
    return normalized_symbol in ctx.protocol_rows


def _update_snapshot_state(
    current_quantities: dict[str, Decimal],
    group: pd.DataFrame | None,
) -> None:
    if group is None:
        return

    for _, snap in group.iterrows():
        symbol = sanitize_symbol(str(snap["Coin"]))
        if not symbol:
            continue
        current_quantities[symbol] = Decimal(str(snap["Quantity"]))


def _expand_carried_protocol_positions(
    *,
    date: pd.Timestamp,
    current_quantities: dict[str, Decimal],
    ctx: ExpansionContext,
    out: dict[str, Decimal],
) -> None:
    for symbol, quantity in current_quantities.items():
        if abs(quantity) <= DUST:
            continue
        if symbol in ctx.aave_wrapper_symbols:
            continue
        if not _should_carry_protocol_position(symbol=symbol, ctx=ctx):
            continue
        _expand_symbol(symbol=symbol, quantity=quantity, date=date, ctx=ctx, out=out)


def _expand_snapshot_rows_for_date(
    *,
    date: pd.Timestamp,
    group: pd.DataFrame | None,
    ctx: ExpansionContext,
    out: dict[str, Decimal],
) -> None:
    if group is None:
        return

    for _, snap in group.iterrows():
        symbol = sanitize_symbol(str(snap["Coin"]))
        if not symbol or symbol in ctx.aave_wrapper_symbols:
            continue
        quantity = Decimal(str(snap["Quantity"]))
        if abs(quantity) <= DUST:
            continue
        if _should_carry_protocol_position(symbol=symbol, ctx=ctx):
            continue
        _expand_symbol(symbol=symbol, quantity=quantity, date=date, ctx=ctx, out=out)


def compose_base_ingredients(chain: str) -> Path:
    clear_price_cache()

    snapshots_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv"
    df = _normalize_snapshot_df(pd.read_csv(snapshots_path))
    token_metadata = load_token_metadata(
        chain=chain,
        tokens_folder=TOKENS_FOLDER,
    )
    symbol_family = build_symbol_family_map(token_metadata=token_metadata)
    protocol_rows = _load_protocol_rows(chain=chain)

    ctx = ExpansionContext(
        chain=chain,
        protocol_rows=protocol_rows,
        symbol_protocol=build_symbol_protocol_map(token_metadata=token_metadata),
        protocol_derived_symbols=set(protocol_rows.keys()),
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
    composition_dates = _collect_composition_dates(snapshots=df, ctx=ctx)
    snapshot_groups = _build_snapshot_groups(df=df)
    current_quantities: dict[str, Decimal] = {}

    for date in composition_dates:
        group = snapshot_groups.get(date)
        _update_snapshot_state(current_quantities=current_quantities, group=group)
        out: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        unknown_count, dust_count = _apply_aave_overlay(
            out=out,
            date=date,
            ctx=ctx,
            exceptions=exceptions,
        )
        unknown_overlay_symbols += unknown_count
        dust_overlay_values += dust_count

        _expand_carried_protocol_positions(
            date=date,
            current_quantities=current_quantities,
            ctx=ctx,
            out=out,
        )
        _expand_snapshot_rows_for_date(
            date=date,
            group=group,
            ctx=ctx,
            out=out,
        )

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
