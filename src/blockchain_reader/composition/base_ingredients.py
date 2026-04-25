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
UNPRICED_QUANTITY_DUST = Decimal("0.000001")
MAX_EXPANSION_DEPTH = 8
AAVE_SYMBOL_ALIASES: dict[str, str] = {"USD0": "USDT", "USDT0": "USDT", "USDT": "USDT"}
BASE_INGREDIENT_COLUMNS = [
    "Date",
    "Coin",
    "Quantity",
    "ValuationRoute",
    "PriceSymbol",
    "PriceEUR",
    "EstimatedValueEUR",
    "HasDirectExposure",
    "HasProtocolExposure",
    "HasAaveExposure",
]
EXCEPTION_COLUMNS = ["Date", "Coin", "Quantity", "Reason", "EstimatedValueEUR", "Action"]


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


@dataclass
class AggregatedExposure:
    quantity: Decimal = Decimal("0")
    has_direct_exposure: bool = False
    has_protocol_exposure: bool = False
    has_aave_exposure: bool = False


@dataclass(frozen=True)
class PriceResolution:
    route: ValuationRoute
    price_symbol: str | None
    price_eur: Decimal | None


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
        if df.empty:
            continue
        df["date"] = df["date"].dt.normalize()
        protocol_rows[symbol] = df.sort_values("date").reset_index(drop=True)
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
    if df.empty:
        return None
    df["date"] = df["date"].dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _find_row_for_date(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    target = pd.Timestamp(date).normalize()
    eligible = df[df["date"] <= target]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _normalize_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["_row_order"] = range(len(normalized))
    normalized["Date"] = pd.to_datetime(
        normalized["Date"].map(parse_daily_datetime),
        errors="coerce",
    )
    normalized["Coin"] = normalized["Coin"].map(sanitize_symbol)
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce").fillna(0.0)
    normalized = normalized.dropna(subset=["Date"])
    normalized["Date"] = normalized["Date"].dt.normalize()
    normalized = normalized[normalized["Coin"] != ""]
    normalized = normalized.sort_values(["Date", "_row_order"])
    normalized = normalized.groupby(["Date", "Coin"], as_index=False).tail(1)
    return (
        normalized[["Date", "Coin", "Quantity"]]
        .sort_values(["Date", "Coin"])
        .reset_index(drop=True)
    )


def _resolve_compose_end_date(
    *,
    snapshots: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    if as_of_date is None:
        requested_end_date = pd.Timestamp.today().normalize()
    else:
        requested_end_date = pd.Timestamp(as_of_date).normalize()

    if snapshots.empty:
        return requested_end_date

    latest_snapshot_date = pd.Timestamp(snapshots["Date"].max()).normalize()
    return max(latest_snapshot_date, requested_end_date)


def _build_daily_holdings_state(
    snapshots: pd.DataFrame,
    *,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()

    start_date = pd.Timestamp(snapshots["Date"].min()).normalize()
    latest_snapshot_date = pd.Timestamp(snapshots["Date"].max()).normalize()
    effective_end_date = latest_snapshot_date
    if end_date is not None:
        effective_end_date = max(latest_snapshot_date, pd.Timestamp(end_date).normalize())
    calendar = pd.date_range(start=start_date, end=effective_end_date, freq="D")

    pivoted = snapshots.pivot(index="Date", columns="Coin", values="Quantity")
    dense = pivoted.reindex(calendar).sort_index().ffill().fillna(0.0)
    dense.index.name = "Date"
    return dense


def _to_decimal(value: object) -> Decimal:
    if pd.isna(value):
        return Decimal("0")
    return Decimal(str(value))


def _add_exposure(
    exposures: dict[str, AggregatedExposure],
    *,
    symbol: str,
    quantity: Decimal,
    has_direct_exposure: bool,
    has_protocol_exposure: bool,
    has_aave_exposure: bool,
) -> None:
    normalized_symbol = sanitize_symbol(symbol)
    if not normalized_symbol or abs(quantity) <= DUST:
        return

    exposure = exposures[normalized_symbol]
    exposure.quantity += quantity
    exposure.has_direct_exposure = exposure.has_direct_exposure or has_direct_exposure
    exposure.has_protocol_exposure = exposure.has_protocol_exposure or has_protocol_exposure
    exposure.has_aave_exposure = exposure.has_aave_exposure or has_aave_exposure


def _expand_symbol(
    *,
    symbol: str,
    quantity: Decimal,
    date: pd.Timestamp,
    ctx: ExpansionContext,
    exposures: dict[str, AggregatedExposure],
    has_direct_exposure: bool,
    has_protocol_exposure: bool,
    has_aave_exposure: bool,
    depth: int = 0,
) -> None:
    normalized_symbol = sanitize_symbol(symbol)
    if not normalized_symbol or abs(quantity) <= DUST:
        return

    route = _resolve_route(symbol=normalized_symbol, ctx=ctx)
    current_has_protocol = has_protocol_exposure or route == ValuationRoute.PROTOCOL_DERIVED
    terminal_symbol = normalized_symbol
    if route == ValuationRoute.DIRECT:
        terminal_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)
        terminal_symbol = terminal_symbol or normalized_symbol

    if depth > MAX_EXPANSION_DEPTH:
        _add_exposure(
            exposures,
            symbol=terminal_symbol,
            quantity=quantity,
            has_direct_exposure=has_direct_exposure,
            has_protocol_exposure=current_has_protocol,
            has_aave_exposure=has_aave_exposure,
        )
        return

    protocol_history = ctx.protocol_rows.get(normalized_symbol)
    if protocol_history is None:
        _add_exposure(
            exposures,
            symbol=terminal_symbol,
            quantity=quantity,
            has_direct_exposure=has_direct_exposure,
            has_protocol_exposure=current_has_protocol,
            has_aave_exposure=has_aave_exposure,
        )
        return

    row = _find_row_for_date(df=protocol_history, date=date)
    if row is None:
        _add_exposure(
            exposures,
            symbol=terminal_symbol,
            quantity=quantity,
            has_direct_exposure=has_direct_exposure,
            has_protocol_exposure=current_has_protocol,
            has_aave_exposure=has_aave_exposure,
        )
        return

    expanded = False
    for column in row.index:
        if not isinstance(column, str) or not column.startswith("asset_"):
            continue
        if pd.isna(row[column]):
            continue
        per_unit = _to_decimal(row[column])
        if abs(per_unit) <= DUST:
            continue
        expanded = True
        _expand_symbol(
            symbol=column.replace("asset_", "", 1),
            quantity=quantity * per_unit,
            date=date,
            ctx=ctx,
            exposures=exposures,
            has_direct_exposure=has_direct_exposure,
            has_protocol_exposure=current_has_protocol,
            has_aave_exposure=has_aave_exposure,
            depth=depth + 1,
        )

    if expanded:
        return

    _add_exposure(
        exposures,
        symbol=terminal_symbol,
        quantity=quantity,
        has_direct_exposure=has_direct_exposure,
        has_protocol_exposure=current_has_protocol,
        has_aave_exposure=has_aave_exposure,
    )


def _resolve_price(
    *,
    symbol: str,
    date: pd.Timestamp,
    ctx: ExpansionContext,
) -> PriceResolution:
    normalized_symbol = sanitize_symbol(symbol)
    route = _resolve_route(symbol=normalized_symbol, ctx=ctx)
    canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)

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
            chain=ctx.chain,
            use_lp_prices=route == ValuationRoute.PROTOCOL_DERIVED,
            fallback_to_oldest=False,
        )
        if price is not None:
            return PriceResolution(route=route, price_symbol=candidate, price_eur=price)

    return PriceResolution(route=route, price_symbol=None, price_eur=None)


def _is_known_symbol(symbol: str, ctx: ExpansionContext) -> bool:
    if not ctx.known_symbols:
        return True

    normalized_symbol = sanitize_symbol(symbol)
    canonical_symbol = canonicalize_symbol(normalized_symbol, symbol_family=ctx.symbol_family)
    return normalized_symbol in ctx.known_symbols or canonical_symbol in ctx.known_symbols


def _is_material_output(
    *,
    quantity: Decimal,
    estimated_value_eur: Decimal | None,
) -> bool:
    if estimated_value_eur is not None:
        return abs(estimated_value_eur) >= VALUE_DUST_EUR
    return abs(quantity) > UNPRICED_QUANTITY_DUST


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


def _build_row_issue(
    *,
    symbol: str,
    resolution: PriceResolution,
    ctx: ExpansionContext,
) -> tuple[str | None, str | None]:
    if not _is_known_symbol(symbol=symbol, ctx=ctx):
        return "unknown_symbol_material", "add token metadata or protocol mapping"

    if resolution.price_eur is not None:
        return None, None

    if resolution.route == ValuationRoute.PROTOCOL_DERIVED:
        return "protocol_symbol_missing_price", "run protocol pipeline or add protocol adapter"
    if resolution.route == ValuationRoute.AAVE:
        return (
            "aave_symbol_missing_overlay_price",
            "fix aave overlay mapping, then rerun aave/composer",
        )
    return "known_symbol_missing_price", "add direct price file or direct symbol metadata"


def _apply_aave_overlay(
    *,
    exposures: dict[str, AggregatedExposure],
    date: pd.Timestamp,
    ctx: ExpansionContext,
    exceptions: list[dict[str, object]],
) -> None:
    if ctx.aave_overlay is None:
        return

    row = _find_row_for_date(df=ctx.aave_overlay, date=date)
    if row is None:
        return

    for column in row.index:
        if not isinstance(column, str) or not column.startswith("net_"):
            continue
        if pd.isna(row[column]):
            continue

        quantity = _to_decimal(row[column])
        if abs(quantity) <= DUST:
            continue

        raw_symbol = column.replace("net_", "", 1)
        normalized_symbol = _normalize_aave_symbol(raw_symbol)
        if not normalized_symbol:
            exceptions.append(
                _build_exception_row(
                    date=date,
                    symbol=raw_symbol or "<empty>",
                    quantity=quantity,
                    reason="unknown_aave_overlay_symbol",
                    action="fix aave overlay header or token metadata",
                    estimated_value_eur=None,
                )
            )
            continue

        if not _is_known_symbol(symbol=normalized_symbol, ctx=ctx):
            resolution = _resolve_price(symbol=normalized_symbol, date=date, ctx=ctx)
            estimated_value_eur = (
                quantity * resolution.price_eur if resolution.price_eur is not None else None
            )
            if _is_material_output(quantity=quantity, estimated_value_eur=estimated_value_eur):
                exceptions.append(
                    _build_exception_row(
                        date=date,
                        symbol=normalized_symbol,
                        quantity=quantity,
                        reason="unknown_aave_overlay_symbol",
                        action="fix aave overlay header or token metadata",
                        estimated_value_eur=estimated_value_eur,
                    )
                )
            continue

        _expand_symbol(
            symbol=normalized_symbol,
            quantity=quantity,
            date=date,
            ctx=ctx,
            exposures=exposures,
            has_direct_exposure=False,
            has_protocol_exposure=False,
            has_aave_exposure=True,
        )


def _build_daily_rows(
    *,
    date: pd.Timestamp,
    exposures: dict[str, AggregatedExposure],
    ctx: ExpansionContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []

    for symbol, exposure in sorted(exposures.items()):
        quantity = exposure.quantity
        if abs(quantity) <= DUST:
            continue

        resolution = _resolve_price(symbol=symbol, date=date, ctx=ctx)
        estimated_value_eur = (
            quantity * resolution.price_eur if resolution.price_eur is not None else None
        )
        if not _is_material_output(quantity=quantity, estimated_value_eur=estimated_value_eur):
            continue

        reason, action = _build_row_issue(
            symbol=symbol,
            resolution=resolution,
            ctx=ctx,
        )

        rows.append(
            {
                "Date": format_daily_datetime(date),
                "Coin": symbol,
                "Quantity": float(quantity),
                "ValuationRoute": resolution.route.value,
                "PriceSymbol": resolution.price_symbol or "",
                "PriceEUR": float(resolution.price_eur) if resolution.price_eur is not None else "",
                "EstimatedValueEUR": (
                    float(estimated_value_eur) if estimated_value_eur is not None else ""
                ),
                "HasDirectExposure": exposure.has_direct_exposure,
                "HasProtocolExposure": exposure.has_protocol_exposure,
                "HasAaveExposure": exposure.has_aave_exposure,
            }
        )

        if reason is None or action is None:
            continue

        exceptions.append(
            _build_exception_row(
                date=date,
                symbol=symbol,
                quantity=quantity,
                reason=reason,
                action=action,
                estimated_value_eur=estimated_value_eur,
            )
        )

    return rows, exceptions


def compose_base_ingredients(
    chain: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> Path:
    clear_price_cache()

    snapshots_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv"
    snapshots = _normalize_snapshot_df(pd.read_csv(snapshots_path))
    end_date = _resolve_compose_end_date(snapshots=snapshots, as_of_date=as_of_date)
    daily_holdings = _build_daily_holdings_state(snapshots=snapshots, end_date=end_date)
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

    for date, holdings in daily_holdings.iterrows():
        exposures: dict[str, AggregatedExposure] = defaultdict(AggregatedExposure)
        _apply_aave_overlay(
            exposures=exposures,
            date=pd.Timestamp(date).normalize(),
            ctx=ctx,
            exceptions=exceptions,
        )

        for symbol, quantity_value in holdings.items():
            quantity = _to_decimal(quantity_value)
            if abs(quantity) <= DUST:
                continue

            normalized_symbol = sanitize_symbol(symbol)
            route = _resolve_route(symbol=normalized_symbol, ctx=ctx)
            if normalized_symbol in ctx.aave_wrapper_symbols or route == ValuationRoute.AAVE:
                continue

            _expand_symbol(
                symbol=normalized_symbol,
                quantity=quantity,
                date=pd.Timestamp(date).normalize(),
                ctx=ctx,
                exposures=exposures,
                has_direct_exposure=route == ValuationRoute.DIRECT,
                has_protocol_exposure=route == ValuationRoute.PROTOCOL_DERIVED,
                has_aave_exposure=False,
            )

        date_rows, date_exceptions = _build_daily_rows(
            date=pd.Timestamp(date).normalize(),
            exposures=exposures,
            ctx=ctx,
        )
        rows.extend(date_rows)
        exceptions.extend(date_exceptions)

    output_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / f"{chain}_base_ingredients.csv"
    pd.DataFrame(rows, columns=BASE_INGREDIENT_COLUMNS).to_csv(output_path, index=False)

    exceptions_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / f"{chain}_base_ingredients_exceptions.csv"
    pd.DataFrame(exceptions, columns=EXCEPTION_COLUMNS).to_csv(exceptions_path, index=False)
    print(f"[compose] Saved to {output_path}")
    print(f"[compose] Exceptions saved to {exceptions_path} ({len(exceptions)} rows)")
    return output_path
