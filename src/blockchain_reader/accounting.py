from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_reader.composition.core import (
    DUST,
    CompositionContext,
    ExposureExpander,
    PriceResolver,
    build_composition_context,
    component_value_weights,
)
from blockchain_reader.datetime_utils import format_daily_datetime, parse_daily_datetime
from blockchain_reader.shared.prices import clear_price_cache
from blockchain_reader.shared.token_metadata import load_token_metadata
from blockchain_reader.shared.valuation_routes import ValuationRoute
from blockchain_reader.symbols import canonicalize_symbol, price_proxy_symbol, sanitize_symbol
from file_paths import (
    BLOCKCHAIN_ACCOUNTING_FOLDER,
    BLOCKCHAIN_SNAPSHOT_FOLDER,
    PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
)

MATERIAL_QUANTITY_THRESHOLD = Decimal("0.0000000001")
MATERIAL_VALUE_THRESHOLD_EUR = Decimal("1.0")
VALUE_DUST_EUR = Decimal("0.01")
AAVE_EXPOSURE_PREFIXES = ("variableDebtArb", "stableDebtArb", "aArb")
AAVE_DEBT_PREFIXES = ("variableDebtArb", "stableDebtArb")
AAVE_SYMBOL_ALIASES: dict[str, str] = {"USD0": "USDT", "USDT0": "USDT", "USDT": "USDT"}

SOURCE_BASE_DAILY_COLUMNS = [
    "Date",
    "Source",
    "BaseCoin",
    "Quantity",
    "MarketValueEUR",
    "PrincipalInvestedEUR",
    "RealizedPnLEUR",
    "ValuationRoute",
    "HasDirectExposure",
    "HasProtocolExposure",
    "HasAaveExposure",
]
BASE_DAILY_COLUMNS = [
    "Date",
    "Coin",
    "Quantity",
    "ValuationRoute",
    "PriceSymbol",
    "PriceEUR",
    "MarketValueEUR",
    "PrincipalInvestedEUR",
    "RealizedPnLEUR",
    "ProfitLossEUR",
    "HasDirectExposure",
    "HasProtocolExposure",
    "HasAaveExposure",
]
ACCOUNTING_ISSUE_COLUMNS = [
    "Date",
    "Source",
    "BaseCoin",
    "Quantity",
    "Reason",
    "Action",
]
INTERNAL_SOURCE_BASE_COLUMNS = [*SOURCE_BASE_DAILY_COLUMNS, "_ResolvedMarketValue"]


@dataclass(frozen=True)
class AccountingArtifactPaths:
    source_base_daily: Path
    base_daily: Path
    issues: Path


@dataclass(frozen=True)
class AccountingBuildResult:
    paths: AccountingArtifactPaths
    rows_written: dict[str, int]
    errors: list[str]


def accounting_paths(chain: str) -> AccountingArtifactPaths:
    root = BLOCKCHAIN_ACCOUNTING_FOLDER / chain
    return AccountingArtifactPaths(
        source_base_daily=root / "source_base_daily.csv",
        base_daily=root / "base_daily.csv",
        issues=root / "issues.csv",
    )


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty(columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns].copy()


def _parse_daily_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.map(parse_daily_datetime), errors="coerce").dt.normalize()


def _normalize_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "Coin", "Quantity", "Principal Invested"]
    if frame.empty or not set(columns).issubset(frame.columns):
        return _empty(columns)

    normalized = frame.copy()
    normalized["_row_order"] = range(len(normalized))
    normalized["Date"] = _parse_daily_series(normalized["Date"])
    normalized["Coin"] = normalized["Coin"].map(sanitize_symbol)
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce").fillna(0.0)
    normalized["Principal Invested"] = pd.to_numeric(
        normalized["Principal Invested"],
        errors="coerce",
    ).fillna(0.0)
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized[normalized["Coin"] != ""]
    normalized = normalized.sort_values(["Date", "Coin", "_row_order"])
    normalized = normalized.groupby(["Date", "Coin"], as_index=False).tail(1)
    return normalized[columns].reset_index(drop=True)


def _dense_snapshot_state(
    *,
    snapshots: pd.DataFrame,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return _empty(["Date", "Coin", "Quantity", "Principal Invested"])

    latest_snapshot_date = snapshots["Date"].max()
    final_date = (
        max(latest_snapshot_date, end_date) if end_date is not None else latest_snapshot_date
    )
    calendar = pd.date_range(start=snapshots["Date"].min(), end=final_date, freq="D")
    dense_frames: list[pd.DataFrame] = []
    for value_column in ("Quantity", "Principal Invested"):
        pivot = snapshots.pivot(index="Date", columns="Coin", values=value_column)
        dense = pivot.reindex(calendar).sort_index().ffill().fillna(0.0)
        melted = dense.reset_index().melt(
            id_vars="index",
            var_name="Coin",
            value_name=value_column,
        )
        dense_frames.append(melted.rename(columns={"index": "Date"}))

    merged = pd.merge(
        left=dense_frames[0],
        right=dense_frames[1],
        on=["Date", "Coin"],
        how="outer",
    )
    return merged.sort_values(["Date", "Coin"]).reset_index(drop=True)


def _load_aave_overlay(chain: str) -> pd.DataFrame | None:
    overlay_path = PROTOCOL_UNDERLYING_TOKEN_FOLDER / "aave" / f"{chain}_aave_daily_exposure.csv"
    if not overlay_path.exists():
        return None

    frame = pd.read_csv(overlay_path)
    if "date" not in frame.columns:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"].map(parse_daily_datetime), errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return None
    frame["date"] = frame["date"].dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)


def _load_aave_wrapper_symbols(token_metadata: dict[str, dict[str, Any]]) -> set[str]:
    wrappers: set[str] = set()
    for meta in token_metadata.values():
        if not isinstance(meta, dict) or meta.get("protocol") != "aave":
            continue
        symbol = sanitize_symbol(meta.get("symbol"))
        if symbol:
            wrappers.add(symbol)
    return wrappers


def _build_context(*, chain: str, token_metadata: dict[str, dict[str, Any]]) -> CompositionContext:
    return build_composition_context(
        chain=chain,
        token_metadata=token_metadata,
        protocol_root=PROTOCOL_UNDERLYING_TOKEN_FOLDER,
        include_aave=False,
        aave_overlay=_load_aave_overlay(chain=chain),
        aave_wrapper_symbols=_load_aave_wrapper_symbols(token_metadata=token_metadata),
    )


def _metadata_by_symbol(token_metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue
        symbol = sanitize_symbol(meta.get("symbol"))
        if symbol and symbol not in by_symbol:
            by_symbol[symbol] = meta
    return by_symbol


def _normalize_aave_symbol(symbol: str) -> str:
    normalized = sanitize_symbol(symbol)
    if not normalized:
        return ""
    return AAVE_SYMBOL_ALIASES.get(normalized.upper(), normalized)


def _is_aave_debt_symbol(symbol: str) -> bool:
    normalized = sanitize_symbol(symbol).lower()
    return any(normalized.startswith(prefix.lower()) for prefix in AAVE_DEBT_PREFIXES)


def _aave_multiplier(symbol: str) -> Decimal:
    return Decimal("-1") if _is_aave_debt_symbol(symbol) else Decimal("1")


def _aave_base_symbol(symbol: str, meta: dict[str, Any] | None) -> str:
    explicit = ""
    if meta:
        explicit = sanitize_symbol(meta.get("price_source")) or sanitize_symbol(meta.get("family"))
    if not explicit:
        upper_symbol = sanitize_symbol(symbol).upper()
        for prefix in AAVE_EXPOSURE_PREFIXES:
            if upper_symbol.startswith(prefix.upper()):
                explicit = sanitize_symbol(symbol[len(prefix) :])
                break
    normalized = _normalize_aave_symbol(explicit or symbol)
    return price_proxy_symbol(normalized) or normalized


def _display_direct_symbol(symbol: str, ctx: CompositionContext) -> str:
    canonical = canonicalize_symbol(symbol, symbol_family=ctx.symbol_family)
    return canonical or symbol


def _component_weights(
    *,
    exposures: dict[str, Any],
    date_value: pd.Timestamp,
    price_resolver: PriceResolver,
) -> dict[str, float]:
    weights = component_value_weights(
        exposures=exposures,
        date_value=date_value,
        price_resolver=price_resolver,
    )
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {coin: weight / total for coin, weight in weights.items() if weight > 0}


def _is_material(
    *,
    quantity: Decimal,
    market_value: Decimal | None,
    principal: Decimal,
    realized_pnl: Decimal,
) -> bool:
    if abs(principal) >= MATERIAL_VALUE_THRESHOLD_EUR:
        return True
    if abs(realized_pnl) >= MATERIAL_VALUE_THRESHOLD_EUR:
        return True
    if market_value is None:
        return abs(quantity) > MATERIAL_QUANTITY_THRESHOLD
    return abs(market_value) >= MATERIAL_VALUE_THRESHOLD_EUR


def _issue_row(
    *,
    date_value: pd.Timestamp,
    source: str,
    base_coin: str,
    quantity: Decimal,
    reason: str,
    action: str,
) -> dict[str, object]:
    return {
        "Date": format_daily_datetime(date_value),
        "Source": source,
        "BaseCoin": base_coin,
        "Quantity": float(quantity),
        "Reason": reason,
        "Action": action,
    }


def _source_base_row(
    *,
    date_value: pd.Timestamp,
    source: str,
    base_coin: str,
    quantity: Decimal,
    market_value: Decimal | None,
    principal: Decimal,
    realized_pnl: Decimal,
    route: ValuationRoute,
    has_direct_exposure: bool,
    has_protocol_exposure: bool,
    has_aave_exposure: bool,
) -> dict[str, object]:
    return {
        "Date": format_daily_datetime(date_value),
        "Source": source,
        "BaseCoin": base_coin,
        "Quantity": float(quantity),
        "MarketValueEUR": float(market_value) if market_value is not None else "",
        "PrincipalInvestedEUR": float(principal),
        "RealizedPnLEUR": float(realized_pnl),
        "ValuationRoute": route.value,
        "HasDirectExposure": has_direct_exposure,
        "HasProtocolExposure": has_protocol_exposure,
        "HasAaveExposure": has_aave_exposure,
        "_ResolvedMarketValue": market_value is not None,
    }


def _active_rows_for_source(
    *,
    date_value: pd.Timestamp,
    source: str,
    quantity: Decimal,
    principal: Decimal,
    route: ValuationRoute,
    ctx: CompositionContext,
    price_resolver: PriceResolver,
    previous_shares: dict[str, dict[str, float]],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    has_protocol = route == ValuationRoute.PROTOCOL_DERIVED
    has_direct = route == ValuationRoute.DIRECT
    exposures = ExposureExpander(ctx=ctx).expand(
        symbol=source,
        quantity=quantity,
        date_value=date_value,
        has_direct_exposure=has_direct,
        has_protocol_exposure=has_protocol,
        has_aave_exposure=False,
    )
    if not exposures:
        return []

    shares = _component_weights(
        exposures=exposures,
        date_value=date_value,
        price_resolver=price_resolver,
    )
    if shares:
        previous_shares[source] = shares
    else:
        equal_share = 1 / len(exposures)
        previous_shares[source] = {coin: equal_share for coin in exposures}

    rows: list[dict[str, object]] = []
    remaining_principal = principal
    component_items = list(exposures.items())
    for index, (base_coin, exposure) in enumerate(component_items):
        resolution = price_resolver.resolve(symbol=base_coin, target_date=date_value)
        market_value = (
            exposure.quantity * resolution.price_eur if resolution.price_eur is not None else None
        )
        share = previous_shares[source].get(base_coin, 0.0)
        if index == len(component_items) - 1:
            component_principal = remaining_principal
        else:
            component_principal = principal * Decimal(str(share))
            remaining_principal -= component_principal

        if market_value is None and _is_material(
            quantity=exposure.quantity,
            market_value=None,
            principal=component_principal,
            realized_pnl=Decimal("0"),
        ):
            reason = "missing_material_price"
            action = "add price history or protocol decomposition before " "rebuilding accounting"
            if route == ValuationRoute.PROTOCOL_DERIVED and source not in ctx.protocol_rows:
                reason = "protocol_decomposition_missing"
                action = "run protocol adapter before rebuilding accounting"
            else:
                canonical_base = canonicalize_symbol(
                    base_coin,
                    symbol_family=ctx.symbol_family,
                )
                if ctx.known_symbols and (
                    base_coin not in ctx.known_symbols and canonical_base not in ctx.known_symbols
                ):
                    reason = "unknown_symbol_material"
                    action = "add token metadata or protocol mapping"
                elif route == ValuationRoute.DIRECT:
                    reason = "known_symbol_missing_price"
                    action = "add direct price file or direct symbol metadata"
            issues.append(
                _issue_row(
                    date_value=date_value,
                    source=source,
                    base_coin=base_coin,
                    quantity=exposure.quantity,
                    reason=reason,
                    action=action,
                )
            )

        rows.append(
            _source_base_row(
                date_value=date_value,
                source=source,
                base_coin=base_coin,
                quantity=exposure.quantity,
                market_value=market_value,
                principal=component_principal,
                realized_pnl=Decimal("0"),
                route=route,
                has_direct_exposure=exposure.has_direct_exposure,
                has_protocol_exposure=exposure.has_protocol_exposure,
                has_aave_exposure=exposure.has_aave_exposure,
            )
        )
    return rows


def _closed_rows_for_source(
    *,
    date_value: pd.Timestamp,
    source: str,
    principal: Decimal,
    route: ValuationRoute,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    previous_shares: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    if abs(principal) < VALUE_DUST_EUR:
        return []

    if route == ValuationRoute.AAVE:
        base_coin = _aave_base_symbol(symbol=source, meta=metadata.get(source))
        signed_principal = principal * _aave_multiplier(source)
        shares = {base_coin: 1.0}
    elif route == ValuationRoute.PROTOCOL_DERIVED:
        shares = previous_shares.get(source, {})
        signed_principal = principal
        if not shares:
            return []
    else:
        base_coin = _display_direct_symbol(source, ctx=ctx)
        signed_principal = principal
        shares = {base_coin: 1.0}

    rows: list[dict[str, object]] = []
    remaining_realized = -signed_principal
    share_items = list(shares.items())
    for index, (base_coin, share) in enumerate(share_items):
        if index == len(share_items) - 1:
            realized_pnl = remaining_realized
        else:
            realized_pnl = -signed_principal * Decimal(str(share))
            remaining_realized -= realized_pnl
        rows.append(
            _source_base_row(
                date_value=date_value,
                source=source,
                base_coin=base_coin,
                quantity=Decimal("0"),
                market_value=Decimal("0"),
                principal=Decimal("0"),
                realized_pnl=realized_pnl,
                route=route,
                has_direct_exposure=route == ValuationRoute.DIRECT,
                has_protocol_exposure=route == ValuationRoute.PROTOCOL_DERIVED,
                has_aave_exposure=route == ValuationRoute.AAVE,
            )
        )
    return rows


def _build_aave_overlay_rows(
    *,
    ctx: CompositionContext,
    price_resolver: PriceResolver,
    date_value: pd.Timestamp,
    source_state: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    if ctx.aave_overlay is None or ctx.aave_overlay.empty:
        return []

    eligible = ctx.aave_overlay[ctx.aave_overlay["date"] <= date_value]
    if eligible.empty:
        return []
    overlay_row = eligible.iloc[-1]

    principal_by_base: dict[str, Decimal] = {}
    realized_by_base: dict[str, Decimal] = {}
    aave_state = source_state[
        source_state["Coin"].map(lambda value: ctx.route_for(str(value)) == ValuationRoute.AAVE)
    ]
    for _, row in aave_state.iterrows():
        source = sanitize_symbol(row["Coin"])
        if not source:
            continue
        base_coin = _aave_base_symbol(symbol=source, meta=metadata.get(source))
        signed_principal = Decimal(str(row["Principal Invested"])) * _aave_multiplier(source)
        quantity = Decimal(str(row["Quantity"]))
        if abs(quantity) > DUST:
            principal_by_base[base_coin] = (
                principal_by_base.get(base_coin, Decimal("0")) + signed_principal
            )
        elif abs(signed_principal) >= VALUE_DUST_EUR:
            realized_by_base[base_coin] = (
                realized_by_base.get(base_coin, Decimal("0")) - signed_principal
            )

    rows: list[dict[str, object]] = []
    for column in overlay_row.index:
        if not isinstance(column, str) or not column.startswith("net_"):
            continue
        raw_quantity = overlay_row[column]
        if pd.isna(raw_quantity):
            continue
        quantity = Decimal(str(raw_quantity))
        overlay_symbol = _normalize_aave_symbol(column.replace("net_", "", 1))
        if not overlay_symbol:
            continue
        canonical_overlay = canonicalize_symbol(
            overlay_symbol,
            symbol_family=ctx.symbol_family,
        )
        if ctx.known_symbols and (
            overlay_symbol not in ctx.known_symbols and canonical_overlay not in ctx.known_symbols
        ):
            issues.append(
                _issue_row(
                    date_value=date_value,
                    source="Aave",
                    base_coin=overlay_symbol,
                    quantity=quantity,
                    reason="unknown_aave_overlay_symbol",
                    action="fix aave overlay header or token metadata",
                )
            )
            continue
        exposures = ExposureExpander(ctx=ctx).expand(
            symbol=overlay_symbol,
            quantity=quantity,
            date_value=date_value,
            has_direct_exposure=False,
            has_protocol_exposure=False,
            has_aave_exposure=True,
        )
        for base_coin, exposure in exposures.items():
            principal = principal_by_base.get(base_coin, Decimal("0"))
            realized = realized_by_base.get(base_coin, Decimal("0"))
            if (
                abs(exposure.quantity) <= DUST
                and abs(principal) < VALUE_DUST_EUR
                and abs(realized) < VALUE_DUST_EUR
            ):
                continue

            resolution = price_resolver.resolve(symbol=base_coin, target_date=date_value)
            market_value = (
                exposure.quantity * resolution.price_eur
                if resolution.price_eur is not None
                else None
            )
            if market_value is None and _is_material(
                quantity=exposure.quantity,
                market_value=None,
                principal=principal,
                realized_pnl=realized,
            ):
                issues.append(
                    _issue_row(
                        date_value=date_value,
                        source="Aave",
                        base_coin=base_coin,
                        quantity=exposure.quantity,
                        reason="missing_material_price",
                        action="add price history for Aave base asset",
                    )
                )

            rows.append(
                _source_base_row(
                    date_value=date_value,
                    source="Aave",
                    base_coin=base_coin,
                    quantity=exposure.quantity,
                    market_value=market_value,
                    principal=principal,
                    realized_pnl=realized,
                    route=ValuationRoute.AAVE,
                    has_direct_exposure=exposure.has_direct_exposure,
                    has_protocol_exposure=exposure.has_protocol_exposure,
                    has_aave_exposure=True,
                )
            )
    return rows


def _build_aave_source_position_rows(
    *,
    ctx: CompositionContext,
    price_resolver: PriceResolver,
    date_value: pd.Timestamp,
    source_state: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aave_state = source_state[
        source_state["Coin"].map(lambda value: ctx.route_for(str(value)) == ValuationRoute.AAVE)
    ]
    for _, row in aave_state.iterrows():
        source = sanitize_symbol(row["Coin"])
        if not source:
            continue
        quantity = Decimal(str(row["Quantity"])) * _aave_multiplier(source)
        raw_principal = Decimal(str(row["Principal Invested"])) * _aave_multiplier(source)
        if abs(quantity) <= DUST and abs(raw_principal) < VALUE_DUST_EUR:
            continue

        base_coin = _aave_base_symbol(symbol=source, meta=metadata.get(source))
        resolution = price_resolver.resolve(symbol=base_coin, target_date=date_value)
        market_value = quantity * resolution.price_eur if resolution.price_eur is not None else None
        principal = raw_principal if abs(quantity) > DUST else Decimal("0")
        realized = Decimal("0") if abs(quantity) > DUST else -raw_principal
        if market_value is None and _is_material(
            quantity=quantity,
            market_value=None,
            principal=principal,
            realized_pnl=realized,
        ):
            issues.append(
                _issue_row(
                    date_value=date_value,
                    source=source,
                    base_coin=base_coin,
                    quantity=quantity,
                    reason="missing_material_price",
                    action="add price history for Aave base asset",
                )
            )

        rows.append(
            _source_base_row(
                date_value=date_value,
                source=source,
                base_coin=base_coin,
                quantity=quantity,
                market_value=market_value,
                principal=principal,
                realized_pnl=realized,
                route=ValuationRoute.AAVE,
                has_direct_exposure=False,
                has_protocol_exposure=False,
                has_aave_exposure=True,
            )
        )
    return rows


def _build_source_base_daily(
    *,
    snapshots: pd.DataFrame,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshots.empty:
        return _empty(SOURCE_BASE_DAILY_COLUMNS), _empty(ACCOUNTING_ISSUE_COLUMNS)

    dense = _dense_snapshot_state(snapshots=snapshots, end_date=end_date)
    price_resolver = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="eur")
    previous_shares: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []

    for date_value, source_state in dense.groupby("Date", sort=True):
        date_ts = pd.Timestamp(date_value).normalize()
        if ctx.aave_overlay is None:
            rows.extend(
                _build_aave_source_position_rows(
                    ctx=ctx,
                    price_resolver=price_resolver,
                    date_value=date_ts,
                    source_state=source_state,
                    metadata=metadata,
                    issues=issues,
                )
            )
        else:
            rows.extend(
                _build_aave_overlay_rows(
                    ctx=ctx,
                    price_resolver=price_resolver,
                    date_value=date_ts,
                    source_state=source_state,
                    metadata=metadata,
                    issues=issues,
                )
            )
        for _, state in source_state.iterrows():
            source = sanitize_symbol(state["Coin"])
            if not source:
                continue

            route = ctx.route_for(source)
            if route == ValuationRoute.AAVE:
                continue

            quantity = Decimal(str(state["Quantity"]))
            principal = Decimal(str(state["Principal Invested"]))
            if abs(quantity) > DUST:
                rows.extend(
                    _active_rows_for_source(
                        date_value=date_ts,
                        source=source,
                        quantity=quantity,
                        principal=principal,
                        route=route,
                        ctx=ctx,
                        price_resolver=price_resolver,
                        previous_shares=previous_shares,
                        issues=issues,
                    )
                )
            else:
                rows.extend(
                    _closed_rows_for_source(
                        date_value=date_ts,
                        source=source,
                        principal=principal,
                        route=route,
                        ctx=ctx,
                        metadata=metadata,
                        previous_shares=previous_shares,
                    )
                )

    source_base = pd.DataFrame(rows, columns=INTERNAL_SOURCE_BASE_COLUMNS)
    if source_base.empty:
        return _empty(SOURCE_BASE_DAILY_COLUMNS), pd.DataFrame(
            issues,
            columns=ACCOUNTING_ISSUE_COLUMNS,
        )

    for column in ("Quantity", "MarketValueEUR", "PrincipalInvestedEUR", "RealizedPnLEUR"):
        source_base[column] = pd.to_numeric(source_base[column], errors="coerce")
    source_base["_ResolvedMarketValue"] = source_base["_ResolvedMarketValue"].map(bool)
    source_base = source_base[
        source_base.apply(
            lambda row: _is_material(
                quantity=Decimal(str(row["Quantity"])),
                market_value=(
                    None if pd.isna(row["MarketValueEUR"]) else Decimal(str(row["MarketValueEUR"]))
                ),
                principal=Decimal(str(row["PrincipalInvestedEUR"])),
                realized_pnl=Decimal(str(row["RealizedPnLEUR"])),
            ),
            axis=1,
        )
    ].copy()
    source_base["MarketValueEUR"] = source_base["MarketValueEUR"].fillna(0.0)
    source_base = source_base.sort_values(["Date", "Source", "BaseCoin"]).reset_index(drop=True)
    return source_base, pd.DataFrame(
        issues,
        columns=ACCOUNTING_ISSUE_COLUMNS,
    )


def _exposure_route(row: pd.Series) -> str:
    if bool(row["HasAaveExposure"]):
        return ValuationRoute.AAVE.value
    if bool(row["HasProtocolExposure"]) and not bool(row["HasDirectExposure"]):
        return ValuationRoute.PROTOCOL_DERIVED.value
    return ValuationRoute.DIRECT.value


def _build_base_daily(source_base: pd.DataFrame) -> pd.DataFrame:
    if source_base.empty:
        return _empty(BASE_DAILY_COLUMNS)

    frame = source_base.copy()
    if "_ResolvedMarketValue" not in frame.columns:
        frame["_ResolvedMarketValue"] = True
    grouped = (
        frame.groupby(["Date", "BaseCoin"], as_index=False, sort=True)
        .agg(
            Quantity=("Quantity", "sum"),
            MarketValueEUR=("MarketValueEUR", "sum"),
            ActivePrincipalEUR=("PrincipalInvestedEUR", "sum"),
            RealizedPnLEUR=("RealizedPnLEUR", "sum"),
            HasResolvedMarketValue=("_ResolvedMarketValue", "any"),
            HasDirectExposure=("HasDirectExposure", "any"),
            HasProtocolExposure=("HasProtocolExposure", "any"),
            HasAaveExposure=("HasAaveExposure", "any"),
        )
        .rename(columns={"BaseCoin": "Coin"})
    )
    if grouped.empty:
        return _empty(BASE_DAILY_COLUMNS)

    grouped["PrincipalInvestedEUR"] = grouped["ActivePrincipalEUR"] - grouped["RealizedPnLEUR"]
    grouped["ProfitLossEUR"] = grouped["MarketValueEUR"] - grouped["PrincipalInvestedEUR"]
    grouped["ValuationRoute"] = grouped.apply(_exposure_route, axis=1)
    grouped["PriceSymbol"] = grouped["Coin"].map(
        lambda symbol: price_proxy_symbol(symbol) or symbol
    )
    grouped["PriceEUR"] = grouped.apply(
        lambda row: row["MarketValueEUR"] / row["Quantity"]
        if (
            bool(row["HasResolvedMarketValue"])
            and abs(float(row["Quantity"])) > float(MATERIAL_QUANTITY_THRESHOLD)
        )
        else pd.NA,
        axis=1,
    )
    return grouped[BASE_DAILY_COLUMNS].sort_values(["Date", "Coin"]).reset_index(drop=True)


def _write_csv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[columns]
    output.to_csv(path, index=False)


def build_accounting_artifacts(
    *,
    chain: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> AccountingBuildResult:
    clear_price_cache()
    paths = accounting_paths(chain=chain)
    token_metadata = load_token_metadata(chain=chain, tokens_folder=TOKENS_FOLDER)
    metadata = _metadata_by_symbol(token_metadata=token_metadata)
    ctx = _build_context(chain=chain, token_metadata=token_metadata)

    snapshots = _normalize_snapshot_frame(
        _read_csv(
            BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv",
            ["Date", "Coin", "Quantity", "Principal Invested"],
        )
    )
    end_date = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else None
    if end_date is None and not snapshots.empty:
        end_date = pd.Timestamp(snapshots["Date"].max()).normalize()

    source_base, issues = _build_source_base_daily(
        snapshots=snapshots,
        ctx=ctx,
        metadata=metadata,
        end_date=end_date,
    )
    base_daily = _build_base_daily(source_base=source_base)

    _write_csv(paths.source_base_daily, source_base, SOURCE_BASE_DAILY_COLUMNS)
    _write_csv(paths.base_daily, base_daily, BASE_DAILY_COLUMNS)
    _write_csv(paths.issues, issues, ACCOUNTING_ISSUE_COLUMNS)
    errors = [
        f"{row['Date']} {row['Source']}->{row['BaseCoin']}: {row['Reason']}"
        for _, row in issues.iterrows()
    ]
    return AccountingBuildResult(
        paths=paths,
        rows_written={
            "source_base_daily": len(source_base),
            "base_daily": len(base_daily),
            "issues": len(issues),
        },
        errors=errors,
    )
