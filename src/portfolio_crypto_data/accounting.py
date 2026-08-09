from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from portfolio_core import active_context, atomic_write_csv

from portfolio_crypto_data.composition.core import (
    DUST,
    CompositionContext,
    ExposureExpander,
    PriceResolver,
    build_composition_context,
)
from portfolio_crypto_data.datetime_utils import format_daily_datetime, parse_daily_datetime
from portfolio_crypto_data.principal_ledger import PRINCIPAL_DAILY_COLUMNS, PRINCIPAL_EVENT_COLUMNS
from portfolio_crypto_data.shared.prices import clear_price_cache
from portfolio_crypto_data.shared.token_metadata import load_token_metadata
from portfolio_crypto_data.shared.valuation_routes import ValuationRoute
from portfolio_crypto_data.symbols import canonicalize_symbol, price_proxy_symbol, sanitize_symbol

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


@dataclass(frozen=True)
class AccountingArtifactPaths:
    principal_events: Path
    principal_daily: Path
    source_base_daily: Path
    base_daily: Path


@dataclass(frozen=True)
class AccountingBuildResult:
    paths: AccountingArtifactPaths
    rows_written: dict[str, int]


def accounting_paths(chain: str) -> AccountingArtifactPaths:
    root = active_context().paths.accounting / chain
    return AccountingArtifactPaths(
        principal_events=root / "principal_events.csv",
        principal_daily=root / "principal_daily.csv",
        source_base_daily=root / "source_base_daily.csv",
        base_daily=root / "base_daily.csv",
    )


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
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


def _normalize_principal_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(PRINCIPAL_DAILY_COLUMNS).issubset(frame.columns):
        return _empty(PRINCIPAL_DAILY_COLUMNS)

    normalized = frame.copy()
    normalized["_row_order"] = range(len(normalized))
    normalized["Date"] = _parse_daily_series(normalized["Date"])
    normalized["Coin"] = normalized["Coin"].map(sanitize_symbol)
    normalized["PrincipalInvestedEUR"] = pd.to_numeric(
        normalized["PrincipalInvestedEUR"],
        errors="coerce",
    ).fillna(0.0)
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized[normalized["Coin"] != ""]
    normalized = normalized.sort_values(["Date", "Coin", "_row_order"])
    normalized = normalized.groupby(["Date", "Coin"], as_index=False).tail(1)
    return normalized[PRINCIPAL_DAILY_COLUMNS].reset_index(drop=True)


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


def _dense_principal_state(
    *,
    principal_daily: pd.DataFrame,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if principal_daily.empty:
        return _empty(PRINCIPAL_DAILY_COLUMNS)

    latest_date = principal_daily["Date"].max()
    final_date = max(latest_date, end_date) if end_date is not None else latest_date
    calendar = pd.date_range(start=principal_daily["Date"].min(), end=final_date, freq="D")
    pivot = principal_daily.pivot(index="Date", columns="Coin", values="PrincipalInvestedEUR")
    dense = pivot.reindex(calendar).sort_index().ffill().fillna(0.0)
    melted = dense.reset_index().melt(
        id_vars="index",
        var_name="Coin",
        value_name="PrincipalInvestedEUR",
    )
    return (
        melted.rename(columns={"index": "Date"})[PRINCIPAL_DAILY_COLUMNS]
        .sort_values(["Date", "Coin"])
        .reset_index(drop=True)
    )


def _load_aave_overlay(chain: str) -> pd.DataFrame | None:
    overlay_path = (
        active_context().paths.protocol_underlying_tokens
        / "aave"
        / f"{chain}_aave_daily_exposure.csv"
    )
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
    paths = active_context().paths
    return build_composition_context(
        chain=chain,
        token_metadata=token_metadata,
        protocol_root=paths.protocol_underlying_tokens,
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


def _is_material(
    *,
    market_value: Decimal,
    principal: Decimal,
    realized_pnl: Decimal,
) -> bool:
    if abs(principal) >= MATERIAL_VALUE_THRESHOLD_EUR:
        return True
    if abs(realized_pnl) >= MATERIAL_VALUE_THRESHOLD_EUR:
        return True
    return abs(market_value) >= MATERIAL_VALUE_THRESHOLD_EUR


def _source_base_row(
    *,
    date_value: pd.Timestamp,
    source: str,
    base_coin: str,
    quantity: Decimal,
    market_value: Decimal,
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
        "MarketValueEUR": float(market_value),
        "PrincipalInvestedEUR": float(principal),
        "RealizedPnLEUR": float(realized_pnl),
        "ValuationRoute": route.value,
        "HasDirectExposure": has_direct_exposure,
        "HasProtocolExposure": has_protocol_exposure,
        "HasAaveExposure": has_aave_exposure,
    }


def _active_rows_for_source(
    *,
    date_value: pd.Timestamp,
    source: str,
    quantity: Decimal,
    route: ValuationRoute,
    ctx: CompositionContext,
    price_resolver: PriceResolver,
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

    rows: list[dict[str, object]] = []
    for base_coin, exposure in exposures.items():
        resolution = price_resolver.resolve(symbol=base_coin, target_date=date_value)
        if resolution.price_eur is None:
            if abs(exposure.quantity) > MATERIAL_QUANTITY_THRESHOLD:
                raise ValueError(f"Missing EUR price for {base_coin} on {date_value.date()}")
            market_value = Decimal("0")
        else:
            market_value = exposure.quantity * resolution.price_eur

        rows.append(
            _source_base_row(
                date_value=date_value,
                source=source,
                base_coin=base_coin,
                quantity=exposure.quantity,
                market_value=market_value,
                principal=Decimal("0"),
                realized_pnl=Decimal("0"),
                route=route,
                has_direct_exposure=exposure.has_direct_exposure,
                has_protocol_exposure=exposure.has_protocol_exposure,
                has_aave_exposure=exposure.has_aave_exposure,
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
            raise ValueError(f"Unknown Aave overlay symbol: {overlay_symbol}")
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
            if resolution.price_eur is None:
                raise ValueError(f"Missing EUR price for {base_coin} on {date_value.date()}")
            market_value = exposure.quantity * resolution.price_eur

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


def _build_source_base_daily(
    *,
    snapshots: pd.DataFrame,
    principal_daily: pd.DataFrame,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty and principal_daily.empty:
        return _empty(SOURCE_BASE_DAILY_COLUMNS)

    dense = _dense_snapshot_state(snapshots=snapshots, end_date=end_date)
    dense_principal = _dense_principal_state(
        principal_daily=principal_daily,
        end_date=end_date,
    )
    price_resolver = PriceResolver(ctx=ctx, mode="eur")
    rows: list[dict[str, object]] = []

    material_aave = dense[
        dense["Coin"].map(lambda value: ctx.route_for(str(value)) == ValuationRoute.AAVE)
        & (
            (dense["Quantity"].abs() > float(DUST))
            | (dense["Principal Invested"].abs() >= float(VALUE_DUST_EUR))
        )
    ]
    if ctx.aave_overlay is None and not material_aave.empty:
        raise ValueError("Aave positions require aave_daily_exposure.csv")

    for date_value, source_state in dense.groupby("Date", sort=True):
        date_ts = pd.Timestamp(date_value).normalize()
        if ctx.aave_overlay is not None:
            rows.extend(
                _build_aave_overlay_rows(
                    ctx=ctx,
                    price_resolver=price_resolver,
                    date_value=date_ts,
                    source_state=source_state,
                    metadata=metadata,
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
            if abs(quantity) > DUST:
                rows.extend(
                    _active_rows_for_source(
                        date_value=date_ts,
                        source=source,
                        quantity=quantity,
                        route=route,
                        ctx=ctx,
                        price_resolver=price_resolver,
                    )
                )

    source_base = pd.DataFrame(rows, columns=SOURCE_BASE_DAILY_COLUMNS)
    if source_base.empty:
        source_base = _empty(SOURCE_BASE_DAILY_COLUMNS)
    else:
        for column in ("Quantity", "MarketValueEUR", "PrincipalInvestedEUR", "RealizedPnLEUR"):
            source_base[column] = pd.to_numeric(source_base[column], errors="coerce")
        source_base = source_base[
            source_base.apply(
                lambda row: _is_material(
                    market_value=Decimal(str(row["MarketValueEUR"])),
                    principal=Decimal(str(row["PrincipalInvestedEUR"])),
                    realized_pnl=Decimal(str(row["RealizedPnLEUR"])),
                ),
                axis=1,
            )
        ].copy()

    source_base = _allocate_principal_to_sources(
        source_base=source_base,
        principal_daily=dense_principal,
    )
    if source_base.empty:
        return _empty(SOURCE_BASE_DAILY_COLUMNS)

    source_base = source_base.sort_values(["Date", "Source", "BaseCoin"]).reset_index(drop=True)
    return source_base


def _allocate_principal_to_sources(
    *,
    source_base: pd.DataFrame,
    principal_daily: pd.DataFrame,
) -> pd.DataFrame:
    if principal_daily.empty:
        return source_base

    output = source_base.copy()
    if output.empty:
        output = _empty(SOURCE_BASE_DAILY_COLUMNS)
    for column in ("Quantity", "MarketValueEUR", "PrincipalInvestedEUR", "RealizedPnLEUR"):
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    appended: list[dict[str, object]] = []
    principal_frame = principal_daily.copy()
    principal_frame["Date"] = pd.to_datetime(
        principal_frame["Date"],
        errors="coerce",
    ).dt.normalize()
    principal_frame["PrincipalInvestedEUR"] = pd.to_numeric(
        principal_frame["PrincipalInvestedEUR"],
        errors="coerce",
    ).fillna(0.0)
    principal_frame = principal_frame.dropna(subset=["Date"])

    for _, row in principal_frame.iterrows():
        date_value = pd.Timestamp(row["Date"]).normalize()
        base_coin = sanitize_symbol(row["Coin"])
        principal = Decimal(str(row["PrincipalInvestedEUR"]))
        if not base_coin or abs(principal) < VALUE_DUST_EUR:
            continue

        mask = (pd.to_datetime(output["Date"], errors="coerce").dt.normalize() == date_value) & (
            output["BaseCoin"] == base_coin
        )
        active = output[mask & (output["MarketValueEUR"].abs() > float(VALUE_DUST_EUR))]
        if active.empty:
            appended.append(
                _source_base_row(
                    date_value=date_value,
                    source="Closed",
                    base_coin=base_coin,
                    quantity=Decimal("0"),
                    market_value=Decimal("0"),
                    principal=principal,
                    realized_pnl=Decimal("0"),
                    route=ValuationRoute.DIRECT,
                    has_direct_exposure=False,
                    has_protocol_exposure=False,
                    has_aave_exposure=False,
                )
            )
            continue

        weights = active["MarketValueEUR"].abs()
        total_weight = Decimal(str(weights.sum()))
        remaining = principal
        active_indexes = active.index.tolist()
        for index, row_index in enumerate(active_indexes):
            if index == len(active_indexes) - 1:
                share = remaining
            else:
                share = principal * Decimal(str(weights.loc[row_index])) / total_weight
                remaining -= share
            output.loc[row_index, "PrincipalInvestedEUR"] += float(share)

    if appended:
        output = pd.concat(
            [output, pd.DataFrame(appended, columns=SOURCE_BASE_DAILY_COLUMNS)],
            ignore_index=True,
            sort=False,
        )
    output = output[
        output.apply(
            lambda row: _is_material(
                market_value=Decimal(str(row["MarketValueEUR"])),
                principal=Decimal(str(row["PrincipalInvestedEUR"])),
                realized_pnl=Decimal(str(row["RealizedPnLEUR"])),
            ),
            axis=1,
        )
    ].copy()
    return output[SOURCE_BASE_DAILY_COLUMNS].reset_index(drop=True)


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
    grouped = (
        frame.groupby(["Date", "BaseCoin"], as_index=False, sort=True)
        .agg(
            Quantity=("Quantity", "sum"),
            MarketValueEUR=("MarketValueEUR", "sum"),
            ActivePrincipalEUR=("PrincipalInvestedEUR", "sum"),
            RealizedPnLEUR=("RealizedPnLEUR", "sum"),
            HasDirectExposure=("HasDirectExposure", "any"),
            HasProtocolExposure=("HasProtocolExposure", "any"),
            HasAaveExposure=("HasAaveExposure", "any"),
        )
        .rename(columns={"BaseCoin": "Coin"})
    )
    if grouped.empty:
        return _empty(BASE_DAILY_COLUMNS)

    grouped["PrincipalInvestedEUR"] = grouped["ActivePrincipalEUR"]
    grouped["ProfitLossEUR"] = grouped["MarketValueEUR"] - grouped["PrincipalInvestedEUR"]
    grouped["ValuationRoute"] = grouped.apply(_exposure_route, axis=1)
    grouped["PriceSymbol"] = grouped["Coin"].map(
        lambda symbol: price_proxy_symbol(symbol) or symbol
    )
    grouped["PriceEUR"] = grouped.apply(
        lambda row: row["MarketValueEUR"] / row["Quantity"]
        if abs(float(row["Quantity"])) > float(MATERIAL_QUANTITY_THRESHOLD)
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
    atomic_write_csv(frame=output, path=path)


def build_accounting_artifacts(
    *,
    chain: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> AccountingBuildResult:
    clear_price_cache()
    paths = accounting_paths(chain=chain)
    runtime_paths = active_context().paths
    token_metadata = load_token_metadata(chain=chain, tokens_folder=runtime_paths.tokens)
    metadata = _metadata_by_symbol(token_metadata=token_metadata)
    ctx = _build_context(chain=chain, token_metadata=token_metadata)

    snapshots = _normalize_snapshot_frame(
        _read_csv(
            runtime_paths.crypto_snapshots / f"{chain}_raw_snapshots.csv",
            ["Date", "Coin", "Quantity", "Principal Invested"],
        )
    )
    principal_daily = _normalize_principal_daily_frame(
        _read_csv(paths.principal_daily, PRINCIPAL_DAILY_COLUMNS)
    )
    end_date = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else None
    if end_date is None and not snapshots.empty:
        end_date = pd.Timestamp(snapshots["Date"].max()).normalize()

    source_base = _build_source_base_daily(
        snapshots=snapshots,
        principal_daily=principal_daily,
        ctx=ctx,
        metadata=metadata,
        end_date=end_date,
    )
    base_daily = _build_base_daily(source_base=source_base)

    _write_csv(paths.source_base_daily, source_base, SOURCE_BASE_DAILY_COLUMNS)
    _write_csv(paths.base_daily, base_daily, BASE_DAILY_COLUMNS)
    stale_issues = paths.base_daily.parent / "issues.csv"
    if stale_issues.exists():
        stale_issues.unlink()
    if not paths.principal_events.exists():
        _write_csv(paths.principal_events, _empty(PRINCIPAL_EVENT_COLUMNS), PRINCIPAL_EVENT_COLUMNS)
    _write_csv(paths.principal_daily, principal_daily, PRINCIPAL_DAILY_COLUMNS)
    return AccountingBuildResult(
        paths=paths,
        rows_written={
            "principal_events": _read_csv(paths.principal_events, PRINCIPAL_EVENT_COLUMNS).shape[0],
            "principal_daily": len(principal_daily),
            "source_base_daily": len(source_base),
            "base_daily": len(base_daily),
        },
    )
