from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_reader.accounting import (
    BASE_DAILY_COLUMNS,
    SOURCE_BASE_DAILY_COLUMNS,
    accounting_paths,
)
from blockchain_reader.composition.core import (
    CompositionContext,
    ExposureExpander,
    PriceResolver,
    component_value_weights,
)
from blockchain_reader.datetime_utils import parse_daily_datetime, parse_transaction_datetime_series
from blockchain_reader.shared.token_metadata import load_token_metadata
from blockchain_reader.shared.valuation_routes import ValuationRoute
from blockchain_reader.symbols import canonicalize_symbol, price_proxy_symbol, sanitize_symbol
from file_paths import (
    BLOCKCHAIN_DASHBOARD_FOLDER,
    BLOCKCHAIN_TRANSACTIONS_FOLDER,
    PRICES_FOLDER,
    TOKENS_FOLDER,
)

CHAIN = "arbitrum"
MATERIAL_QUANTITY_THRESHOLD = 1e-10
MATERIAL_VALUE_THRESHOLD_EUR = 1.0
PRINCIPAL_SOURCE_SEPARATOR = "||"
AAVE_EXPOSURE_PREFIXES = ("variableDebtArb", "stableDebtArb", "aArb")
AAVE_DEBT_PREFIXES = ("variableDebtArb", "stableDebtArb")

ASSET_DAILY_COLUMNS = [
    "Date",
    "Selection",
    "AssetLayer",
    "Coin",
    "Quantity",
    "PriceEUR",
    "MarketValueEUR",
    "PrincipalInvestedEUR",
    "ProfitLossEUR",
    "ValuationRoute",
    "HasDirectExposure",
    "HasProtocolExposure",
    "HasAaveExposure",
    "MissingPrice",
    "IsMaterial",
]
TIMESERIES_DAILY_COLUMNS = [
    "Date",
    "Selection",
    "MarketValueEUR",
    "PrincipalInvestedEUR",
    "ProfitLossEUR",
    "Quantity",
    "TxCount",
]
COMPOSITION_DAILY_COLUMNS = [
    "Date",
    "Selection",
    "CompositionMode",
    "Label",
    "ValueEUR",
]
SOURCE_DAILY_COLUMNS = [
    "Date",
    "Selection",
    "Source",
    "Coin",
    "Quantity",
    "MarketValueEUR",
    "PrincipalInvestedEUR",
    "ProfitLossEUR",
    "ValuationRoute",
    "HasDirectExposure",
    "HasProtocolExposure",
    "HasAaveExposure",
    "IsMaterial",
]
TRANSACTIONS_DASHBOARD_COLUMNS = [
    "Date",
    "Type",
    "Token in",
    "Qty in",
    "Token out",
    "Qty out",
    "Fee",
    "Fee Token",
    "TX Hash",
    "AssetKeys",
]
ASSETS_COLUMNS = ["Label", "Value"]


@dataclass(frozen=True)
class ArbitrumDashboardArtifactPaths:
    asset_daily: Path
    timeseries_daily: Path
    composition_daily: Path
    source_daily: Path
    transactions_dashboard: Path
    assets: Path


def artifact_paths(chain: str = CHAIN) -> ArbitrumDashboardArtifactPaths:
    root = BLOCKCHAIN_DASHBOARD_FOLDER / chain
    return ArbitrumDashboardArtifactPaths(
        asset_daily=root / "asset_daily.csv",
        timeseries_daily=root / "timeseries_daily.csv",
        composition_daily=root / "composition_daily.csv",
        source_daily=root / "source_daily.csv",
        transactions_dashboard=root / "transactions_dashboard.csv",
        assets=root / "assets.csv",
    )


def _selection_key(value: object) -> str:
    return sanitize_symbol(value).upper()


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


def _normalize_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _normalize_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty(
            [
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
        )

    normalized = frame.copy()
    normalized["Date"] = _parse_daily_series(normalized["Date"])
    normalized["Coin"] = normalized["Coin"].map(sanitize_symbol)
    normalized["Quantity"] = pd.to_numeric(normalized["Quantity"], errors="coerce").fillna(0.0)
    normalized["PriceEUR"] = pd.to_numeric(normalized["PriceEUR"], errors="coerce")
    normalized["EstimatedValueEUR"] = pd.to_numeric(
        normalized["EstimatedValueEUR"],
        errors="coerce",
    )
    normalized["MarketValueEUR"] = normalized["EstimatedValueEUR"]
    missing_value = normalized["MarketValueEUR"].isna() & normalized["PriceEUR"].notna()
    normalized.loc[missing_value, "MarketValueEUR"] = (
        normalized.loc[missing_value, "Quantity"] * normalized.loc[missing_value, "PriceEUR"]
    )
    for column in ("HasDirectExposure", "HasProtocolExposure", "HasAaveExposure"):
        if column not in normalized.columns:
            normalized[column] = False
        normalized[column] = normalized[column].map(_normalize_bool)
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized[normalized["Coin"] != ""]
    return normalized.reset_index(drop=True)


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


def _normalize_transactions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty(TRANSACTIONS_DASHBOARD_COLUMNS[:-1])

    normalized = frame.copy()
    for column in TRANSACTIONS_DASHBOARD_COLUMNS[:-1]:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["Date"] = parse_transaction_datetime_series(normalized["Date"])
    normalized = normalized.dropna(subset=["Date"])
    for column in TRANSACTIONS_DASHBOARD_COLUMNS[:-1]:
        if column == "Date":
            continue
        normalized[column] = normalized[column].fillna("").astype(str)
    return normalized[TRANSACTIONS_DASHBOARD_COLUMNS[:-1]].reset_index(drop=True)


def _metadata_by_symbol(token_metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue
        symbol_key = _selection_key(meta.get("symbol"))
        if symbol_key and symbol_key not in by_symbol:
            by_symbol[symbol_key] = meta
    return by_symbol


def _aave_exposure_symbol(symbol: str, meta: dict[str, Any] | None = None) -> str:
    normalized = sanitize_symbol(symbol)
    explicit = ""
    if meta:
        explicit = sanitize_symbol(meta.get("price_source")) or sanitize_symbol(meta.get("family"))

    if not explicit:
        upper_symbol = normalized.upper()
        for prefix in AAVE_EXPOSURE_PREFIXES:
            if upper_symbol.startswith(prefix.upper()):
                explicit = sanitize_symbol(normalized[len(prefix) :])
                break

    return price_proxy_symbol(explicit or normalized) or explicit or normalized


def _is_aave_debt_symbol(symbol: str) -> bool:
    normalized = sanitize_symbol(symbol).lower()
    return any(normalized.startswith(prefix.lower()) for prefix in AAVE_DEBT_PREFIXES)


def _aave_quantity_multiplier(symbol: str) -> Decimal:
    return Decimal("-1") if _is_aave_debt_symbol(symbol) else Decimal("1")


def _aave_principal_multiplier(symbol: str) -> float:
    return -1.0 if _is_aave_debt_symbol(symbol) else 1.0


def _display_symbol(
    *,
    symbol: str,
    route: ValuationRoute,
    meta: dict[str, Any] | None,
    ctx: CompositionContext,
) -> str:
    if route == ValuationRoute.AAVE:
        return _aave_exposure_symbol(symbol=symbol, meta=meta)
    if route == ValuationRoute.DIRECT:
        return canonicalize_symbol(symbol, symbol_family=ctx.symbol_family) or symbol
    return symbol


def _price_symbol(
    *,
    symbol: str,
    route: ValuationRoute,
    meta: dict[str, Any] | None,
) -> str:
    if route == ValuationRoute.AAVE:
        return _aave_exposure_symbol(symbol=symbol, meta=meta)
    return symbol


def _expand_principal_row(
    *,
    symbol: str,
    quantity: float,
    principal: float,
    date_value: pd.Timestamp,
    ctx: CompositionContext,
    price_resolver: PriceResolver,
) -> list[dict[str, object]]:
    route = ctx.route_for(symbol)
    if route != ValuationRoute.PROTOCOL_DERIVED:
        return []

    exposures = ExposureExpander(ctx=ctx).expand(
        symbol=symbol,
        quantity=Decimal(str(quantity)),
        date_value=date_value,
        has_direct_exposure=False,
        has_protocol_exposure=True,
        has_aave_exposure=False,
    )
    if not exposures:
        return []

    weights = component_value_weights(
        exposures=exposures,
        date_value=date_value,
        price_resolver=price_resolver,
    )
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return []

    rows: list[dict[str, object]] = []
    remaining_principal = principal
    weighted_components = [(coin, weight) for coin, weight in weights.items() if weight > 0]
    for index, (component_symbol, weight) in enumerate(weighted_components):
        component_weight_share = weight / total_weight
        if index == len(weighted_components) - 1:
            component_principal = remaining_principal
        else:
            component_principal = principal * component_weight_share
            remaining_principal -= component_principal
        rows.append(
            {
                "Date": date_value,
                "Source": symbol,
                "Coin": component_symbol,
                "PrincipalInvestedEUR": component_principal,
                "_WeightShare": component_weight_share,
            }
        )
    return rows


def _build_principal_source_components(
    *,
    snapshots: pd.DataFrame,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    columns = ["Date", "Source", "Coin", "PrincipalInvestedEUR"]
    if snapshots.empty:
        return _empty(columns)

    price_resolver = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="eur")
    rows: list[dict[str, object]] = []
    previous_protocol_component_shares: dict[str, dict[str, float]] = {}
    for _, row in snapshots.iterrows():
        date_value = pd.Timestamp(row["Date"]).normalize()
        symbol = sanitize_symbol(row["Coin"])
        quantity = float(row["Quantity"])
        principal = float(row["Principal Invested"])
        if not symbol:
            continue

        expanded_rows = _expand_principal_row(
            symbol=symbol,
            quantity=quantity,
            principal=principal,
            date_value=date_value,
            ctx=ctx,
            price_resolver=price_resolver,
        )
        if expanded_rows:
            rows.extend(expanded_rows)
            previous_protocol_component_shares[symbol] = {
                str(expanded["Coin"]): float(expanded.get("_WeightShare", 0.0))
                for expanded in expanded_rows
                if float(expanded.get("_WeightShare", 0.0)) > 0
            }
            continue

        route = ctx.route_for(symbol)
        previous_shares = previous_protocol_component_shares.get(symbol)
        if route == ValuationRoute.PROTOCOL_DERIVED and previous_shares:
            remaining_principal = principal
            weighted_components = [
                (coin, share) for coin, share in previous_shares.items() if share > 0
            ]
            for index, (component_symbol, share) in enumerate(weighted_components):
                if index == len(weighted_components) - 1:
                    component_principal = remaining_principal
                else:
                    component_principal = principal * share
                    remaining_principal -= component_principal
                rows.append(
                    {
                        "Date": date_value,
                        "Source": symbol,
                        "Coin": component_symbol,
                        "PrincipalInvestedEUR": component_principal,
                    }
                )
            continue

        display_symbol = _display_symbol(
            symbol=symbol,
            route=route,
            meta=metadata.get(_selection_key(symbol)),
            ctx=ctx,
        )
        if route == ValuationRoute.AAVE:
            principal *= _aave_principal_multiplier(symbol)
        rows.append(
            {
                "Date": date_value,
                "Source": symbol,
                "Coin": display_symbol,
                "PrincipalInvestedEUR": principal,
            }
        )

    if not rows:
        return _empty(columns)

    components = pd.DataFrame(rows)
    components["Date"] = pd.to_datetime(components["Date"], errors="coerce").dt.normalize()
    components = components.dropna(subset=["Date"])
    if components.empty:
        return _empty(columns)

    final_date = (
        pd.Timestamp(end_date).normalize() if end_date is not None else components["Date"].max()
    )
    calendar = pd.date_range(start=components["Date"].min(), end=final_date, freq="D")
    components["_key"] = (
        components["Source"].astype(str)
        + PRINCIPAL_SOURCE_SEPARATOR
        + components["Coin"].astype(str)
    )
    grouped = components.groupby(["Date", "_key"], as_index=False)["PrincipalInvestedEUR"].sum()
    pivot = grouped.pivot(index="Date", columns="_key", values="PrincipalInvestedEUR")
    dense = pivot.reindex(calendar).sort_index().ffill().fillna(0.0)
    melted = dense.reset_index().melt(
        id_vars="index",
        var_name="_key",
        value_name="PrincipalInvestedEUR",
    )
    melted = melted.rename(columns={"index": "Date"})
    melted["Source"] = melted["_key"].map(
        lambda value: str(value).rsplit(PRINCIPAL_SOURCE_SEPARATOR, 1)[0]
    )
    melted["Coin"] = melted["_key"].map(
        lambda value: str(value).rsplit(PRINCIPAL_SOURCE_SEPARATOR, 1)[-1]
    )
    out = melted.groupby(["Date", "Source", "Coin"], as_index=False)["PrincipalInvestedEUR"].sum()
    return out[columns]


def _build_principal_components(principal_by_source: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "Coin", "PrincipalInvestedEUR"]
    if principal_by_source.empty:
        return _empty(columns)
    out = principal_by_source.groupby(["Date", "Coin"], as_index=False)[
        "PrincipalInvestedEUR"
    ].sum()
    return out[columns]


def _exposure_label(row: pd.Series) -> str:
    flags = []
    if _normalize_bool(row.get("HasDirectExposure")):
        flags.append("Direct")
    if _normalize_bool(row.get("HasProtocolExposure")):
        flags.append("Protocol")
    if _normalize_bool(row.get("HasAaveExposure")):
        flags.append("Aave")
    if len(flags) > 1:
        return "Mixed Exposure"
    if flags:
        return f"{flags[0]} Exposure"
    return "Unclassified"


def _build_base_asset_rows(
    *,
    base: pd.DataFrame,
    principal_by_coin: pd.DataFrame,
) -> pd.DataFrame:
    if base.empty:
        return _empty(ASSET_DAILY_COLUMNS)

    rows = base.copy()
    rows = pd.merge(
        left=rows,
        right=principal_by_coin,
        on=["Date", "Coin"],
        how="left",
    )
    rows["PrincipalInvestedEUR"] = pd.to_numeric(
        rows["PrincipalInvestedEUR"],
        errors="coerce",
    ).fillna(0.0)
    rows["ProfitLossEUR"] = (
        pd.to_numeric(rows["MarketValueEUR"], errors="coerce").fillna(0.0)
        - rows["PrincipalInvestedEUR"]
    )
    rows["MissingPrice"] = rows["PriceEUR"].isna() & (
        (rows["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (
            pd.to_numeric(rows["MarketValueEUR"], errors="coerce").abs()
            >= MATERIAL_VALUE_THRESHOLD_EUR
        )
    )
    rows["IsMaterial"] = (
        (rows["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (
            pd.to_numeric(rows["MarketValueEUR"], errors="coerce").abs()
            >= MATERIAL_VALUE_THRESHOLD_EUR
        )
        | (rows["PrincipalInvestedEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
    )

    base_rows = rows.rename(
        columns={
            "EstimatedValueEUR": "_EstimatedValueEUR",
            "ValuationRoute": "ValuationRoute",
        }
    )
    base_rows["AssetLayer"] = "base"
    full_rows = base_rows.copy()
    full_rows["Selection"] = "ALL"
    single_rows = base_rows.copy()
    single_rows["Selection"] = single_rows["Coin"]
    out = pd.concat([full_rows, single_rows], ignore_index=True, sort=False)
    return out[ASSET_DAILY_COLUMNS]


def _normalize_accounting_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty(BASE_DAILY_COLUMNS)

    normalized = frame.copy()
    normalized["Date"] = _parse_daily_series(normalized["Date"])
    normalized["Coin"] = normalized["Coin"].map(sanitize_symbol)
    for column in (
        "Quantity",
        "PriceEUR",
        "MarketValueEUR",
        "PrincipalInvestedEUR",
        "RealizedPnLEUR",
        "ProfitLossEUR",
    ):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if column != "PriceEUR":
            normalized[column] = normalized[column].fillna(0.0)
    for column in ("HasDirectExposure", "HasProtocolExposure", "HasAaveExposure"):
        normalized[column] = normalized[column].map(_normalize_bool)
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized[normalized["Coin"] != ""]
    return normalized[BASE_DAILY_COLUMNS].reset_index(drop=True)


def _build_asset_rows_from_accounting(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return _empty(ASSET_DAILY_COLUMNS)

    rows = base.copy()
    rows["MissingPrice"] = rows["PriceEUR"].isna() & (
        (rows["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (rows["MarketValueEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
    )
    rows["IsMaterial"] = (
        (rows["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (rows["MarketValueEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
        | (rows["PrincipalInvestedEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
        | (rows["ProfitLossEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
    )
    rows["AssetLayer"] = "base"
    full_rows = rows.copy()
    full_rows["Selection"] = "ALL"
    single_rows = rows.copy()
    single_rows["Selection"] = single_rows["Coin"]
    out = pd.concat([full_rows, single_rows], ignore_index=True, sort=False)
    return out[ASSET_DAILY_COLUMNS]


def _dense_snapshot_state(
    *,
    snapshots: pd.DataFrame,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return _empty(["Date", "Coin", "Quantity", "PrincipalInvestedEUR"])

    latest_snapshot_date = snapshots["Date"].max()
    final_date = (
        max(latest_snapshot_date, end_date) if end_date is not None else latest_snapshot_date
    )
    calendar = pd.date_range(start=snapshots["Date"].min(), end=final_date, freq="D")
    value_columns = ["Quantity", "Principal Invested"]
    dense_frames: list[pd.DataFrame] = []
    for value_column in value_columns:
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
    return merged.rename(columns={"Principal Invested": "PrincipalInvestedEUR"})


def _build_position_asset_rows(
    *,
    snapshots: pd.DataFrame,
    base_symbols: set[str],
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty:
        return _empty(ASSET_DAILY_COLUMNS)

    dense = _dense_snapshot_state(snapshots=snapshots, end_date=end_date)
    if dense.empty:
        return _empty(ASSET_DAILY_COLUMNS)

    price_resolver = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="eur")
    rows: list[dict[str, object]] = []
    for _, row in dense.iterrows():
        source_symbol = sanitize_symbol(row["Coin"])
        route = ctx.route_for(source_symbol)
        display_symbol = _display_symbol(
            symbol=source_symbol,
            route=route,
            meta=metadata.get(_selection_key(source_symbol)),
            ctx=ctx,
        )
        if _selection_key(display_symbol) in base_symbols:
            continue

        date_value = pd.Timestamp(row["Date"]).normalize()
        price_lookup_symbol = _price_symbol(
            symbol=source_symbol,
            route=route,
            meta=metadata.get(_selection_key(source_symbol)),
        )
        resolution = price_resolver.resolve(symbol=price_lookup_symbol, target_date=date_value)
        price_eur = resolution.price
        quantity = float(row["Quantity"])
        market_value = float(Decimal(str(quantity)) * price_eur) if price_eur is not None else None
        principal = float(row["PrincipalInvestedEUR"])
        is_material = (
            abs(quantity) > MATERIAL_QUANTITY_THRESHOLD
            or abs(principal) >= MATERIAL_VALUE_THRESHOLD_EUR
            or abs(market_value or 0.0) >= MATERIAL_VALUE_THRESHOLD_EUR
        )
        rows.append(
            {
                "Date": date_value,
                "Selection": display_symbol,
                "AssetLayer": "position",
                "Coin": display_symbol,
                "Quantity": quantity,
                "PriceEUR": float(price_eur) if price_eur is not None else pd.NA,
                "MarketValueEUR": market_value,
                "PrincipalInvestedEUR": principal,
                "ProfitLossEUR": (market_value if market_value is not None else 0.0) - principal,
                "ValuationRoute": route.value,
                "HasDirectExposure": route == ValuationRoute.DIRECT,
                "HasProtocolExposure": route == ValuationRoute.PROTOCOL_DERIVED,
                "HasAaveExposure": route == ValuationRoute.AAVE,
                "MissingPrice": price_eur is None and is_material,
                "IsMaterial": is_material,
            }
        )

    if not rows:
        return _empty(ASSET_DAILY_COLUMNS)
    return pd.DataFrame(rows, columns=ASSET_DAILY_COLUMNS)


def _source_route_flags(route: ValuationRoute) -> dict[str, bool]:
    return {
        "HasDirectExposure": route == ValuationRoute.DIRECT,
        "HasProtocolExposure": route == ValuationRoute.PROTOCOL_DERIVED,
        "HasAaveExposure": route == ValuationRoute.AAVE,
    }


def _build_source_quantity_rows(
    *,
    snapshots: pd.DataFrame,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    columns = [
        "Date",
        "Source",
        "Coin",
        "Quantity",
        "MarketValueEUR",
        "ValuationRoute",
        "HasDirectExposure",
        "HasProtocolExposure",
        "HasAaveExposure",
    ]
    if snapshots.empty:
        return _empty(columns)

    dense = _dense_snapshot_state(snapshots=snapshots, end_date=end_date)
    if dense.empty:
        return _empty(columns)

    expander = ExposureExpander(ctx=ctx)
    price_resolver = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="eur")
    rows: list[dict[str, object]] = []
    for _, row in dense.iterrows():
        source_symbol = sanitize_symbol(row["Coin"])
        source_quantity = Decimal(str(row["Quantity"]))
        if not source_symbol or abs(source_quantity) <= Decimal(str(MATERIAL_QUANTITY_THRESHOLD)):
            continue

        date_value = pd.Timestamp(row["Date"]).normalize()
        route = ctx.route_for(source_symbol)
        if route == ValuationRoute.AAVE:
            component_symbol = _aave_exposure_symbol(
                symbol=source_symbol,
                meta=metadata.get(_selection_key(source_symbol)),
            )
            signed_quantity = source_quantity * _aave_quantity_multiplier(source_symbol)
            resolution = price_resolver.resolve(
                symbol=component_symbol,
                target_date=date_value,
            )
            market_value = (
                float(signed_quantity * resolution.price) if resolution.price is not None else pd.NA
            )
            rows.append(
                {
                    "Date": date_value,
                    "Source": source_symbol,
                    "Coin": component_symbol,
                    "Quantity": float(signed_quantity),
                    "MarketValueEUR": market_value,
                    "ValuationRoute": route.value,
                    **_source_route_flags(route),
                }
            )
            continue

        exposures = expander.expand(
            symbol=source_symbol,
            quantity=source_quantity,
            date_value=date_value,
            has_direct_exposure=route == ValuationRoute.DIRECT,
            has_protocol_exposure=route == ValuationRoute.PROTOCOL_DERIVED,
            has_aave_exposure=route == ValuationRoute.AAVE,
        )
        flags = _source_route_flags(route)
        for component_symbol, exposure in exposures.items():
            resolution = price_resolver.resolve(
                symbol=component_symbol,
                target_date=date_value,
            )
            market_value = (
                float(exposure.quantity * resolution.price)
                if resolution.price is not None
                else pd.NA
            )
            rows.append(
                {
                    "Date": date_value,
                    "Source": source_symbol,
                    "Coin": component_symbol,
                    "Quantity": float(exposure.quantity),
                    "MarketValueEUR": market_value,
                    "ValuationRoute": route.value,
                    **flags,
                }
            )

    if not rows:
        return _empty(columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .groupby(
            [
                "Date",
                "Source",
                "Coin",
                "ValuationRoute",
                "HasDirectExposure",
                "HasProtocolExposure",
                "HasAaveExposure",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                "Quantity": "sum",
                "MarketValueEUR": lambda values: values.sum(min_count=1),
            }
        )
    )


def _build_source_daily(
    *,
    snapshots: pd.DataFrame,
    principal_by_source: pd.DataFrame,
    ctx: CompositionContext,
    metadata: dict[str, dict[str, Any]],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if snapshots.empty and principal_by_source.empty:
        return _empty(SOURCE_DAILY_COLUMNS)

    quantities = _build_source_quantity_rows(
        snapshots=snapshots,
        ctx=ctx,
        metadata=metadata,
        end_date=end_date,
    )
    if principal_by_source.empty:
        principal = _empty(["Date", "Source", "Coin", "PrincipalInvestedEUR"])
    else:
        principal = principal_by_source.copy()

    source = pd.merge(
        left=quantities,
        right=principal,
        on=["Date", "Source", "Coin"],
        how="outer",
    )
    if source.empty:
        return _empty(SOURCE_DAILY_COLUMNS)

    for column in ("Quantity", "MarketValueEUR", "PrincipalInvestedEUR"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["Quantity"] = source["Quantity"].fillna(0.0)
    source["PrincipalInvestedEUR"] = source["PrincipalInvestedEUR"].fillna(0.0)

    missing_route = source["ValuationRoute"].isna()
    if missing_route.any():
        source.loc[missing_route, "ValuationRoute"] = source.loc[missing_route, "Source"].map(
            lambda symbol: ctx.route_for(str(symbol)).value
        )
    for column, route_value in {
        "HasDirectExposure": ValuationRoute.DIRECT.value,
        "HasProtocolExposure": ValuationRoute.PROTOCOL_DERIVED.value,
        "HasAaveExposure": ValuationRoute.AAVE.value,
    }.items():
        if column not in source.columns:
            source[column] = False
        missing_flags = source[column].isna()
        source.loc[missing_flags, column] = (
            source.loc[missing_flags, "ValuationRoute"].astype(str) == route_value
        )
        source[column] = source[column].map(_normalize_bool)

    source["ProfitLossEUR"] = source["MarketValueEUR"].fillna(0.0) - source["PrincipalInvestedEUR"]
    source["Selection"] = source["Coin"]
    source["IsMaterial"] = (
        (source["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (source["MarketValueEUR"].fillna(0.0).abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
        | (source["PrincipalInvestedEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
    )
    source = source[source["IsMaterial"].map(_normalize_bool)].copy()
    if source.empty:
        return _empty(SOURCE_DAILY_COLUMNS)
    source = source.sort_values(["Selection", "Date", "Source", "Coin"])
    return source[SOURCE_DAILY_COLUMNS]


def _normalize_source_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty(SOURCE_BASE_DAILY_COLUMNS)

    normalized = frame.copy()
    normalized["Date"] = _parse_daily_series(normalized["Date"])
    normalized["Source"] = normalized["Source"].map(sanitize_symbol)
    normalized["BaseCoin"] = normalized["BaseCoin"].map(sanitize_symbol)
    for column in ("Quantity", "MarketValueEUR", "PrincipalInvestedEUR", "RealizedPnLEUR"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
    for column in ("HasDirectExposure", "HasProtocolExposure", "HasAaveExposure"):
        normalized[column] = normalized[column].map(_normalize_bool)
    normalized = normalized.dropna(subset=["Date"])
    normalized = normalized[(normalized["Source"] != "") & (normalized["BaseCoin"] != "")]
    return normalized[SOURCE_BASE_DAILY_COLUMNS].reset_index(drop=True)


def _build_source_daily_from_accounting(source_base: pd.DataFrame) -> pd.DataFrame:
    if source_base.empty:
        return _empty(SOURCE_DAILY_COLUMNS)

    source = source_base.copy()
    source["Selection"] = source["BaseCoin"]
    source["Coin"] = source["BaseCoin"]
    source["PrincipalInvestedEUR"] = source["PrincipalInvestedEUR"] - source["RealizedPnLEUR"]
    source["ProfitLossEUR"] = source["MarketValueEUR"] - source["PrincipalInvestedEUR"]
    source["IsMaterial"] = (
        (source["Quantity"].abs() > MATERIAL_QUANTITY_THRESHOLD)
        | (source["MarketValueEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
        | (source["PrincipalInvestedEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
        | (source["ProfitLossEUR"].abs() >= MATERIAL_VALUE_THRESHOLD_EUR)
    )
    source = source[source["IsMaterial"].map(_normalize_bool)].copy()
    if source.empty:
        return _empty(SOURCE_DAILY_COLUMNS)
    source = source.sort_values(["Selection", "Date", "Source", "Coin"])
    return source[SOURCE_DAILY_COLUMNS]


def _split_symbols(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [
        symbol for symbol in (sanitize_symbol(part) for part in str(value).split(",")) if symbol
    ]


def _source_base_keys_for_symbol(
    *,
    source_base: pd.DataFrame,
    symbol: str,
    date_value: pd.Timestamp,
) -> set[str]:
    if source_base.empty:
        return set()

    source_key = _selection_key(symbol)
    frame = source_base[
        (source_base["Source"].map(_selection_key) == source_key)
        & (source_base["Date"] <= date_value)
    ]
    if frame.empty:
        return set()

    latest_date = frame["Date"].max()
    latest = frame[frame["Date"] == latest_date]
    return {_selection_key(value) for value in latest["BaseCoin"].dropna().tolist()}


def _asset_keys_for_symbol(
    *,
    symbol: str,
    date_value: pd.Timestamp,
    metadata: dict[str, dict[str, Any]],
    source_base: pd.DataFrame,
) -> set[str]:
    keys = {_selection_key(symbol)}
    keys.add(
        _selection_key(
            _aave_exposure_symbol(
                symbol=symbol,
                meta=metadata.get(_selection_key(symbol)),
            )
        )
    )
    keys.update(
        _source_base_keys_for_symbol(
            source_base=source_base,
            symbol=symbol,
            date_value=date_value,
        )
    )
    return {key for key in keys if key}


def _build_transactions_dashboard(
    *,
    transactions: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    source_base: pd.DataFrame,
) -> pd.DataFrame:
    if transactions.empty:
        return _empty(TRANSACTIONS_DASHBOARD_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in transactions.iterrows():
        keys = {"ALL"}
        date_value = pd.Timestamp(row["Date"]).normalize()
        for column in ("Token in", "Token out", "Fee Token"):
            for symbol in _split_symbols(row.get(column)):
                keys.update(
                    _asset_keys_for_symbol(
                        symbol=symbol,
                        date_value=date_value,
                        metadata=metadata,
                        source_base=source_base,
                    )
                )
        out_row = {column: row.get(column, "") for column in TRANSACTIONS_DASHBOARD_COLUMNS[:-1]}
        out_row["Date"] = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d %H:%M:%S")
        out_row["AssetKeys"] = ";".join(sorted(key for key in keys if key))
        rows.append(out_row)
    return pd.DataFrame(rows, columns=TRANSACTIONS_DASHBOARD_COLUMNS)


def _filter_transactions_for_selection(transactions: pd.DataFrame, selection: str) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()
    key = _selection_key(selection)
    if key == "ALL":
        return transactions.copy()
    return transactions[
        transactions["AssetKeys"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .map(lambda keys: key in keys)
    ].copy()


def _build_composition_daily(asset_daily: pd.DataFrame) -> pd.DataFrame:
    if asset_daily.empty:
        return _empty(COMPOSITION_DAILY_COLUMNS)

    rows: list[pd.DataFrame] = []
    frame = asset_daily.copy()
    frame["ValueEUR"] = pd.to_numeric(frame["MarketValueEUR"], errors="coerce").abs().fillna(0.0)
    frame = frame[frame["ValueEUR"] > 0]
    if frame.empty:
        return _empty(COMPOSITION_DAILY_COLUMNS)

    for mode, label_series in {
        "name": frame["Coin"].fillna("Unknown").astype(str),
        "route": frame["ValuationRoute"].fillna("Unknown").astype(str),
        "exposure": frame.apply(_exposure_label, axis=1),
    }.items():
        grouped = (
            frame.assign(CompositionMode=mode, Label=label_series)
            .groupby(["Date", "Selection", "CompositionMode", "Label"], as_index=False)["ValueEUR"]
            .sum()
        )
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True, sort=False)[COMPOSITION_DAILY_COLUMNS]


def _build_timeseries_daily(
    *,
    asset_daily: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    if asset_daily.empty:
        return _empty(TIMESERIES_DAILY_COLUMNS)

    grouped = (
        asset_daily.groupby(["Date", "Selection"], as_index=False)
        .agg(
            {
                "MarketValueEUR": "sum",
                "PrincipalInvestedEUR": "sum",
                "Quantity": "sum",
            }
        )
        .sort_values(["Selection", "Date"])
    )
    grouped["ProfitLossEUR"] = grouped["MarketValueEUR"] - grouped["PrincipalInvestedEUR"]
    final_date = grouped["Date"].max()
    dense_frames: list[pd.DataFrame] = []
    for selection, selection_rows in grouped.groupby("Selection", sort=False):
        selection_rows = selection_rows.sort_values("Date")
        calendar = pd.date_range(
            start=selection_rows["Date"].min(),
            end=final_date,
            freq="D",
        )
        dense = selection_rows.set_index("Date").reindex(calendar).rename_axis("Date").reset_index()
        dense["Selection"] = selection
        for column in ("MarketValueEUR", "PrincipalInvestedEUR", "Quantity"):
            dense[column] = pd.to_numeric(dense[column], errors="coerce").fillna(0.0)
        dense["ProfitLossEUR"] = (
            pd.to_numeric(dense["ProfitLossEUR"], errors="coerce").ffill().fillna(0.0)
        )
        dense_frames.append(dense)

    if dense_frames:
        grouped = pd.concat(dense_frames, ignore_index=True, sort=False)

    count_frames: list[pd.DataFrame] = []
    for selection in sorted(grouped["Selection"].dropna().unique().tolist()):
        filtered_tx = _filter_transactions_for_selection(
            transactions=transactions,
            selection=selection,
        )
        if filtered_tx.empty:
            continue
        tx_counts = (
            filtered_tx.assign(
                Date=pd.to_datetime(filtered_tx["Date"], errors="coerce").dt.normalize()
            )
            .dropna(subset=["Date"])
            .groupby("Date", as_index=False)
            .size()
            .rename(columns={"size": "TxCount"})
        )
        tx_counts["Selection"] = selection
        count_frames.append(tx_counts)

    if count_frames:
        tx_daily = pd.concat(count_frames, ignore_index=True, sort=False)
        grouped = pd.merge(grouped, tx_daily, on=["Date", "Selection"], how="left")
    else:
        grouped["TxCount"] = 0
    grouped["TxCount"] = pd.to_numeric(grouped["TxCount"], errors="coerce").fillna(0).astype(int)

    return grouped[TIMESERIES_DAILY_COLUMNS]


def _build_assets(asset_daily: pd.DataFrame) -> pd.DataFrame:
    if asset_daily.empty:
        return _empty(ASSETS_COLUMNS)

    frame = asset_daily[asset_daily["Selection"] != "ALL"].copy()
    frame = frame[frame["IsMaterial"].map(_normalize_bool)]
    assets = sorted({str(selection) for selection in frame["Selection"].dropna().tolist()})
    return pd.DataFrame(
        [{"Label": asset, "Value": asset} for asset in assets],
        columns=ASSETS_COLUMNS,
    )


def _write_artifact(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[columns]
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            if column == "Date":
                output[column] = output[column].dt.strftime("%Y-%m-%d")
            else:
                output[column] = output[column].astype(str)
    output.to_csv(path, index=False)


def build_arbitrum_dashboard_artifacts(chain: str = CHAIN) -> ArbitrumDashboardArtifactPaths:
    """
    Builds dashboard-ready Arbitrum CSV artifacts from generated blockchain CSVs.

    args:
        chain: Chain identifier.

    returns:
        Paths for all generated dashboard artifacts.
    """
    paths = artifact_paths(chain=chain)
    token_metadata = load_token_metadata(chain=chain, tokens_folder=TOKENS_FOLDER)
    metadata = _metadata_by_symbol(token_metadata)
    accounting = accounting_paths(chain=chain)
    base = _normalize_accounting_base_frame(
        _read_csv(
            accounting.base_daily,
            BASE_DAILY_COLUMNS,
        )
    )
    source_base = _normalize_source_base_frame(
        _read_csv(
            accounting.source_base_daily,
            SOURCE_BASE_DAILY_COLUMNS,
        )
    )
    base_asset_rows = _build_asset_rows_from_accounting(base=base)
    source_daily = _build_source_daily_from_accounting(source_base=source_base)
    asset_daily = base_asset_rows.copy()
    if asset_daily.empty:
        asset_daily = _empty(ASSET_DAILY_COLUMNS)
    else:
        asset_daily["Date"] = pd.to_datetime(asset_daily["Date"], errors="coerce").dt.normalize()
        asset_daily = asset_daily.dropna(subset=["Date"])
        asset_daily = asset_daily.sort_values(["Selection", "Date", "Coin", "AssetLayer"])

    transactions = _build_transactions_dashboard(
        transactions=_normalize_transactions_frame(
            _read_csv(
                BLOCKCHAIN_TRANSACTIONS_FOLDER / f"{chain}_transactions.csv",
                TRANSACTIONS_DASHBOARD_COLUMNS[:-1],
            )
        ),
        metadata=metadata,
        source_base=source_base,
    )
    composition_daily = _build_composition_daily(asset_daily=asset_daily)
    timeseries_daily = _build_timeseries_daily(
        asset_daily=asset_daily,
        transactions=transactions,
    )
    assets = _build_assets(asset_daily=asset_daily)

    _write_artifact(paths.asset_daily, asset_daily, ASSET_DAILY_COLUMNS)
    _write_artifact(paths.timeseries_daily, timeseries_daily, TIMESERIES_DAILY_COLUMNS)
    _write_artifact(paths.composition_daily, composition_daily, COMPOSITION_DAILY_COLUMNS)
    _write_artifact(paths.source_daily, source_daily, SOURCE_DAILY_COLUMNS)
    _write_artifact(
        paths.transactions_dashboard,
        transactions,
        TRANSACTIONS_DASHBOARD_COLUMNS,
    )
    _write_artifact(paths.assets, assets, ASSETS_COLUMNS)
    stale_data_quality_path = paths.asset_daily.parent / "data_quality.csv"
    if stale_data_quality_path.exists():
        stale_data_quality_path.unlink()
    print(f"[dashboard_artifacts] Saved Arbitrum artifacts to {paths.asset_daily.parent}")
    return paths
