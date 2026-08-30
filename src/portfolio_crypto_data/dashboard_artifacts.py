from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from portfolio_core import active_context, atomic_write_csv

from portfolio_crypto_data.accounting import (
    BASE_DAILY_COLUMNS,
    SOURCE_BASE_DAILY_COLUMNS,
    accounting_paths,
)
from portfolio_crypto_data.datetime_utils import (
    parse_daily_datetime,
    parse_transaction_datetime_series,
)
from portfolio_crypto_data.shared.aave_symbols import aave_base_symbol
from portfolio_crypto_data.shared.token_metadata import load_token_metadata
from portfolio_crypto_data.symbols import sanitize_symbol

MATERIAL_QUANTITY_THRESHOLD = 1e-10
MATERIAL_VALUE_THRESHOLD_EUR = 1.0

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
class ChainDashboardArtifactPaths:
    asset_daily: Path
    timeseries_daily: Path
    composition_daily: Path
    source_daily: Path
    transactions_dashboard: Path
    assets: Path


def artifact_paths(chain: str) -> ChainDashboardArtifactPaths:
    root = active_context().paths.dashboard_artifacts / chain
    return ChainDashboardArtifactPaths(
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
    frame = pd.read_csv(path)
    return frame[columns].copy()


def _parse_daily_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.map(parse_daily_datetime), errors="coerce").dt.normalize()


def _normalize_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


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
    return aave_base_symbol(symbol=symbol, meta=meta)


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
    atomic_write_csv(frame=output, path=path)


def build_chain_dashboard_artifacts(chain: str) -> ChainDashboardArtifactPaths:
    """
    Builds dashboard-ready CSV artifacts for one EVM chain.

    args:
        chain: Chain identifier.

    returns:
        Paths for all generated dashboard artifacts.
    """
    paths = artifact_paths(chain=chain)
    runtime_paths = active_context().paths
    token_metadata = load_token_metadata(chain=chain, tokens_folder=runtime_paths.tokens)
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
                runtime_paths.crypto_transactions / f"{chain}_transactions.csv",
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
    print(f"[dashboard_artifacts] Saved {chain} artifacts to {paths.asset_daily.parent}")
    return paths
