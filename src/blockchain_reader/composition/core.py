from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from blockchain_reader.datetime_utils import parse_daily_datetime
from blockchain_reader.shared.prices import (
    STABLE_PRICE_SYMBOLS,
    get_price_eur_on_or_before,
)
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
    price_proxy_symbol,
    sanitize_symbol,
)
from file_paths import (
    PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
    get_direct_price_file_path,
    get_lp_price_file_path,
)
from price_history.price_data_utils import load_price_csv

DUST = Decimal("0.000000000001")
MAX_EXPANSION_DEPTH = 8
MAX_PRICE_RECURSION_DEPTH = 10
PriceMode = Literal["eur", "native"]


@dataclass(frozen=True)
class SymbolMetadata:
    price_source: str = ""
    family: str = ""
    protocol: str = ""


@dataclass(frozen=True)
class PriceResolution:
    route: ValuationRoute
    price_symbol: str | None
    price: Decimal | None
    currency: str

    @property
    def price_eur(self) -> Decimal | None:
        return self.price


@dataclass
class AggregatedExposure:
    quantity: Decimal = Decimal("0")
    has_direct_exposure: bool = False
    has_protocol_exposure: bool = False
    has_aave_exposure: bool = False


@dataclass
class ProtocolStore:
    chain: str
    rows: dict[str, pd.DataFrame]

    @classmethod
    def load(
        cls,
        *,
        chain: str,
        root: Path = PROTOCOL_UNDERLYING_TOKEN_FOLDER,
        include_aave: bool = False,
    ) -> ProtocolStore:
        rows: dict[str, pd.DataFrame] = {}
        if not root.exists():
            return cls(chain=chain, rows=rows)

        excluded_names = {f"{chain}_aave_daily_exposure.csv"}
        for csv_path in root.rglob(f"{chain}_*.csv"):
            if not include_aave and csv_path.parent.name == "aave":
                continue
            if csv_path.name in excluded_names:
                continue

            symbol = sanitize_symbol(csv_path.stem[len(chain) + 1 :])
            if not symbol:
                continue

            df = pd.read_csv(csv_path)
            if "date" not in df.columns:
                continue

            df = df.copy()
            df["date"] = pd.to_datetime(
                df["date"].map(parse_daily_datetime),
                errors="coerce",
            )
            df = df.dropna(subset=["date"])
            if df.empty:
                continue

            df["date"] = df["date"].dt.normalize()
            rows[symbol] = df.sort_values("date").reset_index(drop=True)

        return cls(chain=chain, rows=rows)

    @property
    def symbols(self) -> set[str]:
        return set(self.rows.keys())

    def find_row(self, *, symbol: str, target_date: object) -> pd.Series | None:
        return find_protocol_row(
            protocol_rows=self.rows,
            symbol=symbol,
            target_date=target_date,
        )


@dataclass
class CompositionContext:
    chain: str
    protocol_rows: dict[str, pd.DataFrame]
    symbol_protocol: dict[str, str]
    protocol_derived_symbols: set[str]
    symbol_family: dict[str, str]
    aave_overlay: pd.DataFrame | None = None
    aave_wrapper_symbols: set[str] = field(default_factory=set)
    known_symbols: set[str] = field(default_factory=set)
    symbol_metadata: dict[str, SymbolMetadata] = field(default_factory=dict)
    price_cache: dict[tuple[str, bool], pd.DataFrame] = field(default_factory=dict)

    def route_for(self, symbol: str) -> ValuationRoute:
        return classify_valuation_route(
            symbol=symbol,
            symbol_protocol=self.symbol_protocol,
            protocol_derived_symbols=self.protocol_derived_symbols,
        )


def build_symbol_metadata(
    token_metadata: dict[str, dict[str, Any]],
) -> dict[str, SymbolMetadata]:
    merged: dict[str, dict[str, str]] = {}

    for meta in token_metadata.values():
        if not isinstance(meta, dict):
            continue

        symbol = sanitize_symbol(meta.get("symbol"))
        if not symbol:
            continue

        price_source = sanitize_symbol(meta.get("price_source"))
        family = sanitize_symbol(meta.get("family"))
        protocol = sanitize_symbol(meta.get("protocol")).lower()
        current = merged.get(symbol, {"price_source": "", "family": "", "protocol": ""})

        if not current["price_source"] and price_source:
            current["price_source"] = price_source
        if not current["family"] and family:
            current["family"] = family
        if not current["protocol"] and protocol:
            current["protocol"] = protocol
        merged[symbol] = current

    return {
        symbol: SymbolMetadata(
            price_source=meta["price_source"],
            family=meta["family"],
            protocol=meta["protocol"],
        )
        for symbol, meta in merged.items()
    }


def build_composition_context(
    *,
    chain: str,
    token_metadata: dict[str, dict[str, Any]] | None = None,
    protocol_root: Path = PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    include_aave: bool = False,
    aave_overlay: pd.DataFrame | None = None,
    aave_wrapper_symbols: set[str] | None = None,
) -> CompositionContext:
    metadata = token_metadata
    if metadata is None:
        metadata = load_token_metadata(chain=chain, tokens_folder=TOKENS_FOLDER)

    symbol_family = build_symbol_family_map(token_metadata=metadata)
    protocol_store = ProtocolStore.load(
        chain=chain,
        root=protocol_root,
        include_aave=include_aave,
    )
    return CompositionContext(
        chain=chain,
        protocol_rows=protocol_store.rows,
        symbol_protocol=build_symbol_protocol_map(token_metadata=metadata),
        protocol_derived_symbols=protocol_store.symbols,
        symbol_family=symbol_family,
        aave_overlay=aave_overlay,
        aave_wrapper_symbols=aave_wrapper_symbols or set(),
        known_symbols=build_known_canonical_symbols(
            token_metadata=metadata,
            symbol_family=symbol_family,
        ),
        symbol_metadata=build_symbol_metadata(token_metadata=metadata),
    )


def find_protocol_row(
    *,
    protocol_rows: dict[str, pd.DataFrame],
    symbol: str,
    target_date: object,
) -> pd.Series | None:
    normalized_symbol = sanitize_symbol(symbol)
    df = protocol_rows.get(normalized_symbol)
    if df is None or df.empty:
        return None

    target = pd.Timestamp(target_date).normalize()
    eligible = df[df["date"] <= target]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _to_decimal(value: object) -> Decimal:
    if pd.isna(value):
        return Decimal("0")
    return Decimal(str(value))


class ExposureExpander:
    def __init__(self, ctx: CompositionContext):
        self.ctx = ctx

    def expand(
        self,
        *,
        symbol: str,
        quantity: Decimal,
        date_value: object,
        has_direct_exposure: bool,
        has_protocol_exposure: bool,
        has_aave_exposure: bool,
    ) -> dict[str, AggregatedExposure]:
        exposures: dict[str, AggregatedExposure] = {}
        self.expand_into(
            symbol=symbol,
            quantity=quantity,
            date_value=date_value,
            exposures=exposures,
            has_direct_exposure=has_direct_exposure,
            has_protocol_exposure=has_protocol_exposure,
            has_aave_exposure=has_aave_exposure,
        )
        return exposures

    def expand_into(
        self,
        *,
        symbol: str,
        quantity: Decimal,
        date_value: object,
        exposures: dict[str, AggregatedExposure],
        has_direct_exposure: bool,
        has_protocol_exposure: bool,
        has_aave_exposure: bool,
        depth: int = 0,
    ) -> None:
        normalized_symbol = sanitize_symbol(symbol)
        if not normalized_symbol or abs(quantity) <= DUST:
            return

        route = self.ctx.route_for(normalized_symbol)
        current_has_direct = has_direct_exposure or route == ValuationRoute.DIRECT
        current_has_protocol = has_protocol_exposure or route == ValuationRoute.PROTOCOL_DERIVED
        terminal_symbol = normalized_symbol
        if route == ValuationRoute.DIRECT:
            terminal_symbol = canonicalize_symbol(
                normalized_symbol,
                symbol_family=self.ctx.symbol_family,
            )
            terminal_symbol = terminal_symbol or normalized_symbol

        if depth > MAX_EXPANSION_DEPTH:
            self._add_exposure(
                exposures=exposures,
                symbol=terminal_symbol,
                quantity=quantity,
                has_direct_exposure=current_has_direct,
                has_protocol_exposure=current_has_protocol,
                has_aave_exposure=has_aave_exposure,
            )
            return

        row = find_protocol_row(
            protocol_rows=self.ctx.protocol_rows,
            symbol=normalized_symbol,
            target_date=date_value,
        )
        if row is None:
            self._add_exposure(
                exposures=exposures,
                symbol=terminal_symbol,
                quantity=quantity,
                has_direct_exposure=current_has_direct,
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
            self.expand_into(
                symbol=column.replace("asset_", "", 1),
                quantity=quantity * per_unit,
                date_value=date_value,
                exposures=exposures,
                has_direct_exposure=current_has_direct,
                has_protocol_exposure=current_has_protocol,
                has_aave_exposure=has_aave_exposure,
                depth=depth + 1,
            )

        if expanded:
            return

        self._add_exposure(
            exposures=exposures,
            symbol=terminal_symbol,
            quantity=quantity,
            has_direct_exposure=current_has_direct,
            has_protocol_exposure=current_has_protocol,
            has_aave_exposure=has_aave_exposure,
        )

    def _add_exposure(
        self,
        *,
        exposures: dict[str, AggregatedExposure],
        symbol: str,
        quantity: Decimal,
        has_direct_exposure: bool,
        has_protocol_exposure: bool,
        has_aave_exposure: bool,
    ) -> None:
        normalized_symbol = sanitize_symbol(symbol)
        if not normalized_symbol or abs(quantity) <= DUST:
            return

        exposure = exposures.setdefault(normalized_symbol, AggregatedExposure())
        exposure.quantity += quantity
        exposure.has_direct_exposure = exposure.has_direct_exposure or has_direct_exposure
        exposure.has_protocol_exposure = exposure.has_protocol_exposure or has_protocol_exposure
        exposure.has_aave_exposure = exposure.has_aave_exposure or has_aave_exposure


class PriceResolver:
    def __init__(
        self,
        *,
        ctx: CompositionContext,
        prices_folder: Path = PRICES_FOLDER,
        mode: PriceMode = "eur",
        fallback_to_oldest: bool = False,
    ):
        self.ctx = ctx
        self.prices_folder = prices_folder
        self.mode = mode
        self.fallback_to_oldest = fallback_to_oldest
        self.currency = "EUR" if mode == "eur" else "native"

    def resolve(
        self,
        *,
        symbol: str,
        target_date: object,
        visited: set[str] | None = None,
        depth: int = 0,
    ) -> PriceResolution:
        normalized = sanitize_symbol(symbol)
        if not normalized:
            return PriceResolution(
                route=ValuationRoute.DIRECT,
                price_symbol=None,
                price=None,
                currency=self.currency,
            )
        if depth > MAX_PRICE_RECURSION_DEPTH:
            return self._empty_resolution(symbol=normalized)

        route = self.ctx.route_for(normalized)
        resolution = self._resolve_direct_candidates(
            symbol=normalized,
            route=route,
            target_date=target_date,
        )
        if resolution.price is not None:
            return resolution

        if route == ValuationRoute.PROTOCOL_DERIVED:
            protocol_resolution = self._resolve_from_protocol(
                symbol=normalized,
                target_date=target_date,
                visited=visited or set(),
                depth=depth,
            )
            if protocol_resolution.price is not None:
                return protocol_resolution

        metadata = self.ctx.symbol_metadata.get(normalized)
        metadata_candidates: list[str] = []
        if metadata and metadata.price_source and metadata.price_source != normalized:
            metadata_candidates.append(metadata.price_source)
        if route == ValuationRoute.DIRECT and metadata and metadata.family:
            if metadata.family not in {normalized, metadata.price_source}:
                metadata_candidates.append(metadata.family)

        for candidate in metadata_candidates:
            price = self._price_from_history(
                symbol=candidate,
                target_date=target_date,
                use_lp_prices=False,
            )
            if price is not None:
                return PriceResolution(
                    route=route,
                    price_symbol=candidate,
                    price=price,
                    currency=self.currency,
                )

        protocol_resolution = self._resolve_from_protocol(
            symbol=normalized,
            target_date=target_date,
            visited=visited or set(),
            depth=depth,
        )
        if protocol_resolution.price is not None:
            return protocol_resolution

        return self._empty_resolution(symbol=normalized, route=route)

    def _empty_resolution(
        self,
        *,
        symbol: str,
        route: ValuationRoute | None = None,
    ) -> PriceResolution:
        return PriceResolution(
            route=route or self.ctx.route_for(symbol),
            price_symbol=None,
            price=None,
            currency=self.currency,
        )

    def _resolve_direct_candidates(
        self,
        *,
        symbol: str,
        route: ValuationRoute,
        target_date: object,
    ) -> PriceResolution:
        candidates = [symbol]
        proxy_symbol = price_proxy_symbol(symbol)
        if proxy_symbol and proxy_symbol not in candidates:
            candidates.append(proxy_symbol)

        if route == ValuationRoute.DIRECT:
            canonical_symbol = canonicalize_symbol(
                symbol,
                symbol_family=self.ctx.symbol_family,
            )
            if canonical_symbol and canonical_symbol not in candidates:
                candidates.append(canonical_symbol)

        for candidate in candidates:
            price = self._price_from_history(
                symbol=candidate,
                target_date=target_date,
                use_lp_prices=candidate in self.ctx.protocol_derived_symbols,
            )
            if price is not None:
                return PriceResolution(
                    route=route,
                    price_symbol=candidate,
                    price=price,
                    currency=self.currency,
                )

        return self._empty_resolution(symbol=symbol, route=route)

    def _resolve_from_protocol(
        self,
        *,
        symbol: str,
        target_date: object,
        visited: set[str],
        depth: int,
    ) -> PriceResolution:
        if depth > MAX_PRICE_RECURSION_DEPTH or symbol in visited:
            return self._empty_resolution(symbol=symbol, route=ValuationRoute.PROTOCOL_DERIVED)

        row = find_protocol_row(
            protocol_rows=self.ctx.protocol_rows,
            symbol=symbol,
            target_date=target_date,
        )
        if row is None:
            return self._empty_resolution(symbol=symbol, route=ValuationRoute.PROTOCOL_DERIVED)

        asset_columns = [
            column
            for column in row.index
            if isinstance(column, str) and column.startswith("asset_")
        ]
        if not asset_columns:
            return self._empty_resolution(symbol=symbol, route=ValuationRoute.PROTOCOL_DERIVED)

        total = Decimal("0")
        next_visited = set(visited)
        next_visited.add(symbol)

        for column in asset_columns:
            raw_quantity = row[column]
            if pd.isna(raw_quantity):
                return self._empty_resolution(
                    symbol=symbol,
                    route=ValuationRoute.PROTOCOL_DERIVED,
                )

            quantity = Decimal(str(raw_quantity))
            if abs(quantity) <= DUST:
                continue

            component_symbol = sanitize_symbol(column.replace("asset_", "", 1))
            if not component_symbol:
                return self._empty_resolution(
                    symbol=symbol,
                    route=ValuationRoute.PROTOCOL_DERIVED,
                )

            component = self.resolve(
                symbol=component_symbol,
                target_date=target_date,
                visited=next_visited,
                depth=depth + 1,
            )
            if component.price is None:
                return self._empty_resolution(
                    symbol=symbol,
                    route=ValuationRoute.PROTOCOL_DERIVED,
                )

            total += quantity * component.price

        return PriceResolution(
            route=ValuationRoute.PROTOCOL_DERIVED,
            price_symbol=symbol,
            price=total,
            currency=self.currency,
        )

    def _price_from_history(
        self,
        *,
        symbol: str,
        target_date: object,
        use_lp_prices: bool,
    ) -> Decimal | None:
        if self.mode == "eur":
            return get_price_eur_on_or_before(
                symbol=symbol,
                as_of_date=target_date,
                prices_folder=self.prices_folder,
                chain=self.ctx.chain,
                use_lp_prices=use_lp_prices,
                fallback_to_oldest=self.fallback_to_oldest,
            )

        stable_price = STABLE_PRICE_SYMBOLS.get(symbol)
        if stable_price is not None:
            return stable_price

        history = self._load_native_history(symbol=symbol, use_lp_prices=use_lp_prices)
        if history.empty:
            return None

        target = _as_date(target_date)
        eligible = history[history["Date"] <= target]
        if eligible.empty:
            return None
        return Decimal(str(eligible.iloc[-1]["Price"]))

    def _load_native_history(self, *, symbol: str, use_lp_prices: bool) -> pd.DataFrame:
        cache_key = (symbol, use_lp_prices)
        if cache_key in self.ctx.price_cache:
            return self.ctx.price_cache[cache_key]

        file_path = (
            get_lp_price_file_path(
                chain=self.ctx.chain,
                symbol=symbol,
                prices_folder=self.prices_folder,
            )
            if use_lp_prices
            else get_direct_price_file_path(symbol=symbol, prices_folder=self.prices_folder)
        )
        frame = load_price_csv(file_path=file_path)
        if frame.empty:
            self.ctx.price_cache[cache_key] = frame
            return frame

        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
        frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Price"])[["Date", "Price"]].sort_values("Date")
        self.ctx.price_cache[cache_key] = frame
        return frame


def _as_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def component_value_weights(
    *,
    exposures: dict[str, AggregatedExposure],
    date_value: object,
    price_resolver: PriceResolver,
) -> dict[str, float]:
    value_weights: dict[str, float] = {}
    quantity_weights: dict[str, float] = {}

    for component_symbol, exposure in exposures.items():
        quantity = exposure.quantity
        quantity_weights[component_symbol] = abs(float(quantity))
        resolution = price_resolver.resolve(
            symbol=component_symbol,
            target_date=date_value,
        )
        if resolution.price is not None:
            value_weights[component_symbol] = abs(float(quantity * resolution.price))
        else:
            value_weights[component_symbol] = 0.0

    if sum(value_weights.values()) > 0:
        return value_weights
    return quantity_weights
