from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_crypto_data.composition.core import (
    DUST,
    ExposureExpander,
    PriceResolver,
    build_composition_context,
    component_value_weights,
)
from portfolio_crypto_data.datetime_utils import format_daily_datetime
from portfolio_crypto_data.shared.aave_symbols import (
    aave_base_symbol,
    is_aave_debt_symbol,
)
from portfolio_crypto_data.shared.valuation_routes import ValuationRoute
from portfolio_crypto_data.symbols import canonicalize_symbol, price_proxy_symbol, sanitize_symbol

PRINCIPAL_EVENT_COLUMNS = [
    "Date",
    "TX Hash",
    "Action",
    "Source",
    "BaseCoin",
    "PrincipalDeltaEUR",
    "PrincipalBalanceEUR",
]
PRINCIPAL_DAILY_COLUMNS = ["Date", "Coin", "PrincipalInvestedEUR"]


@dataclass(frozen=True)
class PrincipalComponent:
    coin: str
    weight: Decimal


class PrincipalResolver:
    def __init__(
        self,
        *,
        chain: str,
        token_metadata: dict[str, dict[str, Any]],
        protocol_root: Path | None = None,
        prices_folder: Path | None = None,
    ):
        self.ctx = build_composition_context(
            chain=chain,
            token_metadata=token_metadata,
            protocol_root=protocol_root,
            include_aave=False,
        )
        self.price_resolver = PriceResolver(ctx=self.ctx, prices_folder=prices_folder, mode="eur")
        self.metadata = self._metadata_by_symbol(token_metadata=token_metadata)

    def components(self, *, symbol: str, date_value: object) -> list[PrincipalComponent]:
        normalized = sanitize_symbol(symbol)
        if not normalized:
            return []

        meta = self.metadata.get(normalized)
        multiplier = (
            Decimal("-1") if is_aave_debt_symbol(normalized, meta) else Decimal("1")
        )
        route = self.ctx.route_for(normalized)
        expansion_symbol = normalized
        has_aave = route == ValuationRoute.AAVE
        if has_aave:
            expansion_symbol = aave_base_symbol(
                symbol=normalized,
                meta=meta,
            )

        exposures = ExposureExpander(ctx=self.ctx).expand(
            symbol=expansion_symbol,
            quantity=Decimal("1"),
            date_value=date_value,
            has_direct_exposure=route == ValuationRoute.DIRECT,
            has_protocol_exposure=route == ValuationRoute.PROTOCOL_DERIVED,
            has_aave_exposure=has_aave,
        )
        if not exposures:
            return [
                PrincipalComponent(
                    coin=self._principal_coin(expansion_symbol, date_value=date_value),
                    weight=multiplier,
                )
            ]

        weights = component_value_weights(
            exposures=exposures,
            date_value=date_value,
            price_resolver=self.price_resolver,
        )
        component_weights: dict[str, float] = {}
        for raw_coin, raw_weight in weights.items():
            coin = self._principal_coin(raw_coin, date_value=date_value)
            component_weights[coin] = component_weights.get(coin, 0.0) + raw_weight

        total_weight = sum(component_weights.values())
        if total_weight <= 0:
            component_coins = sorted(
                {self._principal_coin(coin, date_value=date_value) for coin in exposures}
            )
            component_count = len(component_coins)
            return [
                PrincipalComponent(
                    coin=coin,
                    weight=multiplier / Decimal(str(component_count)),
                )
                for coin in component_coins
            ]

        components: list[PrincipalComponent] = []
        for coin in sorted(component_weights):
            weight = Decimal(str(component_weights.get(coin, 0.0))) / Decimal(str(total_weight))
            if abs(weight) <= DUST:
                continue
            components.append(PrincipalComponent(coin=coin, weight=weight * multiplier))
        return components

    def _principal_coin(
        self,
        symbol: str,
        *,
        date_value: object,
        seen: set[str] | None = None,
        depth: int = 0,
    ) -> str:
        normalized = sanitize_symbol(symbol)
        canonical = canonicalize_symbol(symbol, symbol_family=self.ctx.symbol_family)
        proxy = price_proxy_symbol(canonical)
        if proxy:
            return proxy
        if canonical and canonical != normalized:
            return canonical

        protocol_base = self._single_family_protocol_base(
            symbol=canonical or normalized,
            date_value=date_value,
            seen=seen,
            depth=depth,
        )
        return protocol_base or canonical or normalized

    def _single_family_protocol_base(
        self,
        *,
        symbol: str,
        date_value: object,
        seen: set[str] | None = None,
        depth: int = 0,
    ) -> str:
        normalized = sanitize_symbol(symbol)
        if not normalized or depth > 8:
            return ""

        visited = set(seen or set())
        if normalized in visited:
            return ""
        visited.add(normalized)

        row = self._principal_protocol_row(symbol=normalized, date_value=date_value)
        if row is None:
            return ""

        bases: set[str] = set()
        for column in row.index:
            if not isinstance(column, str) or not column.startswith("asset_"):
                continue
            if pd.isna(row[column]):
                continue
            quantity = Decimal(str(row[column]))
            if abs(quantity) <= DUST:
                continue
            child_base = self._principal_coin(
                column.replace("asset_", "", 1),
                date_value=date_value,
                seen=visited,
                depth=depth + 1,
            )
            if not child_base:
                return ""
            bases.add(child_base)

        if len(bases) == 1:
            return next(iter(bases))
        return ""

    def _principal_protocol_row(self, *, symbol: str, date_value: object) -> pd.Series | None:
        frame = self.ctx.protocol_rows.get(sanitize_symbol(symbol))
        if frame is None or frame.empty:
            return None

        target = pd.Timestamp(date_value).normalize()
        eligible = frame[frame["date"] <= target]
        if not eligible.empty:
            return eligible.iloc[-1]
        return frame.iloc[0]

    @staticmethod
    def _metadata_by_symbol(token_metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        by_symbol: dict[str, dict[str, Any]] = {}
        for meta in token_metadata.values():
            if not isinstance(meta, dict):
                continue
            symbol = sanitize_symbol(meta.get("symbol"))
            if symbol and symbol not in by_symbol:
                by_symbol[symbol] = meta
        return by_symbol


class EconomicPrincipalLedger:
    def __init__(self, resolver: PrincipalResolver):
        self.resolver = resolver
        self.balances: dict[str, Decimal] = {}
        self.events: list[dict[str, object]] = []

    def adjust(
        self,
        *,
        symbol: str,
        amount_eur: float | Decimal,
        date_value: object,
        action: str,
        tx_hash: str = "",
    ) -> None:
        amount = Decimal(str(amount_eur))
        if abs(amount) <= DUST:
            return

        components = self.resolver.components(symbol=symbol, date_value=date_value)
        total_delta = amount * sum((component.weight for component in components), Decimal("0"))
        remaining = total_delta
        for index, component in enumerate(components):
            if index == len(components) - 1:
                delta = remaining
            else:
                delta = amount * component.weight
                remaining -= delta
            if abs(delta) <= DUST:
                continue

            current = self.balances.get(component.coin, Decimal("0")) + delta
            self.balances[component.coin] = current
            self.events.append(
                {
                    "Date": date_value,
                    "TX Hash": tx_hash,
                    "Action": action,
                    "Source": sanitize_symbol(symbol),
                    "BaseCoin": component.coin,
                    "PrincipalDeltaEUR": float(delta),
                    "PrincipalBalanceEUR": float(current),
                }
            )

    def events_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=PRINCIPAL_EVENT_COLUMNS)
        frame = pd.DataFrame(self.events, columns=PRINCIPAL_EVENT_COLUMNS)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        frame["Date"] = frame["Date"].map(format_daily_datetime)
        return frame[PRINCIPAL_EVENT_COLUMNS]

    def daily_frame(self) -> pd.DataFrame:
        events = self.events_frame()
        if events.empty:
            return pd.DataFrame(columns=PRINCIPAL_DAILY_COLUMNS)
        daily = events.copy()
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.normalize()
        daily = daily.dropna(subset=["Date"])
        daily = daily.sort_values(["Date", "BaseCoin"])
        daily = daily.groupby(["Date", "BaseCoin"], as_index=False).tail(1)
        daily = daily.rename(
            columns={
                "BaseCoin": "Coin",
                "PrincipalBalanceEUR": "PrincipalInvestedEUR",
            }
        )
        daily["Date"] = daily["Date"].map(format_daily_datetime)
        return daily[PRINCIPAL_DAILY_COLUMNS].sort_values(["Date", "Coin"]).reset_index(drop=True)
