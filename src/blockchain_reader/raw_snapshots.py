from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from blockchain_reader.datetime_utils import (
    format_daily_datetime,
    parse_transaction_datetime_series,
)
from blockchain_reader.principal_ledger import EconomicPrincipalLedger, PrincipalResolver
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
from blockchain_reader.symbols import canonicalize_symbol, price_proxy_symbol, sanitize_symbol
from file_paths import (
    PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
)
from historical_transactions.portfolio_snapshots import get_forex_rate

MAX_INVALID_DATE_RATIO = 0.1
AAVE_PRICE_SOURCE_PREFIXES = ("variableDebtArb", "stableDebtArb", "aArb")
AAVE_DEBT_PREFIXES = ("variableDebtArb", "stableDebtArb")
MAX_PRINCIPAL_PROXY_DEPTH = 8
SWAP_UNDERVALUED_ALLOCATION_RATIO = 0.01
SWAP_VALUE_DUST_EUR = 0.01


@dataclass(frozen=True)
class UnresolvedPriceEvent:
    date: str
    coin: str
    price_source: str
    action: str


def get_crypto_price(
    coin: str,
    date: str,
    chain: str,
    use_lp_prices: bool = False,
) -> float | None:
    """Retrieves exchange rate of a specific coin on a date.

    args:
        coin: The coin you want the price for.
        date: On which date you want the price.
        chain: Chain identifier used for LP price lookup.
        use_lp_prices: Whether protocol-derived LP prices should be checked first.

    returns:
        Crypto price on the requested date, or None when no price is resolvable.
    """
    candidates = [coin]
    proxy = price_proxy_symbol(coin)
    if proxy and proxy not in candidates:
        candidates.append(proxy)

    lookup_modes = [use_lp_prices]
    if not use_lp_prices:
        lookup_modes.append(True)

    for lookup_lp_prices in lookup_modes:
        for candidate in candidates:
            price = get_price_eur_on_or_before(
                symbol=candidate,
                as_of_date=date,
                prices_folder=PRICES_FOLDER,
                chain=chain,
                use_lp_prices=lookup_lp_prices,
                fallback_to_oldest=False,
            )
            if price is not None:
                return float(price)

    for lookup_lp_prices in lookup_modes:
        for candidate in candidates:
            oldest_price = get_price_eur_on_or_before(
                symbol=candidate,
                as_of_date=date,
                prices_folder=PRICES_FOLDER,
                chain=chain,
                use_lp_prices=lookup_lp_prices,
                fallback_to_oldest=True,
            )
            if oldest_price is not None:
                if candidate not in STABLE_PRICE_SYMBOLS:
                    print(
                        f"Warning: No price found for {candidate} on/before {date}. "
                        "Using oldest known price."
                    )
                return float(oldest_price)

    print(f"Warning: No data for {coin}. Price is unresolved.")
    return None


def _price_or_zero(
    *,
    coin: str,
    date: str,
    chain: str,
    use_lp_prices: bool,
) -> float:
    price = get_crypto_price(
        coin=coin,
        date=date,
        chain=chain,
        use_lp_prices=use_lp_prices,
    )
    return price if price is not None else 0.0


def _derive_aave_price_source(symbol: str, meta: dict[str, Any] | None) -> str:
    if meta:
        explicit = sanitize_symbol(meta.get("price_source")) or sanitize_symbol(meta.get("family"))
        if explicit:
            return explicit

    for prefix in AAVE_PRICE_SOURCE_PREFIXES:
        if symbol.startswith(prefix):
            underlying = sanitize_symbol(symbol.removeprefix(prefix))
            if underlying:
                return underlying

    return symbol


def _is_aave_debt_symbol(symbol: str) -> bool:
    normalized = sanitize_symbol(symbol).lower()
    return any(normalized.startswith(prefix.lower()) for prefix in AAVE_DEBT_PREFIXES)


def _load_protocol_components(
    *,
    chain: str,
    root: Path | None = None,
) -> dict[str, set[str]]:
    """
    Loads protocol token component symbols from protocol-underlying CSV headers.

    args:
        chain: Chain identifier.
        root: Protocol-underlying CSV root.

    returns:
        Mapping of protocol token symbol to its component symbols.
    """
    root = root or PROTOCOL_UNDERLYING_TOKEN_FOLDER
    if not root.exists():
        return {}

    components: dict[str, set[str]] = {}
    for csv_path in root.rglob(f"{chain}_*.csv"):
        if csv_path.parent.name == "aave":
            continue

        symbol = sanitize_symbol(csv_path.stem[len(chain) + 1 :])
        if not symbol:
            continue

        try:
            columns = pd.read_csv(csv_path, nrows=0).columns
        except (OSError, pd.errors.EmptyDataError):
            continue

        asset_columns = [
            sanitize_symbol(column.replace("asset_", "", 1))
            for column in columns
            if isinstance(column, str) and column.startswith("asset_")
        ]
        asset_symbols = {column for column in asset_columns if column}
        if asset_symbols:
            components[symbol] = asset_symbols

    return components


@dataclass
class CryptoPosition:
    """Tracks the running state and calculations of a single crypto position."""

    coin: str
    chain: str
    valuation_route: ValuationRoute
    quantity: Decimal = Decimal(0)
    principal: float = 0.0
    family_proxy: CryptoPosition | None = None
    price_source: str = ""

    def __post_init__(self):
        if not self.price_source:
            self.price_source = self.coin

    def adjust_principal(self, amount: float):
        if self.family_proxy:
            self.family_proxy.adjust_principal(amount)
        else:
            self.principal += amount

    def buy(self, amount_bought: Decimal, fiat_spent: Decimal, currency: str, date: str):
        self.quantity += amount_bought
        rate = get_forex_rate(currency=currency, date=date)
        self.adjust_principal(float(fiat_spent) * rate)

    def sell(self, amount_sold: Decimal, fiat_received: Decimal, currency: str, date: str):
        self.quantity -= amount_sold
        rate = get_forex_rate(currency=currency, date=date)
        self.adjust_principal(-(float(fiat_received) * rate))

    def receive(self, amount_received: Decimal, date: str):
        self.quantity += amount_received
        price = _price_or_zero(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        self.adjust_principal(float(amount_received) * price)

    def send(self, amount_sent: Decimal, date: str):
        self.quantity -= amount_sent
        price = _price_or_zero(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        self.adjust_principal(-(float(amount_sent) * price))

    def reward(self, amount_received: Decimal, source_asset: CryptoPosition, date: str):
        self.quantity += amount_received
        price = _price_or_zero(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        invested = float(amount_received) * price
        self.adjust_principal(invested)
        source_asset.adjust_principal(-invested)

    def to_snapshot(self, date_value) -> dict:
        return {
            "Date": date_value,
            "Coin": self.coin,
            "Quantity": self.quantity,
            "Principal Invested": round(self.principal, 2),
        }


@dataclass
class TxEntry:
    token: str
    quantity: Decimal
    val: float | None = None


def _incoming_principal_allocations(
    *,
    ins: list[TxEntry],
    total_in_value_eur: float,
    principal_transfer_value_eur: float,
) -> list[float]:
    if not ins:
        return []

    if principal_transfer_value_eur == 0:
        return [0.0 for _ in ins]

    values = [max(float(entry.val or 0.0), 0.0) for entry in ins]
    zero_value_indexes = [
        index for index, value in enumerate(values) if value <= SWAP_VALUE_DUST_EUR
    ]
    transfer_abs = abs(principal_transfer_value_eur)
    if (
        zero_value_indexes
        and transfer_abs > 0
        and total_in_value_eur < transfer_abs * SWAP_UNDERVALUED_ALLOCATION_RATIO
    ):
        share = principal_transfer_value_eur / len(zero_value_indexes)
        return [share if index in zero_value_indexes else 0.0 for index in range(len(ins))]

    if total_in_value_eur > 0:
        return [principal_transfer_value_eur * (value / total_in_value_eur) for value in values]

    equal_share = principal_transfer_value_eur / len(ins)
    return [equal_share for _ in ins]


class TransactionParser:
    def parse_entries(self, *, qty_val: object, token_val: object) -> list[TxEntry]:
        if pd.isna(qty_val) or str(qty_val).strip() == "":
            return []

        qty_str = str(qty_val)
        token_str = str(token_val) if pd.notna(token_val) else ""
        quantities = [Decimal(x.strip()) for x in qty_str.split(",") if x.strip()]
        tokens = []
        for raw_token in token_str.split(","):
            candidate = sanitize_symbol(raw_token.strip())
            if candidate:
                tokens.append(candidate)
        return [TxEntry(token=t, quantity=q) for t, q in zip(tokens, quantities)]

    def parse_reward_sources(self, *, tx_type_lower: str) -> list[str]:
        if "|" not in tx_type_lower:
            return []

        _, raw_sources = tx_type_lower.split("|", 1)
        sources: list[str] = []
        for raw_source in raw_sources.split(","):
            source = sanitize_symbol(raw_source.strip())
            if source:
                sources.append(source)
        return sources


class PortfolioLedger:
    def __init__(self, chain: str, token_metadata: dict[str, dict[str, Any]] | None = None):
        self.chain = chain
        self.token_metadata = token_metadata or load_token_metadata(
            chain=chain,
            tokens_folder=TOKENS_FOLDER,
        )
        self.symbol_to_meta: dict[str, dict[str, Any]] = {}
        self.symbol_family: dict[str, str] = {}
        self.symbol_protocol = build_symbol_protocol_map(token_metadata=self.token_metadata)
        self.protocol_components = _load_protocol_components(chain=chain)
        self.use_dual_principal = chain == "arbitrum"
        self.principal_ledger = EconomicPrincipalLedger(
            resolver=PrincipalResolver(
                chain=chain,
                token_metadata=self.token_metadata,
                protocol_root=PROTOCOL_UNDERLYING_TOKEN_FOLDER,
                prices_folder=PRICES_FOLDER,
            )
        )

        for meta in self.token_metadata.values():
            symbol = sanitize_symbol(meta.get("symbol"))
            if not symbol:
                continue
            if symbol not in self.symbol_to_meta:
                self.symbol_to_meta[symbol] = meta

            family = sanitize_symbol(meta.get("family")) or symbol
            self.symbol_family[symbol] = family

        self.assets: dict[str, CryptoPosition] = {}
        self.history: list[dict] = []
        self.daily_coin_cache: dict[str, int] = {}
        self.current_date: date | None = None
        self.unresolved_prices: list[UnresolvedPriceEvent] = []

    def _principal_terminal_families(
        self,
        symbol: str,
        *,
        seen: set[str] | None = None,
        depth: int = 0,
    ) -> set[str] | None:
        normalized = sanitize_symbol(symbol)
        if not normalized or depth > MAX_PRINCIPAL_PROXY_DEPTH:
            return None

        visited = set(seen or set())
        if normalized in visited:
            return None
        visited.add(normalized)

        price_proxy = price_proxy_symbol(normalized)
        if price_proxy and price_proxy != normalized:
            return self._principal_terminal_families(
                price_proxy,
                seen=visited,
                depth=depth + 1,
            )

        components = self.protocol_components.get(normalized)
        if components:
            terminal_families: set[str] = set()
            for component in components:
                component_families = self._principal_terminal_families(
                    component,
                    seen=visited,
                    depth=depth + 1,
                )
                if component_families is None:
                    return None
                terminal_families.update(component_families)
            return terminal_families

        route = classify_valuation_route(
            symbol=normalized,
            symbol_protocol=self.symbol_protocol,
        )
        if route == ValuationRoute.PROTOCOL_DERIVED:
            return None

        terminal = canonicalize_symbol(
            normalized,
            symbol_family=self.symbol_family,
        )
        return {terminal or normalized}

    def _principal_proxy_symbol(
        self,
        *,
        asset_key: str,
        route: ValuationRoute,
        meta: dict[str, Any] | None,
    ) -> str:
        if route == ValuationRoute.AAVE:
            if _is_aave_debt_symbol(asset_key):
                return ""
            base_symbol = _derive_aave_price_source(symbol=asset_key, meta=meta)
        elif route == ValuationRoute.PROTOCOL_DERIVED:
            if asset_key not in self.protocol_components:
                return ""
            base_symbol = asset_key
        else:
            base_symbol = ""
            if meta:
                base_symbol = canonicalize_symbol(
                    meta.get("family"),
                    symbol_family=self.symbol_family,
                ) or sanitize_symbol(meta.get("price_source"))
            if not base_symbol:
                base_symbol = asset_key

        terminal_families = self._principal_terminal_families(base_symbol)
        if not terminal_families or len(terminal_families) != 1:
            return ""

        proxy_symbol = next(iter(terminal_families))
        return proxy_symbol if proxy_symbol and proxy_symbol != asset_key else ""

    def fetch_asset(self, coin: str) -> CryptoPosition:
        normalized_coin = sanitize_symbol(coin)
        asset_key = normalized_coin or str(coin).strip()
        if asset_key not in self.assets:
            meta = self.symbol_to_meta.get(asset_key)
            route = classify_valuation_route(
                symbol=asset_key,
                symbol_protocol=self.symbol_protocol,
            )

            price_source = ""
            if route == ValuationRoute.DIRECT and meta:
                price_source = sanitize_symbol(meta.get("price_source"))
            elif route == ValuationRoute.AAVE:
                price_source = _derive_aave_price_source(symbol=asset_key, meta=meta)
            if not price_source:
                price_source = asset_key

            self.assets[asset_key] = CryptoPosition(
                coin=asset_key,
                chain=self.chain,
                valuation_route=route,
                price_source=price_source,
            )

        return self.assets[asset_key]

    def adjust_principal(
        self,
        *,
        asset: CryptoPosition,
        amount_eur: float,
        date_value: object,
        action: str,
        tx_hash: str = "",
    ) -> None:
        if not self.use_dual_principal:
            asset.adjust_principal(amount_eur)
            return

        self.principal_ledger.adjust(
            symbol=asset.coin,
            amount_eur=amount_eur,
            date_value=date_value,
            action=action,
            tx_hash=tx_hash,
        )

    def collect_snapshots(self, *, asset: CryptoPosition, date_value: str) -> list[dict]:
        snapshots = [asset.to_snapshot(date_value)]
        if asset.family_proxy:
            snapshots.append(asset.family_proxy.to_snapshot(date_value))
        return snapshots

    def update_snapshots(self, *, touched_coins: set[str], date_value: str) -> None:
        new_snapshots = []
        for coin in touched_coins:
            asset = self.assets[coin]
            new_snapshots.extend(self.collect_snapshots(asset=asset, date_value=date_value))

        unique_snapshots = {snapshot["Coin"]: snapshot for snapshot in new_snapshots}
        for snapshot in unique_snapshots.values():
            snap_date = snapshot["Date"].date()
            coin = snapshot["Coin"]

            if self.current_date != snap_date:
                self.daily_coin_cache = {}
                self.current_date = snap_date

            if coin in self.daily_coin_cache:
                idx = self.daily_coin_cache[coin]
                self.history[idx] = snapshot
            else:
                self.history.append(snapshot)
                self.daily_coin_cache[coin] = len(self.history) - 1

    def record_unresolved_price(
        self,
        *,
        asset: CryptoPosition,
        date_value: str,
        action: str,
    ) -> None:
        self.unresolved_prices.append(
            UnresolvedPriceEvent(
                date=str(date_value),
                coin=asset.coin,
                price_source=asset.price_source,
                action=action,
            )
        )


class TransactionApplier:
    def __init__(self, ledger: PortfolioLedger, parser: TransactionParser | None = None):
        self.ledger = ledger
        self.parser = parser or TransactionParser()

    def _price_for_asset(self, *, asset: CryptoPosition, date_value: str, action: str) -> float:
        price = get_crypto_price(
            coin=asset.price_source,
            date=date_value,
            chain=self.ledger.chain,
            use_lp_prices=asset.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        if price is None:
            self.ledger.record_unresolved_price(
                asset=asset,
                date_value=date_value,
                action=action,
            )
            return 0.0
        return price

    def receive(
        self,
        *,
        asset: CryptoPosition,
        amount_received: Decimal,
        date_value: str,
        tx_hash: str = "",
    ):
        asset.quantity += amount_received
        price = self._price_for_asset(asset=asset, date_value=date_value, action="receive")
        self.ledger.adjust_principal(
            asset=asset,
            amount_eur=float(amount_received) * price,
            date_value=date_value,
            action="receive",
            tx_hash=tx_hash,
        )

    def send(
        self,
        *,
        asset: CryptoPosition,
        amount_sent: Decimal,
        date_value: str,
        tx_hash: str = "",
    ):
        asset.quantity -= amount_sent
        price = self._price_for_asset(asset=asset, date_value=date_value, action="send")
        self.ledger.adjust_principal(
            asset=asset,
            amount_eur=-(float(amount_sent) * price),
            date_value=date_value,
            action="send",
            tx_hash=tx_hash,
        )

    def _process_swap(
        self,
        *,
        ins: list[TxEntry],
        outs: list[TxEntry],
        date_value: str,
        touched_coins: set[str],
        tx_hash: str = "",
    ) -> None:
        total_in_value_eur = 0.0
        for entry in ins:
            asset = self.ledger.fetch_asset(entry.token)
            price = self._price_for_asset(asset=asset, date_value=date_value, action="swap_in")
            val = price * float(entry.quantity)
            total_in_value_eur += val
            entry.val = val

        total_out_value_eur = 0.0
        for entry in outs:
            asset = self.ledger.fetch_asset(entry.token)
            price = self._price_for_asset(asset=asset, date_value=date_value, action="swap_out")
            val = price * float(entry.quantity)
            total_out_value_eur += val
            entry.val = val

        principal_transfer_value_eur = (
            total_out_value_eur if total_out_value_eur > 0 else total_in_value_eur
        )
        incoming_principal = _incoming_principal_allocations(
            ins=ins,
            total_in_value_eur=total_in_value_eur,
            principal_transfer_value_eur=principal_transfer_value_eur,
        )

        for entry, principal_addition in zip(ins, incoming_principal, strict=True):
            asset_in = self.ledger.fetch_asset(entry.token)
            asset_in.quantity += entry.quantity
            self.ledger.adjust_principal(
                asset=asset_in,
                amount_eur=principal_addition,
                date_value=date_value,
                action="swap_in",
                tx_hash=tx_hash,
            )
            touched_coins.add(asset_in.coin)

        if not outs:
            out_shares = []
        elif total_out_value_eur == 0:
            out_shares = [1.0 / len(outs) for _ in outs]
        else:
            out_shares = [max(float(entry.val or 0.0), 0.0) / total_out_value_eur for entry in outs]

        for entry, share_of_out in zip(outs, out_shares, strict=True):
            asset_out = self.ledger.fetch_asset(entry.token)
            principal_reduction = principal_transfer_value_eur * share_of_out
            asset_out.quantity -= entry.quantity
            self.ledger.adjust_principal(
                asset=asset_out,
                amount_eur=-principal_reduction,
                date_value=date_value,
                action="swap_out",
                tx_hash=tx_hash,
            )
            touched_coins.add(asset_out.coin)

    def _process_reward(
        self,
        *,
        rewards: list[TxEntry],
        allocate_reward_to: list[str],
        date_value: str,
        touched_coins: set[str],
        tx_hash: str = "",
    ) -> None:
        if not rewards:
            return

        for entry_in in rewards:
            if allocate_reward_to:
                allocations = [(source_coin.upper(), 1.0) for source_coin in allocate_reward_to]
            else:
                allocations = [(None, 1.0)]

            self.apply_reward_with_allocations(
                reward_token=entry_in.token,
                reward_quantity=entry_in.quantity,
                date_value=date_value,
                allocations=allocations,
                touched_coins=touched_coins,
                tx_hash=tx_hash,
            )

    def apply_reward_with_allocations(
        self,
        *,
        reward_token: str,
        reward_quantity: Decimal,
        date_value: str,
        allocations: list[tuple[str | None, float]] | None,
        touched_coins: set[str],
        tx_hash: str = "",
    ) -> None:
        """
        Applies a reward quantity and reallocates principal by weighted source buckets.

        args:
            reward_token: Token received as reward.
            reward_quantity: Reward quantity.
            date_value: Reward datetime.
            allocations: Weighted principal source buckets where None means free allocation.
            touched_coins: Coin set touched by this operation.
        """
        if reward_quantity <= 0:
            return

        asset_in = self.ledger.fetch_asset(reward_token)
        price = self._price_for_asset(asset=asset_in, date_value=date_value, action="reward")
        invested = float(reward_quantity) * price

        asset_in.quantity += reward_quantity
        touched_coins.add(asset_in.coin)

        normalized_allocations: list[tuple[str | None, float]] = []
        for source_coin, weight in allocations or []:
            if weight <= 0:
                continue
            normalized_source = sanitize_symbol(source_coin) if source_coin else None
            normalized_allocations.append((normalized_source, weight))

        if not normalized_allocations:
            return

        total_weight = sum(weight for _, weight in normalized_allocations)
        if total_weight <= 0:
            return

        self.ledger.adjust_principal(
            asset=asset_in,
            amount_eur=invested,
            date_value=date_value,
            action="reward",
            tx_hash=tx_hash,
        )

        remaining_value = invested
        for idx, (source_coin, weight) in enumerate(normalized_allocations):
            if idx == len(normalized_allocations) - 1:
                share = remaining_value
            else:
                share = invested * (weight / total_weight)
                remaining_value -= share

            if source_coin is None:
                self.ledger.adjust_principal(
                    asset=asset_in,
                    amount_eur=-share,
                    date_value=date_value,
                    action="reward_unallocated",
                    tx_hash=tx_hash,
                )
                continue

            source_asset = self.ledger.fetch_asset(source_coin)
            self.ledger.adjust_principal(
                asset=source_asset,
                amount_eur=-share,
                date_value=date_value,
                action="reward_source",
                tx_hash=tx_hash,
            )
            touched_coins.add(source_asset.coin)

    def handle_fees(
        self,
        *,
        row: pd.Series,
        date_value: str,
        ins: list[TxEntry],
        outs: list[TxEntry],
        tx_type_lower: str,
        touched_coins: set[str],
        tx_hash: str = "",
    ) -> None:
        fee_str = row.get("Fee")
        fee_token = row.get("Fee Token")

        if pd.isna(fee_str) or pd.isna(fee_token):
            return

        fee_qty = Decimal(str(fee_str))
        if fee_qty <= 0:
            return

        fee_asset = self.ledger.fetch_asset(str(fee_token))
        fee_price = self._price_for_asset(asset=fee_asset, date_value=date_value, action="fee")
        fee_val_eur = float(fee_qty) * fee_price

        fee_asset.quantity -= fee_qty
        touched_coins.add(fee_asset.coin)

        target_entries = []
        if tx_type_lower in ["swap", "buy", "receive"] and ins:
            target_entries = ins
        elif tx_type_lower in ["sell", "send"] and outs:
            target_entries = outs

        if target_entries:
            self.ledger.adjust_principal(
                asset=fee_asset,
                amount_eur=-fee_val_eur,
                date_value=date_value,
                action="fee",
                tx_hash=tx_hash,
            )
            share_val_eur = fee_val_eur / len(target_entries)
            for entry in target_entries:
                target_asset = self.ledger.fetch_asset(entry.token)
                self.ledger.adjust_principal(
                    asset=target_asset,
                    amount_eur=share_val_eur,
                    date_value=date_value,
                    action="fee_allocation",
                    tx_hash=tx_hash,
                )
                touched_coins.add(target_asset.coin)

    def process_transaction(self, row: pd.Series):
        tx_type: str = row["Type"]
        tx_type_lower = tx_type.lower()
        date_value = row["Date"]
        tx_hash = "" if pd.isna(row.get("TX Hash", "")) else str(row.get("TX Hash", ""))

        ins = self.parser.parse_entries(qty_val=row.get("Qty in"), token_val=row.get("Token in"))
        outs = self.parser.parse_entries(
            qty_val=row.get("Qty out"),
            token_val=row.get("Token out"),
        )
        touched_coins = set()

        if tx_type_lower == "buy":
            entry_in = ins[0]
            entry_out = outs[0]

            asset_in = self.ledger.fetch_asset(entry_in.token)
            asset_in.quantity += entry_in.quantity
            rate = get_forex_rate(currency=entry_out.token, date=date_value)
            self.ledger.adjust_principal(
                asset=asset_in,
                amount_eur=float(entry_out.quantity) * rate,
                date_value=date_value,
                action="buy",
                tx_hash=tx_hash,
            )
            touched_coins.add(asset_in.coin)

        elif tx_type_lower == "receive":
            for entry in ins:
                asset_in = self.ledger.fetch_asset(entry.token)
                self.receive(
                    asset=asset_in,
                    amount_received=entry.quantity,
                    date_value=date_value,
                    tx_hash=tx_hash,
                )
                touched_coins.add(asset_in.coin)

        elif tx_type_lower == "sell":
            entry_in = ins[0]
            entry_out = outs[0]

            asset_out = self.ledger.fetch_asset(entry_out.token)
            asset_out.quantity -= entry_out.quantity
            rate = get_forex_rate(currency=entry_in.token, date=date_value)
            self.ledger.adjust_principal(
                asset=asset_out,
                amount_eur=-(float(entry_in.quantity) * rate),
                date_value=date_value,
                action="sell",
                tx_hash=tx_hash,
            )
            touched_coins.add(asset_out.coin)

        elif tx_type_lower == "send":
            for entry in outs:
                asset_out = self.ledger.fetch_asset(entry.token)
                self.send(
                    asset=asset_out,
                    amount_sent=entry.quantity,
                    date_value=date_value,
                    tx_hash=tx_hash,
                )
                touched_coins.add(asset_out.coin)

        elif tx_type_lower == "swap":
            self._process_swap(
                ins=ins,
                outs=outs,
                date_value=date_value,
                touched_coins=touched_coins,
                tx_hash=tx_hash,
            )

        elif tx_type_lower.startswith("reward"):
            allocate_reward_to = self.parser.parse_reward_sources(tx_type_lower=tx_type_lower)
            self._process_reward(
                rewards=ins,
                allocate_reward_to=allocate_reward_to,
                date_value=date_value,
                touched_coins=touched_coins,
                tx_hash=tx_hash,
            )

        elif tx_type_lower.startswith("approve"):
            return

        elif tx_type_lower == "interaction":
            pass
        else:
            error_msg = f"{tx_type}: {ins} -> {outs} on {date_value} not found."
            print(error_msg)
            return

        self.handle_fees(
            row=row,
            date_value=date_value,
            ins=ins,
            outs=outs,
            tx_type_lower=tx_type_lower,
            touched_coins=touched_coins,
            tx_hash=tx_hash,
        )

        self.ledger.update_snapshots(touched_coins=touched_coins, date_value=date_value)


class SnapshotWriter:
    def save(self, *, history: list[dict], output_path: Path):
        df = pd.DataFrame(history)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].map(format_daily_datetime)
        df.to_csv(output_path, index=False)
        print(f"Portfolio snapshots successfully saved to {output_path}")

    def save_principal(self, *, tracker: CryptoTracker, events_path: Path, daily_path: Path):
        events_path.parent.mkdir(parents=True, exist_ok=True)
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        tracker.ledger.principal_ledger.events_frame().to_csv(events_path, index=False)
        tracker.ledger.principal_ledger.daily_frame().to_csv(daily_path, index=False)


class CryptoTracker:
    def __init__(self, chain: str, token_metadata: dict[str, dict[str, Any]] | None = None):
        self.ledger = PortfolioLedger(chain=chain, token_metadata=token_metadata)
        self.parser = TransactionParser()
        self.applier = TransactionApplier(ledger=self.ledger, parser=self.parser)
        self.writer = SnapshotWriter()

    @property
    def chain(self) -> str:
        return self.ledger.chain

    @property
    def token_metadata(self) -> dict[str, dict[str, Any]]:
        return self.ledger.token_metadata

    @property
    def symbol_to_meta(self) -> dict[str, dict[str, Any]]:
        return self.ledger.symbol_to_meta

    @property
    def symbol_family(self) -> dict[str, str]:
        return self.ledger.symbol_family

    @property
    def symbol_protocol(self) -> dict[str, str]:
        return self.ledger.symbol_protocol

    @property
    def assets(self) -> dict[str, CryptoPosition]:
        return self.ledger.assets

    @assets.setter
    def assets(self, value: dict[str, CryptoPosition]) -> None:
        self.ledger.assets = value

    @property
    def history(self) -> list[dict]:
        return self.ledger.history

    @history.setter
    def history(self, value: list[dict]) -> None:
        self.ledger.history = value

    @property
    def daily_coin_cache(self) -> dict[str, int]:
        return self.ledger.daily_coin_cache

    @daily_coin_cache.setter
    def daily_coin_cache(self, value: dict[str, int]) -> None:
        self.ledger.daily_coin_cache = value

    @property
    def current_date(self) -> date | None:
        return self.ledger.current_date

    @current_date.setter
    def current_date(self, value: date | None) -> None:
        self.ledger.current_date = value

    @property
    def unresolved_prices(self) -> list[UnresolvedPriceEvent]:
        return self.ledger.unresolved_prices

    def fetch_asset(self, coin: str) -> CryptoPosition:
        return self.ledger.fetch_asset(coin=coin)

    def _collect_snapshots(self, asset: CryptoPosition, date: str) -> list[dict]:
        return self.ledger.collect_snapshots(asset=asset, date_value=date)

    def _update_snapshots(self, touched_coins: set[str], date: str) -> None:
        self.ledger.update_snapshots(touched_coins=touched_coins, date_value=date)

    def _process_swap(
        self,
        ins: list[TxEntry],
        outs: list[TxEntry],
        date: str,
        touched_coins: set[str],
    ) -> None:
        self.applier._process_swap(
            ins=ins,
            outs=outs,
            date_value=date,
            touched_coins=touched_coins,
        )

    def _process_reward(
        self,
        rewards: list[TxEntry],
        allocate_reward_to: list[str],
        date: str,
        touched_coins: set[str],
    ) -> None:
        self.applier._process_reward(
            rewards=rewards,
            allocate_reward_to=allocate_reward_to,
            date_value=date,
            touched_coins=touched_coins,
        )

    def apply_reward_with_allocations(
        self,
        *,
        reward_token: str,
        reward_quantity: Decimal,
        date: str,
        allocations: list[tuple[str | None, float]] | None,
        touched_coins: set[str],
    ) -> None:
        self.applier.apply_reward_with_allocations(
            reward_token=reward_token,
            reward_quantity=reward_quantity,
            date_value=date,
            allocations=allocations,
            touched_coins=touched_coins,
        )

    def _parse_reward_sources(self, tx_type_lower: str) -> list[str]:
        return self.parser.parse_reward_sources(tx_type_lower=tx_type_lower)

    def handle_fees(
        self,
        row: pd.Series,
        date: str,
        ins: list[TxEntry],
        outs: list[TxEntry],
        tx_type_lower: str,
        touched_coins: set[str],
    ) -> None:
        self.applier.handle_fees(
            row=row,
            date_value=date,
            ins=ins,
            outs=outs,
            tx_type_lower=tx_type_lower,
            touched_coins=touched_coins,
        )

    def process_transaction(self, row: pd.Series):
        self.applier.process_transaction(row=row)

    def save_to_csv(self, output_path: Path):
        self.writer.save(history=self.history, output_path=output_path)

    def save_principal_to_csv(self, *, events_path: Path, daily_path: Path):
        self.writer.save_principal(tracker=self, events_path=events_path, daily_path=daily_path)


def generate_raw_snapshots(
    input_csv: Path,
    output_csv: Path,
    chain: str,
    principal_events_csv: Path | None = None,
    principal_daily_csv: Path | None = None,
) -> None:
    df = pd.read_csv(input_csv, dtype=str)
    parsed_dates = parse_transaction_datetime_series(df["Date"])
    invalid_date_count = int(parsed_dates.isna().sum())
    total_rows = len(df)
    if total_rows > 0 and (invalid_date_count / total_rows) > MAX_INVALID_DATE_RATIO:
        raise ValueError(
            f"Aborting snapshot generation: invalid dates={invalid_date_count}/{total_rows} "
            f"({invalid_date_count / total_rows:.1%})."
        )
    if invalid_date_count:
        print(f"[raw_snapshots] Dropping {invalid_date_count} rows with invalid Date values.")

    df["Date"] = parsed_dates
    df = df.dropna(subset=["Date"])
    df = df.sort_values(by=["Date"], ascending=True)

    tracker = CryptoTracker(chain=chain)
    for _, row in df.iterrows():
        tracker.process_transaction(row)

    tracker.save_to_csv(output_csv)
    if principal_events_csv is not None and principal_daily_csv is not None:
        tracker.save_principal_to_csv(
            events_path=principal_events_csv,
            daily_path=principal_daily_csv,
        )
    if tracker.unresolved_prices:
        print(f"[raw_snapshots] Unresolved price events: {len(tracker.unresolved_prices)}")
