from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from portfolio_core import active_context, atomic_write_csv

from portfolio_crypto_data.datetime_utils import (
    format_daily_datetime,
    parse_transaction_datetime_series,
)
from portfolio_crypto_data.principal_ledger import EconomicPrincipalLedger, PrincipalResolver
from portfolio_crypto_data.shared.aave_symbols import aave_base_symbol
from portfolio_crypto_data.shared.prices import (
    STABLE_PRICE_SYMBOLS,
    get_price_eur_on_or_before,
)
from portfolio_crypto_data.shared.token_metadata import load_token_metadata
from portfolio_crypto_data.shared.valuation_routes import (
    ValuationRoute,
    build_symbol_protocol_map,
    classify_valuation_route,
)
from portfolio_crypto_data.symbols import price_proxy_symbol, sanitize_symbol

MAX_INVALID_DATE_RATIO = 0.1
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
    context = active_context()
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
                prices_folder=context.paths.prices,
                chain=chain,
                use_lp_prices=lookup_lp_prices,
                fallback_to_oldest=False,
                currency_metadata=context.currency_metadata(),
            )
            if price is not None:
                return float(price)

    for lookup_lp_prices in lookup_modes:
        for candidate in candidates:
            oldest_price = get_price_eur_on_or_before(
                symbol=candidate,
                as_of_date=date,
                prices_folder=context.paths.prices,
                chain=chain,
                use_lp_prices=lookup_lp_prices,
                fallback_to_oldest=True,
                currency_metadata=context.currency_metadata(),
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


def _derive_aave_price_source(symbol: str, meta: dict[str, Any] | None) -> str:
    return aave_base_symbol(symbol, meta)


@dataclass
class CryptoPosition:
    """Tracks the running state and calculations of a single crypto position."""

    coin: str
    chain: str
    valuation_route: ValuationRoute
    quantity: Decimal = Decimal(0)
    principal: float = 0.0
    price_source: str = ""

    def __post_init__(self):
        if not self.price_source:
            self.price_source = self.coin

    def adjust_principal(self, amount: float):
        self.principal += amount

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
        if len(tokens) != len(quantities):
            raise ValueError(f"Mismatched token and quantity fields: {token_val!r}, {qty_val!r}")
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
    def __init__(
        self,
        chain: str,
        token_metadata: dict[str, dict[str, Any]] | None = None,
        *,
        track_economic_principal: bool = False,
    ):
        self.chain = chain
        paths = active_context().paths
        self.token_metadata = (
            load_token_metadata(chain=chain, tokens_folder=paths.tokens)
            if token_metadata is None
            else token_metadata
        )
        self.symbol_to_meta: dict[str, dict[str, Any]] = {}
        self.symbol_protocol = build_symbol_protocol_map(token_metadata=self.token_metadata)
        self.track_economic_principal = track_economic_principal
        self.principal_ledger = EconomicPrincipalLedger(
            resolver=PrincipalResolver(
                chain=chain,
                token_metadata=self.token_metadata,
                protocol_root=paths.protocol_underlying_tokens,
                prices_folder=paths.prices,
            )
        )

        for meta in self.token_metadata.values():
            symbol = sanitize_symbol(meta.get("symbol"))
            if not symbol:
                continue
            if symbol not in self.symbol_to_meta:
                self.symbol_to_meta[symbol] = meta

        self.assets: dict[str, CryptoPosition] = {}
        self.history: list[dict] = []
        self.daily_coin_cache: dict[str, int] = {}
        self.current_date: date | None = None
        self.unresolved_prices: list[UnresolvedPriceEvent] = []

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
        if not self.track_economic_principal:
            asset.adjust_principal(amount_eur)
            return

        self.principal_ledger.adjust(
            symbol=asset.coin,
            amount_eur=amount_eur,
            date_value=date_value,
            action=action,
            tx_hash=tx_hash,
        )

    def update_snapshots(self, *, touched_coins: set[str], date_value: str) -> None:
        for coin in touched_coins:
            snapshot = self.assets[coin].to_snapshot(date_value)
            snap_date = snapshot["Date"].date()

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

    def apply_swap(
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

    def apply_rewards(
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
        if tx_type_lower in ["swap", "receive"] and ins:
            target_entries = ins
        elif tx_type_lower == "send" and outs:
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

        if tx_type_lower == "receive":
            for entry in ins:
                asset_in = self.ledger.fetch_asset(entry.token)
                self.receive(
                    asset=asset_in,
                    amount_received=entry.quantity,
                    date_value=date_value,
                    tx_hash=tx_hash,
                )
                touched_coins.add(asset_in.coin)

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
            self.apply_swap(
                ins=ins,
                outs=outs,
                date_value=date_value,
                touched_coins=touched_coins,
                tx_hash=tx_hash,
            )

        elif tx_type_lower.startswith("reward"):
            allocate_reward_to = self.parser.parse_reward_sources(tx_type_lower=tx_type_lower)
            self.apply_rewards(
                rewards=ins,
                allocate_reward_to=allocate_reward_to,
                date_value=date_value,
                touched_coins=touched_coins,
                tx_hash=tx_hash,
            )

        elif tx_type_lower.startswith("approve"):
            # Approval transactions have no portfolio asset movement, but they still
            # consume the native token as gas. Continue into fee handling so the
            # wallet balance and daily snapshot include that cost.
            pass

        elif tx_type_lower == "interaction":
            pass
        else:
            raise ValueError(f"Unsupported transaction type: {tx_type}")

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


def _save_snapshots(*, history: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(history)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame["Date"] = frame["Date"].map(format_daily_datetime)
    atomic_write_csv(frame=frame, path=output_path)


def _save_principal(*, ledger: PortfolioLedger, events_path: Path, daily_path: Path) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame=ledger.principal_ledger.events_frame(), path=events_path)
    atomic_write_csv(frame=ledger.principal_ledger.daily_frame(), path=daily_path)


def generate_raw_snapshots(
    input_csv: Path,
    output_csv: Path,
    chain: str,
    principal_events_csv: Path | None = None,
    principal_daily_csv: Path | None = None,
    token_metadata: dict[str, dict[str, Any]] | None = None,
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

    ledger = PortfolioLedger(
        chain=chain,
        token_metadata=token_metadata,
        track_economic_principal=True,
    )
    applier = TransactionApplier(ledger=ledger)
    for _, row in df.iterrows():
        applier.process_transaction(row)

    _save_snapshots(history=ledger.history, output_path=output_csv)
    if principal_events_csv is not None and principal_daily_csv is not None:
        _save_principal(
            ledger=ledger,
            events_path=principal_events_csv,
            daily_path=principal_daily_csv,
        )
    if ledger.unresolved_prices:
        print(f"[raw_snapshots] Unresolved price events: {len(ledger.unresolved_prices)}")
