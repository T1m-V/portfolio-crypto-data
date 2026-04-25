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
from blockchain_reader.symbols import canonicalize_symbol, sanitize_symbol
from file_paths import (
    BLOCKCHAIN_SNAPSHOT_FOLDER,
    BLOCKCHAIN_TRANSACTIONS_FOLDER,
    PRICES_FOLDER,
    TOKENS_FOLDER,
)
from historical_transactions.portfolio_snapshots import get_forex_rate

MAX_INVALID_DATE_RATIO = 0.1


def get_crypto_price(
    coin: str,
    date: str,
    chain: str,
    use_lp_prices: bool = False,
) -> float:
    """Retrieves exchange rate of a specific coin on a date.

    Args:
        coin: The coin you want the price for.
        date: On which date you want the price.

    Returns:
        Crypto price on the requested date.
    """
    price = get_price_eur_on_or_before(
        symbol=coin,
        as_of_date=date,
        prices_folder=PRICES_FOLDER,
        chain=chain,
        use_lp_prices=use_lp_prices,
        fallback_to_oldest=False,
    )
    if price is not None:
        return float(price)

    oldest_price = get_price_eur_on_or_before(
        symbol=coin,
        as_of_date=date,
        prices_folder=PRICES_FOLDER,
        chain=chain,
        use_lp_prices=use_lp_prices,
        fallback_to_oldest=True,
    )
    if oldest_price is not None:
        if coin not in STABLE_PRICE_SYMBOLS:
            print(f"Warning: No price found for {coin} on/before {date}. Using oldest known price.")
        return float(oldest_price)

    print(f"Warning: No data for {coin}. Assuming value is 0.")
    return 0.0


@dataclass
class CryptoPosition:
    """Tracks the running state and calculations of a single crypto position."""

    coin: str
    chain: str
    valuation_route: ValuationRoute
    quantity: Decimal = Decimal(0)
    principal: float = 0.0
    family_proxy: "CryptoPosition | None" = None
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
        price = get_crypto_price(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        self.adjust_principal(float(amount_received) * price)

    def send(self, amount_sent: Decimal, date: str):
        self.quantity -= amount_sent
        price = get_crypto_price(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        self.adjust_principal(-(float(amount_sent) * price))

    def reward(self, amount_received: Decimal, source_asset: "CryptoPosition", date: str):
        self.quantity += amount_received
        price = get_crypto_price(
            coin=self.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=self.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        invested = float(amount_received) * price
        self.adjust_principal(invested)
        source_asset.adjust_principal(-invested)

    def to_snapshot(self, date) -> dict:
        return {
            "Date": date,
            "Coin": self.coin,
            "Quantity": self.quantity,
            "Principal Invested": round(self.principal, 2),
        }


@dataclass
class TxEntry:
    token: str
    quantity: Decimal
    val: float | None = None


class CryptoTracker:
    def __init__(self, chain: str, token_metadata: dict[str, dict[str, Any]] | None = None):
        self.chain = chain
        self.token_metadata = token_metadata or load_token_metadata(
            chain=chain,
            tokens_folder=TOKENS_FOLDER,
        )
        self.symbol_to_meta: dict[str, dict[str, Any]] = {}
        self.symbol_family: dict[str, str] = {}
        self.symbol_protocol = build_symbol_protocol_map(token_metadata=self.token_metadata)

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
            if not price_source:
                price_source = asset_key

            family_coin = asset_key
            if route == ValuationRoute.DIRECT:
                family_coin = canonicalize_symbol(
                    meta.get("family") if meta else asset_key,
                    symbol_family=self.symbol_family,
                )
                if not family_coin:
                    family_coin = asset_key

            self.assets[asset_key] = CryptoPosition(
                coin=asset_key,
                chain=self.chain,
                valuation_route=route,
                price_source=price_source,
            )
            if route == ValuationRoute.DIRECT and family_coin != asset_key:
                self.assets[asset_key].family_proxy = self.fetch_asset(family_coin)

        return self.assets[asset_key]

    def _collect_snapshots(self, asset: CryptoPosition, date: str) -> list[dict]:
        snaps = [asset.to_snapshot(date)]
        if asset.family_proxy:
            snaps.append(asset.family_proxy.to_snapshot(date))
        return snaps

    def _process_swap(
        self, ins: list[TxEntry], outs: list[TxEntry], date: str, touched_coins: set[str]
    ) -> None:
        # 1. Calculate Value of all Ins
        total_in_value_eur = 0.0

        for entry in ins:
            asset = self.fetch_asset(entry.token)
            price = get_crypto_price(
                coin=asset.price_source,
                date=date,
                chain=self.chain,
                use_lp_prices=asset.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
            )
            val = price * float(entry.quantity)
            total_in_value_eur += val
            entry.val = val

        # 2. Calculate Value of all Outs (for weighting)
        total_out_value_eur = 0.0
        for entry in outs:
            asset = self.fetch_asset(entry.token)
            price = get_crypto_price(
                coin=asset.price_source,
                date=date,
                chain=self.chain,
                use_lp_prices=asset.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
            )
            val = price * float(entry.quantity)
            total_out_value_eur += val
            entry.val = val

        if total_out_value_eur == 0:
            total_out_value_eur = 1
            equal_share = 1.0 / len(outs)
            for entry in outs:
                entry.val = equal_share

        # 3. Process Ins (Increase Quantity, Increase Principal)
        for entry in ins:
            asset_in = self.fetch_asset(entry.token)
            asset_in.quantity += entry.quantity
            if entry.val is not None:
                asset_in.adjust_principal(entry.val)
            touched_coins.add(asset_in.coin)

        # 4. Process Outs (Decrease Quantity, Decrease Principal)
        for entry in outs:
            asset_out = self.fetch_asset(entry.token)
            val_out = entry.val if entry.val is not None else 0.0
            share_of_out = val_out / total_out_value_eur
            principal_reduction = total_in_value_eur * share_of_out
            asset_out.quantity -= entry.quantity
            asset_out.adjust_principal(-principal_reduction)
            touched_coins.add(asset_out.coin)

    def _process_reward(
        self,
        rewards: list[TxEntry],
        allocate_reward_to: list[str],
        date: str,
        touched_coins: set[str],
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
                date=date,
                allocations=allocations,
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
        """
        Applies a reward quantity and reallocates principal by weighted source buckets.

        args:
            reward_token: Token received as reward.
            reward_quantity: Reward quantity.
            date: Reward datetime.
            allocations: Weighted principal source buckets where None means free allocation.
            touched_coins: Coin set touched by this operation.
        """
        if reward_quantity <= 0:
            return

        asset_in = self.fetch_asset(reward_token)
        price = get_crypto_price(
            coin=asset_in.price_source,
            date=date,
            chain=self.chain,
            use_lp_prices=asset_in.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
        )
        invested = float(reward_quantity) * price

        asset_in.quantity += reward_quantity
        asset_in.adjust_principal(invested)
        touched_coins.add(asset_in.coin)

        normalized_allocations: list[tuple[str | None, float]] = []
        for source_coin, weight in allocations or []:
            if weight <= 0:
                continue
            normalized_source = sanitize_symbol(source_coin) if source_coin else None
            normalized_allocations.append((normalized_source, weight))

        if not normalized_allocations:
            asset_in.adjust_principal(-invested)
            return

        total_weight = sum(weight for _, weight in normalized_allocations)
        if total_weight <= 0:
            asset_in.adjust_principal(-invested)
            return

        remaining_value = invested
        for idx, (source_coin, weight) in enumerate(normalized_allocations):
            if idx == len(normalized_allocations) - 1:
                share = remaining_value
            else:
                share = invested * (weight / total_weight)
                remaining_value -= share

            if source_coin is None:
                asset_in.adjust_principal(-share)
                continue

            source_asset = self.fetch_asset(source_coin)
            source_asset.adjust_principal(-share)
            touched_coins.add(source_asset.coin)

    def _parse_reward_sources(self, tx_type_lower: str) -> list[str]:
        if "|" not in tx_type_lower:
            return []

        _, raw_sources = tx_type_lower.split("|", 1)
        sources: list[str] = []
        for raw_source in raw_sources.split(","):
            source = sanitize_symbol(raw_source.strip())
            if source:
                sources.append(source)
        return sources

    def handle_fees(
        self,
        row: pd.Series,
        date: str,
        ins: list[TxEntry],
        outs: list[TxEntry],
        tx_type_lower: str,
        touched_coins: set[str],
    ) -> None:
        fee_str = row.get("Fee")
        fee_token = row.get("Fee Token")

        if pd.notna(fee_str) and pd.notna(fee_token):
            fee_qty = Decimal(str(fee_str))
            if fee_qty > 0:
                fee_asset = self.fetch_asset(str(fee_token))

                fee_price = get_crypto_price(
                    coin=fee_asset.price_source,
                    date=date,
                    chain=self.chain,
                    use_lp_prices=fee_asset.valuation_route == ValuationRoute.PROTOCOL_DERIVED,
                )
                fee_val_eur = float(fee_qty) * fee_price

                # 1. Deduct from Fee Asset (Qty decreases, Principal decreases)
                fee_asset.quantity -= fee_qty
                touched_coins.add(fee_asset.coin)

                # 2. Map Cost to Target Asset (Principal increases)
                target_entries = []
                if tx_type_lower in ["swap", "buy", "receive"] and ins:
                    target_entries = ins

                elif tx_type_lower in ["sell", "send"] and outs:
                    target_entries = outs

                if target_entries:
                    fee_asset.adjust_principal(-fee_val_eur)
                    share_val_eur = fee_val_eur / len(target_entries)
                    for entry in target_entries:
                        t_asset = self.fetch_asset(entry.token)
                        t_asset.adjust_principal(share_val_eur)
                        touched_coins.add(t_asset.coin)

    def _update_snapshots(self, touched_coins: set[str], date: str) -> None:
        new_snapshots = []
        for coin in touched_coins:
            asset = self.assets[coin]
            new_snapshots.extend(self._collect_snapshots(asset=asset, date=date))

        # 1. Deduplicate: Keep only the last snapshot per coin for this transaction
        unique_snapshots = {s["Coin"]: s for s in new_snapshots}

        # 2. Update History: Overwrite if exists for same Date+Coin, else Append
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

    def process_transaction(self, row: pd.Series):
        tx_type: str = row["Type"]
        tx_type_lower = tx_type.lower()
        date = row["Date"]

        def _parse_entries(qty_val: str, token_val: str) -> list[TxEntry]:
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

        ins = _parse_entries(qty_val=row.get("Qty in"), token_val=row.get("Token in"))
        outs = _parse_entries(qty_val=row.get("Qty out"), token_val=row.get("Token out"))
        touched_coins = set()

        if tx_type_lower == "buy":
            entry_in = ins[0]
            entry_out = outs[0]

            asset_in = self.fetch_asset(entry_in.token)
            asset_in.buy(
                amount_bought=entry_in.quantity,
                fiat_spent=entry_out.quantity,
                currency=entry_out.token,
                date=date,
            )
            touched_coins.add(asset_in.coin)

        elif tx_type_lower == "receive":
            for entry in ins:
                asset_in = self.fetch_asset(entry.token)
                asset_in.receive(amount_received=entry.quantity, date=date)
                touched_coins.add(asset_in.coin)

        elif tx_type_lower == "sell":
            entry_in = ins[0]
            entry_out = outs[0]

            asset_out = self.fetch_asset(entry_out.token)
            asset_out.sell(
                amount_sold=entry_out.quantity,
                fiat_received=entry_in.quantity,
                currency=entry_in.token,
                date=date,
            )
            touched_coins.add(asset_out.coin)

        elif tx_type_lower == "send":
            for entry in outs:
                asset_out = self.fetch_asset(entry.token)
                asset_out.send(amount_sent=entry.quantity, date=date)
                touched_coins.add(asset_out.coin)

        elif tx_type_lower == "swap":
            self._process_swap(ins=ins, outs=outs, date=date, touched_coins=touched_coins)

        elif tx_type_lower.startswith("reward"):
            allocate_reward_to = self._parse_reward_sources(tx_type_lower=tx_type_lower)
            self._process_reward(
                rewards=ins,
                allocate_reward_to=allocate_reward_to,
                date=date,
                touched_coins=touched_coins,
            )

        elif tx_type_lower.startswith("approve"):
            return

        elif tx_type_lower == "interaction":
            pass
        else:
            error_msg = f"{tx_type}: {ins} -> {outs} on {date} not found."
            print(error_msg)
            return

        # Process Gas Fee (if applicable)
        self.handle_fees(
            row=row,
            date=date,
            ins=ins,
            outs=outs,
            tx_type_lower=tx_type_lower,
            touched_coins=touched_coins,
        )

        self._update_snapshots(touched_coins=touched_coins, date=date)

    def save_to_csv(self, output_path: Path):
        df = pd.DataFrame(self.history)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Date"] = df["Date"].map(format_daily_datetime)
        df.to_csv(output_path, index=False)
        print(f"Portfolio snapshots successfully saved to {output_path}")


def generate_raw_snapshots(input_csv: Path, output_csv: Path, chain: str) -> None:
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


if __name__ == "__main__":
    generate_raw_snapshots(
        input_csv=BLOCKCHAIN_TRANSACTIONS_FOLDER / "arbitrum_transactions.csv",
        output_csv=BLOCKCHAIN_SNAPSHOT_FOLDER / "arbitrum_raw_snapshots.csv",
        chain="arbitrum",
    )
