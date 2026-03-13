from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from tqdm import tqdm
from web3 import Web3

from blockchain_reader.protocols.common import (
    load_block_map,
    load_chain_config,
    load_chain_web3,
    load_tokens,
    write_protocol_history_csv,
)
from blockchain_reader.symbols import (
    build_address_symbol_map,
    build_symbol_family_map,
    canonicalize_symbol,
    sanitize_symbol,
)
from file_paths import BLOCKCHAIN_TRANSACTIONS_FOLDER

ATOKEN_ABI = [
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "UNDERLYING_ASSET_ADDRESS",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

DUST = Decimal("0.000000000001")


@dataclass(frozen=True)
class AaveTokenDescriptor:
    token_address: str
    token_symbol: str
    token_decimals: int
    underlying_address: str
    underlying_symbol: str
    leg: str


def _classify_aave_leg(symbol: str) -> str:
    if symbol.lower().startswith("variabledebt"):
        return "debt"
    return "supply"


def _compute_leg_columns(
    supply_by_symbol: dict[str, Decimal], debt_by_symbol: dict[str, Decimal]
) -> dict[str, Decimal]:
    columns: dict[str, Decimal] = {}
    symbols = sorted(set(supply_by_symbol.keys()) | set(debt_by_symbol.keys()))
    for symbol in symbols:
        supply = supply_by_symbol.get(symbol, Decimal(0))
        debt = debt_by_symbol.get(symbol, Decimal(0))
        columns[f"supply_{symbol}"] = supply
        columns[f"debt_{symbol}"] = debt
        columns[f"net_{symbol}"] = supply - debt
    return columns


def _build_aave_field_order(history: list[dict[str, object]]) -> list[str]:
    base = ["date", "block", "queried_token_count", "missing_contract_count", "rpc_error_count"]
    symbols = set()
    for row in history:
        for key in row.keys():
            if key.startswith("supply_"):
                symbols.add(key.replace("supply_", "", 1))
            elif key.startswith("debt_"):
                symbols.add(key.replace("debt_", "", 1))
            elif key.startswith("net_"):
                symbols.add(key.replace("net_", "", 1))

    ordered = list(base)
    for symbol in sorted(symbols):
        ordered.extend([f"supply_{symbol}", f"debt_{symbol}", f"net_{symbol}"])
    return ordered


def _build_aave_descriptors(
    w3: Web3,
    tokens: dict[str, dict[str, Any]],
    symbol_family: dict[str, str],
    address_symbol_map: dict[str, str],
) -> tuple[list[AaveTokenDescriptor], int]:
    aave_addresses = [
        addr
        for addr, info in tokens.items()
        if addr != "native" and isinstance(info, dict) and info.get("protocol") == "aave"
    ]

    descriptors: list[AaveTokenDescriptor] = []
    unresolved_count = 0
    progress = tqdm(
        total=len(aave_addresses),
        desc="[aave] token discovery",
        unit="token",
        leave=False,
    )
    for addr in aave_addresses:
        info = tokens.get(addr, {})
        try:
            token_address = w3.to_checksum_address(addr)
            token_contract = w3.eth.contract(address=token_address, abi=ATOKEN_ABI)

            token_symbol = token_contract.functions.symbol().call()
            token_decimals = int(token_contract.functions.decimals().call())
            underlying_address = token_contract.functions.UNDERLYING_ASSET_ADDRESS().call()
            underlying_contract = w3.eth.contract(address=underlying_address, abi=ERC20_ABI)
            raw_underlying_symbol = underlying_contract.functions.symbol().call()
            underlying_symbol = address_symbol_map.get(str(underlying_address).lower())
            if not underlying_symbol:
                underlying_symbol = canonicalize_symbol(
                    raw_underlying_symbol, symbol_family=symbol_family
                )
            if not underlying_symbol:
                underlying_symbol = f"UNK_{str(underlying_address)[:8]}"

            descriptors.append(
                AaveTokenDescriptor(
                    token_address=token_address,
                    token_symbol=token_symbol,
                    token_decimals=token_decimals,
                    underlying_address=underlying_address,
                    underlying_symbol=underlying_symbol,
                    leg=_classify_aave_leg(token_symbol),
                )
            )
        except Exception:
            unresolved_count += 1
            fallback_symbol = info.get("symbol", addr)
            print(f"[aave] unresolved token metadata for {fallback_symbol} ({addr})")
        progress.update(1)

    progress.close()
    return descriptors, unresolved_count


def _sorted_block_days(block_map: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(block_map.items(), key=lambda x: x[0])


def _is_on_or_after_start_date(date_str: str, start_date: str | None = None) -> bool:
    if start_date and date_str < start_date:
        return False
    return True


def _parse_date_value(date_value: str) -> datetime | None:
    raw_value = str(date_value or "").strip()
    if not raw_value:
        return None
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(raw_value, fmt)
        except ValueError:
            continue
    return None


def _all_leg_values_within_dust(
    supply_by_symbol: dict[str, Decimal], debt_by_symbol: dict[str, Decimal]
) -> bool:
    return all(abs(value) <= DUST for value in supply_by_symbol.values()) and all(
        abs(value) <= DUST for value in debt_by_symbol.values()
    )


def _parse_entries_from_row(
    row: pd.Series, qty_col: str, token_col: str
) -> list[tuple[str, Decimal]]:
    qty_raw = row.get(qty_col, "")
    token_raw = row.get(token_col, "")
    if pd.isna(qty_raw) or pd.isna(token_raw):
        return []

    qty_val = str(qty_raw or "").strip()
    token_val = str(token_raw or "").strip()
    if not qty_val or not token_val:
        return []

    qty_parts = [x.strip() for x in qty_val.split(",")]
    token_parts = [x.strip() for x in token_val.split(",")]

    assert len(qty_parts) == len(
        token_parts
    ), "Unequal amount of qty values and Token values found in transaction data"

    entries: list[tuple[str, Decimal]] = []
    for idx, token in enumerate(token_parts):
        qty = Decimal(qty_parts[idx])
        entries.append((token, qty))
    return entries


def _derive_aave_bounds_from_transactions(chain: str) -> tuple[str | None, str | None]:
    tx_path = BLOCKCHAIN_TRANSACTIONS_FOLDER / f"{chain}_transactions.csv"
    tokens = load_tokens(chain=chain)
    wrappers = {
        sanitize_symbol(info.get("symbol"))
        for info in tokens.values()
        if isinstance(info, dict) and info.get("protocol") == "aave" and info.get("symbol")
    }

    earliest: datetime | None = None
    latest: datetime | None = None
    if tx_path.exists():
        tx_df = pd.read_csv(tx_path, dtype=str)
        for _, row in tx_df.iterrows():
            entries = _parse_entries_from_row(
                row=row, qty_col="Qty in", token_col="Token in"
            ) + _parse_entries_from_row(row=row, qty_col="Qty out", token_col="Token out")

            if not any(sanitize_symbol(symbol) in wrappers for symbol, _ in entries):
                continue

            dt = _parse_date_value(str(row.get("Date", "")))
            if dt is None:
                continue

            if earliest is None or dt < earliest:
                earliest = dt
            if latest is None or dt > latest:
                latest = dt

    start_date = earliest.strftime("%Y-%m-%d") if earliest else None
    end_date = latest.strftime("%Y-%m-%d") if latest else None

    return start_date, end_date


def get_aave_daily_exposure(
    chain: str, start_date: str | None = None, end_date: str | None = None
) -> None:
    cfg = load_chain_config(chain=chain)
    w3 = load_chain_web3(chain=chain)
    wallet = w3.to_checksum_address(cfg["my_address"])
    tokens = load_tokens(chain=chain)
    block_map = load_block_map(chain=chain)
    symbol_family = build_symbol_family_map(token_metadata=tokens)
    address_symbol_map = build_address_symbol_map(
        token_metadata=tokens, symbol_family=symbol_family
    )

    descriptors, unresolved_count = _build_aave_descriptors(
        w3=w3,
        tokens=tokens,
        symbol_family=symbol_family,
        address_symbol_map=address_symbol_map,
    )
    if not descriptors:
        print("[aave] no Aave token descriptors resolved; skipping")
        return

    token_contracts = {
        d.token_address: w3.eth.contract(address=d.token_address, abi=ATOKEN_ABI)
        for d in descriptors
    }

    day_items = [
        (date_str, block_num)
        for date_str, block_num in _sorted_block_days(block_map=block_map)
        if _is_on_or_after_start_date(date_str=date_str, start_date=start_date)
    ]

    history: list[dict[str, object]] = []
    non_zero_net_days = 0
    rpc_error_total = 0
    missing_contract_total = 0
    first_processed_day: str | None = None
    last_processed_day: str | None = None

    day_progress = tqdm(
        total=len(day_items),
        desc="[aave] daily balance query",
        unit="day",
        leave=False,
    )
    for date_str, block_num in day_items:
        supply_by_symbol: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
        debt_by_symbol: dict[str, Decimal] = defaultdict(lambda: Decimal(0))

        queried_token_count = 0
        missing_contract_count = 0
        rpc_error_count = 0

        for desc in descriptors:
            queried_token_count += 1
            contract = token_contracts[desc.token_address]
            try:
                if len(w3.eth.get_code(contract.address, block_identifier=block_num)) == 0:
                    missing_contract_count += 1
                    continue

                raw_balance = contract.functions.balanceOf(wallet).call(block_identifier=block_num)
                balance = Decimal(raw_balance) / Decimal(10**desc.token_decimals)
                if abs(balance) <= DUST:
                    continue

                if desc.leg == "supply":
                    supply_by_symbol[desc.underlying_symbol] += balance
                else:
                    debt_by_symbol[desc.underlying_symbol] += balance
            except Exception:
                rpc_error_count += 1

        leg_columns = _compute_leg_columns(
            supply_by_symbol=supply_by_symbol,
            debt_by_symbol=debt_by_symbol,
        )
        has_non_zero_net = any(
            abs(value) > DUST for key, value in leg_columns.items() if key.startswith("net_")
        )
        if has_non_zero_net:
            non_zero_net_days += 1

        row: dict[str, object] = {
            "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
            "block": block_num,
            "queried_token_count": queried_token_count,
            "missing_contract_count": missing_contract_count,
            "rpc_error_count": rpc_error_count,
        }
        for col_name, value in leg_columns.items():
            row[col_name] = float(value)
        history.append(row)
        if first_processed_day is None:
            first_processed_day = date_str
        last_processed_day = date_str

        rpc_error_total += rpc_error_count
        missing_contract_total += missing_contract_count
        day_progress.update(1)

        is_post_end_day = end_date is not None and date_str > end_date
        should_stop_after_day = (
            is_post_end_day
            and rpc_error_count == 0
            and _all_leg_values_within_dust(
                supply_by_symbol=supply_by_symbol,
                debt_by_symbol=debt_by_symbol,
            )
        )
        if should_stop_after_day:
            break

    day_progress.close()

    output = write_protocol_history_csv(
        protocol="aave",
        chain=chain,
        symbol="aave_daily_exposure",
        history_data=history,
        fieldnames=_build_aave_field_order(history=history),
    )
    if output:
        first_day = first_processed_day or "-"
        last_day = last_processed_day or "-"
        print(f"[aave] Saved to {output}")
        print(
            "[aave] Audit "
            f"first_day={first_day}, last_day={last_day}, days={len(history)}, "
            f"non_zero_net_days={non_zero_net_days}, unresolved_mappings={unresolved_count}, "
            f"missing_contract_total={missing_contract_total}, rpc_error_total={rpc_error_total}"
        )


def process_all_aave_tokens(chain: str) -> None:
    start_date, end_date = _derive_aave_bounds_from_transactions(chain=chain)
    get_aave_daily_exposure(chain=chain, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    default_start = "2023-03-24"
    default_end = "2025-05-10"
    get_aave_daily_exposure(chain="arbitrum", start_date=default_start, end_date=default_end)
