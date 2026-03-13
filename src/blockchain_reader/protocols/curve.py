from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from web3 import Web3

from blockchain_reader.protocols.common import (
    load_block_map,
    load_chain_web3,
    load_snapshot_ranges,
    load_tokens,
    resolve_date_window,
    resolve_effective_start_date,
    should_skip_date_window,
    write_protocol_history_csv,
)

LP_TOKEN_ABI = [
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
        "name": "totalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "minter",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

CURVE_POOL_ABI = [
    {
        "inputs": [{"internalType": "uint256", "name": "i", "type": "uint256"}],
        "name": "coins",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "int128", "name": "i", "type": "int128"}],
        "name": "coins",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "arg0", "type": "uint256"}],
        "name": "balances",
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

MAX_CURVE_POOL_COINS = 8


@dataclass(frozen=True)
class CurvePoolToken:
    address: str
    balance: int
    symbol: str
    decimals: int


def _curve_underlying_from_balances(
    one_unit: int,
    total_supply: int,
    pool_tokens: list[CurvePoolToken],
) -> dict[str, Decimal]:
    if total_supply == 0:
        return {}
    share = Decimal(one_unit) / Decimal(total_supply)
    out: dict[str, Decimal] = {}
    for token in pool_tokens:
        human = (Decimal(token.balance) * share) / Decimal(10**token.decimals)
        out[token.symbol] = human
    return out


def _read_curve_pool_tokens(w3: Web3, pool_address: str, block_number: int) -> list[CurvePoolToken]:
    pool = w3.eth.contract(address=pool_address, abi=CURVE_POOL_ABI)
    token_addresses: list[str] = []
    # Curve pools expose `coins(i)` by index and revert when `i` is out of bounds.
    for i in range(MAX_CURVE_POOL_COINS):
        try:
            addr = pool.functions.coins(i).call(block_identifier=block_number)
            token_addresses.append(addr)
        except Exception:
            break

    pool_tokens: list[CurvePoolToken] = []
    for i, addr in enumerate(token_addresses):
        balance = int(pool.functions.balances(i).call(block_identifier=block_number))
        token = w3.eth.contract(address=addr, abi=ERC20_ABI)
        symbol = token.functions.symbol().call(block_identifier=block_number)
        decimals = int(token.functions.decimals().call(block_identifier=block_number))
        pool_tokens.append(
            CurvePoolToken(
                address=addr,
                balance=balance,
                symbol=symbol,
                decimals=decimals,
            )
        )
    return pool_tokens


def get_curve_underlying(
    w3: Web3, lp_token_address: str, one_unit: int, block_number: int
) -> dict[str, Decimal]:
    lp = w3.eth.contract(address=lp_token_address, abi=LP_TOKEN_ABI)
    total_supply = lp.functions.totalSupply().call(block_identifier=block_number)
    pool_address = lp.functions.minter().call(block_identifier=block_number)
    pool_tokens = _read_curve_pool_tokens(
        w3=w3, pool_address=pool_address, block_number=block_number
    )

    return _curve_underlying_from_balances(
        one_unit=one_unit,
        total_supply=total_supply,
        pool_tokens=pool_tokens,
    )


def get_curve_history(chain: str, token_address: str, start_date: str, end_date: str) -> None:
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    token = w3.eth.contract(address=w3.to_checksum_address(token_address), abi=LP_TOKEN_ABI)
    token_decimals = token.functions.decimals().call()
    token_symbol = token.functions.symbol().call()
    history_data: list[dict[str, object]] = []

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        block_num = block_map.get(date_str)
        if block_num is None:
            current_dt += timedelta(days=1)
            continue

        try:
            if len(w3.eth.get_code(token.address, block_identifier=block_num)) == 0:
                current_dt += timedelta(days=1)
                continue

            assets = get_curve_underlying(
                w3=w3,
                lp_token_address=token.address,
                one_unit=10**token_decimals,
                block_number=block_num,
            )
            row: dict[str, object] = {
                "date": current_dt.date(),
                "block": block_num,
                "curve_lp_balance": 1.0,
            }
            for sym, amount in assets.items():
                row[f"asset_{sym}"] = float(amount)
            history_data.append(row)
        except Exception as e:
            print(f"[curve] Error on {current_dt.date()} for {token_symbol}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="curve", chain=chain, symbol=token_symbol, history_data=history_data
    )
    if output:
        print(f"[curve] Saved to {output}")


def process_all_curve_tokens(chain: str, start_date: str | None = None) -> None:
    tokens = load_tokens(chain=chain)
    token_ranges = load_snapshot_ranges(chain=chain)

    for address, info in tokens.items():
        if info.get("protocol") != "curve":
            continue
        symbol = info.get("symbol", address)
        if symbol not in token_ranges:
            continue

        rng = token_ranges[symbol]
        fallback_start_date = rng["start"].strftime("%Y-%m-%d")
        resolved_start_date = resolve_effective_start_date(
            protocol="curve",
            chain=chain,
            symbol=symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = "now" if rng["qty"] > 0 else rng["end"].strftime("%Y-%m-%d")
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            print(f"[curve] Skipping {symbol}: start={resolved_start_date} is after end={end_date}")
            continue

        if resolved_start_date is None:
            continue

        print(f"[curve] Processing {symbol} ({resolved_start_date} -> {end_date})")
        get_curve_history(
            chain=chain,
            token_address=address,
            start_date=resolved_start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    process_all_curve_tokens(chain="arbitrum")
