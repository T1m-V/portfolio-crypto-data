from datetime import timedelta
from decimal import Decimal

from web3 import Web3

from blockchain_reader.protocols.balancer import BALANCER_VAULT_ADDR, get_balancer_underlying
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

TOKEN_ABI = [
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
        "name": "asset",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "stakingToken",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "lp_token",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "want",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "pricePerShare",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getPricePerFullShare",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "shares", "type": "uint256"}],
        "name": "convertToAssets",
        "outputs": [{"internalType": "uint256", "name": "assets", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getPoolId",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _call_optional(contract, fn: str, *args, block_number: int):
    try:
        return getattr(contract.functions, fn)(*args).call(block_identifier=block_number)
    except Exception:
        return None


def _resolve_underlying_address(contract, block_number: int) -> str | None:
    for fn in ("asset", "stakingToken", "lp_token", "want"):
        value = _call_optional(contract, fn, block_number=block_number)
        if isinstance(value, str) and value.startswith("0x"):
            return value
    return None


def _resolve_conversion_ratio(
    contract, shares: int, wrapper_decimals: int, underlying_decimals: int, block_number: int
) -> Decimal:
    converted = _call_optional(contract, "convertToAssets", shares, block_number=block_number)
    if isinstance(converted, int):
        return Decimal(converted) / Decimal(10**underlying_decimals)

    pps = _call_optional(contract, "pricePerShare", block_number=block_number)
    if not isinstance(pps, int):
        pps = _call_optional(contract, "getPricePerFullShare", block_number=block_number)
    if isinstance(pps, int):
        human_shares = Decimal(shares) / Decimal(10**wrapper_decimals)
        return human_shares * (Decimal(pps) / Decimal(10**18))

    # Gauge-like wrappers are typically 1:1 with stake token.
    return Decimal(shares) / Decimal(10**wrapper_decimals)


def get_aura_underlying(
    w3: Web3, token_address: str, one_unit: int, block_number: int
) -> dict[str, Decimal]:
    wrapper = w3.eth.contract(address=token_address, abi=TOKEN_ABI)
    wrapper_decimals = wrapper.functions.decimals().call(block_identifier=block_number)
    underlying_addr = _resolve_underlying_address(contract=wrapper, block_number=block_number)
    if underlying_addr is None:
        return {}

    underlying = w3.eth.contract(address=underlying_addr, abi=TOKEN_ABI)
    underlying_symbol = underlying.functions.symbol().call(block_identifier=block_number)
    underlying_decimals = underlying.functions.decimals().call(block_identifier=block_number)

    underlying_human = _resolve_conversion_ratio(
        contract=wrapper,
        shares=one_unit,
        wrapper_decimals=wrapper_decimals,
        underlying_decimals=underlying_decimals,
        block_number=block_number,
    )

    underlying_one_unit = int(underlying_human * Decimal(10**underlying_decimals))

    # If the unwrapped token is a Balancer BPT, decompose all the way to base tokens.
    pool_id = _call_optional(underlying, "getPoolId", block_number=block_number)
    if pool_id is not None:
        return get_balancer_underlying(
            w3=w3,
            bpt_address=underlying_addr,
            one_unit=underlying_one_unit,
            block_number=block_number,
            vault_address=BALANCER_VAULT_ADDR,
        )

    return {underlying_symbol: underlying_human}


def get_aura_history(chain: str, token_address: str, start_date: str, end_date: str) -> None:
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    token = w3.eth.contract(address=w3.to_checksum_address(token_address), abi=TOKEN_ABI)
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

            assets = get_aura_underlying(
                w3=w3,
                token_address=token.address,
                one_unit=10**token_decimals,
                block_number=block_num,
            )
            row: dict[str, object] = {
                "date": current_dt.date(),
                "block": block_num,
                "aura_balance": 1.0,
            }
            for sym, amount in assets.items():
                row[f"asset_{sym}"] = float(amount)
            history_data.append(row)
        except Exception as e:
            print(f"[aura] Error on {current_dt.date()} for {token_symbol}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="aura", chain=chain, symbol=token_symbol, history_data=history_data
    )
    if output:
        print(f"[aura] Saved to {output}")


def process_all_aura_tokens(chain: str, start_date: str | None = None) -> None:
    tokens = load_tokens(chain=chain)
    token_ranges = load_snapshot_ranges(chain=chain)

    for address, info in tokens.items():
        if info.get("protocol") != "aura":
            continue
        symbol = info.get("symbol", address)
        if symbol not in token_ranges:
            continue

        rng = token_ranges[symbol]
        fallback_start_date = rng["start"].strftime("%Y-%m-%d")
        resolved_start_date = resolve_effective_start_date(
            protocol="aura",
            chain=chain,
            symbol=symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = "now" if rng["qty"] > 0 else rng["end"].strftime("%Y-%m-%d")
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            print(f"[aura] Skipping {symbol}: start={resolved_start_date} is after end={end_date}")
            continue

        if resolved_start_date is None:
            continue

        print(f"[aura] Processing {symbol} ({resolved_start_date} -> {end_date})")
        get_aura_history(
            chain=chain,
            token_address=address,
            start_date=resolved_start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    process_all_aura_tokens(chain="arbitrum")
