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

# ==========================================
# ABIS
# ==========================================

BEEFY_VAULT_ABI = [
    {
        "inputs": [],
        "name": "want",
        "outputs": [{"internalType": "contract IERC20", "name": "", "type": "address"}],
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
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
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


# ==========================================
# CORE BEEFY LOGIC
# ==========================================
def get_beefy_underlying(
    w3: Web3,
    vault_address: str,
    one_unit: int,
    block_number: int,
) -> dict[str, Decimal]:
    """
    Calculates the underlying assets for a given Beefy MooToken balance.
    """
    vault_contract = w3.eth.contract(address=vault_address, abi=BEEFY_VAULT_ABI)

    # 1. Get Price Per Full Share (Ratio of Want Token per MooToken)
    # Beefy PPFS is always scaled to 1e18, regardless of token decimals.
    ppfs = vault_contract.functions.getPricePerFullShare().call(block_identifier=block_number)

    # 2. Get Want Token Address
    want_addr = vault_contract.functions.want().call(block_identifier=block_number)

    # 3. Calculate Underlying Amount in Wei
    # Formula: (MooAmount * PPFS) / 1e18
    underlying_wei = (Decimal(one_unit) * Decimal(ppfs)) / Decimal(10**18)

    assets = {}

    # 4. Standard Unwrapping
    want_contract = w3.eth.contract(address=want_addr, abi=ERC20_ABI)
    sym = want_contract.functions.symbol().call(block_identifier=block_number)
    dec = want_contract.functions.decimals().call(block_identifier=block_number)

    readable_balance = underlying_wei / Decimal(10**dec)
    assets[sym] = readable_balance

    return assets


# ==========================================
# MAIN LOOP
# ==========================================
def get_beefy_history(
    chain: str,
    vault_address: str,
    start_date: str,
    end_date: str,
) -> None:
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    history_data = []
    vault = w3.to_checksum_address(vault_address)
    vault_contract = w3.eth.contract(address=vault, abi=BEEFY_VAULT_ABI)

    try:
        vault_decimals = vault_contract.functions.decimals().call()
        vault_symbol = vault_contract.functions.symbol().call()
    except Exception:
        vault_decimals = 18
        vault_symbol = "MOO"

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        if date_str not in block_map:
            current_dt += timedelta(days=1)
            continue

        block_num = block_map[date_str]
        print(f"Date: {date_str} | Block: {block_num}")

        try:
            if len(w3.eth.get_code(vault, block_identifier=block_num)) == 0:
                print("  -> Contract not deployed yet.")
                current_dt += timedelta(days=1)
                continue

            one_unit = 10**vault_decimals
            assets = get_beefy_underlying(w3, vault, one_unit, block_num)

            row = {
                "date": current_dt.date(),
                "block": block_num,
                "moo_balance": 1.0,  # Representing 1 unit of the vault token
            }
            for sym, amt in assets.items():
                row[f"asset_{sym}"] = float(amt)

            history_data.append(row)

        except Exception as e:
            print(f"Error on {current_dt.date()}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="beefy",
        chain=chain,
        symbol=vault_symbol,
        history_data=history_data,
    )
    if output:
        print(f"[beefy] Saved to {output}")


def process_all_beefy_tokens(chain: str, start_date: str | None = None) -> None:
    tokens = load_tokens(chain=chain)
    token_ranges = load_snapshot_ranges(chain=chain)
    for address, info in tokens.items():
        if info.get("protocol") != "beefy":
            continue

        symbol = info.get("symbol", address)
        if symbol not in token_ranges:
            continue

        rng = token_ranges[symbol]
        fallback_start_date = rng["start"].strftime("%Y-%m-%d")
        resolved_start_date = resolve_effective_start_date(
            protocol="beefy",
            chain=chain,
            symbol=symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = "now" if rng["qty"] > 0 else rng["end"].strftime("%Y-%m-%d")
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            print(f"[beefy] Skipping {symbol}: start={resolved_start_date} is after end={end_date}")
            continue

        if resolved_start_date is None:
            continue

        print(f"[beefy] Processing {symbol} ({resolved_start_date} -> {end_date})")
        get_beefy_history(
            chain=chain,
            vault_address=address,
            start_date=resolved_start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    process_all_beefy_tokens(chain="arbitrum")
