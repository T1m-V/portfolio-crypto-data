from datetime import timedelta
from decimal import Decimal

from web3 import Web3

from blockchain_reader.datetime_utils import format_daily_datetime
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
# CONFIGURATION
# ==========================================

# Balancer V2 Vault (Constant on all chains)
BALANCER_VAULT_ADDR = "0xBA12222222228d8Ba445958a75a0704d566BF2C8"

# ==========================================
# ABIS (INTERFACES)
# ==========================================

# Standard ERC20 + Balancer Specific extensions
TOKEN_ABI = [
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
    {
        "inputs": [],
        "name": "getPoolId",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
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
    # 'getActualSupply' is crucial for newer "Composable" pools
    {
        "inputs": [],
        "name": "getActualSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

BALANCER_VAULT_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "poolId", "type": "bytes32"}],
        "name": "getPoolTokens",
        "outputs": [
            {"internalType": "contract IERC20[]", "name": "tokens", "type": "address[]"},
            {"internalType": "uint256[]", "name": "balances", "type": "uint256[]"},
            {"internalType": "uint256", "name": "lastChangeBlock", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


# ==========================================
# CORE BALANCER LOGIC
# ==========================================
def get_balancer_underlying(
    w3: Web3,
    bpt_address: str,
    one_unit: int,
    block_number: int,
    vault_address: str,
) -> dict[str, Decimal]:
    """
    Calculates the underlying assets for a given BPT balance.

    args:
        w3: Web3 instance.
        bpt_address: Address of the Balancer Pool Token.
        one_unit: User's BPT balance in Wei.
        block_number: Block number for historical lookup.
        vault_address: Address of the Balancer Vault.

    returns:
        Dictionary mapping token symbols to readable decimal amounts.
    """
    # Initialize Contracts
    bpt_contract = w3.eth.contract(address=bpt_address, abi=TOKEN_ABI)
    vault_contract = w3.eth.contract(address=vault_address, abi=BALANCER_VAULT_ABI)

    # 1. Get Pool Metadata
    pool_id = bpt_contract.functions.getPoolId().call(block_identifier=block_number)

    # 2. Get Total Supply (Handling Phantom BPTs)
    total_supply = bpt_contract.functions.getActualSupply().call(block_identifier=block_number)

    # 3. Get User's Share
    # (User Balance / Total Circulating Supply)
    share_ratio = Decimal(value=one_unit) / Decimal(value=total_supply)

    # 4. Get Pool Assets from Vault
    # Returns ([addresses], [balances], lastChangeBlock)
    pool_tokens_data = vault_contract.functions.getPoolTokens(poolId=pool_id).call(
        block_identifier=block_number
    )
    token_addrs = pool_tokens_data[0]
    token_bals = pool_tokens_data[1]

    underlying_assets = {}

    for i, token_addr in enumerate(token_addrs):
        # SKIP the BPT itself.
        if token_addr.lower() == bpt_address.lower():
            continue

        # Get Token Details
        t_contract = w3.eth.contract(address=token_addr, abi=TOKEN_ABI)
        sym = t_contract.functions.symbol().call(block_identifier=block_number)
        dec = t_contract.functions.decimals().call(block_identifier=block_number)

        # Calculate User Portion
        # Total Asset Balance * User Share Ratio
        user_asset_wei = Decimal(value=token_bals[i]) * share_ratio
        readable_balance = user_asset_wei / Decimal(value=10**dec)

        underlying_assets[sym] = readable_balance

    return underlying_assets


def get_balancer_history(
    chain: str,
    pool_address: str,
    start_date: str,
    end_date: str,
    vault_address: str = BALANCER_VAULT_ADDR,
) -> None:
    """
    Main execution loop for fetching Balancer history.

    args:
        chain: The blockchain network name.
        pool_address: The address of the Balancer Pool Token (BPT).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD or 'now').
        vault_address: The address of the Balancer Vault (defaults to standard V2 Vault).
    """
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    history_data = []
    bpt = w3.to_checksum_address(pool_address)

    bpt_contract = w3.eth.contract(address=bpt, abi=TOKEN_ABI)

    bpt_decimals = bpt_contract.functions.decimals().call()
    bpt_symbol = bpt_contract.functions.symbol().call()

    current_dt = start_dt
    while current_dt <= end_dt:
        # 1. Find Block
        date_str = format_daily_datetime(current_dt)

        if date_str in block_map:
            block_num = block_map[date_str]
        else:
            print(f"Skipping {date_str}: Block not found in map.")
            current_dt += timedelta(days=1)
            continue

        print(f"Date: {date_str} | Block: {block_num}")

        try:
            # Check if contract exists at this block
            if len(w3.eth.get_code(bpt, block_identifier=block_num)) == 0:
                print("  -> Contract not deployed yet.")
                current_dt += timedelta(days=1)
                continue

            # 2. Calculate Underlying
            assets = get_balancer_underlying(
                w3=w3,
                bpt_address=bpt,
                one_unit=10**bpt_decimals,
                block_number=block_num,
                vault_address=vault_address,
            )

            # 3. Save
            row = {
                "date": format_daily_datetime(current_dt),
                "block": block_num,
                "bpt_balance": float(Decimal(value=10**bpt_decimals) / Decimal(value=10**18)),
            }
            # Flatten assets into the row
            for sym, amt in assets.items():
                row[f"asset_{sym}"] = float(amt)

            history_data.append(row)

        except Exception as e:
            err_str = str(e)
            if "missing trie node" in err_str or "not available" in err_str:
                print(
                    f"Error on {current_dt.date()}: Missing historical state. "
                    f"Ensure your RPC URL for '{chain}' supports Archive Node mode."
                )
            else:
                print(f"Error on {current_dt.date()}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="balancer",
        chain=chain,
        symbol=bpt_symbol,
        history_data=history_data,
    )
    if output:
        print(f"[balancer] Saved to {output}")


def process_all_balancer_tokens(chain: str, start_date: str | None = None) -> None:
    tokens = load_tokens(chain=chain)
    token_ranges = load_snapshot_ranges(chain=chain)
    for address, info in tokens.items():
        if info.get("protocol") != "balancer":
            continue

        symbol = info.get("symbol", address)
        if symbol not in token_ranges:
            print(f"[balancer] Skipping {symbol}: no snapshot data found.")
            continue

        rng = token_ranges[symbol]
        fallback_start_date = format_daily_datetime(rng["start"])
        resolved_start_date = resolve_effective_start_date(
            protocol="balancer",
            chain=chain,
            symbol=symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = "now" if rng["qty"] > 0 else format_daily_datetime(rng["end"])
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            print(
                f"[balancer] Skipping {symbol}: start={resolved_start_date} is after end={end_date}"
            )
            continue

        if resolved_start_date is None:
            continue

        print(f"[balancer] Processing {symbol} ({resolved_start_date} -> {end_date})")
        get_balancer_history(
            chain=chain,
            pool_address=address,
            start_date=resolved_start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    process_all_balancer_tokens(chain="arbitrum")
