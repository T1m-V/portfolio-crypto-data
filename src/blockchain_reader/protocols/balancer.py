import csv
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
from web3 import Web3

from file_paths import (
    BLOCKCHAIN_BLOCK_MAP_FOLDER,
    BLOCKHAIN_SNAPSHOT_FOLDER,
    CHAIN_INFO_PATH,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
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
    if not os.path.exists(CHAIN_INFO_PATH):
        print(f"Config '{CHAIN_INFO_PATH}' not found.")
        return

    with open(file=CHAIN_INFO_PATH, mode="r") as f:
        config_data = json.load(fp=f)

    if chain not in config_data:
        print(f"Chain '{chain}' not found in config.")
        return

    cfg = config_data[chain]
    rpc_url = cfg["alchemy_url"]

    w3 = Web3(provider=Web3.HTTPProvider(endpoint_uri=rpc_url))
    if not w3.is_connected():
        print("Connection Failed")
        return

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if end_date.lower() == "now":
        end_dt = datetime.now(tz=timezone.utc)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    history_data = []
    bpt = w3.to_checksum_address(pool_address)

    bpt_contract = w3.eth.contract(address=bpt, abi=TOKEN_ABI)

    bpt_decimals = bpt_contract.functions.decimals().call()
    bpt_symbol = bpt_contract.functions.symbol().call()

    # Load Block Map
    map_file_path = BLOCKCHAIN_BLOCK_MAP_FOLDER / f"block_map_{chain}.csv"
    block_map = {}
    if os.path.exists(path=map_file_path):
        print(f"Loading block map from {map_file_path}...")
        with open(file=map_file_path, mode="r") as f:
            reader = csv.DictReader(f=f)
            for row in reader:
                block_map[row["date"]] = int(row["block"])

    current_dt = start_dt
    while current_dt <= end_dt:
        # 1. Find Block
        date_str = current_dt.strftime("%Y-%m-%d")

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
                "date": current_dt.date(),
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

    # Write CSV
    if history_data:
        output_file = PROTOCOL_UNDERLYING_TOKEN_FOLDER / "balancer" / f"{chain}_{bpt_symbol}.csv"
        keys = set().union(*(d.keys() for d in history_data))
        with open(file=output_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f=f, fieldnames=sorted(list(keys)))
            writer.writeheader()
            writer.writerows(history_data)
        print(f"Saved to {output_file}")


def process_all_balancer_tokens(chain: str) -> None:
    tokens_file_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    if not os.path.exists(tokens_file_path):
        print(f"Config '{tokens_file_path}' not found.")
        return

    with open(file=tokens_file_path, mode="r") as f:
        tokens: dict[str, dict[str, str]] = json.load(fp=f)

    snapshots_file_path = BLOCKHAIN_SNAPSHOT_FOLDER / f"{chain}_snapshots.csv"
    if not os.path.exists(snapshots_file_path):
        print(f"Snapshots '{snapshots_file_path}' not found.")
        return

    df = pd.read_csv(snapshots_file_path)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    token_ranges = {}
    for coin, group in df.groupby("Coin"):
        token_ranges[coin] = {
            "start": group["Date"].min(),
            "end": group["Date"].max(),
            "qty": group["Quantity"].iloc[-1],
        }

    for address, info in tokens.items():
        if info.get("protocol") == "balancer":
            symbol = info.get("symbol", address)
            if symbol in token_ranges:
                rng = token_ranges[symbol]
                start_date = rng["start"].strftime("%Y-%m-%d")
                end_date = "now" if rng["qty"] > 0 else rng["end"].strftime("%Y-%m-%d")
                print(f"Processing Balancer Token: {symbol} ({start_date} -> {end_date})")
                get_balancer_history(
                    chain=chain, pool_address=address, start_date=start_date, end_date=end_date
                )
            else:
                print(f"Skipping {symbol}: No snapshot data found.")


if __name__ == "__main__":
    process_all_balancer_tokens(chain="arbitrum")
