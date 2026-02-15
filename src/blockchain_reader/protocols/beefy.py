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
    end_dt = (
        datetime.now(tz=timezone.utc)
        if end_date.lower() == "now"
        else datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    )

    history_data = []
    vault = w3.to_checksum_address(vault_address)
    vault_contract = w3.eth.contract(address=vault, abi=BEEFY_VAULT_ABI)

    try:
        vault_decimals = vault_contract.functions.decimals().call()
        vault_symbol = vault_contract.functions.symbol().call()
    except Exception:
        vault_decimals = 18
        vault_symbol = "MOO"

    map_file_path = BLOCKCHAIN_BLOCK_MAP_FOLDER / f"block_map_{chain}.csv"
    block_map = {}
    if os.path.exists(path=map_file_path):
        with open(file=map_file_path, mode="r") as f:
            reader = csv.DictReader(f=f)
            for row in reader:
                block_map[row["date"]] = int(row["block"])

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

    if history_data:
        output_file = PROTOCOL_UNDERLYING_TOKEN_FOLDER / "beefy" / f"{chain}_{vault_symbol}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        keys = set().union(*(d.keys() for d in history_data))
        with open(file=output_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f=f, fieldnames=sorted(list(keys)))
            writer.writeheader()
            writer.writerows(history_data)
        print(f"Saved to {output_file}")


def process_all_beefy_tokens(chain: str) -> None:
    tokens_file_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    if not os.path.exists(tokens_file_path):
        print(f"Config '{tokens_file_path}' not found.")
        return

    with open(file=tokens_file_path, mode="r") as f:
        tokens = json.load(fp=f)

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
        if info.get("protocol") == "beefy":
            symbol = info.get("symbol", address)
            if symbol in token_ranges:
                rng = token_ranges[symbol]
                start_date = rng["start"].strftime("%Y-%m-%d")
                end_date = "now" if rng["qty"] > 0 else rng["end"].strftime("%Y-%m-%d")
                print(f"Processing Beefy Token: {symbol} ({start_date} -> {end_date})")
                get_beefy_history(
                    chain=chain, vault_address=address, start_date=start_date, end_date=end_date
                )


if __name__ == "__main__":
    process_all_beefy_tokens(chain="arbitrum")
