import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from web3 import Web3

from file_paths import (
    BLOCKCHAIN_BLOCK_MAP_FOLDER,
    BLOCKCHAIN_SNAPSHOT_FOLDER,
    CHAIN_INFO_PATH,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
)


def load_chain_config(chain: str) -> dict[str, str]:
    if not os.path.exists(CHAIN_INFO_PATH):
        raise FileNotFoundError(f"Config '{CHAIN_INFO_PATH}' not found.")

    with open(file=CHAIN_INFO_PATH, mode="r") as f:
        config_data = json.load(fp=f)

    if chain not in config_data:
        raise ValueError(f"Chain '{chain}' not found in config.")

    return config_data[chain]


def load_chain_web3(chain: str) -> Web3:
    cfg = load_chain_config(chain=chain)
    rpc_url = cfg.get("alchemy_url") or cfg.get("rpc_url")
    if not rpc_url:
        raise ValueError(f"Chain '{chain}' is missing both 'alchemy_url' and 'rpc_url'.")

    w3 = Web3(provider=Web3.HTTPProvider(endpoint_uri=rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Connection failed for chain '{chain}'.")
    return w3


def load_tokens(chain: str) -> dict[str, dict[str, str]]:
    tokens_file_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    if not os.path.exists(tokens_file_path):
        raise FileNotFoundError(f"Config '{tokens_file_path}' not found.")

    with open(file=tokens_file_path, mode="r") as f:
        return json.load(fp=f)


def load_snapshot_ranges(chain: str) -> dict[str, dict[str, object]]:
    snapshots_file_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_snapshots.csv"
    if not os.path.exists(snapshots_file_path):
        raise FileNotFoundError(f"Snapshots '{snapshots_file_path}' not found.")

    df = pd.read_csv(snapshots_file_path)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    token_ranges: dict[str, dict[str, object]] = {}
    for coin, group in df.groupby("Coin"):
        token_ranges[coin] = {
            "start": group["Date"].min(),
            "end": group["Date"].max(),
            "qty": group["Quantity"].iloc[-1],
        }
    return token_ranges


def resolve_date_window(
    start_date: str,
    end_date: str,
) -> tuple[datetime, datetime]:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_date.lower() == "now":
        end_dt = datetime.now(tz=timezone.utc)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start_dt, end_dt


def load_block_map(chain: str) -> dict[str, int]:
    map_file_path = BLOCKCHAIN_BLOCK_MAP_FOLDER / f"block_map_{chain}.csv"
    block_map = {}
    if os.path.exists(path=map_file_path):
        with open(file=map_file_path, mode="r") as f:
            reader = csv.DictReader(f=f)
            for row in reader:
                block_map[row["date"]] = int(row["block"])
    return block_map


def write_protocol_history_csv(
    protocol: str,
    chain: str,
    symbol: str,
    history_data: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> Path | None:
    if not history_data:
        return None

    output_file = PROTOCOL_UNDERLYING_TOKEN_FOLDER / protocol / f"{chain}_{symbol}.csv"
    os.makedirs(Path(output_file).parent, exist_ok=True)
    keys = set().union(*(d.keys() for d in history_data))
    ordered_fieldnames = sorted(list(keys))
    if fieldnames:
        preferred = [name for name in fieldnames if name in keys]
        remaining = sorted([name for name in keys if name not in set(preferred)])
        ordered_fieldnames = preferred + remaining

    with open(file=output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f=f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(history_data)
    return output_file
