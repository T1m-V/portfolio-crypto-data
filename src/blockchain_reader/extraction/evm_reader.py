import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any

import pandas as pd
import requests
from tqdm.asyncio import tqdm_asyncio
from web3 import Web3

from blockchain_reader.datetime_utils import (
    TRANSACTION_DATETIME_FORMAT,
    parse_transaction_datetime,
)
from blockchain_reader.datetime_utils import (
    parse_transaction_datetime_series as parse_transaction_datetime_series_compat,
)
from blockchain_reader.extraction.token_manager import TokenManager
from blockchain_reader.extraction.transaction_analyzer import analyze_transaction
from file_paths import BLOCKCHAIN_TRANSACTIONS_FOLDER, CHAIN_INFO_PATH, TOKENS_FOLDER

ctx = Context(prec=78, rounding=ROUND_HALF_UP)
DEFAULT_START_DATE = "01/01/2000 00:00:00"
OUTPUT_COLUMNS = [
    "TX Hash",
    "Date",
    "Qty in",
    "Token in",
    "Qty out",
    "Token out",
    "Type",
    "Fee",
    "Fee Token",
]


def _fetch_explorer_data(
    api_url: str,
    params: dict[str, Any],
    result_key: str = "result",
    timeout_seconds: int = 20,
    max_retries: int = 3,
) -> list[Any]:
    """
    Executes a GET request to the Explorer API.

    args:
        api_url: API endpoint URL.
        params: Query parameters.
        result_key: JSON key containing the result data.

    returns:
        List of result items.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(api_url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == max_retries:
                print(f"[!] API Request Error for action={params.get('action')}: {e}")
                return []
            time.sleep(0.4 * (2**attempt))
            continue

        status = data.get("status")
        message = str(data.get("message", "")).lower()
        result = data.get(result_key, [])
        result_str = str(result).lower()

        if status == "1":
            if isinstance(result, list):
                return result
            return []

        if status == "0" and "no transactions found" in (message + result_str):
            return []

        if attempt == max_retries:
            print(f"[!] Unexpected explorer payload for action={params.get('action')}: {data}")
            return []
        time.sleep(0.4 * (2**attempt))

    return []


def _parse_input_date_to_utc(date_str: str, end_of_day: bool) -> datetime:
    parsed = parse_transaction_datetime(date_str)
    if parsed is None:
        raise ValueError(f"Invalid date input: {date_str}")

    text = str(date_str or "").strip()
    has_explicit_time = " " in text
    parsed = parsed.replace(tzinfo=timezone.utc)

    if has_explicit_time:
        return parsed
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=0)
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def _derive_start_date(output_path: str, overlap_days: int = 1) -> str:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return DEFAULT_START_DATE
    try:
        df_dates = pd.read_csv(output_path, usecols=["Date"])
        if df_dates.empty:
            return DEFAULT_START_DATE
        latest = _parse_transaction_datetime_series(df_dates["Date"]).max()
        if pd.isna(latest):
            return DEFAULT_START_DATE
        start_dt = (latest - timedelta(days=overlap_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start_dt.strftime(TRANSACTION_DATETIME_FORMAT)
    except (ValueError, KeyError, pd.errors.ParserError):
        return DEFAULT_START_DATE


def _parse_transaction_datetime_series(series: pd.Series) -> pd.Series:
    return parse_transaction_datetime_series_compat(series=series)


def _normalize_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for col in OUTPUT_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized = normalized[OUTPUT_COLUMNS]
    for col in OUTPUT_COLUMNS:
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized


def _safe_timestamp(tx: dict[str, Any]) -> int:
    try:
        return int(tx.get("timeStamp", 0))
    except (TypeError, ValueError):
        return 0


def get_all_transaction_hashes(
    api_url: str, api_key: str, chain_id: str, address: str, start_ts: int, end_ts: int
) -> tuple[set[str], set[str], list[Any]]:
    """
    Retrieves transaction hashes from multiple API endpoints.

    args:
        api_url: Explorer API URL.
        api_key: Explorer API key.
        chain_id: Chain ID.
        address: User's wallet address.
        start_ts: Start timestamp.
        end_ts: End timestamp.

    returns:
        Tuple of (standard_hashes, all_hashes, internal_txs).
    """
    base_params = {
        "module": "account",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key,
        "chainid": chain_id,
    }

    # 1. Standard TX List
    p_std = {**base_params, "action": "txlist"}
    txs_std = _fetch_explorer_data(api_url=api_url, params=p_std)
    hashes_std = {tx["hash"] for tx in txs_std if start_ts <= _safe_timestamp(tx) <= end_ts}

    # 2. Token Transfers (ERC20)
    p_tok = {**base_params, "action": "tokentx"}
    txs_tok = _fetch_explorer_data(api_url=api_url, params=p_tok)
    hashes_tok = {tx["hash"] for tx in txs_tok if start_ts <= _safe_timestamp(tx) <= end_ts}

    # 3. Internal Transactions
    # We return the raw list to map values later, but here we just need hashes
    p_int = {**base_params, "action": "txlistinternal"}
    txs_int = _fetch_explorer_data(api_url=api_url, params=p_int)
    hashes_int = {tx["hash"] for tx in txs_int if start_ts <= _safe_timestamp(tx) <= end_ts}

    all_hashes = hashes_std | hashes_tok | hashes_int
    return hashes_std, all_hashes, txs_int


def build_internal_eth_map(txs_internal: list[dict], my_address: str) -> dict[str, Decimal]:
    """
    Maps internal transaction hashes to ETH values.

    args:
        txs_internal: List of raw internal transactions.
        my_address: User's wallet address.

    returns:
        Dictionary mapping tx_hash to ETH amount.
    """
    internal_map = {}
    for tx in txs_internal:
        # Check if 'to' exists (contract creation has None/empty 'to')
        value = Decimal(str(tx.get("value", "0")))
        if tx.get("to") and tx["to"].lower() == my_address and value > 0:
            tx_hash = tx["hash"]
            amount = value / Decimal(10**18)
            internal_map[tx_hash] = internal_map.get(tx_hash, Decimal(0)) + amount
    return internal_map


async def retrieve_transactions(
    chain: str, start_date: str | None = None, end_date: str | None = None
) -> None:
    """
    Main entry point to fetch and analyze transactions for a chain.

    args:
        chain: Chain identifier (e.g., 'arbitrum').
        start_date: Start date (DD/MM/YYYY HH:MM:SS; legacy formats supported).
        end_date: End date (DD/MM/YYYY HH:MM:SS; legacy formats supported).
    """
    print(f"--- START PROCESSING: {chain.upper()} ---")

    # 1. Load Config
    if not os.path.exists(CHAIN_INFO_PATH):
        raise FileNotFoundError(f"Config '{CHAIN_INFO_PATH}' not found.")

    with open(CHAIN_INFO_PATH, "r") as f:
        config_data = json.load(f)

    if chain not in config_data:
        raise ValueError(f"Chain '{chain}' not found.")

    cfg: dict[str, str] = config_data[chain]
    my_address = cfg["my_address"].lower()
    api_url = cfg.get("api_url")
    api_key = cfg.get("api_key")
    chain_id = cfg.get("chain_id")
    if not api_url:
        raise ValueError(f"Chain '{chain}' missing 'api_url' in config.")

    # Setup Paths & Connection
    token_path = TOKENS_FOLDER / f"{chain}_tokens.json"
    output_path = BLOCKCHAIN_TRANSACTIONS_FOLDER / f"{chain}_transactions.csv"
    os.makedirs(BLOCKCHAIN_TRANSACTIONS_FOLDER, exist_ok=True)
    os.makedirs(TOKENS_FOLDER, exist_ok=True)

    w3 = Web3(Web3.HTTPProvider(cfg["rpc_url"]))
    if not w3.is_connected():
        print("No RPC connection.")
        return

    # Determine dates if not provided
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc).strftime(TRANSACTION_DATETIME_FORMAT)

    if start_date is None:
        start_date = _derive_start_date(output_path=output_path, overlap_days=1)

    # 2. Parse Dates
    start_ts = int(_parse_input_date_to_utc(start_date, end_of_day=False).timestamp())
    end_ts = int(_parse_input_date_to_utc(end_date, end_of_day=True).timestamp())

    # 3. Fetch Hashes
    print("Fetching transaction lists...")
    std_hashes, all_hashes, raw_internal_txs = get_all_transaction_hashes(
        api_url=api_url,
        api_key=api_key,
        chain_id=chain_id,
        address=my_address,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    internal_map = build_internal_eth_map(txs_internal=raw_internal_txs, my_address=my_address)

    print(f"-> Found {len(all_hashes)} unique transactions.")
    if not all_hashes:
        print("No transactions found.")
        return

    # 4. Processing Setup
    std_list = list(std_hashes)
    others_list = list(all_hashes - std_hashes)

    token_manager = TokenManager(token_path=token_path, w3=w3)
    results = []

    # RPC Rate Limit Protection
    semaphore = asyncio.Semaphore(5)

    # Helper function to wrap the sync analysis in a thread with a semaphore
    async def analyze_wrapper(tx_hash: str, fetch_meta: bool) -> dict[str, Any] | None:
        async with semaphore:
            # Run the synchronous function in a separate thread
            return await asyncio.to_thread(
                analyze_transaction,
                tx_hash=tx_hash,
                w3=w3,
                my_address=my_address,
                token_manager=token_manager,
                internal_eth_map=internal_map,
                fetch_metadata=fetch_meta,
            )

    # 5. Phase A: Process Standard Transactions
    # We process these first and ALLOW fetching metadata (updating token DB)
    if std_list:
        print(f"Processing {len(std_list)} Standard TXs (Async)...")
        tasks_std = [analyze_wrapper(tx_hash=tx, fetch_meta=True) for tx in std_list]

        # tqdm_asyncio.gather displays a progress bar for async tasks
        batch_results = await tqdm_asyncio.gather(*tasks_std, desc="Standard TXs", unit="tx")
        results.extend([r for r in batch_results if r])

    # 6. Phase B: Process Passive Transactions
    # We process these second and DENY fetching metadata (use only cached tokens)
    if others_list:
        print(f"Processing {len(others_list)} Passive TXs (Async)...")
        tasks_others = [analyze_wrapper(tx_hash=tx, fetch_meta=False) for tx in others_list]

        batch_results = await tqdm_asyncio.gather(*tasks_others, desc="Passive TXs", unit="tx")
        results.extend([r for r in batch_results if r])

    # 7. Export
    try:
        if results:
            new_results_df = _normalize_results_frame(pd.DataFrame(results))

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                existing_results_df = _normalize_results_frame(pd.read_csv(output_path, dtype=str))
                results_df = pd.concat([existing_results_df, new_results_df]).drop_duplicates(
                    subset=["TX Hash"], keep="last"
                )
            else:
                results_df = new_results_df

            results_df["_sort_helper"] = _parse_transaction_datetime_series(results_df["Date"])
            results_df = results_df.sort_values(by="_sort_helper", ascending=True).drop(
                columns=["_sort_helper"]
            )
            results_df = _normalize_results_frame(results_df)
            results_df.to_csv(output_path, index=False)
            print(f"Done. Saved {len(results_df)} rows to {output_path}")
        else:
            print("No results generated.")
    finally:
        token_manager.flush()


if __name__ == "__main__":
    asyncio.run(
        retrieve_transactions(chain="arbitrum", start_date="01/01/2020", end_date="01/02/2026")
    )
