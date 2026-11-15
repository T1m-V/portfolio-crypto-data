import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from portfolio_core import active_context, atomic_write_csv
from tqdm.asyncio import tqdm_asyncio
from web3 import Web3

from portfolio_crypto_data.chain_config import EvmChainConfig
from portfolio_crypto_data.datetime_utils import (
    TRANSACTION_DATETIME_FORMAT,
    parse_transaction_datetime,
    parse_transaction_datetime_series,
)
from portfolio_crypto_data.extraction.token_manager import TokenManager
from portfolio_crypto_data.extraction.transaction_analyzer import analyze_transaction

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


@dataclass(frozen=True)
class TransactionAnalysisFailure:
    """Machine-readable details for a transaction that could not be analyzed."""

    tx_hash: str
    error_type: str
    message: str


@dataclass(frozen=True)
class TransactionRetrievalReport:
    """Summary returned after processing every transaction in a refresh."""

    chain: str
    transaction_count: int
    failures: tuple[TransactionAnalysisFailure, ...]

    @property
    def successful_count(self) -> int:
        return self.transaction_count - len(self.failures)


class ExplorerAPIError(RuntimeError):
    """Raised when an explorer request cannot be completed reliably."""


def _fetch_explorer_data(
    api_url: str,
    params: dict[str, Any],
    result_key: str = "result",
    timeout_seconds: int = 20,
    max_retries: int = 3,
) -> list[Any]:
    """
    Executes a GET request to the Explorer API (Etherscan v2 schema).

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
                action = params.get("action")
                raise ExplorerAPIError(
                    f"Explorer request failed for action={action}: {e}"
                ) from e
            time.sleep(0.4 * (2**attempt))
            continue

        if not isinstance(data, dict):
            if attempt == max_retries:
                action = params.get("action")
                raise ExplorerAPIError(
                    "Unexpected explorer response "
                    f"for action={action}: expected an object, got {type(data).__name__}."
                )
            time.sleep(0.4 * (2**attempt))
            continue

        status = data.get("status")
        message = str(data.get("message", "")).lower()
        result = data.get(result_key, [])
        result_str = str(result).lower()

        if status == "1" and isinstance(result, list):
            return result

        if status == "0" and "no transactions found" in (message + result_str):
            return []

        if attempt == max_retries:
            action = params.get("action")
            raise ExplorerAPIError(
                "Unexpected explorer response "
                f"for action={action}: status={status!r}, message={data.get('message')!r}"
            )
        time.sleep(0.4 * (2**attempt))

    return []


def _fetch_cronos_table_data(
    endpoint_url: str,
    address: str,
    start_ts: int = 0,
    timeout_seconds: int = 20,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    """
    Fetches paginated transactions from explorer.cronos.org/table/* Next.js SSR endpoints.
    """
    all_rows: list[dict[str, Any]] = []
    page = 1
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    while True:
        params = {"address": address, "p": page}
        data = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(
                    endpoint_url,
                    params=params,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                resp.raise_for_status()
                match = re.search(
                    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                    resp.text,
                    re.DOTALL,
                )
                if match:
                    nd = json.loads(match.group(1))
                    data = nd.get("props", {}).get("pageProps", {}).get("data", {})
                else:
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                break
            except Exception as e:
                if attempt == max_retries:
                    raise ExplorerAPIError(
                        f"Cronos explorer request failed for {endpoint_url}: {e}"
                    ) from e
                time.sleep(0.5 * (2**attempt))

        if not isinstance(data, dict):
            break
        rows = data.get("result") or data.get("data")
        if not isinstance(rows, list) or not rows:
            break
        all_rows.extend(rows)
        if start_ts > 0:
            timestamps = [_safe_timestamp(tx) for tx in rows if _safe_timestamp(tx) > 0]
            if timestamps and min(timestamps) < start_ts:
                break
        pag = data.get("pagination") or {}
        total_page = pag.get("totalPage", 1)
        if page >= total_page:
            break
        page += 1
        time.sleep(0.1)

    return all_rows


def _parse_input_date_to_utc(date_str: str) -> datetime:
    parsed = parse_transaction_datetime(date_str)
    if parsed is None:
        raise ValueError(f"Invalid date input: {date_str}")

    return parsed.replace(tzinfo=timezone.utc)


def _derive_start_date(output_path: Path, overlap_days: int = 1) -> str:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return DEFAULT_START_DATE
    try:
        df_dates = pd.read_csv(output_path, usecols=["Date"])
        if df_dates.empty:
            return DEFAULT_START_DATE
        latest = parse_transaction_datetime_series(df_dates["Date"]).max()
        if pd.isna(latest):
            return DEFAULT_START_DATE
        start_dt = (latest - timedelta(days=overlap_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start_dt.strftime(TRANSACTION_DATETIME_FORMAT)
    except (ValueError, KeyError, pd.errors.ParserError):
        return DEFAULT_START_DATE


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
        return int(tx.get("timeStamp") or tx.get("blockTimestamp") or tx.get("timestamp") or 0)
    except (TypeError, ValueError):
        return 0


def get_all_transaction_hashes(
    *,
    api_url: str,
    api_key: str,
    chain_id: str,
    address: str,
    start_ts: int,
    end_ts: int,
    include_chain_id: bool,
    supplementary_hashes: tuple[str, ...] | list[str] | set[str] = (),
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
    is_cronos = "explorer.cronos.org" in api_url.lower()

    if is_cronos:
        base_url = api_url.rstrip("/")
        if not base_url.endswith("/table"):
            base_url = f"{base_url}/table"

        std_rows = _fetch_cronos_table_data(f"{base_url}/getTransactions", address, start_ts=start_ts)
        tok_rows = _fetch_cronos_table_data(f"{base_url}/addressTransfers", address, start_ts=start_ts)
        int_rows = _fetch_cronos_table_data(f"{base_url}/getInternalTransactions", address, start_ts=start_ts)

        hashes_std = set()
        for tx in std_rows:
            tx_hash = tx.get("hash") or tx.get("transactionHash")
            ts = _safe_timestamp(tx)
            if tx_hash and start_ts <= ts <= end_ts:
                hashes_std.add(tx_hash)

        hashes_tok = set()
        for tx in tok_rows:
            tx_hash = tx.get("transactionHash") or tx.get("hash")
            ts = _safe_timestamp(tx)
            if tx_hash and start_ts <= ts <= end_ts:
                hashes_tok.add(tx_hash)

        hashes_int = set()
        normalized_int_txs = []
        seen_internal = set()
        for tx in int_rows:
            tx_hash = tx.get("transactionHash") or tx.get("hash")
            if not tx_hash:
                continue
            idx = str(tx.get("index", 0))
            dedup_key = (tx_hash.lower(), idx)
            if dedup_key in seen_internal:
                continue
            seen_internal.add(dedup_key)

            ts = _safe_timestamp(tx)
            if start_ts <= ts <= end_ts:
                hashes_int.add(tx_hash)
                to_val = tx.get("to")
                to_addr = to_val.get("address") if isinstance(to_val, dict) else str(to_val or "")
                normalized_int_txs.append(
                    {
                        "hash": tx_hash,
                        "to": to_addr,
                        "value": tx.get("value", "0"),
                    }
                )

        supplementary = set(supplementary_hashes)
        all_hashes = hashes_std | hashes_tok | hashes_int | supplementary
        return hashes_std, all_hashes, normalized_int_txs

    base_params: dict[str, Any] = {
        "module": "account",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key,
    }
    if include_chain_id:
        base_params["chainid"] = chain_id

    # 1. Standard TX List
    p_std = {**base_params, "action": "txlist"}
    txs_std = _fetch_explorer_data(api_url=api_url, params=p_std)
    hashes_std = {tx["hash"] for tx in txs_std if start_ts <= _safe_timestamp(tx) <= end_ts}

    # 2. Token Transfers (ERC20)
    p_tok = {**base_params, "action": "tokentx"}
    txs_tok = _fetch_explorer_data(api_url=api_url, params=p_tok)
    hashes_tok = {tx["hash"] for tx in txs_tok if start_ts <= _safe_timestamp(tx) <= end_ts}

    # 3. Internal Transactions
    p_int = {**base_params, "action": "txlistinternal"}
    txs_int = _fetch_explorer_data(api_url=api_url, params=p_int)
    hashes_int = {tx["hash"] for tx in txs_int if start_ts <= _safe_timestamp(tx) <= end_ts}

    all_hashes = hashes_std | hashes_tok | hashes_int | set(supplementary_hashes)
    return hashes_std, all_hashes, txs_int


def build_internal_native_map(
    *,
    txs_internal: list[dict],
    my_address: str,
    native_decimals: int,
) -> dict[str, Decimal]:
    """
    Maps internal transaction hashes to native-asset values.

    args:
        txs_internal: List of raw internal transactions.
        my_address: User's wallet address.
        native_decimals: Decimal precision of the chain's native asset.

    returns:
        Dictionary mapping transaction hashes to native-asset amounts.
    """
    internal_map: dict[str, Decimal] = {}
    for tx in txs_internal:
        value = Decimal(str(tx.get("value", "0")))
        to_addr = tx.get("to")
        if isinstance(to_addr, dict):
            to_addr = to_addr.get("address")
        to_str = str(to_addr or "")
        if to_str and to_str.lower() == my_address.lower() and value > 0:
            tx_hash = tx.get("hash") or tx.get("transactionHash")
            if tx_hash:
                amount = value / (Decimal(10) ** native_decimals)
                internal_map[tx_hash] = internal_map.get(tx_hash, Decimal(0)) + amount
    return internal_map


async def retrieve_transactions(config: EvmChainConfig) -> TransactionRetrievalReport:
    """
    Main entry point to fetch and analyze transactions for a chain.

    args:
        config: Runtime configuration for one EVM chain.
    """
    chain = config.chain
    print(f"--- START PROCESSING: {chain.upper()} ---")

    paths = active_context().paths
    my_address = config.wallet_address

    # Setup Paths & Connection
    token_path = paths.tokens / f"{chain}_tokens.json"
    output_path = paths.crypto_transactions / f"{chain}_transactions.csv"
    paths.crypto_transactions.mkdir(parents=True, exist_ok=True)
    paths.tokens.mkdir(parents=True, exist_ok=True)

    w3 = Web3(Web3.HTTPProvider(config.rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"RPC connection failed for chain '{chain}'.")

    end_date = datetime.now(tz=timezone.utc).strftime(TRANSACTION_DATETIME_FORMAT)
    start_date = _derive_start_date(output_path=output_path, overlap_days=1)

    # 2. Parse Dates
    start_ts = int(_parse_input_date_to_utc(start_date).timestamp())
    end_ts = int(_parse_input_date_to_utc(end_date).timestamp())

    # 3. Fetch Hashes
    print("Fetching transaction lists...")
    std_hashes, all_hashes, raw_internal_txs = get_all_transaction_hashes(
        api_url=config.explorer_api_url,
        api_key=config.explorer_api_key,
        chain_id=config.chain_id,
        address=my_address,
        start_ts=start_ts,
        end_ts=end_ts,
        include_chain_id=config.explorer_include_chain_id,
        supplementary_hashes=config.supplementary_transactions,
    )

    internal_map = build_internal_native_map(
        txs_internal=raw_internal_txs,
        my_address=my_address,
        native_decimals=config.native_decimals,
    )

    print(f"-> Found {len(all_hashes)} unique transactions.")
    if not all_hashes:
        print("No transactions found.")
        return TransactionRetrievalReport(chain=chain, transaction_count=0, failures=())

    # 4. Processing Setup
    std_list = list(std_hashes)
    others_list = list(all_hashes - std_hashes)

    token_manager = TokenManager(
        token_path=token_path,
        w3=w3,
        native_symbol=config.native_symbol,
        native_decimals=config.native_decimals,
    )
    results = []
    failures: list[TransactionAnalysisFailure] = []

    overrides_path = paths.crypto_transactions / f"{chain}_overrides.json"
    tx_overrides: dict[str, dict[str, Any]] = {}
    if overrides_path.exists():
        try:
            overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
            tx_overrides = overrides_data.get("transaction_overrides", {})
        except Exception as e:
            print(f"Warning: could not read overrides from {overrides_path}: {e}")

    semaphore = asyncio.Semaphore(config.rpc_concurrency)

    # Wrap sync analysis with retry/backoff on RPC rate limits
    async def analyze_wrapper(
        tx_hash: str,
        fetch_meta: bool,
    ) -> tuple[str, dict[str, Any] | None, Exception | None]:
        async with semaphore:
            for attempt in range(5):
                try:
                    result = await asyncio.to_thread(
                        analyze_transaction,
                        tx_hash=tx_hash,
                        w3=w3,
                        my_address=my_address,
                        token_manager=token_manager,
                        internal_native_map=internal_map,
                        fetch_metadata=fetch_meta,
                        native_symbol=config.native_symbol,
                        native_decimals=config.native_decimals,
                        transaction_overrides=tx_overrides,
                    )
                    return tx_hash, result, None
                except Exception as exc:
                    if "429" in str(exc) and attempt < 4:
                        await asyncio.sleep(1.0 * (2**attempt))
                        continue
                    if attempt == 4:
                        return tx_hash, None, exc
                    await asyncio.sleep(0.5)
            return tx_hash, None, RuntimeError(f"Exhausted retries for {tx_hash}")

    async def analyze_batch(
        tx_hashes: list[str],
        *,
        fetch_meta: bool,
        description: str,
    ) -> tuple[list[dict[str, Any]], list[TransactionAnalysisFailure]]:
        outcomes = await tqdm_asyncio.gather(
            *(analyze_wrapper(tx_hash=tx_hash, fetch_meta=fetch_meta) for tx_hash in tx_hashes),
            desc=description,
            unit="tx",
        )
        batch_failures = [
            TransactionAnalysisFailure(
                tx_hash=tx_hash,
                error_type=type(error).__name__,
                message=str(error),
            )
            for tx_hash, _, error in outcomes
            if error is not None
        ]
        batch_results = [result for _, result, _ in outcomes if result is not None]
        return batch_results, batch_failures

    try:
        # 5. Phase A: Process Standard Transactions
        if std_list:
            print(f"Processing {len(std_list)} Standard TXs (Async)...")
            batch_results, batch_failures = await analyze_batch(
                std_list,
                fetch_meta=True,
                description="Standard TXs",
            )
            results.extend(batch_results)
            failures.extend(batch_failures)

        # 6. Phase B: Process Passive Transactions
        if others_list:
            print(f"Processing {len(others_list)} Passive TXs (Async)...")
            batch_results, batch_failures = await analyze_batch(
                others_list,
                fetch_meta=False,
                description="Passive TXs",
            )
            results.extend(batch_results)
            failures.extend(batch_failures)

        # 7. Export every successfully analyzed transaction.
        if results:
            new_results_df = _normalize_results_frame(pd.DataFrame(results))

            if output_path.exists() and output_path.stat().st_size > 0:
                existing_results_df = _normalize_results_frame(pd.read_csv(output_path, dtype=str))
                results_df = pd.concat([existing_results_df, new_results_df]).drop_duplicates(
                    subset=["TX Hash"], keep="last"
                )
            else:
                results_df = new_results_df

            results_df["_sort_helper"] = parse_transaction_datetime_series(results_df["Date"])
            results_df = results_df.sort_values(by="_sort_helper", ascending=True).drop(
                columns=["_sort_helper"]
            )
            results_df = _normalize_results_frame(results_df)
            atomic_write_csv(frame=results_df, path=output_path)
            print(f"Done. Saved {len(results_df)} rows to {output_path}")
        else:
            print("No results generated.")

        if failures:
            print(f"[!] Failed to analyze {len(failures)} transaction(s):")
            for failure in failures:
                print(
                    f"[!] {failure.tx_hash}: "
                    f"{failure.error_type}: {failure.message}"
                )
    finally:
        token_manager.flush()

    return TransactionRetrievalReport(
        chain=chain,
        transaction_count=len(all_hashes),
        failures=tuple(failures),
    )
