import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from portfolio_core import active_context, atomic_write_csv, atomic_write_text
from web3 import Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

from portfolio_crypto_data.chain_config import EvmChainConfig, load_evm_chain_config
from portfolio_crypto_data.datetime_utils import (
    format_daily_datetime,
    parse_daily_datetime,
    parse_transaction_datetime_series,
)
from portfolio_crypto_data.pipeline_logging import PipelineLogger

ACTIVE_PROTOCOL_QUANTITY_THRESHOLD = Decimal("0.0001")


def is_transient_rpc_error(exc: Exception) -> bool:
    if isinstance(exc, (ContractLogicError, BadFunctionCallOutput)):
        return False

    msg = str(exc).lower()
    if any(
        hint in msg
        for hint in ("429", "rate limit", "too many requests", "timeout", "timed out", "retry")
    ):
        return True

    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True

    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return code in {429, 500, 502, 503, 504}

    if isinstance(exc, (TimeoutError, ConnectionResetError, ConnectionRefusedError)):
        return True

    return False


def call_with_retry(
    fn: Any, *args: Any, max_retries: int = 5, backoff: float = 0.5, **kwargs: Any
) -> Any:
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_transient_rpc_error(exc):
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff * (2**attempt))


@dataclass(frozen=True)
class ProtocolTokenRun:
    address: str
    symbol: str
    start_date: str
    end_date: str


def load_chain_config(chain: str) -> EvmChainConfig:
    return load_evm_chain_config(chain=chain)


def load_chain_web3(chain: str, config: EvmChainConfig | None = None) -> Web3:
    cfg = config or load_chain_config(chain=chain)
    w3 = Web3(provider=Web3.HTTPProvider(endpoint_uri=cfg.rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Connection failed for chain '{chain}'.")
    return w3


def load_tokens(chain: str) -> dict[str, dict[str, Any]]:
    tokens_file_path = active_context().paths.tokens / f"{chain}_tokens.json"
    if not tokens_file_path.exists():
        raise FileNotFoundError(f"Config '{tokens_file_path}' not found.")

    with open(file=tokens_file_path, mode="r", encoding="utf-8") as f:
        return json.load(fp=f)


def load_snapshot_ranges(chain: str) -> dict[str, dict[str, object]]:
    snapshots_file_path = active_context().paths.crypto_snapshots / f"{chain}_raw_snapshots.csv"
    if not snapshots_file_path.exists():
        raise FileNotFoundError(f"Snapshots '{snapshots_file_path}' not found.")

    df = pd.read_csv(snapshots_file_path)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Date"] = pd.to_datetime(df["Date"].map(parse_daily_datetime), errors="coerce")
    df = df.dropna(subset=["Date"])
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
    parsed_start = parse_daily_datetime(start_date)
    if parsed_start is None:
        raise ValueError(f"Invalid start_date: {start_date}")
    start_dt = parsed_start.replace(tzinfo=timezone.utc)

    if end_date.lower() == "now":
        end_dt = datetime.now(tz=timezone.utc)
    else:
        parsed_end = parse_daily_datetime(end_date)
        if parsed_end is None:
            raise ValueError(f"Invalid end_date: {end_date}")
        end_dt = parsed_end.replace(tzinfo=timezone.utc)
    return start_dt, end_dt


def load_block_map(chain: str) -> dict[str, int]:
    map_file_path = active_context().paths.block_map / f"block_map_{chain}.csv"
    block_map = {}
    if map_file_path.exists():
        with open(file=map_file_path, mode="r") as f:
            reader = csv.DictReader(f=f)
            for row in reader:
                try:
                    normalized = format_daily_datetime(row["date"])
                    block_map[normalized] = int(row["block"])
                except (ValueError, TypeError):
                    continue
    return block_map


def sync_block_map(chain: str, config: EvmChainConfig | None = None) -> dict[str, int]:
    """
    Ensures block_map_{chain}.csv is up to date through today.
    """
    map_file_path = active_context().paths.block_map / f"block_map_{chain}.csv"
    block_map = load_block_map(chain=chain)

    today_dt = datetime.now(tz=timezone.utc).date()
    if block_map:
        last_date_str = max(block_map.keys())
        parsed_last = parse_daily_datetime(last_date_str)
        if parsed_last is not None and parsed_last.date() >= today_dt:
            return block_map

    try:
        w3 = load_chain_web3(chain=chain, config=config)
    except Exception:
        return block_map

    def safe_get_block(identifier: Any) -> Any:
        for attempt in range(6):
            try:
                return w3.eth.get_block(identifier)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(0.5 * (2**attempt))

    try:
        latest_b = safe_get_block("latest")
        latest_block = int(latest_b.number)
        latest_ts = int(latest_b.timestamp)
    except Exception:
        return block_map

    if not block_map:
        start_date = today_dt - timedelta(days=30)
        tx_path = active_context().paths.crypto_transactions / f"{chain}_transactions.csv"
        if tx_path.exists():
            try:
                df = pd.read_csv(tx_path, usecols=["Date"])
                min_dt = parse_transaction_datetime_series(df["Date"]).min()
                if pd.notna(min_dt):
                    start_date = min_dt.date()
            except Exception:
                pass
        curr_block = 1
    else:
        last_date_str = max(block_map.keys())
        parsed_last = parse_daily_datetime(last_date_str)
        assert parsed_last is not None
        start_date = parsed_last.date() + timedelta(days=1)
        curr_block = block_map[last_date_str]

    if start_date > today_dt:
        return block_map

    curr = start_date
    while curr <= today_dt:
        target_dt = datetime.combine(curr, datetime.min.time(), tzinfo=timezone.utc)
        target_ts = int(target_dt.timestamp())
        if target_ts >= latest_ts:
            block = latest_block
        else:
            try:
                curr_b = safe_get_block(curr_block)
                curr_ts = int(curr_b.timestamp)
                if curr_ts >= latest_ts or curr_block >= latest_block:
                    block = latest_block
                else:
                    ratio = (target_ts - curr_ts) / (latest_ts - curr_ts)
                    block = min(
                        latest_block,
                        max(curr_block, int(curr_block + ratio * (latest_block - curr_block))),
                    )
            except Exception:
                block = curr_block
        date_str = f"{curr.strftime('%Y-%m-%d')} 00:00:00"
        block_map[date_str] = block
        curr_block = block
        curr += timedelta(days=1)

    frame = pd.DataFrame(
        [{"date": ds, "block": block_map[ds]} for ds in sorted(block_map)],
        columns=["date", "block"],
    )
    atomic_write_csv(frame=frame, path=map_file_path)

    return block_map


def protocol_history_output_path(
    protocol: str,
    chain: str,
    symbol: str,
    *,
    protocol_root: Path | None = None,
) -> Path:
    root = protocol_root or active_context().paths.protocol_underlying_tokens
    return root / protocol / f"{chain}_{symbol}.csv"


def _parse_history_date(raw_value: object) -> datetime | None:
    if isinstance(raw_value, datetime):
        return raw_value

    value = str(raw_value or "").strip()
    if not value:
        return None

    return parse_daily_datetime(value)


def _normalize_history_date(raw_value: object) -> str | None:
    parsed = _parse_history_date(raw_value=raw_value)
    if parsed is None:
        return None
    return format_daily_datetime(parsed)


def get_output_max_processed_date(
    protocol: str,
    chain: str,
    symbol: str,
    *,
    protocol_root: Path | None = None,
) -> datetime | None:
    output_file = protocol_history_output_path(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    if not output_file.exists():
        return None

    with open(file=output_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f=f)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            return None

        max_date: datetime | None = None
        for row in reader:
            parsed = _parse_history_date(raw_value=row.get("date"))
            if parsed is None:
                continue
            if max_date is None or parsed > max_date:
                max_date = parsed
        return max_date


def earliest_unresolved_error_date(
    protocol: str,
    chain: str,
    symbol: str,
    *,
    protocol_root: Path | None = None,
) -> datetime | None:
    output_file = protocol_history_output_path(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    if not output_file.exists():
        return None

    with open(file=output_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f=f)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            return None

        min_error_date: datetime | None = None
        for row in reader:
            parsed = _parse_history_date(raw_value=row.get("date"))
            if parsed is None:
                continue
            if int(row.get("rpc_error_count") or 0) > 0:
                if min_error_date is None or parsed < min_error_date:
                    min_error_date = parsed
        return min_error_date


def resolve_effective_start_date(
    *,
    protocol: str,
    chain: str,
    symbol: str,
    fallback_start_date: str | None,
    protocol_root: Path | None = None,
) -> str | None:
    normalized_fallback_start: str | None = None
    if fallback_start_date:
        normalized_fallback_start = format_daily_datetime(fallback_start_date)

    error_date = earliest_unresolved_error_date(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    if error_date is not None:
        inferred_error_start = format_daily_datetime(error_date)
        if normalized_fallback_start:
            return max(inferred_error_start, normalized_fallback_start)
        return inferred_error_start

    max_processed_date = get_output_max_processed_date(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    inferred_start_date: str | None = None
    if max_processed_date is not None:
        inferred_start_date = format_daily_datetime(max_processed_date + timedelta(days=1))

    if inferred_start_date and normalized_fallback_start:
        return max(inferred_start_date, normalized_fallback_start)
    return inferred_start_date or normalized_fallback_start


def should_skip_date_window(start_date: str | None, end_date: str | None) -> bool:
    if not start_date or not end_date:
        return False
    if str(end_date).lower() == "now":
        return False

    start = _parse_history_date(raw_value=start_date)
    end = _parse_history_date(raw_value=end_date)
    if start is None or end is None:
        return False
    return start > end


def is_active_protocol_quantity(quantity: object) -> bool:
    try:
        value = Decimal(str(quantity))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return value > ACTIVE_PROTOCOL_QUANTITY_THRESHOLD


def resolve_protocol_end_date(range_info: dict[str, object] | None) -> str:
    if range_info is None:
        return "now"

    # Tiny positive balances are common dust after closing protocol positions; only
    # a material positive balance should extend non-Aave protocol queries to today.
    if is_active_protocol_quantity(range_info.get("qty")):
        return "now"
    return format_daily_datetime(range_info["end"])


def protocol_token_runs(
    *,
    protocol: str,
    chain: str,
    logger: PipelineLogger,
) -> list[ProtocolTokenRun]:
    """Return configured tokens that still need protocol-history rows."""
    ranges = load_snapshot_ranges(chain=chain)
    runs: list[ProtocolTokenRun] = []
    for address, metadata in load_tokens(chain=chain).items():
        if metadata.get("protocol") != protocol:
            continue

        symbol = metadata.get("symbol", address)
        snapshot_range = ranges.get(symbol)
        if snapshot_range is None:
            logger.protocol_skip(protocol, symbol, "no snapshot data found")
            continue

        start_date = resolve_effective_start_date(
            protocol=protocol,
            chain=chain,
            symbol=symbol,
            fallback_start_date=format_daily_datetime(snapshot_range["start"]),
        )
        end_date = resolve_protocol_end_date(snapshot_range)
        if start_date is None or should_skip_date_window(start_date, end_date):
            logger.protocol_skip(protocol, symbol, f"start={start_date} is after end={end_date}")
            continue

        runs.append(
            ProtocolTokenRun(
                address=address,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
        )
    return runs


def _normalize_history_row(row: dict[str, object]) -> tuple[str | None, dict[str, object]]:
    normalized = dict(row)
    date_key = _normalize_history_date(raw_value=normalized.get("date"))
    if date_key is not None:
        normalized["date"] = date_key
    return date_key, normalized


def _read_existing_history_rows(output_file: Path) -> list[dict[str, object]]:
    if not output_file.exists():
        return []

    with open(file=output_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f=f)
        return [dict(row) for row in reader]


def load_existing_clean_history_by_date(
    protocol: str,
    chain: str,
    symbol: str,
    *,
    protocol_root: Path | None = None,
) -> dict[str, dict[str, object]]:
    output_file = protocol_history_output_path(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    if not output_file.exists():
        return {}

    clean_by_date: dict[str, dict[str, object]] = {}
    for row in _read_existing_history_rows(output_file=output_file):
        date_key, normalized = _normalize_history_row(row=row)
        if not date_key:
            continue
        err_count = int(normalized.get("rpc_error_count") or 0)
        if err_count > 0:
            continue
        if protocol == "aave":
            has_val = any(
                k.startswith(("supply_", "debt_")) and normalized.get(k) is not None
                for k in normalized
            )
            if not has_val and int(normalized.get("queried_token_count") or 0) > 0:
                continue
        clean_by_date[date_key] = normalized
    return clean_by_date


def write_protocol_history_csv(
    protocol: str,
    chain: str,
    symbol: str,
    history_data: list[dict[str, object]],
    fieldnames: list[str] | None = None,
    protocol_root: Path | None = None,
) -> Path | None:
    output_file = protocol_history_output_path(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
        protocol_root=protocol_root,
    )
    if not history_data:
        return None
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_by_date: dict[str, dict[str, object]] = {}
    for row in [*_read_existing_history_rows(output_file=output_file), *history_data]:
        date_key, normalized_row = _normalize_history_row(row=row)
        if date_key is not None:
            merged_by_date[date_key] = normalized_row

    if not merged_by_date:
        return None

    merged_rows = [merged_by_date[date_key] for date_key in sorted(merged_by_date)]
    keys = set().union(*(d.keys() for d in merged_rows))
    ordered_fieldnames = sorted(list(keys))
    if fieldnames:
        preferred = [name for name in fieldnames if name in keys]
        remaining = sorted([name for name in keys if name not in set(preferred)])
        ordered_fieldnames = preferred + remaining

    buffer = StringIO(newline="")
    writer = csv.DictWriter(f=buffer, fieldnames=ordered_fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)
    atomic_write_text(text=buffer.getvalue(), path=output_file)
    return output_file
