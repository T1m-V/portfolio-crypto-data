import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from web3 import Web3

from blockchain_reader.datetime_utils import (
    format_daily_datetime,
    parse_daily_datetime,
)
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
    snapshots_file_path = BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv"
    if not os.path.exists(snapshots_file_path):
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
    map_file_path = BLOCKCHAIN_BLOCK_MAP_FOLDER / f"block_map_{chain}.csv"
    block_map = {}
    if os.path.exists(path=map_file_path):
        with open(file=map_file_path, mode="r") as f:
            reader = csv.DictReader(f=f)
            for row in reader:
                try:
                    normalized = format_daily_datetime(row["date"])
                    block_map[normalized] = int(row["block"])
                except (ValueError, TypeError):
                    continue
    return block_map


def protocol_history_output_path(protocol: str, chain: str, symbol: str) -> Path:
    return PROTOCOL_UNDERLYING_TOKEN_FOLDER / protocol / f"{chain}_{symbol}.csv"


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


def get_output_max_processed_date(protocol: str, chain: str, symbol: str) -> datetime | None:
    output_file = protocol_history_output_path(protocol=protocol, chain=chain, symbol=symbol)
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


def resolve_effective_start_date(
    *,
    protocol: str,
    chain: str,
    symbol: str,
    explicit_start_date: str | None,
    fallback_start_date: str | None,
) -> str | None:
    if explicit_start_date:
        return format_daily_datetime(explicit_start_date)

    normalized_fallback_start: str | None = None
    if fallback_start_date:
        normalized_fallback_start = format_daily_datetime(fallback_start_date)

    max_processed_date = get_output_max_processed_date(
        protocol=protocol,
        chain=chain,
        symbol=symbol,
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


def write_protocol_history_csv(
    protocol: str,
    chain: str,
    symbol: str,
    history_data: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> Path | None:
    if not history_data:
        return None

    output_file = protocol_history_output_path(protocol=protocol, chain=chain, symbol=symbol)
    os.makedirs(output_file.parent, exist_ok=True)

    existing_rows = _read_existing_history_rows(output_file=output_file)
    existing_by_date: dict[str, dict[str, object]] = {}
    for row in existing_rows:
        date_key, normalized_row = _normalize_history_row(row=row)
        if date_key is None:
            continue
        # Existing rows win when incoming data overlaps on the same date.
        existing_by_date.setdefault(date_key, normalized_row)

    incoming_by_date: dict[str, dict[str, object]] = {}
    for row in history_data:
        date_key, normalized_row = _normalize_history_row(row=row)
        if date_key is None:
            continue
        incoming_by_date[date_key] = normalized_row

    merged_by_date = dict(existing_by_date)
    for date_key, row in incoming_by_date.items():
        if date_key not in merged_by_date:
            merged_by_date[date_key] = row

    if not merged_by_date:
        return None

    merged_rows = [merged_by_date[date_key] for date_key in sorted(merged_by_date)]
    keys = set().union(*(d.keys() for d in merged_rows))
    ordered_fieldnames = sorted(list(keys))
    if fieldnames:
        preferred = [name for name in fieldnames if name in keys]
        remaining = sorted([name for name in keys if name not in set(preferred)])
        ordered_fieldnames = preferred + remaining

    with open(file=output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f=f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return output_file
