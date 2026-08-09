from __future__ import annotations

from datetime import date, datetime

import pandas as pd

TRANSACTION_DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DAILY_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_TRANSACTION_INPUT_FORMATS = (TRANSACTION_DATETIME_FORMAT, "%d/%m/%Y %H:%M")
_DAILY_INPUT_FORMATS = (
    DAILY_DATETIME_FORMAT,
    "%Y-%m-%d",
    *_TRANSACTION_INPUT_FORMATS,
)


def _parse(value: object, formats: tuple[str, ...]) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    text = str(value or "").strip()
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            pass
    return None


def parse_transaction_datetime(value: object) -> datetime | None:
    return _parse(value, _TRANSACTION_INPUT_FORMATS)


def parse_transaction_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.map(parse_transaction_datetime), errors="coerce")


def parse_daily_datetime(value: object) -> datetime | None:
    return _parse(value, _DAILY_INPUT_FORMATS)


def format_daily_datetime(value: object) -> str:
    parsed = parse_daily_datetime(value)
    if parsed is None:
        raise ValueError(f"Invalid daily datetime: {value}")
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
        DAILY_DATETIME_FORMAT
    )
