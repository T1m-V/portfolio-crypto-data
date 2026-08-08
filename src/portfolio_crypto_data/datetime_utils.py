from __future__ import annotations

from datetime import date, datetime

import pandas as pd

TRANSACTION_DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"
DAILY_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_datetime_value(
    value: object,
    *,
    formats: tuple[str, ...],
    dayfirst_fallback: bool,
) -> datetime | None:
    """
    Parses a single datetime value with explicit formats first, then pandas fallback.

    args:
        value: Raw value to parse.
        formats: Ordered datetime formats to try first.
        dayfirst_fallback: Whether pandas fallback should parse with day-first semantics.

    returns:
        Parsed datetime, or None when parsing fails.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    text = str(value or "").strip()
    if not text:
        return None

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst_fallback)
    if pd.isna(parsed):
        return None

    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return None


def parse_datetime_series(
    series: pd.Series,
    *,
    formats: tuple[str, ...],
    dayfirst_fallback: bool,
) -> pd.Series:
    """
    Parses a datetime series while tolerating legacy formats.

    args:
        series: Series to parse.
        formats: Ordered datetime formats to try first.
        dayfirst_fallback: Whether pandas fallback should parse with day-first semantics.

    returns:
        Parsed datetime series with NaT for unparsable values.
    """
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for fmt in formats:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], format=fmt, errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            errors="coerce",
            dayfirst=dayfirst_fallback,
        )

    return parsed


def parse_transaction_datetime(value: object) -> datetime | None:
    """
    Parses transaction datetime input.

    args:
        value: Raw transaction datetime value.

    returns:
        Parsed datetime, or None when parsing fails.
    """
    return parse_datetime_value(
        value=value,
        formats=(
            TRANSACTION_DATETIME_FORMAT,
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ),
        dayfirst_fallback=True,
    )


def parse_transaction_datetime_series(series: pd.Series) -> pd.Series:
    """
    Parses transaction datetimes with support for legacy precision.

    args:
        series: Raw transaction datetime series.

    returns:
        Parsed datetime series with NaT for unparsable values.
    """
    return parse_datetime_series(
        series=series,
        formats=(
            TRANSACTION_DATETIME_FORMAT,
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ),
        dayfirst_fallback=True,
    )


def parse_daily_datetime(value: object) -> datetime | None:
    """
    Parses a day-granular blockchain datetime value.

    args:
        value: Raw date/datetime value.

    returns:
        Parsed datetime, or None when parsing fails.
    """
    return parse_datetime_value(
        value=value,
        formats=(
            DAILY_DATETIME_FORMAT,
            "%Y-%m-%d",
            TRANSACTION_DATETIME_FORMAT,
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ),
        dayfirst_fallback=False,
    )


def normalize_to_midnight(value: object) -> datetime | None:
    """
    Normalizes a date-like value to midnight.

    args:
        value: Input date or datetime value.

    returns:
        Midnight datetime, or None when parsing fails.
    """
    parsed = parse_daily_datetime(value=value)
    if parsed is None:
        return None
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def format_transaction_datetime(value: object) -> str:
    """
    Formats a transaction datetime with second precision.

    args:
        value: Datetime-like input.

    returns:
        Canonical transaction datetime string.
    """
    parsed = parse_transaction_datetime(value=value)
    if parsed is None:
        raise ValueError(f"Invalid transaction datetime: {value}")
    return parsed.strftime(TRANSACTION_DATETIME_FORMAT)


def format_daily_datetime(value: object) -> str:
    """
    Formats a day-granular datetime with second precision at midnight.

    args:
        value: Datetime-like input.

    returns:
        Canonical daily datetime string.
    """
    parsed = normalize_to_midnight(value=value)
    if parsed is None:
        raise ValueError(f"Invalid daily datetime: {value}")
    return parsed.strftime(DAILY_DATETIME_FORMAT)
