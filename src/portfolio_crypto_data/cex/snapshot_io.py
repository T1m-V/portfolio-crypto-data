from __future__ import annotations

from pathlib import Path

import pandas as pd
from portfolio_core import atomic_write_csv

from portfolio_crypto_data.datetime_utils import format_daily_datetime

SNAPSHOT_COLUMNS = ["Date", "Coin", "Quantity", "Principal Invested"]


def save_cex_snapshot_history(*, history: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        atomic_write_csv(frame=pd.DataFrame(columns=SNAPSHOT_COLUMNS), path=output_csv)
        return

    frame = pd.DataFrame(history)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame["Date"] = frame["Date"].map(format_daily_datetime)
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["Principal Invested"] = pd.to_numeric(
        frame["Principal Invested"], errors="coerce"
    )
    frame = frame.dropna(subset=["Quantity", "Principal Invested"])
    frame = frame.sort_values(["Date", "Coin"])[SNAPSHOT_COLUMNS]
    atomic_write_csv(frame=frame, path=output_csv)
