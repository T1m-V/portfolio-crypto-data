import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from blockchain_reader.extraction.evm_reader import (
    OUTPUT_COLUMNS,
    _derive_start_date,
    _fetch_explorer_data,
    _normalize_results_frame,
    _parse_transaction_datetime_series,
)


def test_derive_start_date_uses_latest_date_with_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "tx.csv"
        pd.DataFrame(
            {
                "Date": [
                    "01/01/2026 08:00:00",
                    "05/01/2026 11:30:00",
                    "03/01/2026 09:45:00",
                ]
            }
        ).to_csv(csv_path, index=False)

        derived = _derive_start_date(output_path=csv_path, overlap_days=1)
        assert derived == "04/01/2026 00:00:00"


def test_derive_start_date_handles_minute_precision() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "tx.csv"
        pd.DataFrame({"Date": ["01/01/2026 08:00", "05/01/2026 11:30"]}).to_csv(
            csv_path, index=False
        )

        derived = _derive_start_date(output_path=csv_path, overlap_days=1)
        assert derived == "04/01/2026 00:00:00"


def test_parse_transaction_datetime_series_mixed_formats() -> None:
    series = pd.Series(["05/01/2026 11:30:00", "03/01/2026 09:45", "invalid"])
    parsed = _parse_transaction_datetime_series(series)

    assert parsed.notna().sum() == 2
    assert parsed.max() == pd.Timestamp("2026-01-05 11:30:00")


def test_normalize_results_frame_enforces_columns() -> None:
    raw = pd.DataFrame([{"TX Hash": "0x1", "Date": "01/01/2026 10:00:00", "Fee": 1.2}])
    normalized = _normalize_results_frame(raw)

    assert list(normalized.columns) == OUTPUT_COLUMNS
    assert normalized.iloc[0]["Fee"] == "1.2"
    assert normalized.iloc[0]["Token in"] == ""


def test_fetch_explorer_data_handles_no_transactions() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "0",
        "message": "No transactions found",
        "result": "No transactions found",
    }

    with patch(
        "blockchain_reader.extraction.evm_reader.requests.get",
        return_value=response,
    ) as get_mock:
        data = _fetch_explorer_data(
            api_url="https://example.test",
            params={"action": "txlist"},
        )

    assert data == []
    get_mock.assert_called_once()
