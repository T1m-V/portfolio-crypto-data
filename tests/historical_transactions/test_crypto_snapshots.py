from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from blockchain_reader.raw_snapshots import CryptoTracker


class TestCryptoSnapshots:
    def test_plain_reward_does_not_create_fake_reward_asset(self) -> None:
        tracker = CryptoTracker(chain="arbitrum", token_metadata={})
        row = pd.Series(
            {
                "Date": pd.Timestamp("2025-01-01 10:00:00"),
                "Type": "Reward",
                "Qty in": "1.5",
                "Token in": "ETH",
                "Qty out": "",
                "Token out": "",
                "Fee": pd.NA,
                "Fee Token": pd.NA,
            }
        )

        tracker.process_transaction(row=row)

        assert "REWARD" not in tracker.assets
        assert tracker.assets["ETH"].quantity == Decimal("1.5")
        assert tracker.assets["ETH"].principal == 0.0
        assert [snapshot["Coin"] for snapshot in tracker.history] == ["ETH"]

    def test_reward_with_explicit_source_keeps_reallocation_behavior(self) -> None:
        tracker = CryptoTracker(chain="arbitrum", token_metadata={})
        tracker.fetch_asset("GLP").principal = 100.0
        row = pd.Series(
            {
                "Date": pd.Timestamp("2025-01-01 10:00:00"),
                "Type": "Reward|GLP",
                "Qty in": "2",
                "Token in": "ETH",
                "Qty out": "",
                "Token out": "",
                "Fee": pd.NA,
                "Fee Token": pd.NA,
            }
        )

        tracker.process_transaction(row=row)

        assert tracker.assets["ETH"].quantity == Decimal("2")
        assert tracker.assets["GLP"].principal < 100.0
        assert "REWARD" not in tracker.assets

    def test_save_to_csv_uses_second_precision_daily_dates(self) -> None:
        tracker = CryptoTracker(chain="arbitrum", token_metadata={})
        tracker.history = [
            {
                "Date": pd.Timestamp("2025-01-01 10:15:31"),
                "Coin": "ETH",
                "Quantity": Decimal("1"),
                "Principal Invested": 42.0,
            }
        ]

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "snapshots.csv"
            tracker.save_to_csv(output_path=output_path)

            result = pd.read_csv(output_path)
            assert result.loc[0, "Date"] == "2025-01-01 00:00:00"
