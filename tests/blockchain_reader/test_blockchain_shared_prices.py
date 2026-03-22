from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from blockchain_reader.shared.prices import clear_price_cache, get_price_on_or_before


class TestBlockchainSharedPrices:
    def test_lp_lookup_requires_chain_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            prices_root = Path(tmp_dir) / "prices"
            (prices_root / "lp_prices" / "arbitrum").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"Date": "2024-01-02", "Price": 100.0}]).to_csv(
                prices_root / "lp_prices" / "arbitrum" / "LP.csv",
                index=False,
            )

            clear_price_cache()
            with_chain = get_price_on_or_before(
                symbol="LP",
                as_of_date="2024-01-02 00:00:00",
                prices_folder=prices_root,
                chain="arbitrum",
                use_lp_prices=True,
            )
            with pytest.raises(ValueError, match="requires `chain`"):
                get_price_on_or_before(
                    symbol="LP",
                    as_of_date="2024-01-02 00:00:00",
                    prices_folder=prices_root,
                    use_lp_prices=True,
                )

        assert with_chain == Decimal("100.0")

    def test_lp_lookup_uses_chain_specific_folder_without_cross_chain_fallback(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            prices_root = Path(tmp_dir) / "prices"
            (prices_root / "lp_prices" / "arbitrum").mkdir(parents=True, exist_ok=True)
            (prices_root / "lp_prices" / "base").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"Date": "2024-01-02", "Price": 100.0}]).to_csv(
                prices_root / "lp_prices" / "arbitrum" / "LP.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 50.0}]).to_csv(
                prices_root / "LP.csv",
                index=False,
            )

            clear_price_cache()
            base_chain = get_price_on_or_before(
                symbol="LP",
                as_of_date="2024-01-02 00:00:00",
                prices_folder=prices_root,
                chain="base",
                use_lp_prices=True,
            )
            arbitrum_chain = get_price_on_or_before(
                symbol="LP",
                as_of_date="2024-01-02 00:00:00",
                prices_folder=prices_root,
                chain="arbitrum",
                use_lp_prices=True,
            )

        assert base_chain is None
        assert arbitrum_chain == Decimal("100.0")

    def test_direct_lookup_ignores_lp_folder(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            prices_root = Path(tmp_dir) / "prices"
            (prices_root / "lp_prices" / "arbitrum").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"Date": "2024-01-02", "Price": 100.0}]).to_csv(
                prices_root / "lp_prices" / "arbitrum" / "LP.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 50.0}]).to_csv(
                prices_root / "LP.csv",
                index=False,
            )

            clear_price_cache()
            direct = get_price_on_or_before(
                symbol="LP",
                as_of_date="2024-01-02 00:00:00",
                prices_folder=prices_root,
                chain="arbitrum",
                use_lp_prices=False,
            )

        assert direct == Decimal("50.0")
