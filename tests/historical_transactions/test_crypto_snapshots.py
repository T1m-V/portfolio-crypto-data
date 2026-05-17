from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import blockchain_reader.raw_snapshots as raw_snapshots
from blockchain_reader.raw_snapshots import CryptoTracker


def _patch_protocol_underlying_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "protocol_underlying_tokens"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        raw_snapshots,
        "PROTOCOL_UNDERLYING_TOKEN_FOLDER",
        root,
        raising=False,
    )
    return root


def _write_single_underlying(
    *,
    root: Path,
    chain: str,
    protocol: str,
    symbol: str,
    underlying: str,
) -> None:
    protocol_dir = root / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                f"asset_{underlying}": 1.0,
            }
        ]
    ).to_csv(protocol_dir / f"{chain}_{symbol}.csv", index=False)


def _write_underlyings(
    *,
    root: Path,
    chain: str,
    protocol: str,
    symbol: str,
    underlyings: dict[str, float],
) -> None:
    protocol_dir = root / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    row = {"date": "2025-01-01"}
    row.update({f"asset_{underlying}": quantity for underlying, quantity in underlyings.items()})
    pd.DataFrame([row]).to_csv(protocol_dir / f"{chain}_{symbol}.csv", index=False)


def _patch_prices(monkeypatch, prices: dict[str, float]) -> None:
    monkeypatch.setattr(raw_snapshots, "get_forex_rate", lambda **kwargs: 1.0)
    monkeypatch.setattr(
        raw_snapshots,
        "get_crypto_price",
        lambda coin, **kwargs: prices.get(coin, 0.0),
    )


def _tx(
    *,
    tx_type: str,
    qty_in: str = "",
    token_in: str = "",
    qty_out: str = "",
    token_out: str = "",
    hour: int = 10,
) -> pd.Series:
    return pd.Series(
        {
            "Date": pd.Timestamp(f"2025-01-01 {hour}:00:00"),
            "Type": tx_type,
            "Qty in": qty_in,
            "Token in": token_in,
            "Qty out": qty_out,
            "Token out": token_out,
            "Fee": pd.NA,
            "Fee Token": pd.NA,
        }
    )


def _wsteth_tracker(monkeypatch, tmp_path: Path) -> CryptoTracker:
    root = _patch_protocol_underlying_root(monkeypatch, tmp_path)
    _write_single_underlying(
        root=root,
        chain="arbitrum",
        protocol="liquid_staking",
        symbol="wstETH",
        underlying="ETH",
    )
    _patch_prices(monkeypatch, {"ETH": 1000.0, "wstETH": 1000.0})
    return CryptoTracker(
        chain="arbitrum",
        token_metadata={
            "0xeth": {"symbol": "ETH"},
            "0xwsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
            "0xaavewsteth": {"symbol": "aArbwstETH", "protocol": "aave"},
        },
    )


def _deposit_eth_to_aave_wsteth(tracker: CryptoTracker) -> None:
    tracker.process_transaction(_tx(tx_type="Receive", qty_in="10", token_in="ETH", hour=10))
    tracker.process_transaction(
        _tx(
            tx_type="Swap",
            qty_in="10",
            token_in="wstETH",
            qty_out="10",
            token_out="ETH",
            hour=11,
        )
    )
    tracker.process_transaction(
        _tx(
            tx_type="Swap",
            qty_in="10",
            token_in="aArbwstETH",
            qty_out="10",
            token_out="wstETH",
            hour=12,
        )
    )


def _nested_eth_wrapper_tracker(monkeypatch, tmp_path: Path) -> CryptoTracker:
    root = _patch_protocol_underlying_root(monkeypatch, tmp_path)
    _write_single_underlying(
        root=root,
        chain="arbitrum",
        protocol="liquid_staking",
        symbol="wstETH",
        underlying="ETH",
    )
    _write_underlyings(
        root=root,
        chain="arbitrum",
        protocol="balancer",
        symbol="wstETH-WETH-BPT",
        underlyings={"WETH": 0.5, "wstETH": 0.5},
    )
    _write_single_underlying(
        root=root,
        chain="arbitrum",
        protocol="beefy",
        symbol="mooBalancerArbwstETH-ETHV3",
        underlying="wstETH-WETH-BPT",
    )
    _patch_prices(
        monkeypatch,
        {
            "ETH": 1000.0,
            "WETH": 1000.0,
            "wstETH": 1000.0,
            "wstETH-WETH-BPT": 1000.0,
            "mooBalancerArbwstETH-ETHV3": 1000.0,
        },
    )
    return CryptoTracker(
        chain="arbitrum",
        token_metadata={
            "0xeth": {"symbol": "ETH"},
            "0xweth": {"symbol": "WETH", "family": "ETH"},
            "0xwsteth": {"symbol": "wstETH", "protocol": "liquid_staking"},
            "0xbpt": {
                "symbol": "wstETH-WETH-BPT",
                "family": "ETH",
                "protocol": "balancer",
            },
            "0xmoo": {
                "symbol": "mooBalancerArbwstETH-ETHV3",
                "family": "ETH",
                "protocol": "beefy",
            },
            "0xaavewsteth": {"symbol": "aArbwstETH", "protocol": "aave"},
        },
    )


class TestCryptoSnapshots:
    def test_protocol_symbols_do_not_use_family_proxy_in_raw_snapshots(self) -> None:
        tracker = CryptoTracker(
            chain="arbitrum",
            token_metadata={
                "0xwrap": {"symbol": "WRAP", "family": "ETH", "protocol": "beefy"},
                "0xeth": {"symbol": "ETH"},
            },
        )

        wrap = tracker.fetch_asset("WRAP")
        assert wrap.family_proxy is None
        assert wrap.price_source == "WRAP"

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

    def test_aave_wrapper_principal_uses_underlying_price_source(self, monkeypatch) -> None:
        tracker = CryptoTracker(
            chain="arbitrum",
            token_metadata={
                "0xusdt": {"symbol": "USDT", "decimals": 6},
                "0xausdt": {"symbol": "aArbUSDT", "decimals": 6, "protocol": "aave"},
            },
        )
        monkeypatch.setattr(raw_snapshots, "get_forex_rate", lambda **kwargs: 1.0)

        def fake_price(coin: str, **kwargs):
            if coin == "USDT":
                return 1.0
            return 0.0

        monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_price)
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 10:00:00"),
                    "Type": "Receive",
                    "Qty in": "1000",
                    "Token in": "USDT",
                    "Qty out": "",
                    "Token out": "",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 11:00:00"),
                    "Type": "Swap",
                    "Qty in": "1000",
                    "Token in": "aArbUSDT",
                    "Qty out": "1000",
                    "Token out": "USDT",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )

        assert tracker.assets["aArbUSDT"].price_source == "USDT"
        assert tracker.assets["USDT"].quantity == Decimal("0")
        assert tracker.assets["USDT"].principal == 1000.0
        assert tracker.assets["aArbUSDT"].quantity == Decimal("1000")
        assert tracker.assets["aArbUSDT"].principal == 0.0
        assert tracker.assets["aArbUSDT"].family_proxy is tracker.assets["USDT"]

    def test_wrapper_supply_principal_stays_on_base_family_through_aave(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        tracker = _wsteth_tracker(monkeypatch=monkeypatch, tmp_path=tmp_path)

        _deposit_eth_to_aave_wsteth(tracker=tracker)

        assert tracker.assets["ETH"].quantity == Decimal("0")
        assert tracker.assets["ETH"].principal == 10000.0
        assert tracker.assets["wstETH"].quantity == Decimal("0")
        assert tracker.assets["wstETH"].principal == 0.0
        assert tracker.assets["wstETH"].family_proxy is tracker.assets["ETH"]
        assert tracker.assets["aArbwstETH"].quantity == Decimal("10")
        assert tracker.assets["aArbwstETH"].principal == 0.0
        assert tracker.assets["aArbwstETH"].family_proxy is tracker.assets["ETH"]

    def test_aave_supply_withdraw_unwrap_leaves_no_wrapper_principal(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        tracker = _wsteth_tracker(monkeypatch=monkeypatch, tmp_path=tmp_path)
        _deposit_eth_to_aave_wsteth(tracker=tracker)

        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="10",
                token_in="wstETH",
                qty_out="10",
                token_out="aArbwstETH",
                hour=13,
            )
        )
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="10",
                token_in="ETH",
                qty_out="10",
                token_out="wstETH",
                hour=14,
            )
        )

        assert tracker.assets["ETH"].quantity == Decimal("10")
        assert tracker.assets["ETH"].principal == 10000.0
        assert tracker.assets["wstETH"].quantity == Decimal("0")
        assert tracker.assets["wstETH"].principal == 0.0
        assert tracker.assets["aArbwstETH"].quantity == Decimal("0")
        assert tracker.assets["aArbwstETH"].principal == 0.0

    def test_partial_aave_withdrawal_keeps_principal_on_base_family(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        tracker = _wsteth_tracker(monkeypatch=monkeypatch, tmp_path=tmp_path)
        _deposit_eth_to_aave_wsteth(tracker=tracker)

        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="4",
                token_in="wstETH",
                qty_out="4",
                token_out="aArbwstETH",
                hour=13,
            )
        )

        assert tracker.assets["ETH"].quantity == Decimal("0")
        assert tracker.assets["ETH"].principal == 10000.0
        assert tracker.assets["wstETH"].quantity == Decimal("4")
        assert tracker.assets["wstETH"].principal == 0.0
        assert tracker.assets["aArbwstETH"].quantity == Decimal("6")
        assert tracker.assets["aArbwstETH"].principal == 0.0

    def test_nested_single_family_protocol_principal_stays_on_base_family(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        tracker = _nested_eth_wrapper_tracker(monkeypatch=monkeypatch, tmp_path=tmp_path)

        tracker.process_transaction(_tx(tx_type="Receive", qty_in="2", token_in="ETH", hour=10))
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="wstETH-WETH-BPT",
                qty_out="2",
                token_out="ETH",
                hour=11,
            )
        )
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="mooBalancerArbwstETH-ETHV3",
                qty_out="2",
                token_out="wstETH-WETH-BPT",
                hour=12,
            )
        )

        assert tracker.assets["ETH"].quantity == Decimal("0")
        assert tracker.assets["ETH"].principal == 2000.0
        assert tracker.assets["wstETH-WETH-BPT"].quantity == Decimal("0")
        assert tracker.assets["wstETH-WETH-BPT"].principal == 0.0
        assert tracker.assets["wstETH-WETH-BPT"].family_proxy is tracker.assets["ETH"]
        assert tracker.assets["mooBalancerArbwstETH-ETHV3"].quantity == Decimal("2")
        assert tracker.assets["mooBalancerArbwstETH-ETHV3"].principal == 0.0
        assert tracker.assets["mooBalancerArbwstETH-ETHV3"].family_proxy is tracker.assets["ETH"]

    def test_nested_single_family_wrapper_closure_keeps_no_intermediate_principal(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        tracker = _nested_eth_wrapper_tracker(monkeypatch=monkeypatch, tmp_path=tmp_path)
        tracker.process_transaction(_tx(tx_type="Receive", qty_in="2", token_in="ETH", hour=10))
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="wstETH-WETH-BPT",
                qty_out="2",
                token_out="ETH",
                hour=11,
            )
        )
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="mooBalancerArbwstETH-ETHV3",
                qty_out="2",
                token_out="wstETH-WETH-BPT",
                hour=12,
            )
        )
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="wstETH",
                qty_out="2",
                token_out="mooBalancerArbwstETH-ETHV3",
                hour=13,
            )
        )
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="2",
                token_in="aArbwstETH",
                qty_out="2",
                token_out="wstETH",
                hour=14,
            )
        )

        assert tracker.assets["ETH"].quantity == Decimal("0")
        assert tracker.assets["ETH"].principal == 2000.0
        assert tracker.assets["wstETH-WETH-BPT"].principal == 0.0
        assert tracker.assets["mooBalancerArbwstETH-ETHV3"].quantity == Decimal("0")
        assert tracker.assets["mooBalancerArbwstETH-ETHV3"].principal == 0.0
        assert tracker.assets["wstETH"].quantity == Decimal("0")
        assert tracker.assets["wstETH"].principal == 0.0
        assert tracker.assets["aArbwstETH"].quantity == Decimal("2")
        assert tracker.assets["aArbwstETH"].principal == 0.0

    def test_mixed_protocol_wrapper_keeps_principal_on_active_wrapper(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        root = _patch_protocol_underlying_root(monkeypatch, tmp_path)
        _write_underlyings(
            root=root,
            chain="arbitrum",
            protocol="beefy",
            symbol="mooFishUSDT-USDC",
            underlyings={"USDC": 0.5, "USDT": 0.5},
        )
        _patch_prices(
            monkeypatch,
            {"USDC": 1.0, "USDT": 1.0, "mooFishUSDT-USDC": 1.0},
        )
        tracker = CryptoTracker(
            chain="arbitrum",
            token_metadata={
                "0xusdc": {"symbol": "USDC"},
                "0xusdt": {"symbol": "USDT"},
                "0xmoo": {"symbol": "mooFishUSDT-USDC", "protocol": "beefy"},
            },
        )

        tracker.process_transaction(_tx(tx_type="Receive", qty_in="100", token_in="USDC", hour=10))
        tracker.process_transaction(
            _tx(
                tx_type="Swap",
                qty_in="100",
                token_in="mooFishUSDT-USDC",
                qty_out="100",
                token_out="USDC",
                hour=11,
            )
        )

        assert tracker.assets["USDC"].quantity == Decimal("0")
        assert tracker.assets["USDC"].principal == 0.0
        assert tracker.assets["mooFishUSDT-USDC"].quantity == Decimal("100")
        assert tracker.assets["mooFishUSDT-USDC"].principal == 100.0
        assert tracker.assets["mooFishUSDT-USDC"].family_proxy is None

    def test_swap_clears_outgoing_principal_using_outgoing_value(self, monkeypatch) -> None:
        tracker = CryptoTracker(chain="arbitrum", token_metadata={})
        monkeypatch.setattr(raw_snapshots, "get_forex_rate", lambda **kwargs: 1.0)

        def fake_price(coin: str, **kwargs):
            return {"USDC": 1.0, "GLP": 0.5}.get(coin, 0.0)

        monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_price)
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 10:00:00"),
                    "Type": "Receive",
                    "Qty in": "100",
                    "Token in": "USDC",
                    "Qty out": "",
                    "Token out": "",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 11:00:00"),
                    "Type": "Swap",
                    "Qty in": "100",
                    "Token in": "GLP",
                    "Qty out": "100",
                    "Token out": "USDC",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )

        assert tracker.assets["USDC"].quantity == Decimal("0")
        assert tracker.assets["USDC"].principal == 0.0
        assert tracker.assets["GLP"].quantity == Decimal("100")
        assert tracker.assets["GLP"].principal == 100.0

    def test_swap_assigns_principal_to_unpriced_incoming_leg(self, monkeypatch) -> None:
        tracker = CryptoTracker(
            chain="arbitrum",
            token_metadata={
                "0xusdc": {"symbol": "USDC"},
                "0xweth": {"symbol": "WETH"},
                "0xlp": {"symbol": "UNPRICED-LP"},
            },
        )
        monkeypatch.setattr(raw_snapshots, "get_forex_rate", lambda **kwargs: 1.0)

        def fake_price(coin: str, **kwargs):
            return {"USDC": 1.0, "WETH": 2000.0, "UNPRICED-LP": 0.0}.get(coin, 0.0)

        monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_price)
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 10:00:00"),
                    "Type": "Receive",
                    "Qty in": "1000",
                    "Token in": "USDC",
                    "Qty out": "",
                    "Token out": "",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 11:00:00"),
                    "Type": "Swap",
                    "Qty in": "0.00001, 2e-9",
                    "Token in": "WETH, UNPRICED-LP",
                    "Qty out": "1000",
                    "Token out": "USDC",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )

        assert tracker.assets["USDC"].quantity == Decimal("0")
        assert tracker.assets["USDC"].principal == 0.0
        assert tracker.assets["WETH"].principal == 0.0
        assert tracker.assets["UNPRICED-LP"].principal == 1000.0

    def test_swap_records_missing_incoming_price_and_keeps_principal_transfer(
        self, monkeypatch
    ) -> None:
        tracker = CryptoTracker(
            chain="arbitrum",
            token_metadata={
                "0xusdc": {"symbol": "USDC"},
                "0xlp": {"symbol": "UNPRICED-LP"},
            },
        )
        monkeypatch.setattr(raw_snapshots, "get_forex_rate", lambda **kwargs: 1.0)

        def fake_price(coin: str, **kwargs):
            if coin == "UNPRICED-LP":
                return None
            return {"USDC": 1.0}.get(coin, 0.0)

        monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_price)
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 10:00:00"),
                    "Type": "Receive",
                    "Qty in": "1000",
                    "Token in": "USDC",
                    "Qty out": "",
                    "Token out": "",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )
        tracker.process_transaction(
            row=pd.Series(
                {
                    "Date": pd.Timestamp("2025-01-01 11:00:00"),
                    "Type": "Swap",
                    "Qty in": "1",
                    "Token in": "UNPRICED-LP",
                    "Qty out": "1000",
                    "Token out": "USDC",
                    "Fee": pd.NA,
                    "Fee Token": pd.NA,
                }
            )
        )

        assert tracker.assets["USDC"].principal == 0.0
        assert tracker.assets["UNPRICED-LP"].principal == 1000.0
        assert tracker.unresolved_prices[0].coin == "UNPRICED-LP"
        assert tracker.unresolved_prices[0].action == "swap_in"

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
