import csv
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from blockchain_reader.protocols import aave, common, curve, lp_pricing


class DummyProgress:
    def update(self, _: int) -> None:
        return None

    def close(self) -> None:
        return None


class FakeCall:
    def __init__(self, fn):
        self.fn = fn

    def call(self, block_identifier=None):
        return self.fn(block_identifier)


class FakeAaveFunctions:
    def __init__(self, contract):
        self.contract = contract

    def balanceOf(self, _wallet):
        def _call(block_identifier):
            self.contract.balance_call_blocks.append(block_identifier)
            value = self.contract.balance_by_block[block_identifier]
            if isinstance(value, Exception):
                raise value
            return value

        return FakeCall(_call)


class FakeAaveContract:
    def __init__(self, address: str, balance_by_block: dict[int, int | Exception]):
        self.address = address
        self.balance_by_block = balance_by_block
        self.balance_call_blocks: list[int] = []
        self.functions = FakeAaveFunctions(contract=self)


class FakeAaveEth:
    def __init__(self, contract: FakeAaveContract):
        self.contract_obj = contract

    def contract(self, address, abi):
        return self.contract_obj

    def get_code(self, address, block_identifier):
        return b"\x01"


class FakeAaveWeb3:
    def __init__(self, contract: FakeAaveContract):
        self.eth = FakeAaveEth(contract=contract)

    def to_checksum_address(self, address: str) -> str:
        return address


class FakePoolFunctions:
    def __init__(self, pool):
        self.pool = pool

    def coins(self, idx: int):
        def _call(_block_identifier):
            self.pool.coins_calls.append(idx)
            if idx not in self.pool.coin_addresses:
                raise RuntimeError("out of range")
            return self.pool.coin_addresses[idx]

        return FakeCall(_call)

    def balances(self, idx: int):
        return FakeCall(lambda _block_identifier: self.pool.coin_balances[idx])


class FakePoolContract:
    def __init__(self, coin_addresses: dict[int, str], coin_balances: dict[int, int]):
        self.coin_addresses = coin_addresses
        self.coin_balances = coin_balances
        self.coins_calls: list[int] = []
        self.functions = FakePoolFunctions(pool=self)


class FakeTokenFunctions:
    def __init__(self, symbol: str, decimals: int):
        self._symbol = symbol
        self._decimals = decimals

    def symbol(self):
        return FakeCall(lambda _block_identifier: self._symbol)

    def decimals(self):
        return FakeCall(lambda _block_identifier: self._decimals)


class FakeTokenContract:
    def __init__(self, symbol: str, decimals: int):
        self.functions = FakeTokenFunctions(symbol=symbol, decimals=decimals)


class FakeLPFunctions:
    def __init__(self, total_supply: int, pool_address: str):
        self.total_supply = total_supply
        self.pool_address = pool_address

    def totalSupply(self):
        return FakeCall(lambda _block_identifier: self.total_supply)

    def minter(self):
        return FakeCall(lambda _block_identifier: self.pool_address)


class FakeLPContract:
    def __init__(self, total_supply: int, pool_address: str):
        self.functions = FakeLPFunctions(total_supply=total_supply, pool_address=pool_address)


class FakeCurveEth:
    def __init__(self, contracts: dict[str, object]):
        self.contracts = contracts

    def contract(self, address, abi):
        return self.contracts[address]


class FakeCurveWeb3:
    def __init__(self, contracts: dict[str, object]):
        self.eth = FakeCurveEth(contracts=contracts)


class TestBlockchainProtocols:
    def _run_aave_daily_exposure(
        self,
        block_map: dict[str, int],
        balance_by_block: dict[int, int | Exception],
        end_date: str,
    ) -> tuple[list[dict[str, object]], list[int]]:
        descriptor = aave.AaveTokenDescriptor(
            token_address="0xtoken",
            token_symbol="aToken",
            token_decimals=18,
            underlying_address="0xunderlying",
            underlying_symbol="USDC",
            leg="supply",
        )
        contract = FakeAaveContract(address="0xtoken", balance_by_block=balance_by_block)
        fake_w3 = FakeAaveWeb3(contract=contract)
        write_mock = Mock(return_value=Path("aave_out.csv"))

        with (
            patch(
                "blockchain_reader.protocols.aave.load_chain_config",
                return_value={"my_address": "0xwallet"},
            ),
            patch("blockchain_reader.protocols.aave.load_chain_web3", return_value=fake_w3),
            patch("blockchain_reader.protocols.aave.load_tokens", return_value={}),
            patch("blockchain_reader.protocols.aave.load_block_map", return_value=block_map),
            patch("blockchain_reader.protocols.aave.build_symbol_family_map", return_value={}),
            patch("blockchain_reader.protocols.aave.build_address_symbol_map", return_value={}),
            patch(
                "blockchain_reader.protocols.aave._build_aave_descriptors",
                return_value=([descriptor], 0),
            ),
            patch("blockchain_reader.protocols.aave.write_protocol_history_csv", write_mock),
            patch("blockchain_reader.protocols.aave.tqdm", return_value=DummyProgress()),
        ):
            aave.get_aave_daily_exposure(
                chain="arbitrum",
                start_date="2026-01-01",
                end_date=end_date,
            )

        history = write_mock.call_args.kwargs["history_data"]
        return history, contract.balance_call_blocks

    def test_parse_date_value_supports_arbitrum_formats(self) -> None:
        minute_value = aave._parse_date_value("25/08/2022 17:35")
        second_value = aave._parse_date_value("03/08/2025 15:32:09")

        assert minute_value == datetime(2022, 8, 25, 17, 35)
        assert second_value == datetime(2025, 8, 3, 15, 32, 9)
        assert aave._parse_date_value("2025-08-03") is None

    def test_aave_daily_exposure_extends_past_end_until_terminal_zero_day(self) -> None:
        block_map = {
            "2026-01-01": 11,
            "2026-01-02": 12,
            "2026-01-03": 13,
            "2026-01-04": 14,
        }
        history, queried_blocks = self._run_aave_daily_exposure(
            block_map=block_map,
            balance_by_block={
                11: 5 * 10**18,
                12: 2 * 10**18,
                13: 0,
                14: 9 * 10**18,
            },
            end_date="2026-01-02",
        )

        assert queried_blocks == [11, 12, 13]
        assert [row["date"] for row in history] == [
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 3),
        ]
        assert history[-1]["rpc_error_count"] == 0

    def test_aave_terminal_zero_requires_zero_rpc_errors(self) -> None:
        block_map = {
            "2026-01-01": 11,
            "2026-01-02": 12,
            "2026-01-03": 13,
            "2026-01-04": 14,
            "2026-01-05": 15,
        }
        history, queried_blocks = self._run_aave_daily_exposure(
            block_map=block_map,
            balance_by_block={
                11: 5 * 10**18,
                12: 2 * 10**18,
                13: RuntimeError("rpc error"),
                14: 0,
                15: 9 * 10**18,
            },
            end_date="2026-01-02",
        )

        assert queried_blocks == [11, 12, 13, 14]
        assert history[2]["date"] == date(2026, 1, 3)
        assert history[2]["rpc_error_count"] == 1
        assert history[-1]["date"] == date(2026, 1, 4)

    def test_read_curve_pool_tokens_returns_dataclass_list_and_stops_on_revert(self) -> None:
        pool_address = "0xpool"
        token_a = "0xA"
        token_b = "0xB"
        pool = FakePoolContract(
            coin_addresses={0: token_a, 1: token_b},
            coin_balances={0: 1000, 1: 2000},
        )
        w3 = FakeCurveWeb3(
            contracts={
                pool_address: pool,
                token_a: FakeTokenContract(symbol="USDC", decimals=6),
                token_b: FakeTokenContract(symbol="WETH", decimals=18),
            }
        )

        result = curve._read_curve_pool_tokens(w3=w3, pool_address=pool_address, block_number=123)

        assert pool.coins_calls == [0, 1, 2]
        assert len(result) == 2
        assert isinstance(result[0], curve.CurvePoolToken)
        assert result[0] == curve.CurvePoolToken(
            address=token_a,
            balance=1000,
            symbol="USDC",
            decimals=6,
        )
        assert result[1] == curve.CurvePoolToken(
            address=token_b,
            balance=2000,
            symbol="WETH",
            decimals=18,
        )

    def test_get_curve_underlying_uses_curve_pool_token_dataclass(self) -> None:
        lp_address = "0xlp"
        pool_address = "0xpool"
        w3 = FakeCurveWeb3(
            contracts={
                lp_address: FakeLPContract(total_supply=200, pool_address=pool_address),
            }
        )
        pool_tokens = [
            curve.CurvePoolToken(address="0xA", balance=2000, symbol="USDC", decimals=6),
            curve.CurvePoolToken(address="0xB", balance=4 * 10**18, symbol="WETH", decimals=18),
        ]

        with patch(
            "blockchain_reader.protocols.curve._read_curve_pool_tokens",
            return_value=pool_tokens,
        ) as read_pool_mock:
            result = curve.get_curve_underlying(
                w3=w3,
                lp_token_address=lp_address,
                one_unit=100,
                block_number=123,
            )

        read_pool_mock.assert_called_once_with(w3=w3, pool_address=pool_address, block_number=123)
        assert result["USDC"] == Decimal("0.001")
        assert result["WETH"] == Decimal("2")

    def test_resolve_effective_start_date_prefers_existing_output_plus_one(self) -> None:
        with patch(
            "blockchain_reader.protocols.common.get_output_max_processed_date",
            return_value=date(2026, 1, 5),
        ):
            result = common.resolve_effective_start_date(
                protocol="curve",
                chain="arbitrum",
                symbol="LP",
                explicit_start_date=None,
                fallback_start_date="2026-01-01",
            )
        assert result == "2026-01-06"

    def test_resolve_effective_start_date_respects_explicit_start(self) -> None:
        with patch(
            "blockchain_reader.protocols.common.get_output_max_processed_date",
            return_value=date(2026, 1, 5),
        ):
            result = common.resolve_effective_start_date(
                protocol="curve",
                chain="arbitrum",
                symbol="LP",
                explicit_start_date="2025-09-01",
                fallback_start_date="2026-01-01",
            )
        assert result == "2025-09-01"

    def test_resolve_effective_start_date_uses_fallback_without_existing_output(self) -> None:
        with patch(
            "blockchain_reader.protocols.common.get_output_max_processed_date",
            return_value=None,
        ):
            result = common.resolve_effective_start_date(
                protocol="curve",
                chain="arbitrum",
                symbol="LP",
                explicit_start_date=None,
                fallback_start_date="2026-01-01",
            )
        assert result == "2026-01-01"

    def test_resolve_effective_start_date_clamps_to_fallback_floor(self) -> None:
        with patch(
            "blockchain_reader.protocols.common.get_output_max_processed_date",
            return_value=date(2026, 1, 1),
        ):
            result = common.resolve_effective_start_date(
                protocol="curve",
                chain="arbitrum",
                symbol="LP",
                explicit_start_date=None,
                fallback_start_date="2026-01-10",
            )
        assert result == "2026-01-10"

    def test_write_protocol_history_csv_merges_rows_and_keeps_existing_overlap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_dir = root / "curve"
            protocol_dir.mkdir(parents=True, exist_ok=True)
            output_path = protocol_dir / "arbitrum_LP.csv"

            with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f=f,
                    fieldnames=["date", "block", "asset_A", "legacy_col"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "date": "2026-01-01",
                        "block": 10,
                        "asset_A": 1.1,
                        "legacy_col": "keep",
                    }
                )
                writer.writerow(
                    {
                        "date": "2026-01-02",
                        "block": 20,
                        "asset_A": 2.2,
                        "legacy_col": "keep2",
                    }
                )

            with patch("blockchain_reader.protocols.common.PROTOCOL_UNDERLYING_TOKEN_FOLDER", root):
                output = common.write_protocol_history_csv(
                    protocol="curve",
                    chain="arbitrum",
                    symbol="LP",
                    history_data=[
                        {"date": "2026-01-02", "block": 999, "asset_A": 9.9, "asset_B": 99},
                        {"date": "2026-01-03", "block": 30, "asset_A": 3.3, "asset_B": 33},
                    ],
                    fieldnames=["date", "block", "asset_A"],
                )

            assert output == output_path
            with open(output_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f=f)
                rows = list(reader)
                assert reader.fieldnames == ["date", "block", "asset_A", "asset_B", "legacy_col"]

            assert [row["date"] for row in rows] == ["2026-01-01", "2026-01-02", "2026-01-03"]
            assert rows[1]["block"] == "20"
            assert rows[1]["legacy_col"] == "keep2"
            assert rows[2]["block"] == "30"
            assert rows[2]["asset_B"] == "33"

    def test_process_all_curve_tokens_passes_resolved_incremental_start(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.curve.load_tokens",
                return_value={"0xpool": {"protocol": "curve", "symbol": "CurveLP"}},
            ),
            patch(
                "blockchain_reader.protocols.curve.load_snapshot_ranges",
                return_value={
                    "CurveLP": {
                        "start": pd.Timestamp("2024-01-01"),
                        "end": pd.Timestamp("2024-01-10"),
                        "qty": 1,
                    }
                },
            ),
            patch(
                "blockchain_reader.protocols.curve.resolve_effective_start_date",
                return_value="2024-01-05",
            ),
            patch("blockchain_reader.protocols.curve.get_curve_history") as history_mock,
        ):
            curve.process_all_curve_tokens(chain="arbitrum")

        history_mock.assert_called_once_with(
            chain="arbitrum",
            token_address="0xpool",
            start_date="2024-01-05",
            end_date="now",
        )

    def test_process_all_curve_tokens_skips_when_resolved_start_after_end(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.curve.load_tokens",
                return_value={"0xpool": {"protocol": "curve", "symbol": "CurveLP"}},
            ),
            patch(
                "blockchain_reader.protocols.curve.load_snapshot_ranges",
                return_value={
                    "CurveLP": {
                        "start": pd.Timestamp("2024-01-01"),
                        "end": pd.Timestamp("2024-01-10"),
                        "qty": 0,
                    }
                },
            ),
            patch(
                "blockchain_reader.protocols.curve.resolve_effective_start_date",
                return_value="2024-01-20",
            ),
            patch("blockchain_reader.protocols.curve.get_curve_history") as history_mock,
        ):
            curve.process_all_curve_tokens(chain="arbitrum")

        history_mock.assert_not_called()

    def test_process_all_aave_tokens_uses_resolved_incremental_start(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.aave._derive_aave_bounds_from_transactions",
                return_value=("2024-01-01", "2024-01-10"),
            ),
            patch(
                "blockchain_reader.protocols.aave.resolve_effective_start_date",
                return_value="2024-01-06",
            ),
            patch("blockchain_reader.protocols.aave.get_aave_daily_exposure") as exposure_mock,
        ):
            aave.process_all_aave_tokens(chain="arbitrum")

        exposure_mock.assert_called_once_with(
            chain="arbitrum",
            start_date="2024-01-06",
            end_date="2024-01-10",
        )

    def test_process_all_aave_tokens_skips_when_resolved_start_after_end(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.aave._derive_aave_bounds_from_transactions",
                return_value=("2024-01-01", "2024-01-10"),
            ),
            patch(
                "blockchain_reader.protocols.aave.resolve_effective_start_date",
                return_value="2024-01-20",
            ),
            patch("blockchain_reader.protocols.aave.get_aave_daily_exposure") as exposure_mock,
        ):
            aave.process_all_aave_tokens(chain="arbitrum")

        exposure_mock.assert_not_called()

    def test_generate_protocol_lp_price_files_merges_and_keeps_canonical_schema(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            tokens_root = root / "tokens"
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"date": "2024-01-02", "asset_ETH": 1.2, "asset_BTC": 0.5},
                    {"date": "03/01/2024", "asset_ETH": 1.0, "asset_BTC": 0.25},
                ]
            ).to_csv(protocol_root / "balancer" / "arbitrum_LP.csv", index=False)

            pd.DataFrame(
                [
                    {"Date": "2024-01-03", "Price": 2000},
                    {"Date": "2024-01-02", "Price": 1900},
                ]
            ).to_csv(prices_root / "ETH.csv", index=False)
            pd.DataFrame(
                [
                    {"Date": "2024-01-03", "Price": 40000},
                    {"Date": "2024-01-02", "Price": 39000},
                ]
            ).to_csv(prices_root / "BTC.csv", index=False)
            pd.DataFrame([{"Date": "2024-01-04", "Price": 100}]).to_csv(
                prices_root / "LP.csv",
                index=False,
            )

            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.protocols.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.protocols.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.protocols.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [prices_root / "LP.csv"]
            result = pd.read_csv(prices_root / "LP.csv")
            assert list(result.columns) == ["Date", "Price"]
            assert list(result["Date"]) == ["2024-01-04", "2024-01-03", "2024-01-02"]
            assert result.loc[result["Date"] == "2024-01-03", "Price"].iloc[0] == 12000.0
            assert result.loc[result["Date"] == "2024-01-02", "Price"].iloc[0] == 21780.0
            assert result.loc[result["Date"] == "2024-01-04", "Price"].iloc[0] == 100.0

    def test_generate_protocol_lp_price_files_handles_nested_beefy_lp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            tokens_root = root / "tokens"
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            (protocol_root / "beefy").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_ETH": 2.0}]).to_csv(
                protocol_root / "balancer" / "arbitrum_LP.csv",
                index=False,
            )
            pd.DataFrame([{"date": "2024-01-02", "asset_LP": 1.5}]).to_csv(
                protocol_root / "beefy" / "arbitrum_MOO.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 2000}]).to_csv(
                prices_root / "ETH.csv",
                index=False,
            )

            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.protocols.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.protocols.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.protocols.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert set(updated) == {prices_root / "LP.csv", prices_root / "MOO.csv"}
            lp_frame = pd.read_csv(prices_root / "LP.csv")
            moo_frame = pd.read_csv(prices_root / "MOO.csv")
            assert lp_frame.loc[0, "Price"] == 4000.0
            assert moo_frame.loc[0, "Price"] == 6000.0

    def test_generate_protocol_lp_price_files_skips_rows_with_unresolved_assets(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            tokens_root = root / "tokens"
            (protocol_root / "curve").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_UNKNOWN": 1.0}]).to_csv(
                protocol_root / "curve" / "arbitrum_BAD.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.protocols.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.protocols.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.protocols.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == []
            assert not (prices_root / "BAD.csv").exists()

    def test_generate_protocol_lp_price_files_excludes_aave_inputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            tokens_root = root / "tokens"
            (protocol_root / "aave").mkdir(parents=True, exist_ok=True)
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_ETH": 9.0}]).to_csv(
                protocol_root / "aave" / "arbitrum_AAVEWRAP.csv",
                index=False,
            )
            pd.DataFrame([{"date": "2024-01-02", "asset_ETH": 1.0}]).to_csv(
                protocol_root / "balancer" / "arbitrum_LP.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 2000}]).to_csv(
                prices_root / "ETH.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.protocols.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.protocols.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.protocols.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [prices_root / "LP.csv"]
            assert (prices_root / "LP.csv").exists()
            assert not (prices_root / "AAVEWRAP.csv").exists()
