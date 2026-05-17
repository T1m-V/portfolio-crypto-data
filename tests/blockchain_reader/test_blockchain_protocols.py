import csv
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, Mock, patch

import pandas as pd

from blockchain_reader.composition import lp_pricing
from blockchain_reader.protocols import aave, balancer, beefy, common, curve, liquid_staking


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
    def __init__(
        self,
        total_supply: int,
        pool_address: str,
        minter_error: Exception | None = None,
    ):
        self.total_supply = total_supply
        self.pool_address = pool_address
        self.minter_error = minter_error

    def totalSupply(self):
        return FakeCall(lambda _block_identifier: self.total_supply)

    def minter(self):
        def _call(_block_identifier):
            if self.minter_error is not None:
                raise self.minter_error
            return self.pool_address

        return FakeCall(_call)


class FakeLPContract:
    def __init__(
        self,
        total_supply: int,
        pool_address: str,
        minter_error: Exception | None = None,
    ):
        self.functions = FakeLPFunctions(
            total_supply=total_supply,
            pool_address=pool_address,
            minter_error=minter_error,
        )


class FakeCurveEth:
    def __init__(self, contracts: dict[str, object]):
        self.contracts = contracts

    def contract(self, address, abi):
        return self.contracts[address]


class FakeCurveWeb3:
    def __init__(self, contracts: dict[str, object]):
        self.eth = FakeCurveEth(contracts=contracts)


class FakeBalancerBptFunctions:
    def __init__(self, pool_id: bytes, total_supply: int):
        self.pool_id = pool_id
        self.total_supply = total_supply
        self.actual_supply_calls = 0
        self.total_supply_calls = 0

    def getPoolId(self):
        return FakeCall(lambda _block_identifier: self.pool_id)

    def getActualSupply(self):
        def _call(_block_identifier):
            self.actual_supply_calls += 1
            raise RuntimeError("execution reverted")

        return FakeCall(_call)

    def totalSupply(self):
        def _call(_block_identifier):
            self.total_supply_calls += 1
            return self.total_supply

        return FakeCall(_call)


class FakeBalancerBptContract:
    def __init__(self, pool_id: bytes, total_supply: int):
        self.functions = FakeBalancerBptFunctions(
            pool_id=pool_id,
            total_supply=total_supply,
        )


class FakeBalancerVaultFunctions:
    def __init__(self, token_addresses: list[str], token_balances: list[int]):
        self.token_addresses = token_addresses
        self.token_balances = token_balances

    def getPoolTokens(self, poolId):
        return FakeCall(lambda _block_identifier: (self.token_addresses, self.token_balances, 0))


class FakeBalancerVaultContract:
    def __init__(self, token_addresses: list[str], token_balances: list[int]):
        self.functions = FakeBalancerVaultFunctions(
            token_addresses=token_addresses,
            token_balances=token_balances,
        )


class FakeBalancerEth:
    def __init__(self, contracts: dict[str, object]):
        self.contracts = contracts

    def contract(self, address, abi):
        return self.contracts[address]


class FakeBalancerWeb3:
    def __init__(self, contracts: dict[str, object]):
        self.eth = FakeBalancerEth(contracts=contracts)


class FakeBeefyVaultFunctions:
    def __init__(self, want_address: str, ppfs: int, decimals: int = 18, symbol: str = "moo"):
        self.want_address = want_address
        self.ppfs = ppfs
        self.decimals_value = decimals
        self.symbol_value = symbol

    def getPricePerFullShare(self):
        return FakeCall(lambda _block_identifier: self.ppfs)

    def want(self):
        return FakeCall(lambda _block_identifier: self.want_address)

    def decimals(self):
        return FakeCall(lambda _block_identifier: self.decimals_value)

    def symbol(self):
        return FakeCall(lambda _block_identifier: self.symbol_value)


class FakeBeefyVaultContract:
    def __init__(self, want_address: str, ppfs: int, decimals: int = 18, symbol: str = "moo"):
        self.functions = FakeBeefyVaultFunctions(
            want_address=want_address,
            ppfs=ppfs,
            decimals=decimals,
            symbol=symbol,
        )


class FakePairFunctions:
    def __init__(
        self,
        token0_address: str,
        token1_address: str,
        reserve0: int,
        reserve1: int,
        total_supply: int,
    ):
        self.token0_address = token0_address
        self.token1_address = token1_address
        self.reserve0 = reserve0
        self.reserve1 = reserve1
        self.total_supply = total_supply

    def token0(self):
        return FakeCall(lambda _block_identifier: self.token0_address)

    def token1(self):
        return FakeCall(lambda _block_identifier: self.token1_address)

    def getReserves(self):
        return FakeCall(lambda _block_identifier: (self.reserve0, self.reserve1, 0))

    def totalSupply(self):
        return FakeCall(lambda _block_identifier: self.total_supply)


class FakePairContract:
    def __init__(
        self,
        token0_address: str,
        token1_address: str,
        reserve0: int,
        reserve1: int,
        total_supply: int,
    ):
        self.functions = FakePairFunctions(
            token0_address=token0_address,
            token1_address=token1_address,
            reserve0=reserve0,
            reserve1=reserve1,
            total_supply=total_supply,
        )


class FakeBeefyEth:
    def __init__(self, contracts: dict[str, object]):
        self.contracts = contracts

    def contract(self, address, abi):
        return self.contracts[address]


class FakeBeefyWeb3:
    def __init__(self, contracts: dict[str, object]):
        self.eth = FakeBeefyEth(contracts=contracts)


class FakeRateProviderFunctions:
    def __init__(self, contract):
        self.contract = contract

    def getRate(self):
        def _call(block_identifier):
            self.contract.rate_call_blocks.append(block_identifier)
            value = self.contract.rate_by_block[block_identifier]
            if isinstance(value, Exception):
                raise value
            return value

        return FakeCall(_call)


class FakeRateProviderContract:
    def __init__(self, address: str, rate_by_block: dict[int, int | Exception]):
        self.address = address
        self.rate_by_block = rate_by_block
        self.rate_call_blocks: list[int] = []
        self.functions = FakeRateProviderFunctions(contract=self)


class FakeLiquidStakingEth:
    def __init__(self, contract: FakeRateProviderContract, deployed_blocks: set[int] | None = None):
        self.contract_obj = contract
        self.deployed_blocks = deployed_blocks or set()

    def contract(self, address, abi):
        return self.contract_obj

    def get_code(self, address, block_identifier):
        if not self.deployed_blocks:
            return b"\x01"
        return b"\x01" if block_identifier in self.deployed_blocks else b""


class FakeLiquidStakingWeb3:
    def __init__(self, contract: FakeRateProviderContract, deployed_blocks: set[int] | None = None):
        self.eth = FakeLiquidStakingEth(contract=contract, deployed_blocks=deployed_blocks)

    def to_checksum_address(self, address: str) -> str:
        return address


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

    def test_normalize_aave_underlying_symbol_maps_usdt_aliases(self) -> None:
        assert aave._normalize_aave_underlying_symbol("USDÃ¢â€šÂ®0") == "USDT"
        assert aave._normalize_aave_underlying_symbol("USDT") == "USDT"
        assert aave._normalize_aave_underlying_symbol("wstETH") == "wstETH"

    def test_balancer_underlying_falls_back_to_total_supply_for_legacy_pools(self) -> None:
        bpt_address = "0xbpt"
        vault_address = "0xvault"
        token_address = "0xtoken"
        bpt = FakeBalancerBptContract(pool_id=b"pool", total_supply=200)
        w3 = FakeBalancerWeb3(
            contracts={
                bpt_address: bpt,
                vault_address: FakeBalancerVaultContract(
                    token_addresses=[bpt_address, token_address],
                    token_balances=[999, 4_000_000],
                ),
                token_address: FakeTokenContract(symbol="USDC", decimals=6),
            }
        )

        result = balancer.get_balancer_underlying(
            w3=w3,
            bpt_address=bpt_address,
            one_unit=100,
            block_number=123,
            vault_address=vault_address,
        )

        assert bpt.functions.actual_supply_calls == 1
        assert bpt.functions.total_supply_calls == 1
        assert result == {"USDC": Decimal("2")}

    def test_beefy_underlying_keeps_single_want_token_fallback(self) -> None:
        vault_address = "0xmoo"
        want_address = "0xusdc"
        w3 = FakeBeefyWeb3(
            contracts={
                vault_address: FakeBeefyVaultContract(
                    want_address=want_address,
                    ppfs=2 * 10**18,
                    decimals=6,
                    symbol="mooUSDC",
                ),
                want_address: FakeTokenContract(symbol="USDC", decimals=6),
            }
        )

        result = beefy.get_beefy_underlying(
            w3=w3,
            vault_address=vault_address,
            one_unit=10**6,
            block_number=123,
        )

        assert result == {"USDC": Decimal("2")}

    def test_beefy_underlying_decomposes_pair_want_tokens(self) -> None:
        vault_address = "0xmoo"
        pair_address = "0xpair"
        usdt_address = "0xusdt"
        usdc_address = "0xusdc"
        w3 = FakeBeefyWeb3(
            contracts={
                vault_address: FakeBeefyVaultContract(
                    want_address=pair_address,
                    ppfs=10**18,
                    symbol="mooFishUSDT-USDC",
                ),
                pair_address: FakePairContract(
                    token0_address=usdt_address,
                    token1_address=usdc_address,
                    reserve0=5_000 * 10**6,
                    reserve1=5_000 * 10**6,
                    total_supply=10 * 10**18,
                ),
                usdt_address: FakeTokenContract(symbol="USDT", decimals=6),
                usdc_address: FakeTokenContract(symbol="USDC", decimals=6),
            }
        )

        result = beefy.get_beefy_underlying(
            w3=w3,
            vault_address=vault_address,
            one_unit=10**18,
            block_number=123,
        )

        assert result == {"USDT": Decimal("500"), "USDC": Decimal("500")}

    def test_merge_disappeared_symbol_zeroes_emits_one_day_clear_markers(self) -> None:
        result = aave._merge_disappeared_symbol_zeroes(
            leg_columns={
                "supply_USDC": Decimal("3"),
                "debt_USDC": Decimal("0"),
                "net_USDC": Decimal("3"),
            },
            current_symbols={"USDC"},
            previous_active_symbols={"USDC", "WBTC", "LINK"},
            current_state_known=True,
        )

        assert result["supply_USDC"] == Decimal("3")
        assert result["net_USDC"] == Decimal("3")
        assert result["supply_WBTC"] == Decimal("0")
        assert result["debt_WBTC"] == Decimal("0")
        assert result["net_WBTC"] == Decimal("0")
        assert result["supply_LINK"] == Decimal("0")
        assert result["debt_LINK"] == Decimal("0")
        assert result["net_LINK"] == Decimal("0")

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
            "2026-01-01 00:00:00",
            "2026-01-02 00:00:00",
            "2026-01-03 00:00:00",
        ]
        assert history[-1]["rpc_error_count"] == 0
        assert history[-1]["supply_USDC"] == 0.0
        assert history[-1]["debt_USDC"] == 0.0
        assert history[-1]["net_USDC"] == 0.0

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
        assert history[2]["date"] == "2026-01-03 00:00:00"
        assert history[2]["rpc_error_count"] == 1
        assert history[-1]["date"] == "2026-01-04 00:00:00"
        assert history[-1]["supply_USDC"] == 0.0
        assert history[-1]["debt_USDC"] == 0.0
        assert history[-1]["net_USDC"] == 0.0

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

    def test_get_curve_underlying_uses_lp_address_when_minter_is_missing(self) -> None:
        lp_address = "0xlp"
        w3 = FakeCurveWeb3(
            contracts={
                lp_address: FakeLPContract(
                    total_supply=200,
                    pool_address="0xunused",
                    minter_error=RuntimeError("execution reverted"),
                ),
            }
        )
        pool_tokens = [
            curve.CurvePoolToken(address="0xA", balance=4 * 10**8, symbol="WBTC", decimals=8),
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

        read_pool_mock.assert_called_once_with(w3=w3, pool_address=lp_address, block_number=123)
        assert result["WBTC"] == Decimal("2")

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
        assert result == "2026-01-06 00:00:00"

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
        assert result == "2025-09-01 00:00:00"

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
        assert result == "2026-01-01 00:00:00"

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
        assert result == "2026-01-10 00:00:00"

    def test_resolve_protocol_end_date_treats_dust_balance_as_closed(self) -> None:
        result = common.resolve_protocol_end_date(
            {"qty": Decimal("0.000000001"), "end": pd.Timestamp("2026-01-10")}
        )

        assert result == "2026-01-10 00:00:00"

    def test_resolve_protocol_end_date_keeps_material_positive_balance_active(self) -> None:
        result = common.resolve_protocol_end_date(
            {"qty": Decimal("0.0000001"), "end": pd.Timestamp("2026-01-10")}
        )

        assert result == "now"

    def test_resolve_protocol_end_date_does_not_extend_negative_wrappers(self) -> None:
        result = common.resolve_protocol_end_date(
            {"qty": Decimal("-1"), "end": pd.Timestamp("2026-01-10")}
        )

        assert result == "2026-01-10 00:00:00"

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

            assert [row["date"] for row in rows] == [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
            ]
            assert rows[1]["block"] == "20"
            assert rows[1]["legacy_col"] == "keep2"
            assert rows[2]["block"] == "30"
            assert rows[2]["asset_B"] == "33"

    def test_write_protocol_history_csv_replace_from_date_removes_overlap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_dir = root / "curve"
            protocol_dir.mkdir(parents=True, exist_ok=True)
            output_path = protocol_dir / "arbitrum_LP.csv"

            pd.DataFrame(
                [
                    {"date": "2026-01-01", "block": 10, "asset_A": 1.1},
                    {"date": "2026-01-02", "block": 20, "asset_A": 2.2},
                    {"date": "2026-01-03", "block": 30, "asset_A": 3.3},
                ]
            ).to_csv(output_path, index=False)

            with patch("blockchain_reader.protocols.common.PROTOCOL_UNDERLYING_TOKEN_FOLDER", root):
                output = common.write_protocol_history_csv(
                    protocol="curve",
                    chain="arbitrum",
                    symbol="LP",
                    history_data=[
                        {"date": "2026-01-02", "block": 200, "asset_A": 22.0},
                        {"date": "2026-01-04", "block": 400, "asset_A": 44.0},
                    ],
                    replace_from_date="2026-01-02",
                )

            assert output == output_path
            result = pd.read_csv(output_path)
            assert result["date"].tolist() == [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-04 00:00:00",
            ]
            assert result["block"].tolist() == [10, 200, 400]

    def test_write_protocol_history_csv_replace_from_date_handles_empty_incoming(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_dir = root / "curve"
            protocol_dir.mkdir(parents=True, exist_ok=True)
            output_path = protocol_dir / "arbitrum_LP.csv"

            pd.DataFrame(
                [
                    {"date": "2026-01-01", "block": 10, "asset_A": 1.1},
                    {"date": "2026-01-02", "block": 20, "asset_A": 2.2},
                ]
            ).to_csv(output_path, index=False)

            with patch("blockchain_reader.protocols.common.PROTOCOL_UNDERLYING_TOKEN_FOLDER", root):
                output = common.write_protocol_history_csv(
                    protocol="curve",
                    chain="arbitrum",
                    symbol="LP",
                    history_data=[],
                    replace_from_date="2026-01-02",
                )

            assert output == output_path
            result = pd.read_csv(output_path)
            assert result["date"].tolist() == ["2026-01-01 00:00:00"]

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
            replace_from_date=None,
            logger=ANY,
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
            replace_from_date=None,
            logger=ANY,
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

    def test_get_liquid_staking_history_writes_scaled_eth_ratio(self) -> None:
        rate_provider = FakeRateProviderContract(
            address="0xrateprovider",
            rate_by_block={
                11: 1_100_000_000_000_000_000,
                12: 1_120_000_000_000_000_000,
            },
        )
        fake_w3 = FakeLiquidStakingWeb3(contract=rate_provider)
        write_mock = Mock(return_value=Path("lst_out.csv"))

        with (
            patch(
                "blockchain_reader.protocols.liquid_staking.load_chain_web3",
                return_value=fake_w3,
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.load_block_map",
                return_value={"2026-01-01 00:00:00": 11, "2026-01-02 00:00:00": 12},
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.write_protocol_history_csv",
                write_mock,
            ),
        ):
            liquid_staking.get_liquid_staking_history(
                chain="arbitrum",
                symbol="wstETH",
                underlying_symbol="ETH",
                rate_provider_address="0xrateprovider",
                start_date="2026-01-01",
                end_date="2026-01-02",
            )

        history = write_mock.call_args.kwargs["history_data"]
        assert rate_provider.rate_call_blocks == [11, 12]
        assert [row["date"] for row in history] == [
            "2026-01-01 00:00:00",
            "2026-01-02 00:00:00",
        ]
        assert history[0]["lst_balance"] == 1.0
        assert history[0]["asset_ETH"] == 1.1
        assert history[1]["asset_ETH"] == 1.12

    def test_process_all_liquid_staking_tokens_passes_resolved_incremental_start(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.liquid_staking.load_snapshot_ranges",
                return_value={
                    "wstETH": {
                        "start": pd.Timestamp("2024-01-01"),
                        "end": pd.Timestamp("2024-01-10"),
                        "qty": 0,
                    }
                },
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.load_block_map",
                return_value={"2024-01-01": 100},
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.resolve_effective_start_date",
                return_value="2024-01-05",
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.get_liquid_staking_history"
            ) as history_mock,
        ):
            liquid_staking.process_all_liquid_staking_tokens(chain="arbitrum")

        history_mock.assert_called_once_with(
            chain="arbitrum",
            symbol="wstETH",
            underlying_symbol="ETH",
            rate_provider_address="0xf7c5c26B574063e7b098ed74fAd6779e65E3F836",
            start_date="2024-01-05",
            end_date="2024-01-10 00:00:00",
            rate_provider_method="getRate",
            rate_scale=10**18,
            replace_from_date=None,
            logger=ANY,
        )

    def test_process_all_liquid_staking_tokens_uses_block_map_fallback_start(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.liquid_staking.load_snapshot_ranges",
                return_value={},
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.load_block_map",
                return_value={"2024-05-01": 12, "2024-01-15": 2},
            ),
            patch(
                "blockchain_reader.protocols.liquid_staking.resolve_effective_start_date",
                return_value=None,
            ) as resolve_start_mock,
        ):
            liquid_staking.process_all_liquid_staking_tokens(chain="arbitrum")

        assert resolve_start_mock.call_args.kwargs["fallback_start_date"] == "2024-01-15 00:00:00"

    def test_process_all_liquid_staking_tokens_skips_unsupported_chain(self) -> None:
        with (
            patch(
                "blockchain_reader.protocols.liquid_staking.load_snapshot_ranges"
            ) as snapshots_mock,
            patch("blockchain_reader.protocols.liquid_staking.load_block_map") as block_map_mock,
        ):
            liquid_staking.process_all_liquid_staking_tokens(chain="ethereum")

        snapshots_mock.assert_not_called()
        block_map_mock.assert_not_called()

    def test_generate_protocol_lp_price_files_merges_and_keeps_canonical_schema(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
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
                lp_prices_root / "LP.csv",
                index=False,
            )

            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [lp_prices_root / "LP.csv"]
            result = pd.read_csv(lp_prices_root / "LP.csv")
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
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            (protocol_root / "beefy").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
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
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert set(updated) == {lp_prices_root / "LP.csv", lp_prices_root / "MOO.csv"}
            lp_frame = pd.read_csv(lp_prices_root / "LP.csv")
            moo_frame = pd.read_csv(lp_prices_root / "MOO.csv")
            assert lp_frame.loc[0, "Price"] == 4000.0
            assert moo_frame.loc[0, "Price"] == 6000.0

    def test_generate_protocol_lp_price_files_prices_moofish_stable_pair(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "beefy").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_USDT": 0.5, "asset_USDC": 0.5}]).to_csv(
                protocol_root / "beefy" / "arbitrum_mooFishUSDT-USDC.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "0xmoo": {"symbol": "mooFishUSDT-USDC", "protocol": "beefy"},
                        "0xusdt": {"symbol": "USDT"},
                        "0xusdc": {"symbol": "USDC"},
                    },
                    f,
                )

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [lp_prices_root / "mooFishUSDT-USDC.csv"]
            frame = pd.read_csv(lp_prices_root / "mooFishUSDT-USDC.csv")
            assert frame.loc[0, "Price"] == 1.0

    def test_generate_protocol_lp_price_files_prices_btc_wrapper_aliases(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_tBTC": 0.5, "asset_renBTC": 0.25}]).to_csv(
                protocol_root / "balancer" / "arbitrum_2BTC.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 40000.0}]).to_csv(
                prices_root / "BTC.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w", encoding="utf-8") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [lp_prices_root / "2BTC.csv"]
            frame = pd.read_csv(lp_prices_root / "2BTC.csv")
            assert frame.loc[0, "Price"] == 30000.0

    def test_generate_protocol_lp_price_files_skips_rows_with_unresolved_assets(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "curve").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_UNKNOWN": 1.0}]).to_csv(
                protocol_root / "curve" / "arbitrum_BAD.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w") as f:
                json.dump({}, f)

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == []
            assert not (lp_prices_root / "BAD.csv").exists()

    def test_generate_protocol_lp_price_files_excludes_aave_inputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "aave").mkdir(parents=True, exist_ok=True)
            (protocol_root / "balancer").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
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
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [lp_prices_root / "LP.csv"]
            assert (lp_prices_root / "LP.csv").exists()
            assert not (lp_prices_root / "AAVEWRAP.csv").exists()

    def test_generate_protocol_lp_price_files_prefers_protocol_rows_over_family_proxy(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "beefy").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_BTC": 1.0}]).to_csv(
                protocol_root / "beefy" / "arbitrum_WRAP.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 2000.0}]).to_csv(
                prices_root / "ETH.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 40000.0}]).to_csv(
                prices_root / "BTC.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "0xwrap": {"symbol": "WRAP", "family": "ETH", "protocol": "beefy"},
                    },
                    f,
                )

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert updated == [lp_prices_root / "WRAP.csv"]
            frame = pd.read_csv(lp_prices_root / "WRAP.csv")
            assert frame.loc[0, "Price"] == 40000.0

    def test_generate_protocol_lp_price_files_values_wsteth_from_liquid_staking_ratio(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            protocol_root = root / "protocol_underlying_tokens"
            prices_root = root / "prices"
            lp_prices_root = prices_root / "lp_prices" / "arbitrum"
            tokens_root = root / "tokens"
            (protocol_root / "liquid_staking").mkdir(parents=True, exist_ok=True)
            (protocol_root / "beefy").mkdir(parents=True, exist_ok=True)
            prices_root.mkdir(parents=True, exist_ok=True)
            lp_prices_root.mkdir(parents=True, exist_ok=True)
            tokens_root.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"date": "2024-01-02", "asset_ETH": 1.1}]).to_csv(
                protocol_root / "liquid_staking" / "arbitrum_wstETH.csv",
                index=False,
            )
            pd.DataFrame([{"date": "2024-01-02", "asset_wstETH": 2.0}]).to_csv(
                protocol_root / "beefy" / "arbitrum_WRAP.csv",
                index=False,
            )
            pd.DataFrame([{"Date": "2024-01-02", "Price": 2000.0}]).to_csv(
                prices_root / "ETH.csv",
                index=False,
            )
            with open(tokens_root / "arbitrum_tokens.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "0xwsteth": {
                            "symbol": "wstETH",
                            "family": "ETH",
                            "protocol": "liquid_staking",
                        },
                        "0xwrap": {"symbol": "WRAP", "family": "ETH", "protocol": "beefy"},
                    },
                    f,
                )

            with (
                patch(
                    "blockchain_reader.composition.lp_pricing.PROTOCOL_UNDERLYING_TOKEN_FOLDER",
                    protocol_root,
                ),
                patch("blockchain_reader.composition.lp_pricing.PRICES_FOLDER", prices_root),
                patch("blockchain_reader.composition.lp_pricing.TOKENS_FOLDER", tokens_root),
            ):
                updated = lp_pricing.generate_protocol_lp_price_files(chain="arbitrum")

            assert set(updated) == {lp_prices_root / "wstETH.csv", lp_prices_root / "WRAP.csv"}
            wsteth_frame = pd.read_csv(lp_prices_root / "wstETH.csv")
            wrap_frame = pd.read_csv(lp_prices_root / "WRAP.csv")
            assert wsteth_frame.loc[0, "Price"] == 2200.0
            assert wrap_frame.loc[0, "Price"] == 4400.0
