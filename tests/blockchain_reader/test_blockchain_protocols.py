import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from blockchain_reader.protocols import aave, curve


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


class BlockchainProtocolTests(unittest.TestCase):
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

        self.assertEqual(minute_value, datetime(2022, 8, 25, 17, 35))
        self.assertEqual(second_value, datetime(2025, 8, 3, 15, 32, 9))
        self.assertIsNone(aave._parse_date_value("2025-08-03"))

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

        self.assertEqual(queried_blocks, [11, 12, 13])
        self.assertEqual(
            [row["date"] for row in history],
            [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
        )
        self.assertEqual(history[-1]["rpc_error_count"], 0)

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

        self.assertEqual(queried_blocks, [11, 12, 13, 14])
        self.assertEqual(history[2]["date"], date(2026, 1, 3))
        self.assertEqual(history[2]["rpc_error_count"], 1)
        self.assertEqual(history[-1]["date"], date(2026, 1, 4))

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

        self.assertEqual(pool.coins_calls, [0, 1, 2])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], curve.CurvePoolToken)
        self.assertEqual(
            result[0],
            curve.CurvePoolToken(
                address=token_a,
                balance=1000,
                symbol="USDC",
                decimals=6,
            ),
        )
        self.assertEqual(
            result[1],
            curve.CurvePoolToken(
                address=token_b,
                balance=2000,
                symbol="WETH",
                decimals=18,
            ),
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
        self.assertEqual(result["USDC"], Decimal("0.001"))
        self.assertEqual(result["WETH"], Decimal("2"))


if __name__ == "__main__":
    unittest.main()
