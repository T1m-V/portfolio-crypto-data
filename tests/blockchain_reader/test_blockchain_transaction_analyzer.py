import unittest
from decimal import Decimal
from unittest.mock import patch

from web3 import Web3

from blockchain_reader import transaction_analyzer as ta


class DummyTokenManager:
    def __init__(self, symbol: str = "USDC", decimals: int = 6):
        self._token = {"symbol": symbol, "decimals": decimals, "resolved": True}

    def get_token(self, address: str, fetch_if_missing: bool = False):  # noqa: ARG002
        return self._token


def _address_topic(address: str) -> bytes:
    return Web3.to_bytes(hexstr=f"0x{'0' * 24}{address.lower()[2:]}")


def _uint256_data(value: int) -> bytes:
    return value.to_bytes(32, byteorder="big")


class BlockchainTransactionAnalyzerTests(unittest.TestCase):
    def test_get_token_movements_transfer_in_and_out(self) -> None:
        my_address = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        other = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        token = "0xcccccccccccccccccccccccccccccccccccccccc"
        transfer_topic = Web3.keccak(text="Transfer(address,address,uint256)")

        receipt = {
            "logs": [
                {
                    "address": token,
                    "topics": [transfer_topic, _address_topic(other), _address_topic(my_address)],
                    "data": _uint256_data(1_500_000),  # 1.5 USDC
                },
                {
                    "address": token,
                    "topics": [transfer_topic, _address_topic(my_address), _address_topic(other)],
                    "data": _uint256_data(250_000),  # 0.25 USDC
                },
            ]
        }

        incoming, outgoing, approvals = ta._get_token_movements(
            receipt=receipt,
            my_address=my_address,
            token_manager=DummyTokenManager(),
            fetch_metadata=False,
        )

        self.assertEqual(len(incoming), 1)
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(approvals, [])
        self.assertEqual(incoming[0].symbol, "USDC")
        self.assertEqual(outgoing[0].symbol, "USDC")
        self.assertEqual(incoming[0].qty, Decimal("1.5"))
        self.assertEqual(outgoing[0].qty, Decimal("0.25"))

    def test_analyze_transaction_formats_utc_date(self) -> None:
        tx_hash = "0xhash"
        my_address = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        tx_context = ta.TransactionContext(
            tx={
                "from": my_address,
                "to": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "value": 10**18,
                "blockNumber": 1,
            },
            receipt={"gasUsed": 21_000, "effectiveGasPrice": 1_000_000_000, "logs": []},
            block={"timestamp": 0},
        )

        with patch("blockchain_reader.transaction_analyzer._fetch_transaction_data") as fetch_mock:
            fetch_mock.return_value = tx_context
            result = ta.analyze_transaction(
                tx_hash=tx_hash,
                w3=None,  # mocked
                my_address=my_address,
                token_manager=DummyTokenManager(),
                internal_eth_map={},
                fetch_metadata=False,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["Date"], "01/01/1970 00:00:00")
        self.assertEqual(result["Type"], "Send")
        self.assertEqual(result["Fee Token"], "ETH")
        self.assertNotEqual(result["Fee"], "0")


if __name__ == "__main__":
    unittest.main()
