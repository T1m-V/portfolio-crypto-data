from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from portfolio_core import active_context
from web3 import Web3

from portfolio_crypto_data.datetime_utils import (
    format_daily_datetime,
    parse_daily_datetime,
    parse_transaction_datetime,
    parse_transaction_datetime_series,
)
from portfolio_crypto_data.extraction import evm_reader, transaction_analyzer
from portfolio_crypto_data.extraction.evm_reader import (
    OUTPUT_COLUMNS,
    ExplorerAPIError,
    _derive_start_date,
    _fetch_explorer_data,
    _normalize_results_frame,
    retrieve_transactions,
)


@pytest.mark.parametrize("value", ["05/01/2026 11:30", "05/01/2026 11:30:12"])
def test_transaction_dates_accept_the_two_persisted_precisions(value: str) -> None:
    assert parse_transaction_datetime(value) == pd.Timestamp("2026-01-05 11:30:12").replace(
        second=12 if value.endswith("12") else 0
    )


def test_daily_dates_are_strict_and_canonical() -> None:
    assert parse_daily_datetime("2026-01-05") == pd.Timestamp("2026-01-05")
    assert parse_daily_datetime("01/05/2026") is None
    assert format_daily_datetime("2026-01-05 17:30:00") == "2026-01-05 00:00:00"
    parsed = parse_transaction_datetime_series(pd.Series(["05/01/2026 11:30", "bad"]))
    assert parsed.notna().tolist() == [True, False]


def test_reader_derives_one_day_overlap_and_normalizes_output(tmp_path) -> None:
    path = tmp_path / "transactions.csv"
    pd.DataFrame({"Date": ["01/01/2026 08:00", "05/01/2026 11:30:00"]}).to_csv(
        path, index=False
    )
    assert _derive_start_date(path) == "04/01/2026 00:00:00"

    frame = _normalize_results_frame(pd.DataFrame([{"TX Hash": "x", "Fee": 1.2}]))
    assert frame.columns.tolist() == OUTPUT_COLUMNS
    assert frame.iloc[0]["Fee"] == "1.2"
    assert frame.iloc[0]["Token in"] == ""


def test_explorer_no_transactions_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "0",
        "message": "No transactions found",
        "result": "No transactions found",
    }
    get = Mock(return_value=response)
    monkeypatch.setattr("portfolio_crypto_data.extraction.evm_reader.requests.get", get)

    assert _fetch_explorer_data("https://example.test", {"action": "txlist"}) == []
    assert get.call_count == 1


def test_explorer_request_failure_aborts_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    get = Mock(side_effect=requests.ConnectionError("temporarily unavailable"))
    monkeypatch.setattr(evm_reader.requests, "get", get)

    with pytest.raises(ExplorerAPIError, match="action=txlist"):
        _fetch_explorer_data(
            "https://example.test",
            {"action": "txlist"},
            max_retries=0,
        )


def test_unexpected_explorer_response_aborts_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "0",
        "message": "NOTOK",
        "result": "Rate limit reached",
    }
    monkeypatch.setattr(evm_reader.requests, "get", Mock(return_value=response))

    with pytest.raises(ExplorerAPIError, match="Unexpected explorer response"):
        _fetch_explorer_data(
            "https://example.test",
            {"action": "tokentx"},
            max_retries=0,
        )


def test_non_object_explorer_response_aborts_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = []
    monkeypatch.setattr(evm_reader.requests, "get", Mock(return_value=response))

    with pytest.raises(ExplorerAPIError, match="expected an object, got list"):
        _fetch_explorer_data(
            "https://example.test",
            {"action": "tokentx"},
            max_retries=0,
        )


def test_rpc_connection_failure_aborts_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = active_context().paths
    paths.chain_config.parent.mkdir(parents=True, exist_ok=True)
    paths.chain_config.write_text(
        json.dumps(
            {
                "arbitrum": {
                    "my_address": "0xabc",
                    "api_url": "https://example.test",
                    "api_key": "test",
                    "chain_id": "42161",
                    "rpc_url": "https://rpc.example.test",
                }
            }
        ),
        encoding="utf-8",
    )

    class DisconnectedWeb3:
        HTTPProvider = staticmethod(lambda endpoint_uri: endpoint_uri)

        def __init__(self, provider: object) -> None:
            self.provider = provider

        def is_connected(self) -> bool:
            return False

    monkeypatch.setattr(evm_reader, "Web3", DisconnectedWeb3)

    with pytest.raises(ConnectionError, match="RPC connection failed"):
        asyncio.run(retrieve_transactions(chain="arbitrum"))


class TokenManager:
    def get_token(self, address: str, fetch_if_missing: bool = False):
        del address, fetch_if_missing
        return {"symbol": "USDC", "decimals": 6, "resolved": True}


def _topic(address: str) -> bytes:
    return Web3.to_bytes(hexstr=f"0x{'0' * 24}{address[2:]}")


def test_transfer_logs_are_netted_in_wallet_direction() -> None:
    wallet = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    other = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    receipt = {
        "logs": [
            {
                "address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "topics": [
                    Web3.keccak(text="Transfer(address,address,uint256)"),
                    _topic(other),
                    _topic(wallet),
                ],
                "data": (1_500_000).to_bytes(32),
            },
            {
                "address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "topics": [
                    Web3.keccak(text="Transfer(address,address,uint256)"),
                    _topic(wallet),
                    _topic(other),
                ],
                "data": (250_000).to_bytes(32),
            },
        ]
    }

    incoming, outgoing, approvals = transaction_analyzer._get_token_movements(
        receipt=receipt,
        my_address=wallet,
        token_manager=TokenManager(),
        fetch_metadata=False,
    )

    assert [(item.symbol, item.qty) for item in incoming] == [("USDC", Decimal("1.5"))]
    assert [(item.symbol, item.qty) for item in outgoing] == [("USDC", Decimal("0.25"))]
    assert approvals == []


def test_analyzer_emits_canonical_utc_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    wallet = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    context = transaction_analyzer.TransactionContext(
        tx={"from": wallet, "to": "0xother", "value": 10**18, "blockNumber": 1},
        receipt={"gasUsed": 21_000, "effectiveGasPrice": 10**9, "logs": []},
        block={"timestamp": 0},
    )
    monkeypatch.setattr(transaction_analyzer, "_fetch_transaction_data", lambda *_, **__: context)

    result = transaction_analyzer.analyze_transaction(
        tx_hash="0xhash",
        w3=None,
        my_address=wallet,
        token_manager=TokenManager(),
        internal_eth_map={},
        fetch_metadata=False,
    )

    assert result is not None
    assert (result["Date"], result["Type"], result["Fee Token"]) == (
        "01/01/1970 00:00:00",
        "Send",
        "ETH",
    )
