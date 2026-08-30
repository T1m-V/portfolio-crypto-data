from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_crypto_data import raw_snapshots
from portfolio_crypto_data.cex.crypto_com_app_snapshots import (
    SUPPORTED_APP_KINDS,
    CryptoComAppTransactionNormalizer,
    _build_dust_actions,
    generate_crypto_com_app_raw_snapshots,
)
from portfolio_crypto_data.cex.nexo_snapshots import (
    NexoTransactionNormalizer,
    _build_manual_repayment_actions,
    generate_nexo_raw_snapshots,
)
from portfolio_crypto_data.principal_ledger import (
    EconomicPrincipalLedger,
    PrincipalComponent,
)
from portfolio_crypto_data.raw_snapshots import (
    PortfolioLedger,
    TransactionApplier,
    TransactionParser,
)


def _chain_row(kind: str, **overrides: object) -> pd.Series:
    row: dict[str, object] = {
        "TX Hash": "hash",
        "Date": pd.Timestamp("2026-01-01 10:00"),
        "Type": kind,
        "Qty in": "",
        "Token in": "",
        "Qty out": "",
        "Token out": "",
        "Fee": pd.NA,
        "Fee Token": pd.NA,
    }
    row.update(overrides)
    return pd.Series(row)


def test_onchain_applier_handles_only_emitted_transaction_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 10.0)
    ledger = PortfolioLedger(chain="arbitrum", token_metadata={})
    applier = TransactionApplier(ledger=ledger)
    rows = [
        _chain_row("Receive", **{"Qty in": "2", "Token in": "ETH"}),
        _chain_row("Send", **{"Qty out": "0.5", "Token out": "ETH"}),
        _chain_row(
            "Swap",
            **{"Qty in": "3", "Token in": "ARB", "Qty out": "1", "Token out": "ETH"},
        ),
        _chain_row("Reward|ETH", **{"Qty in": "0.2", "Token in": "ARB"}),
        _chain_row("Approve ARB"),
        _chain_row("Interaction"),
    ]

    for row in rows:
        applier.process_transaction(row)

    assert ledger.assets["ETH"].quantity == Decimal("0.5")
    assert ledger.assets["ARB"].quantity == Decimal("3.2")
    assert "REWARD" not in ledger.assets
    with pytest.raises(ValueError, match="Unsupported transaction type"):
        applier.process_transaction(_chain_row("Buy"))


def test_approval_fee_reduces_native_token_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 10.0)
    ledger = PortfolioLedger(chain="arbitrum", token_metadata={})
    applier = TransactionApplier(ledger=ledger)

    applier.process_transaction(
        _chain_row("Receive", **{"Qty in": "2", "Token in": "ETH"})
    )
    applier.process_transaction(
        _chain_row("Approve ARB", **{"Fee": "0.01", "Fee Token": "ETH"})
    )

    assert ledger.assets["ETH"].quantity == Decimal("1.99")
    assert ledger.history[-1]["Quantity"] == Decimal("1.99")


def test_transaction_parser_rejects_mismatched_external_fields() -> None:
    parser = TransactionParser()
    assert [(entry.token, entry.quantity) for entry in parser.parse_entries(
        qty_val="1, 2", token_val="ETH, USDC"
    )] == [("ETH", Decimal("1")), ("USDC", Decimal("2"))]
    with pytest.raises(ValueError, match="Mismatched"):
        parser.parse_entries(qty_val="1, 2", token_val="ETH")


class SplitResolver:
    def components(self, *, symbol: str, date_value: object) -> list[PrincipalComponent]:
        del symbol, date_value
        return [
            PrincipalComponent("ETH", Decimal("0.75")),
            PrincipalComponent("BTC", Decimal("0.25")),
        ]


def test_principal_ledger_distributes_exactly_and_keeps_daily_last_state() -> None:
    ledger = EconomicPrincipalLedger(resolver=SplitResolver())
    ledger.adjust(
        symbol="LP", amount_eur=100, date_value="2026-01-01", action="receive", tx_hash="a"
    )
    ledger.adjust(
        symbol="LP", amount_eur=-20, date_value="2026-01-01", action="send", tx_hash="b"
    )

    assert ledger.balances == {"ETH": Decimal("60.00"), "BTC": Decimal("20.00")}
    daily = ledger.daily_frame().set_index("Coin")
    assert daily["PrincipalInvestedEUR"].to_dict() == {"BTC": 20.0, "ETH": 60.0}
    assert set(ledger.events_frame()["TX Hash"]) == {"a", "b"}


ACTUAL_NEXO_TYPES = {
    "assimilation",
    "bonus",
    "cashback",
    "credit card fiatx exchange to withdraw",
    "credit card withdrawal credit",
    "deposit over repayment",
    "deposit to exchange",
    "dividend",
    "exchange",
    "exchange cashback",
    "exchange credit",
    "exchange deposited on",
    "exchange liquidation",
    "exchange to withdraw",
    "fixed term interest",
    "interest",
    "interest additional",
    "loan withdrawal",
    "locking term deposit",
    "manual repayment",
    "manual sell order",
    "nexo card cashback reversal",
    "nexo card purchase",
    "nexo card refund",
    "nexo card transaction fee",
    "referral bonus",
    "top up crypto",
    "transfer in",
    "transfer out",
    "unlocking term deposit",
    "withdraw exchanged",
    "withdrawal",
}


def _nexo_row(kind: str, **overrides: object) -> pd.Series:
    row: dict[str, object] = {
        "Type": kind,
        "Input Currency": "USDC",
        "Input Amount": "2",
        "Output Currency": "BTC",
        "Output Amount": "1",
        "USD Equivalent": "$2",
        "Fee": "0",
        "Fee Currency": "USD",
        "Details": "approved / 0.5 BTC",
        "Date": pd.Timestamp("2026-01-01 10:00"),
        "Date / Time (UTC)": "01/01/2026 10:00",
    }
    row.update(overrides)
    return pd.Series(row)


def test_nexo_support_is_exactly_the_types_present_in_the_exports() -> None:
    normalizer = NexoTransactionNormalizer(known_symbols={"BTC", "USDC", "USD"})
    supported = set(normalizer.handlers)
    assert supported == ACTUAL_NEXO_TYPES


@pytest.mark.parametrize(
    ("kind", "overrides", "expected"),
    [
        ("Assimilation", {}, "receive"),
        ("Top up Crypto", {}, "receive"),
        ("Bonus", {}, "reward"),
        ("Dividend", {}, "reward"),
        ("Referral Bonus", {}, "reward"),
        ("Exchange Cashback", {}, "reward"),
        ("Cashback", {}, "reward"),
        ("Interest", {}, "reward"),
        ("Fixed Term Interest", {}, "reward"),
        ("Exchange", {}, "swap"),
        ("Withdrawal", {"Input Amount": "-2"}, "send"),
        ("Loan Withdrawal", {"Input Amount": "-2"}, "send"),
        ("Nexo Card Transaction Fee", {"Input Amount": "-2"}, "send"),
        ("Deposit Over Repayment", {}, "send"),
        ("Exchange To Withdraw", {}, "send"),
        ("Transfer Out", {}, "skip"),
        ("Locking Term Deposit", {}, "skip"),
        ("Exchange Credit", {}, "skip"),
    ],
)
def test_nexo_rules_are_declarative(
    kind: str,
    overrides: dict[str, object],
    expected: str,
) -> None:
    normalizer = NexoTransactionNormalizer(known_symbols={"BTC", "USDC", "USD"})
    assert normalizer.normalize_row(row=_nexo_row(kind, **overrides)).action == expected


def test_nexo_special_accounting_rules() -> None:
    normalizer = NexoTransactionNormalizer(known_symbols={"BTC", "NEXO", "USD"})
    cashback = normalizer.normalize_row(row=_nexo_row("Cashback", **{"Input Currency": "NEXO"}))
    interest = normalizer.normalize_row(row=_nexo_row("Interest", **{"Input Currency": "NEXO"}))
    debt_interest = normalizer.normalize_row(
        row=_nexo_row("Interest", **{"Input Currency": "xUSD", "Input Amount": "-3"})
    )
    rejected = normalizer.normalize_row(
        row=_nexo_row("Nexo Card Purchase", **{"Details": "rejected / merchant"})
    )

    assert cashback.rewards[0].allocations == [(None, 0.5), ("USD", 0.5)]
    assert interest.rewards[0].allocations == [("BTC", 0.75), ("NEXO", 0.25)]
    assert [(entry.token, entry.quantity) for entry in debt_interest.outs] == [
        ("USD", Decimal("3"))
    ]
    assert debt_interest.principal_overrides == {"USD": 0.0}
    assert rejected.action == "skip"


def test_manual_repayment_pair_becomes_one_swap() -> None:
    frame = pd.DataFrame(
        [
            _nexo_row(
                "Manual Sell Order",
                **{"Input Currency": "BTC", "Input Amount": "-1", "USD Equivalent": "$100"},
            ),
            _nexo_row(
                "Exchange Liquidation",
                **{"Input Currency": "BTC", "Input Amount": "-1", "USD Equivalent": "$100"},
            ),
            _nexo_row(
                "Manual Repayment",
                **{
                    "Input Currency": "USD",
                    "Input Amount": "100",
                    "USD Equivalent": "$100",
                    "Date": pd.Timestamp("2026-01-01 10:01"),
                },
            ),
        ]
    )
    normalizer = NexoTransactionNormalizer.from_dataframe(frame)

    actions = _build_manual_repayment_actions(frame=frame, normalizer=normalizer)

    assert list(actions) == [2]
    action = actions[2]
    assert [(entry.token, entry.quantity) for entry in action.ins] == [("USD", Decimal("100"))]
    assert [(entry.token, entry.quantity) for entry in action.outs] == [("BTC", Decimal("1"))]
    assert action.principal_overrides == {"USD": 100.0, "BTC": -100.0}


def _generate_nexo(
    *,
    rows: list[dict[str, object]],
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pd.DataFrame:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    input_path = folder / "input.csv"
    output_path = folder / "output.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    generate_nexo_raw_snapshots(input_csv=input_path, output_csv=output_path)
    return pd.read_csv(output_path)


def test_nexo_generator_combines_exports_and_overwrites_same_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    for file_name, amount, time in (("a.csv", "1", "10:00"), ("b.csv", "2", "11:00")):
        pd.DataFrame(
            [
                {
                    **_nexo_row("Cashback").to_dict(),
                    "Input Currency": "NEXO",
                    "Input Amount": amount,
                    "Date / Time (UTC)": f"01/01/2026 {time}",
                }
            ]
        ).drop(columns=["Date"]).to_csv(tmp_path / file_name, index=False)

    output = tmp_path / "out" / "snapshots.csv"
    output.parent.mkdir()
    generate_nexo_raw_snapshots(input_csv=tmp_path, output_csv=output)
    frame = pd.read_csv(output)

    nexo = frame[frame["Coin"] == "NEXO"]
    assert len(nexo) == 1
    assert (nexo.iloc[0]["Quantity"], nexo.iloc[0]["Principal Invested"]) == (3.0, 1.5)
    assert frame["Date"].tolist() == sorted(frame["Date"].tolist())


def test_nexo_overlapping_exports_preserve_real_duplicate_occurrences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    duplicate = {
        **_nexo_row("Cashback").to_dict(),
        "Input Currency": "NEXO",
        "Input Amount": "2",
    }
    for file_name in ("first.csv", "second.csv"):
        pd.DataFrame([duplicate, duplicate]).drop(columns=["Date"]).to_csv(
            tmp_path / file_name,
            index=False,
        )

    output = tmp_path / "out" / "snapshots.csv"
    generate_nexo_raw_snapshots(input_csv=tmp_path, output_csv=output)

    snapshot = pd.read_csv(output)
    assert snapshot.loc[snapshot["Coin"] == "NEXO", "Quantity"].iloc[-1] == 4.0


def test_nexo_rejects_malformed_non_empty_amount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Invalid non-empty numeric value"):
        _generate_nexo(
            rows=[
                {
                    **_nexo_row("Cashback").to_dict(),
                    "Input Amount": "not-a-number",
                }
            ],
            folder=tmp_path,
            monkeypatch=monkeypatch,
        )


def test_nexo_generator_rejects_unknown_external_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported NEXO transaction type"):
        _generate_nexo(
            rows=[
                {
                    **_nexo_row("Something New").to_dict(),
                    "Date / Time (UTC)": "01/01/2026 10:00",
                }
            ],
            folder=tmp_path,
            monkeypatch=monkeypatch,
        )


ACTUAL_CRYPTO_COM_APP_KINDS = {
    "card_cashback_reverted",
    "crypto_deposit",
    "crypto_exchange",
    "crypto_to_exchange_transfer",
    "crypto_transfer",
    "crypto_withdrawal",
    "dust_conversion_credited",
    "dust_conversion_debited",
    "exchange_to_crypto_transfer",
    "finance.lockup.dpos_compound_interest.crypto_wallet",
    "finance.lockup.dpos_lock.crypto_wallet",
    "finance.lockup.dpos_lock_upgrade.crypto_wallet",
    "referral_bonus",
    "referral_card_cashback",
    "rewards_platform_deposit_credited",
    "transfer_cashback",
}


def _crypto_com_app_row(kind: str, **overrides: object) -> pd.Series:
    row: dict[str, object] = {
        "Timestamp (UTC)": "2026-01-01 10:00:00",
        "Transaction Description": "Synthetic transaction",
        "Currency": "CRO",
        "Amount": "2",
        "To Currency": "",
        "To Amount": "",
        "Native Currency": "EUR",
        "Native Amount": "1",
        "Native Amount (in USD)": "1.1",
        "Transaction Kind": kind,
        "Transaction Hash": "",
        "Date": pd.Timestamp("2026-01-01 10:00:00"),
    }
    row.update(overrides)
    return pd.Series(row)


def test_crypto_com_app_support_matches_the_sample_export() -> None:
    assert SUPPORTED_APP_KINDS == ACTUAL_CRYPTO_COM_APP_KINDS


@pytest.mark.parametrize(
    ("kind", "overrides", "expected"),
    [
        ("referral_card_cashback", {}, "reward"),
        ("referral_bonus", {}, "reward"),
        ("transfer_cashback", {}, "reward"),
        ("crypto_deposit", {}, "receive"),
        ("exchange_to_crypto_transfer", {}, "receive"),
        ("crypto_withdrawal", {"Amount": "-2"}, "send"),
        ("crypto_transfer", {"Amount": "-2"}, "send"),
        ("crypto_to_exchange_transfer", {"Amount": "-2"}, "send"),
        (
            "finance.lockup.dpos_lock.crypto_wallet",
            {"Amount": "-2"},
            "skip",
        ),
        ("card_cashback_reverted", {"Amount": "-2"}, "send"),
        (
            "crypto_exchange",
            {
                "Currency": "USDC",
                "Amount": "-10",
                "To Currency": "CRO",
                "To Amount": "20",
            },
            "swap",
        ),
    ],
)
def test_crypto_com_app_rules_are_explicit(
    kind: str,
    overrides: dict[str, object],
    expected: str,
) -> None:
    normalizer = CryptoComAppTransactionNormalizer()
    action = normalizer.normalize_row(_crypto_com_app_row(kind, **overrides))
    assert action.action == expected
    if kind == "card_cashback_reverted":
        assert action.principal_overrides == {"CRO": 0.0}


def test_crypto_com_app_dust_legs_become_one_swap() -> None:
    frame = pd.DataFrame(
        [
            _crypto_com_app_row(
                "dust_conversion_credited",
                **{
                    "Transaction Description": "Convert Dust",
                    "Currency": "CRO",
                    "Amount": "0.5",
                },
            ),
            _crypto_com_app_row(
                "dust_conversion_debited",
                **{
                    "Transaction Description": "Convert Dust",
                    "Currency": "USDC",
                    "Amount": "-1",
                },
            ),
        ]
    )

    actions, consumed = _build_dust_actions(
        frame=frame,
        normalizer=CryptoComAppTransactionNormalizer(),
    )

    assert consumed == {0, 1}
    assert list(actions) == [0]
    assert [(entry.token, entry.quantity) for entry in actions[0].outs] == [
        ("USDC", Decimal("1"))
    ]
    assert [(entry.token, entry.quantity) for entry in actions[0].ins] == [
        ("CRO", Decimal("0.5"))
    ]


def _write_crypto_com_app_export(path: Path, rows: list[pd.Series]) -> None:
    pd.DataFrame(rows).drop(columns=["Date"]).to_csv(path, index=False)


def test_crypto_com_app_generator_keeps_entities_separate_and_pairs_dust(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    rows = [
        _crypto_com_app_row("crypto_deposit", **{"Currency": "USDC", "Amount": "2"}),
        _crypto_com_app_row(
            "crypto_exchange",
            **{
                "Currency": "USDC",
                "Amount": "-1",
                "To Currency": "BNB",
                "To Amount": "2",
            },
        ),
        _crypto_com_app_row("referral_card_cashback", **{"Currency": "CRO", "Amount": "3"}),
        _crypto_com_app_row("card_cashback_reverted", **{"Currency": "CRO", "Amount": "-1"}),
        _crypto_com_app_row(
            "crypto_to_exchange_transfer",
            **{"Currency": "BNB", "Amount": "-1"},
        ),
        _crypto_com_app_row(
            "exchange_to_crypto_transfer",
            **{"Currency": "ETH", "Amount": "4"},
        ),
        _crypto_com_app_row(
            "finance.lockup.dpos_lock.crypto_wallet",
            **{"Currency": "CRO", "Amount": "-2"},
        ),
        _crypto_com_app_row(
            "dust_conversion_credited",
            **{
                "Transaction Description": "Convert Dust",
                "Currency": "CRO",
                "Amount": "0.5",
            },
        ),
        _crypto_com_app_row(
            "dust_conversion_debited",
            **{
                "Transaction Description": "Convert Dust",
                "Currency": "USDC",
                "Amount": "-1",
            },
        ),
    ]
    input_csv = tmp_path / "app.csv"
    output_csv = tmp_path / "snapshots" / "app_raw_snapshots.csv"
    _write_crypto_com_app_export(input_csv, rows)

    generate_crypto_com_app_raw_snapshots(input_csv=input_csv, output_csv=output_csv)

    snapshot = pd.read_csv(output_csv).set_index("Coin")
    assert snapshot["Quantity"].to_dict() == {
        "BNB": 1.0,
        "CRO": 2.5,
        "ETH": 4.0,
        "USDC": 0.0,
    }
    assert snapshot["Principal Invested"].to_dict() == {
        "BNB": 0.0,
        "CRO": 1.0,
        "ETH": 4.0,
        "USDC": 0.0,
    }


def test_crypto_com_app_overlapping_exports_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    duplicate = _crypto_com_app_row("referral_card_cashback", **{"Amount": "2"})
    _write_crypto_com_app_export(tmp_path / "first.csv", [duplicate, duplicate])
    _write_crypto_com_app_export(tmp_path / "second.csv", [duplicate, duplicate])
    output_csv = tmp_path / "out" / "app_raw_snapshots.csv"

    generate_crypto_com_app_raw_snapshots(input_csv=tmp_path, output_csv=output_csv)

    snapshot = pd.read_csv(output_csv)
    assert snapshot.loc[snapshot["Coin"] == "CRO", "Quantity"].iloc[-1] == 4.0


def test_crypto_com_app_generator_rejects_unknown_kind_and_unmatched_dust(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.csv"
    _write_crypto_com_app_export(unknown, [_crypto_com_app_row("something_new")])
    with pytest.raises(ValueError, match="Unsupported Crypto.com App transaction kind"):
        generate_crypto_com_app_raw_snapshots(
            input_csv=unknown,
            output_csv=tmp_path / "unknown-output.csv",
        )

    unmatched = tmp_path / "unmatched.csv"
    _write_crypto_com_app_export(
        unmatched,
        [
            _crypto_com_app_row(
                "dust_conversion_credited",
                **{"Transaction Description": "Convert Dust"},
            )
        ],
    )
    with pytest.raises(ValueError, match="Unmatched Crypto.com App dust conversion"):
        generate_crypto_com_app_raw_snapshots(
            input_csv=unmatched,
            output_csv=tmp_path / "unmatched-output.csv",
        )
