from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_crypto_data import raw_snapshots
from portfolio_crypto_data.cex.crypto_com_app_snapshots import (
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

    applier.process_transaction(_chain_row("Receive", **{"Qty in": "2", "Token in": "ETH"}))
    applier.process_transaction(_chain_row("Approve ARB", **{"Fee": "0.01", "Fee Token": "ETH"}))

    assert ledger.assets["ETH"].quantity == Decimal("1.99")
    assert ledger.history[-1]["Quantity"] == Decimal("1.99")


def test_transaction_parser_rejects_mismatched_external_fields() -> None:
    parser = TransactionParser()
    assert [
        (entry.token, entry.quantity)
        for entry in parser.parse_entries(qty_val="1, 2", token_val="ETH, USDC")
    ] == [("ETH", Decimal("1")), ("USDC", Decimal("2"))]
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
    ledger.adjust(symbol="LP", amount_eur=-20, date_value="2026-01-01", action="send", tx_hash="b")

    assert ledger.balances == {"ETH": Decimal("60.00"), "BTC": Decimal("20.00")}
    daily = ledger.daily_frame().set_index("Coin")
    assert daily["PrincipalInvestedEUR"].to_dict() == {"BTC": 20.0, "ETH": 60.0}
    assert set(ledger.events_frame()["TX Hash"]) == {"a", "b"}


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
    assert [(entry.token, entry.quantity) for entry in actions[0].outs] == [("USDC", Decimal("1"))]
    assert [(entry.token, entry.quantity) for entry in actions[0].ins] == [("CRO", Decimal("0.5"))]


def _write_crypto_com_app_export(path: Path, rows: list[pd.Series]) -> None:
    pd.DataFrame(rows).drop(columns=["Date"]).to_csv(path, index=False)


def test_crypto_com_app_historical_accounting(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    rows = [
        _crypto_com_app_row(
            "viban_purchase",
            **{
                "Currency": "EUR",
                "Amount": "-100",
                "To Currency": "CRO",
                "To Amount": "100",
            },
        ),
    ]
    for kind, amount in [
        ("crypto_earn_program_created", "-30"),
        ("crypto_earn_program_withdrawn", "30"),
        ("lockup_lock", "-20"),
        ("lockup_upgrade", "-10"),
        ("lockup_unlock", "30"),
        ("supercharger_deposit", "-10"),
        ("supercharger_withdrawal", "10"),
        ("finance.lockup.dpos_lock.crypto_wallet", "-20"),
        ("finance.lockup.dpos_lock_upgrade.crypto_wallet", "-5"),
        ("admin_wallet_credited", "3"),
        ("crypto_transfer", "4"),
        ("crypto_transfer", "-2"),
        ("card_top_up", "-5"),
        ("crypto_viban_exchange", "-10"),
        ("transfer.p2p_transfer.crypto_wallet.crypto_wallet.debit", "-7"),
    ]:
        rows.append(
            _crypto_com_app_row(
                kind,
                **{
                    "Amount": amount,
                    "To Currency": "CRO",
                    "To Amount": "7",
                },
            )
        )
    for kind in [
        "crypto_earn_interest_paid",
        "finance.lockup.dpos_non_compound_interest.crypto_wallet",
        "mco_stake_reward",
        "referral_gift",
        "reimbursement",
        "supercharger_reward_to_app_credited",
    ]:
        rows.append(_crypto_com_app_row(kind, **{"Amount": "2"}))
    rows.append(_crypto_com_app_row("reimbursement_reverted", **{"Amount": "-1"}))
    # Separate dates ensure internal moves preserve both quantity and principal
    # throughout the history, rather than merely cancelling at the end.
    for idx, row in enumerate(rows):
        row["Timestamp (UTC)"] = str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx))
    source = tmp_path / "history.csv"
    output = tmp_path / "out.csv"
    _write_crypto_com_app_export(source, rows)
    generate_crypto_com_app_raw_snapshots(source, output)
    snapshots = pd.read_csv(output)
    assert set(snapshots["Coin"]) == {"CRO"}
    assert snapshots.iloc[0]["Quantity"] == 100
    assert snapshots.iloc[1]["Quantity"] == 103
    assert snapshots.iloc[-1]["Quantity"] == 94
    assert snapshots.iloc[-1]["Principal Invested"] == 83


@pytest.mark.parametrize(
    ("kind", "amount"),
    [
        ("crypto_transfer", "0"),
        ("crypto_transfer", "NaN"),
        ("crypto_transfer", "Infinity"),
        ("crypto_earn_program_created", "1"),
        ("lockup_unlock", "-1"),
        ("reimbursement_reverted", "1"),
    ],
)
def test_crypto_com_app_rejects_invalid_historical_amounts(kind: str, amount: str) -> None:
    with pytest.raises(ValueError, match="Crypto.com App amount"):
        CryptoComAppTransactionNormalizer().normalize_row(
            _crypto_com_app_row(kind, **{"Amount": amount})
        )


def test_crypto_com_app_multi_asset_dust_preserves_principal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 1.0)
    rows = [
        _crypto_com_app_row("crypto_deposit", **{"Currency": coin, "Amount": "2"})
        for coin in ("BTC", "ETH")
    ]
    rows.extend(
        [
            _crypto_com_app_row(
                "dust_conversion_debited",
                **{
                    "Currency": coin,
                    "Amount": "-2",
                    "Timestamp (UTC)": "2026-01-02 10:00:00",
                },
            )
            for coin in ("BTC", "ETH")
        ]
    )
    rows.append(
        _crypto_com_app_row(
            "dust_conversion_credited",
            **{
                "Amount": "3",
                "Timestamp (UTC)": "2026-01-02 10:00:01",
            },
        )
    )
    source, output = tmp_path / "dust.csv", tmp_path / "out.csv"
    _write_crypto_com_app_export(source, rows)
    generate_crypto_com_app_raw_snapshots(source, output)
    latest = pd.read_csv(output).groupby("Coin").tail(1).set_index("Coin")
    assert latest["Quantity"].to_dict() == {"BTC": 0, "CRO": 3, "ETH": 0}
    assert latest["Principal Invested"].to_dict() == {"BTC": 0, "CRO": 4, "ETH": 0}


@pytest.mark.parametrize("offset", [0, 3])
def test_crypto_com_app_rejects_ambiguous_or_distant_dust(offset: int) -> None:
    rows = [
        _crypto_com_app_row("dust_conversion_credited"),
        _crypto_com_app_row(
            "dust_conversion_debited",
            **{
                "Amount": "-1",
                "Date": pd.Timestamp("2026-01-01 10:00:00") + pd.Timedelta(seconds=offset),
            },
        ),
    ]
    if offset == 0:
        rows.append(_crypto_com_app_row("dust_conversion_credited"))
    with pytest.raises(ValueError, match="Unmatched Crypto.com App dust conversion"):
        _build_dust_actions(
            frame=pd.DataFrame(rows), normalizer=CryptoComAppTransactionNormalizer()
        )


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


def test_compound_v2_supply_tokens_resolve_dynamic_underlying_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    monkeypatch.setattr(raw_snapshots, "get_crypto_price", lambda **_: 2000.0)
    token_metadata = {
        "0x1": {
            "symbol": "CDCETH",
            "decimals": 18,
            "family": "ETH",
        },
        "0x2": {
            "symbol": "tCDCETH",
            "decimals": 8,
            "family": "CDCETH",
            "price_source": "CDCETH",
            "protocol": "aave",
            "position_type": "supply",
            "contract_style": "compound_v2",
            "underlying_symbol": "CDCETH",
            "underlying_decimals": 18,
        },
    }
    ledger = PortfolioLedger(chain="cronos", token_metadata=token_metadata)
    ledger.aave_supply_by_date = {
        date(2026, 1, 2): {"CDCETH": Decimal("1.02")},
    }
    applier = TransactionApplier(ledger=ledger)

    # 1. Mint via Swap: 1.0 CDCETH -> 10.0 tCDCETH (dynamic ratio = 0.10)
    applier.process_transaction(
        _chain_row(
            "Swap",
            Date=pd.Timestamp("2026-01-01 10:00"),
            **{"Qty in": "10.0", "Token in": "tCDCETH", "Qty out": "1.0", "Token out": "CDCETH"},
        )
    )
    assert ledger.resolve_underlying_ratio("tCDCETH") == pytest.approx(0.10)
    assert ledger.assets["tCDCETH"].principal == pytest.approx(2000.0)

    # 2. Next day Send (liquidation of 5.0 tCDCETH) when exposure is 1.02 CDCETH (ratio = 0.102)
    applier.process_transaction(
        _chain_row(
            "Send",
            Date=pd.Timestamp("2026-01-02 10:00"),
            **{"Qty out": "5.0", "Token out": "tCDCETH"},
        )
    )
    assert ledger.resolve_underlying_ratio("tCDCETH") == pytest.approx(0.102)
    # 5.0 tCDCETH * 0.102 * 2000 EUR = 1020 EUR removed -> remaining principal = 980 EUR
    assert ledger.assets["tCDCETH"].principal == pytest.approx(980.0)

