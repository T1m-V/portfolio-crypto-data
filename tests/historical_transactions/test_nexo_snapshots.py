from pathlib import Path

import pandas as pd
import pytest

from blockchain_reader import raw_snapshots
from blockchain_reader.cex.nexo_snapshots import generate_nexo_raw_snapshots


def _run_nexo_generator(
    *,
    rows: list[dict[str, str]],
    tmp_path: Path,
    monkeypatch,
    price_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    prices = price_map or {}

    def fake_get_crypto_price(*, coin: str, date, chain: str, use_lp_prices: bool = False) -> float:
        del date, chain, use_lp_prices
        return prices.get(coin, 1.0)

    monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_get_crypto_price)

    input_path = tmp_path / "nexo_input.csv"
    output_path = tmp_path / "nexo_output.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    generate_nexo_raw_snapshots(input_csv=input_path, output_csv=output_path)
    return pd.read_csv(output_path)


def test_generator_reads_all_csv_files_in_transaction_folder(monkeypatch, tmp_path: Path) -> None:
    def fake_get_crypto_price(*, coin: str, date, chain: str, use_lp_prices: bool = False) -> float:
        del coin, date, chain, use_lp_prices
        return 1.0

    monkeypatch.setattr(raw_snapshots, "get_crypto_price", fake_get_crypto_price)

    pd.DataFrame(
        [
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / first part",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ]
    ).to_csv(tmp_path / "nexo_part_1.csv", index=False)
    pd.DataFrame(
        [
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "2",
                "Output Currency": "NEXO",
                "Output Amount": "2",
                "Details": "approved / second part",
                "Date / Time (UTC)": "01/01/2026 11:00",
            }
        ]
    ).to_csv(tmp_path / "nexo_part_2.csv", index=False)

    output_path = tmp_path / "nexo_output.csv"
    generate_nexo_raw_snapshots(input_csv=tmp_path / "nexo_part_1.csv", output_csv=output_path)
    result = pd.read_csv(output_path)

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    assert nexo_row["Quantity"] == 3.0
    assert nexo_row["Principal Invested"] == 1.5


def test_cashback_uses_half_free_and_half_usd_sourced(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "10",
                "Output Currency": "NEXO",
                "Output Amount": "10",
                "Details": "approved / test cashback",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0, "USD": 1.0},
    )

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert nexo_row["Quantity"] == 10.0
    assert nexo_row["Principal Invested"] == 10.0
    assert usd_row["Quantity"] == 0.0
    assert usd_row["Principal Invested"] == -10.0


def test_interest_uses_parsed_details_source_with_75_25_split(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Interest",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / 0.50000000 BTC",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0, "BTC": 1.0},
    )

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    btc_row = result[result["Coin"] == "BTC"].iloc[0]
    assert nexo_row["Quantity"] == 1.0
    assert nexo_row["Principal Invested"] == 1.5
    assert btc_row["Quantity"] == 0.0
    assert btc_row["Principal Invested"] == -1.5


def test_exchange_cashback_is_treated_as_free_reward(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange Cashback",
                "Input Currency": "EURX",
                "Input Amount": "7.55522139",
                "Output Currency": "EURX",
                "Output Amount": "7.55522139",
                "Details": "approved / 0.25% on top of your Exchange transaction",
                "Date / Time (UTC)": "24/06/2023 07:52",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == 7.55522139
    assert eurx_row["Principal Invested"] == 0.0


@pytest.mark.parametrize("tx_type", ["Bonus", "Dividend", "Referral Bonus"])
def test_bonus_income_types_are_treated_as_free_rewards(
    tx_type: str, monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": tx_type,
                "Input Currency": "NEXO",
                "Input Amount": "3",
                "Output Currency": "NEXO",
                "Output Amount": "3",
                "Details": f"approved / {tx_type}",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0},
    )

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    assert nexo_row["Quantity"] == 3.0
    assert nexo_row["Principal Invested"] == 0.0


def test_top_up_crypto_with_tiny_same_asset_rounding_delta_is_not_double_counted(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Top up Crypto",
                "Input Currency": "USDT",
                "Input Amount": "231.55595",
                "Output Currency": "USDT",
                "Output Amount": "231.5559504",
                "Details": (
                    "approved / "
                    "0x1382e977acd5de3aef23349ae31686ac7bd1681294dcabc4857ee577811ca03e"
                ),
                "Date / Time (UTC)": "16/12/2023 08:55",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDT": 1.0},
    )

    usdt_row = result[result["Coin"] == "USDT"].iloc[0]
    assert usdt_row["Quantity"] == 231.55595
    assert usdt_row["Principal Invested"] == 231.56


def test_interest_falls_back_to_input_coin_when_details_do_not_include_symbol(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Interest",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / Interest",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0},
    )

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    assert nexo_row["Quantity"] == 1.0
    assert nexo_row["Principal Invested"] == 0.0


def test_fixed_term_interest_in_kind_does_not_create_fake_source_coin(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Fixed Term Interest",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / NEXO Term Deposit Interest in kind",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0},
    )

    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    assert nexo_row["Quantity"] == 1.0
    assert nexo_row["Principal Invested"] == 0.0
    assert "kind" not in result["Coin"].astype(str).str.lower().values


def test_assimilation_is_explicit_receive_into_usd(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Assimilation",
                "Input Currency": "xUSD",
                "Input Amount": "2",
                "Output Currency": "-",
                "Output Amount": "0",
                "Details": "approved / Assimilation",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == 2.0
    assert usd_row["Principal Invested"] == 2.0


def test_deposit_over_repayment_reduces_usd_debt_when_repayment_includes_excess(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Purchase",
                "Input Currency": "USDX",
                "Input Amount": "-18.25",
                "Output Currency": "EUR",
                "Output Amount": "18.25",
                "USD Equivalent": "$18.25",
                "Details": "approved / card merchant",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Deposit Over Repayment",
                "Input Currency": "USDX",
                "Input Amount": "0.03",
                "Output Currency": "-",
                "Output Amount": "0",
                "USD Equivalent": "$0.03",
                "Details": "approved / Deposit over Repayment Compensation",
                "Date / Time (UTC)": "01/01/2026 10:01",
            },
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDT",
                "Input Amount": "-18.28",
                "Output Currency": "USDT",
                "Output Amount": "0",
                "USD Equivalent": "$18.28",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:02",
            },
            {
                "Type": "Exchange Liquidation",
                "Input Currency": "USDT",
                "Input Amount": "18.28",
                "Output Currency": "USDX",
                "Output Amount": "18.28",
                "USD Equivalent": "$18.28",
                "Details": "approved / Crypto repayment / Exchange USDT to USDX",
                "Date / Time (UTC)": "01/01/2026 10:03",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "USDX",
                "Input Amount": "18.28",
                "Output Currency": "USDX",
                "Output Amount": "0",
                "USD Equivalent": "$18.28",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:04",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EUR": 1.0, "USD": 1.0, "USDT": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[-1]
    assert usd_row["Quantity"] == 0.0
    assert usd_row["Principal Invested"] == 0.0


def test_unknown_types_raise_instead_of_using_generic_sign_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Unsupported NEXO transaction type"):
        _run_nexo_generator(
            rows=[
                {
                    "Type": "Random Outflow",
                    "Input Currency": "USDC",
                    "Input Amount": "-3",
                    "Output Currency": "USDC",
                    "Output Amount": "0",
                    "Details": "approved / Manual",
                    "Date / Time (UTC)": "01/01/2026 11:00",
                }
            ],
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            price_map={"USDC": 1.0},
        )


def test_output_is_daily_sorted_and_overwrites_same_day_coin(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / tx-2",
                "Date / Time (UTC)": "02/01/2026 10:00",
            },
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "1",
                "Output Currency": "NEXO",
                "Output Amount": "1",
                "Details": "approved / tx-1",
                "Date / Time (UTC)": "01/01/2026 12:00",
            },
            {
                "Type": "Cashback",
                "Input Currency": "NEXO",
                "Input Amount": "2",
                "Output Currency": "NEXO",
                "Output Amount": "2",
                "Details": "approved / tx-1b",
                "Date / Time (UTC)": "01/01/2026 14:00",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"NEXO": 2.0, "USD": 1.0},
    )

    assert list(result["Date"]) == sorted(result["Date"])

    day_one_nexo = result[(result["Date"] == "2026-01-01 00:00:00") & (result["Coin"] == "NEXO")]
    assert len(day_one_nexo) == 1
    assert day_one_nexo.iloc[0]["Quantity"] == 3.0
    assert day_one_nexo.iloc[0]["Principal Invested"] == 3.0


def test_internal_wallet_transfer_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Transfer Out",
                "Input Currency": "USDC",
                "Input Amount": "-10",
                "Output Currency": "USDC",
                "Output Amount": "10",
                "Details": "approved / Transfer from Savings Wallet to Credit Line Wallet",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0},
    )
    assert result.empty


def test_transfer_out_is_ignored_by_type_even_without_internal_wallet_details(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Transfer Out",
                "Input Currency": "USDC",
                "Input Amount": "-10",
                "Output Currency": "USDC",
                "Output Amount": "10",
                "Details": "approved / generic transfer",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0},
    )
    assert result.empty


def test_unlocking_term_deposit_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Unlocking Term Deposit",
                "Input Currency": "BTC",
                "Input Amount": "0.1",
                "Output Currency": "BTC",
                "Output Amount": "0.1",
                "Details": "approved / Transfer from Term Wallet to Savings Wallet",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"BTC": 50000.0},
    )
    assert result.empty


def test_locking_term_deposit_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Locking Term Deposit",
                "Input Currency": "BTC",
                "Input Amount": "-0.1",
                "Output Currency": "BTC",
                "Output Amount": "0.1",
                "Details": "approved / Transfer from Savings Wallet to Term Wallet",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"BTC": 50000.0},
    )
    assert result.empty


def test_withdrawal_is_treated_as_send_even_with_mirrored_output(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Withdrawal",
                "Input Currency": "BTC",
                "Input Amount": "-0.01",
                "Output Currency": "BTC",
                "Output Amount": "0.01",
                "Details": "approved / BTC withdrawal",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"BTC": 50000.0},
    )

    btc_row = result[result["Coin"] == "BTC"].iloc[0]
    assert btc_row["Quantity"] == -0.01
    assert btc_row["Principal Invested"] == -500.0


def test_loan_withdrawal_only_increases_debt_while_top_up_carries_usdc(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Loan Withdrawal",
                "Input Currency": "USD",
                "Input Amount": "-75.0126",
                "Output Currency": "USDC",
                "Output Amount": "75",
                "USD Equivalent": "$75.00",
                "Details": "approved / TIM VOGEL",
                "Date / Time (UTC)": "09/03/2021 21:36",
            },
            {
                "Type": "Top up Crypto",
                "Input Currency": "USDC",
                "Input Amount": "75",
                "Output Currency": "USDC",
                "Output Amount": "75",
                "USD Equivalent": "$75.01",
                "Details": "approved",
                "Date / Time (UTC)": "09/03/2021 21:37",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )

    usdc_row = result[result["Coin"] == "USDC"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdc_row["Quantity"] == 75.0
    assert usdc_row["Principal Invested"] == 75.0
    assert usd_row["Quantity"] == -75.0126
    assert usd_row["Principal Invested"] == -75.01


def test_credit_card_withdrawal_credit_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Credit Card Withdrawal Credit",
                "Input Currency": "xUSD",
                "Input Amount": "-10.06",
                "Output Currency": "xUSD",
                "Output Amount": "10.06",
                "Details": "approved / Nexo Card Loan Withdrawal",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0},
    )
    assert result.empty


def test_manual_repayment_and_manual_sell_order_are_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDC",
                "Input Amount": "-100",
                "Output Currency": "USDC",
                "Output Amount": "0",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "xUSD",
                "Input Amount": "98",
                "Output Currency": "xUSD",
                "Output Amount": "0",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:01",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "xUSD": 1.0},
    )
    assert result.empty


def test_manual_repayment_pair_is_processed_as_synthetic_swap(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDC",
                "Input Amount": "-100",
                "Output Currency": "USDC",
                "Output Amount": "0",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "xUSD",
                "Input Amount": "98",
                "Output Currency": "xUSD",
                "Output Amount": "0",
                "USD Equivalent": "$98.02",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:03",
            },
            {
                "Type": "Exchange Liquidation",
                "Input Currency": "USDC",
                "Input Amount": "100",
                "Output Currency": "xUSD",
                "Output Amount": "98",
                "USD Equivalent": "$98.01",
                "Details": "approved / Crypto repayment / Exchange USDC to xUSD",
                "Date / Time (UTC)": "01/01/2026 10:02",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )

    usdc_row = result[result["Coin"] == "USDC"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdc_row["Quantity"] == -100.0
    assert usdc_row["Principal Invested"] == -98.0
    assert usd_row["Quantity"] == 98.0
    assert usd_row["Principal Invested"] == 98.0

    review_path = tmp_path / "nexo_liquidation_only_review.csv"
    review = pd.read_csv(review_path)
    assert review.empty


def test_manual_repayment_pair_can_match_within_thirty_minutes(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDT",
                "Input Amount": "-4.439778",
                "Output Currency": "USDT",
                "Output Amount": "4.439778",
                "USD Equivalent": "$4.44",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "25/01/2023 18:10",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "USD",
                "Input Amount": "4.43977795",
                "Output Currency": "USD",
                "Output Amount": "0",
                "USD Equivalent": "$4.44",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "25/01/2023 18:37",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDT": 1.0, "USD": 1.0},
    )

    usdt_row = result[result["Coin"] == "USDT"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdt_row["Quantity"] == -4.439778
    assert usdt_row["Principal Invested"] == -4.44
    assert usd_row["Quantity"] == 4.43977795
    assert usd_row["Principal Invested"] == 4.44


def test_exchange_liquidation_without_manual_pair_is_not_applied_and_is_reported(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange Liquidation",
                "Input Currency": "USDC",
                "Input Amount": "100",
                "Output Currency": "xUSD",
                "Output Amount": "98",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto repayment / Exchange USDC to xUSD",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )
    assert result.empty

    review_path = tmp_path / "nexo_liquidation_only_review.csv"
    review = pd.read_csv(review_path)
    assert len(review) == 1
    assert review.iloc[0]["Type"] == "Exchange Liquidation"


def test_manual_repayment_pair_outside_tolerance_is_not_matched(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDC",
                "Input Amount": "-100",
                "Output Currency": "USDC",
                "Output Amount": "0",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "xUSD",
                "Input Amount": "98",
                "Output Currency": "xUSD",
                "Output Amount": "0",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 16:01",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )
    assert result.empty


def test_manual_repayment_pair_within_six_hour_window_is_matched(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "EURX",
                "Input Amount": "-61.1",
                "Output Currency": "EURX",
                "Output Amount": "61.1",
                "USD Equivalent": "$67.05",
                "Details": "approved / Fiat Repayment",
                "Date / Time (UTC)": "07/08/2023 11:22",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "USD",
                "Input Amount": "67.0544394",
                "Output Currency": "USD",
                "Output Amount": "0",
                "USD Equivalent": "$67.05",
                "Details": "approved / Fiat Repayment",
                "Date / Time (UTC)": "07/08/2023 13:36",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0, "USD": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert eurx_row["Quantity"] == -61.1
    assert eurx_row["Principal Invested"] == -67.05
    assert usd_row["Quantity"] == 67.0544394
    assert usd_row["Principal Invested"] == 67.05


def test_manual_pair_matching_prefers_nearest_time_then_usd_delta(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDC",
                "Input Amount": "-100",
                "Output Currency": "USDC",
                "Output Amount": "0",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "xUSD",
                "Input Amount": "98",
                "Output Currency": "xUSD",
                "Output Amount": "0",
                "USD Equivalent": "$98.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:02",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "xUSD",
                "Input Amount": "97",
                "Output Currency": "xUSD",
                "Output Amount": "0",
                "USD Equivalent": "$98.04",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "01/01/2026 10:01",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )

    usdc_row = result[result["Coin"] == "USDC"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdc_row["Quantity"] == -100.0
    assert usd_row["Quantity"] == 97.0


def test_manual_repayment_pair_uses_repayment_fee_in_matching(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDT",
                "Input Amount": "-65.83982",
                "Output Currency": "USDT",
                "Output Amount": "0",
                "USD Equivalent": "$65.90",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "15/04/2024 11:56",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "USD",
                "Input Amount": "65.24",
                "Output Currency": "USD",
                "Output Amount": "0",
                "USD Equivalent": "$65.24",
                "Fee": "0.66",
                "Fee Currency": "USD",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "15/04/2024 11:56",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDT": 1.0, "USD": 1.0},
    )

    usdt_row = result[result["Coin"] == "USDT"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdt_row["Quantity"] == -65.83982
    assert usdt_row["Principal Invested"] == -65.24
    assert usd_row["Quantity"] == 65.24
    assert usd_row["Principal Invested"] == 65.24


def test_manual_repayment_pair_uses_wider_usdc_usd_tolerance(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Manual Sell Order",
                "Input Currency": "USDC",
                "Input Amount": "-199.935021",
                "Output Currency": "USDC",
                "Output Amount": "199.935021",
                "USD Equivalent": "$200.00",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "09/05/2022 10:14",
            },
            {
                "Type": "Manual Repayment",
                "Input Currency": "USD",
                "Input Amount": "199.8649818",
                "Output Currency": "USD",
                "Output Amount": "0",
                "USD Equivalent": "$199.86",
                "Details": "approved / Crypto Repayment",
                "Date / Time (UTC)": "09/05/2022 10:14",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDC": 1.0, "USD": 1.0},
    )

    usdc_row = result[result["Coin"] == "USDC"].iloc[0]
    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usdc_row["Quantity"] == -199.935021
    assert usdc_row["Principal Invested"] == -199.86
    assert usd_row["Quantity"] == 199.8649818
    assert usd_row["Principal Invested"] == 199.86


def test_negative_xusd_interest_maps_to_usd_without_principal_change(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Interest",
                "Input Currency": "xUSD",
                "Input Amount": "-2",
                "Output Currency": "xUSD",
                "Output Amount": "2",
                "Details": "approved / Interest",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "USD": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == -2.0
    assert usd_row["Principal Invested"] == 0.0
    assert "xUSD" not in result["Coin"].values


def test_negative_usdx_interest_maps_to_usd_without_principal_change(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Interest",
                "Input Currency": "USDX",
                "Input Amount": "-3",
                "Output Currency": "USDX",
                "Output Amount": "3",
                "Details": "approved / Interest",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDX": 1.0, "USD": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == -3.0
    assert usd_row["Principal Invested"] == 0.0
    assert "USDX" not in result["Coin"].values


def test_negative_usd_interest_keeps_usd_principal_unchanged(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Interest",
                "Input Currency": "USD",
                "Input Amount": "-4",
                "Output Currency": "USD",
                "Output Amount": "4",
                "Details": "approved / Interest",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USD": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == -4.0
    assert usd_row["Principal Invested"] == 0.0


def test_exchange_is_explicit_swap_between_input_and_output_legs(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange",
                "Input Currency": "EURX",
                "Input Amount": "-500",
                "Output Currency": "BTC",
                "Output Amount": "0.01",
                "Details": "approved / synthetic",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"BTC": 50000.0, "EURX": 1.0},
    )

    btc_row = result[result["Coin"] == "BTC"].iloc[0]
    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert btc_row["Quantity"] == 0.01
    assert btc_row["Principal Invested"] == 500.0
    assert eurx_row["Quantity"] == -500.0
    assert eurx_row["Principal Invested"] == -500.0


def test_nexo_card_purchase_is_treated_as_send(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Purchase",
                "Input Currency": "xUSD",
                "Input Amount": "-10.06",
                "Output Currency": "EUR",
                "Output Amount": "8.75",
                "Details": "approved / card merchant",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "EUR": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == -10.06
    assert usd_row["Principal Invested"] == -8.75
    assert "EUR" not in result["Coin"].values


def test_rejected_nexo_card_purchase_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Purchase",
                "Input Currency": "xUSD",
                "Input Amount": "-10.06",
                "Output Currency": "EUR",
                "Output Amount": "8.75",
                "Details": "rejected / card merchant",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "EUR": 1.0},
    )

    assert result.empty


def test_nexo_card_purchase_in_debit_mode_bundle_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Purchase",
                "Input Currency": "USD",
                "Input Amount": "-13.01",
                "Output Currency": "EUR",
                "Output Amount": "11.97",
                "USD Equivalent": "$13.05",
                "Details": "approved / debit-mode merchant",
                "Date / Time (UTC)": "23/08/2023 16:34",
            },
            {
                "Type": "Credit Card Fiatx Exchange To Withdraw",
                "Input Currency": "EURX",
                "Input Amount": "-11.97",
                "Output Currency": "EUR",
                "Output Amount": "11.97",
                "USD Equivalent": "$13.01",
                "Details": "approved / EUR",
                "Date / Time (UTC)": "23/08/2023 16:34",
            },
            {
                "Type": "Withdraw Exchanged",
                "Input Currency": "EUR",
                "Input Amount": "-11.97",
                "Output Currency": "EUR",
                "Output Amount": "11.97",
                "USD Equivalent": "$13.01",
                "Details": "approved / EUR withdrawal",
                "Date / Time (UTC)": "23/08/2023 16:34",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0, "USD": 1.0, "EUR": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == -11.97
    assert eurx_row["Principal Invested"] == -11.97
    assert "USD" not in result["Coin"].values
    assert "EUR" not in result["Coin"].values


def test_nexo_card_transaction_fee_is_treated_as_send(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Transaction Fee",
                "Input Currency": "xUSD",
                "Input Amount": "-0.54",
                "Output Currency": "xUSD",
                "Output Amount": "0.54",
                "Details": "approved / 0.7% Weekend FX Fee",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == -0.54
    assert usd_row["Principal Invested"] == -0.54


def test_nexo_card_refund_increases_debt_token_and_not_eur(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Refund",
                "Input Currency": "USD",
                "Input Amount": "10.19",
                "Output Currency": "EUR",
                "Output Amount": "8.99",
                "USD Equivalent": "$10.19",
                "Details": "approved / CRV*Amzn Mktp NL | Vilnius",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "EUR": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    assert usd_row["Quantity"] == 10.19
    assert usd_row["Principal Invested"] == 8.99
    assert "EUR" not in result["Coin"].values


def test_rejected_nexo_card_refund_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Refund",
                "Input Currency": "EUR",
                "Input Amount": "8.99",
                "Output Currency": "EUR",
                "Output Amount": "8.99",
                "USD Equivalent": "$10.19",
                "Details": "rejected / Refund",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "EUR": 1.0},
    )

    assert result.empty


def test_eur_mode_nexo_card_refund_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Refund",
                "Input Currency": "EUR",
                "Input Amount": "166.15",
                "Output Currency": "EUR",
                "Output Amount": "166.15",
                "USD Equivalent": "$180.83",
                "Details": "approved / CRV*Hotel at Booking.c | DUBLIN",
                "Date / Time (UTC)": "26/07/2024 23:54",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EUR": 1.0, "USD": 1.0},
    )

    assert result.empty


def test_nexo_card_refund_uses_linked_purchase_debt_token(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Purchase",
                "Input Currency": "USDX",
                "Input Amount": "-10.19",
                "Output Currency": "EUR",
                "Output Amount": "8.99",
                "Details": "approved / CRV*Amzn Mktp NL | Vilnius",
                "Date / Time (UTC)": "22/05/2025 10:51",
            },
            {
                "Type": "Nexo Card Refund",
                "Input Currency": "USD",
                "Input Amount": "10.19",
                "Output Currency": "EUR",
                "Output Amount": "8.99",
                "USD Equivalent": "$10.19",
                "Details": "approved / CRV*Amzn Mktp NL | Vilnius",
                "Date / Time (UTC)": "25/05/2025 23:05",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USDX": 1.0, "EUR": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[-1]
    assert usd_row["Quantity"] == 0.0
    assert usd_row["Principal Invested"] == 0.0
    assert "EUR" not in result["Coin"].values


def test_exchange_credit_card_loan_withdrawal_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange Credit",
                "Input Currency": "xUSD",
                "Input Amount": "-49.01",
                "Output Currency": "EURX",
                "Output Amount": "42.52",
                "Details": "approved / Nexo Card Loan Withdrawal",
                "Date / Time (UTC)": "01/01/2026 10:00",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"xUSD": 1.0, "EURX": 1.0},
    )
    assert result.empty


def test_exchange_to_withdraw_and_withdraw_exchanged_combination_maps_to_single_send(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange To Withdraw",
                "Input Currency": "EURX",
                "Input Amount": "300",
                "Output Currency": "EUR",
                "Output Amount": "300",
                "Details": "approved / EURX to EUR",
                "Date / Time (UTC)": "01/01/2026 10:00",
            },
            {
                "Type": "Withdraw Exchanged",
                "Input Currency": "EUR",
                "Input Amount": "-300",
                "Output Currency": "EUR",
                "Output Amount": "300",
                "Details": "approved / EUR withdrawal",
                "Date / Time (UTC)": "01/01/2026 10:01",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0, "EUR": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == -300.0
    assert eurx_row["Principal Invested"] == -300.0
    assert "EUR" not in result["Coin"].values


def test_credit_card_fiatx_exchange_to_withdraw_is_treated_as_send(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Credit Card Fiatx Exchange To Withdraw",
                "Input Currency": "EURX",
                "Input Amount": "-17.17",
                "Output Currency": "EUR",
                "Output Amount": "17.17",
                "Details": "approved / EUR",
                "Date / Time (UTC)": "23/08/2023 16:34",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0, "EUR": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == -17.17
    assert eurx_row["Principal Invested"] == -17.17
    assert "EUR" not in result["Coin"].values


def test_nexo_card_cashback_reversal_is_debt_only_send(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Cashback Reversal",
                "Input Currency": "USD",
                "Input Amount": "-0.2",
                "Output Currency": "EUR",
                "Output Amount": "0.17",
                "USD Equivalent": "$0.20",
                "Details": "approved",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USD": 1.0, "EUR": 1.0},
    )

    usd_row = result[result["Coin"] == "USD"].iloc[0]
    nexo_row = result[result["Coin"] == "NEXO"].iloc[0]
    assert usd_row["Quantity"] == -0.2
    assert usd_row["Principal Invested"] == -0.17
    assert nexo_row["Quantity"] == 0.0
    assert nexo_row["Principal Invested"] == 0.17
    assert "EUR" not in result["Coin"].values


def test_rejected_nexo_card_cashback_reversal_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Cashback Reversal",
                "Input Currency": "USD",
                "Input Amount": "-0.2",
                "Output Currency": "EUR",
                "Output Amount": "0.17",
                "USD Equivalent": "$0.20",
                "Details": "rejected / cashback reversal",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"USD": 1.0, "EUR": 1.0},
    )

    assert result.empty


def test_eur_mode_refund_bundle_skips_cashback_reversal_and_keeps_only_eurx_top_up(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Nexo Card Refund",
                "Input Currency": "EUR",
                "Input Amount": "166.15",
                "Output Currency": "EUR",
                "Output Amount": "166.15",
                "USD Equivalent": "$180.83",
                "Details": "approved / CRV*Hotel at Booking.c | DUBLIN",
                "Date / Time (UTC)": "26/07/2024 23:54",
            },
            {
                "Type": "Nexo Card Cashback Reversal",
                "Input Currency": "USD",
                "Input Amount": "-3.62",
                "Output Currency": "EUR",
                "Output Amount": "3.32",
                "USD Equivalent": "$3.62",
                "Details": "approved",
                "Date / Time (UTC)": "29/07/2024 02:09",
            },
            {
                "Type": "Deposit To Exchange",
                "Input Currency": "EUR",
                "Input Amount": "162.83",
                "Output Currency": "EURX",
                "Output Amount": "162.82",
                "USD Equivalent": "$177.21",
                "Details": (
                    "approved / CRV*Hotel at Booking.c | " "3 DUBLIN LANDINGS NORTH WALL | DUBLIN"
                ),
                "Date / Time (UTC)": "29/07/2024 02:09",
            },
            {
                "Type": "Exchange Deposited On",
                "Input Currency": "EUR",
                "Input Amount": "-162.83",
                "Output Currency": "EURX",
                "Output Amount": "162.83",
                "USD Equivalent": "$176.96",
                "Details": "approved / EUR to EURX",
                "Date / Time (UTC)": "29/07/2024 02:09",
            },
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EUR": 1.0, "EURX": 1.0, "USD": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[-1]
    assert eurx_row["Quantity"] == 162.83
    assert eurx_row["Principal Invested"] == 162.83
    assert "USD" not in result["Coin"].values
    assert "NEXO" not in result["Coin"].values
    assert "EUR" not in result["Coin"].values


def test_deposit_to_exchange_is_ignored(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Deposit To Exchange",
                "Input Currency": "EUR",
                "Input Amount": "8.81",
                "Output Currency": "EURX",
                "Output Amount": "8.81",
                "USD Equivalent": "$9.98",
                "Details": "approved / top up",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EUR": 1.0, "EURX": 1.0},
    )
    assert result.empty


def test_exchange_deposited_on_is_treated_as_receive(monkeypatch, tmp_path: Path) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange Deposited On",
                "Input Currency": "EUR",
                "Input Amount": "-8.81",
                "Output Currency": "EURX",
                "Output Amount": "8.81",
                "USD Equivalent": "$10.19",
                "Details": "approved / EUR to EURX",
                "Date / Time (UTC)": "25/05/2025 23:05",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == 8.81
    assert eurx_row["Principal Invested"] == 8.81


def test_exchange_deposited_on_uses_eur_input_amount_when_output_is_zero(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run_nexo_generator(
        rows=[
            {
                "Type": "Exchange Deposited On",
                "Input Currency": "EUR",
                "Input Amount": "-250",
                "Output Currency": "EURX",
                "Output Amount": "0",
                "USD Equivalent": "$265.00",
                "Details": "approved / EUR to EURX",
                "Date / Time (UTC)": "19/10/2023 13:28",
            }
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        price_map={"EURX": 1.0},
    )

    eurx_row = result[result["Coin"] == "EURX"].iloc[0]
    assert eurx_row["Quantity"] == 250.0
    assert eurx_row["Principal Invested"] == 250.0
