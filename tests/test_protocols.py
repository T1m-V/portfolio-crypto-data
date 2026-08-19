from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
import pytest
from portfolio_core import active_context

from portfolio_crypto_data.composition.lp_pricing import generate_protocol_lp_price_files
from portfolio_crypto_data.protocols import aave, balancer, beefy, common, curve


class _Call:
    def __init__(self, value: object) -> None:
        self.value = value

    def call(self, block_identifier: int | None = None) -> object:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class _Functions:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def __getattr__(self, name: str):
        if name not in self.values:
            raise AttributeError(name)
        return lambda *args, **kwargs: _Call(self.values[name])


class _Contract:
    def __init__(self, address: str, **values: object) -> None:
        self.address = address
        self.functions = _Functions(values)


class _Web3:
    def __init__(self, *contracts: _Contract) -> None:
        lookup = {contract.address.lower(): contract for contract in contracts}
        self.eth = type(
            "Eth",
            (),
            {"contract": lambda _, address, abi: lookup[address.lower()]},
        )()


class _Logger:
    def __init__(self) -> None:
        self.skipped: list[tuple[str, str, str]] = []

    def protocol_skip(self, protocol: str, symbol: str, reason: str) -> None:
        self.skipped.append((protocol, symbol, reason))


def test_protocol_runs_are_incremental_and_data_driven() -> None:
    paths = active_context().paths
    (paths.tokens / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "0xclosed": {"symbol": "closedLP", "protocol": "beefy"},
                "0xactive": {"symbol": "activeLP", "protocol": "beefy"},
                "0xabsent": {"symbol": "absentLP", "protocol": "beefy"},
                "0xother": {"symbol": "otherLP", "protocol": "curve"},
            }
        )
    )
    pd.DataFrame(
        [
            {"Date": "2025-01-01", "Coin": "closedLP", "Quantity": 1},
            {"Date": "2025-01-03", "Coin": "closedLP", "Quantity": 0},
            {"Date": "2025-02-01", "Coin": "activeLP", "Quantity": 1},
        ]
    ).to_csv(paths.crypto_snapshots / "arbitrum_raw_snapshots.csv", index=False)
    common.write_protocol_history_csv(
        protocol="beefy",
        chain="arbitrum",
        symbol="closedLP",
        history_data=[{"date": "2025-01-01", "asset_USDC": 1}],
    )
    logger = _Logger()

    runs = common.protocol_token_runs(protocol="beefy", chain="arbitrum", logger=logger)

    assert [(run.symbol, run.start_date, run.end_date) for run in runs] == [
        ("closedLP", "2025-01-02 00:00:00", "2025-01-03 00:00:00"),
        ("activeLP", "2025-02-01 00:00:00", "now"),
    ]
    assert logger.skipped == [("beefy", "absentLP", "no snapshot data found")]


def test_protocol_history_merge_is_sorted_and_new_rows_win() -> None:
    first = common.write_protocol_history_csv(
        protocol="curve",
        chain="arbitrum",
        symbol="LP",
        history_data=[
            {"date": "2025-01-02", "asset_USDC": 2},
            {"date": "2025-01-01", "asset_USDC": 1},
        ],
        fieldnames=["date", "asset_USDC"],
    )
    second = common.write_protocol_history_csv(
        protocol="curve",
        chain="arbitrum",
        symbol="LP",
        history_data=[
            {"date": "2025-01-02", "asset_USDC": 20, "asset_USDT": 5},
            {"date": "2025-01-03", "asset_USDC": 3},
        ],
        fieldnames=["date", "asset_USDC", "asset_USDT"],
    )

    assert first == second
    frame = pd.read_csv(second)
    assert frame["date"].tolist() == [
        "2025-01-01 00:00:00",
        "2025-01-02 00:00:00",
        "2025-01-03 00:00:00",
    ]
    assert frame.loc[1, ["asset_USDC", "asset_USDT"]].tolist() == [20, 5]


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        (1, "now"),
        (0, "2025-01-03 00:00:00"),
        (-1, "2025-01-03 00:00:00"),
        (1e-10, "2025-01-03 00:00:00"),
    ],
)
def test_protocol_end_date_only_stays_open_for_material_assets(
    quantity: float, expected: str
) -> None:
    assert common.resolve_protocol_end_date(
        {"qty": quantity, "end": pd.Timestamp("2025-01-03")}
    ) == expected


def test_aave_aliases_and_closed_positions_have_explicit_zeroes() -> None:
    assert aave._normalize_aave_underlying_symbol("USDT0") == "USDT"
    legs = aave._compute_leg_columns(
        supply_by_symbol={"ETH": Decimal("2")},
        debt_by_symbol={"ETH": Decimal("0.5")},
    )

    assert legs == {
        "supply_ETH": Decimal("2"),
        "debt_ETH": Decimal("0.5"),
        "net_ETH": Decimal("1.5"),
    }
    assert aave._merge_disappeared_symbol_zeroes(
        leg_columns={},
        current_symbols=set(),
        previous_active_symbols={"ETH"},
        current_state_known=True,
    ) == {
        "supply_ETH": Decimal(0),
        "debt_ETH": Decimal(0),
        "net_ETH": Decimal(0),
    }


def test_balancer_uses_total_supply_for_standard_pools_and_ignores_phantom_bpt() -> None:
    bpt = "0xbpt"
    vault = "0xvault"
    usdc = "0xusdc"
    w3 = _Web3(
        _Contract(
            bpt,
            getPoolId=b"pool",
            getActualSupply=RuntimeError("unsupported"),
            totalSupply=200,
        ),
        _Contract(vault, getPoolTokens=([bpt, usdc], [999, 4_000_000], 0)),
        _Contract(usdc, symbol="USDC", decimals=6),
    )

    result = balancer.get_balancer_underlying(
        w3=w3,
        bpt_address=bpt,
        one_unit=100,
        block_number=1,
        vault_address=vault,
    )

    assert result == {"USDC": Decimal("2")}


def test_beefy_single_asset_vault_uses_price_per_share() -> None:
    vault = "0xvault"
    want = "0xusdc"
    w3 = _Web3(
        _Contract(vault, getPricePerFullShare=2 * 10**18, want=want),
        _Contract(want, symbol="USDC", decimals=6),
    )

    result = beefy.get_beefy_underlying(
        w3=w3,
        vault_address=vault,
        one_unit=10**6,
        block_number=1,
    )

    assert result == {"USDC": Decimal("2")}


def test_curve_lp_share_uses_pool_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    lp = "0xlp"
    w3 = _Web3(
        _Contract(lp, totalSupply=2 * 10**18, minter=RuntimeError("no minter")),
    )
    monkeypatch.setattr(
        curve,
        "_read_curve_pool_tokens",
        lambda **kwargs: [
            curve.CurvePoolToken(
                address="0xwbtc",
                balance=4 * 10**8,
                symbol="WBTC",
                decimals=8,
            )
        ],
    )

    result = curve.get_curve_underlying(
        w3=w3,
        lp_token_address=lp,
        one_unit=10**18,
        block_number=1,
    )

    assert result == {"WBTC": Decimal("2")}


def test_protocol_ratio_generates_derived_price() -> None:
    paths = active_context().paths
    (paths.tokens / "arbitrum_tokens.json").write_text(
        json.dumps(
            {
                "native": {"symbol": "ETH"},
                "0xlst": {"symbol": "wstETH", "protocol": "liquid_staking"},
            }
        )
    )
    folder = paths.protocol_underlying_tokens / "liquid_staking"
    folder.mkdir(parents=True)
    pd.DataFrame([{"date": "2025-01-01", "asset_ETH": 1.1}]).to_csv(
        folder / "arbitrum_wstETH.csv", index=False
    )
    pd.DataFrame([{"Date": "2025-01-01", "Price": 1_000}]).to_csv(
        paths.prices / "ETH.csv", index=False
    )

    outputs = generate_protocol_lp_price_files(chain="arbitrum")

    assert len(outputs) == 1
    assert pd.read_csv(outputs[0]).to_dict("records") == [
        {"Date": "2025-01-01", "Price": 1_100.0}
    ]
