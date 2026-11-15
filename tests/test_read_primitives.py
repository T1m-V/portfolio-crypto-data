from decimal import Decimal

import pandas as pd
import pytest
from portfolio_core import active_context

from portfolio_crypto_data.shared.prices import (
    clear_price_cache,
    get_price_eur_on_or_before,
    get_price_on_or_before,
)
from portfolio_crypto_data.shared.valuation_routes import (
    ValuationRoute,
    build_symbol_protocol_map,
    classify_valuation_route,
)
from portfolio_crypto_data.symbols import price_proxy_symbol, sanitize_symbol


def test_prices_resolve_direct_price_on_or_before() -> None:
    prices = active_context().paths.prices
    pd.DataFrame(
        [
            {"Date": "2026-01-01", "Price": 50},
            {"Date": "2026-01-03", "Price": 75},
        ]
    ).to_csv(prices / "ETH.csv", index=False)
    clear_price_cache()

    on_date = get_price_on_or_before(
        symbol="ETH", as_of_date="2026-01-01", prices_folder=prices
    )
    carried = get_price_on_or_before(
        symbol="ETH", as_of_date="2026-01-02", prices_folder=prices
    )
    missing_before = get_price_on_or_before(
        symbol="ETH", as_of_date="2025-12-31", prices_folder=prices
    )
    oldest = get_price_on_or_before(
        symbol="ETH", as_of_date="2025-12-31", prices_folder=prices, fallback_to_oldest=True
    )

    assert (on_date, carried, missing_before, oldest) == (
        Decimal("50"),
        Decimal("50"),
        None,
        Decimal("50"),
    )


@pytest.mark.parametrize(
    ("symbol", "protocols", "derived", "expected"),
    [
        ("variableDebtArbLINK", {}, {"variableDebtArbLINK"}, ValuationRoute.AAVE),
        ("wstETH", {}, {"wstETH"}, ValuationRoute.PROTOCOL_DERIVED),
        ("WRAP", {"WRAP": "beefy"}, set(), ValuationRoute.PROTOCOL_DERIVED),
        ("LINK", {"LINK": ""}, set(), ValuationRoute.DIRECT),
    ],
)
def test_valuation_route_is_a_single_explicit_decision(
    symbol: str,
    protocols: dict[str, str],
    derived: set[str],
    expected: ValuationRoute,
) -> None:
    assert (
        classify_valuation_route(
            symbol=symbol,
            symbol_protocol=protocols,
            protocol_derived_symbols=derived,
        )
        == expected
    )


def test_symbol_helpers_cover_persisted_aliases() -> None:
    metadata = {
        "a": {"symbol": "WRAP", "protocol": "beefy"},
        "b": {"symbol": "aArbUSDC", "protocol": "aave"},
    }
    assert build_symbol_protocol_map(metadata) == {"WRAP": "beefy", "aArbUSDC": "aave"}
    assert sanitize_symbol(" USD®0 ") == "USD0"
    assert price_proxy_symbol("WBTC") == "BTC"
    assert price_proxy_symbol("MATIC") == "POL"


def test_busd_resolves_as_usd_stable_symbol_and_matic_proxies_to_pol() -> None:
    from portfolio_crypto_data.raw_snapshots import get_crypto_price

    prices = active_context().paths.prices
    pd.DataFrame([{"Date": "2026-01-01", "Price": 0.9}]).to_csv(
        prices / "USD_EUR.csv", index=False
    )
    pd.DataFrame([{"Date": "2026-01-01", "Price": 2.0}]).to_csv(
        prices / "POL.csv", index=False
    )
    clear_price_cache()

    assert get_price_eur_on_or_before(
        symbol="BUSD", as_of_date="2026-01-01", prices_folder=prices
    ) == Decimal("0.9")
    assert get_crypto_price(coin="MATIC", date="2026-01-01") == pytest.approx(1.8)
