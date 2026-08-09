from decimal import Decimal

import pandas as pd
import pytest
from portfolio_core import active_context

from portfolio_crypto_data.shared.prices import clear_price_cache, get_price_on_or_before
from portfolio_crypto_data.shared.valuation_routes import (
    ValuationRoute,
    build_symbol_protocol_map,
    classify_valuation_route,
)
from portfolio_crypto_data.symbols import price_proxy_symbol, sanitize_symbol


def test_prices_keep_direct_and_chain_lp_namespaces_separate() -> None:
    prices = active_context().paths.prices
    lp = prices / "lp_prices" / "arbitrum"
    lp.mkdir(parents=True)
    pd.DataFrame([{"Date": "2026-01-02", "Price": 50}]).to_csv(prices / "LP.csv", index=False)
    pd.DataFrame([{"Date": "2026-01-02", "Price": 100}]).to_csv(lp / "LP.csv", index=False)
    clear_price_cache()

    direct = get_price_on_or_before(
        symbol="LP", as_of_date="2026-01-02", prices_folder=prices
    )
    derived = get_price_on_or_before(
        symbol="LP",
        as_of_date="2026-01-02",
        prices_folder=prices,
        chain="arbitrum",
        use_lp_prices=True,
    )

    assert (direct, derived) == (Decimal("50"), Decimal("100"))
    with pytest.raises(ValueError, match="requires `chain`"):
        get_price_on_or_before(
            symbol="LP", as_of_date="2026-01-02", prices_folder=prices, use_lp_prices=True
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
