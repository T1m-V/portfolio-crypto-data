from blockchain_reader.shared.valuation_routes import (
    ValuationRoute,
    build_symbol_protocol_map,
    classify_valuation_route,
)


class TestValuationRoutes:
    def test_build_symbol_protocol_map_normalizes_symbols(self) -> None:
        mapping = build_symbol_protocol_map(
            token_metadata={
                "0x1": {"symbol": "WRAP", "protocol": "beefy"},
                "0x2": {"symbol": "aArbUSDC", "protocol": "aave"},
                "0x3": {"symbol": "ARB"},
            }
        )

        assert mapping["WRAP"] == "beefy"
        assert mapping["aArbUSDC"] == "aave"
        assert mapping["ARB"] == ""

    def test_classify_valuation_route_prioritizes_aave_symbols(self) -> None:
        route = classify_valuation_route(
            symbol="variableDebtArbLINK",
            symbol_protocol={"variableDebtArbLINK": ""},
            protocol_derived_symbols={"variableDebtArbLINK"},
        )
        assert route == ValuationRoute.AAVE

    def test_classify_valuation_route_marks_protocol_rows_as_protocol_derived(self) -> None:
        route = classify_valuation_route(
            symbol="wstETH",
            symbol_protocol={},
            protocol_derived_symbols={"wstETH"},
        )
        assert route == ValuationRoute.PROTOCOL_DERIVED

    def test_classify_valuation_route_defaults_to_direct(self) -> None:
        route = classify_valuation_route(
            symbol="LINK",
            symbol_protocol={"LINK": ""},
            protocol_derived_symbols=set(),
        )
        assert route == ValuationRoute.DIRECT
