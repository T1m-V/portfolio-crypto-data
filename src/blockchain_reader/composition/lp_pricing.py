from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from blockchain_reader.composition.core import (
    CompositionContext,
    PriceResolver,
    ProtocolStore,
    SymbolMetadata,
    build_symbol_metadata,
)
from blockchain_reader.shared.token_metadata import load_token_metadata
from blockchain_reader.shared.valuation_routes import build_symbol_protocol_map
from blockchain_reader.symbols import build_symbol_family_map
from file_paths import (
    PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
    TOKENS_FOLDER,
    get_lp_price_file_path,
)
from price_history.price_data_utils import load_price_csv, merge_price_frames, save_price_csv

PricingContext = CompositionContext


def _load_token_metadata(chain: str) -> dict[str, dict[str, object]]:
    return load_token_metadata(chain=chain, tokens_folder=TOKENS_FOLDER)


def _build_symbol_metadata(
    token_metadata: dict[str, dict[str, object]],
) -> dict[str, SymbolMetadata]:
    return build_symbol_metadata(token_metadata=token_metadata)


def _load_protocol_rows(chain: str) -> dict[str, pd.DataFrame]:
    return ProtocolStore.load(
        chain=chain,
        root=PROTOCOL_UNDERLYING_TOKEN_FOLDER,
        include_aave=False,
    ).rows


def resolve_symbol_price(
    symbol: str,
    target_date: date,
    ctx: PricingContext,
    visited: set[str] | None = None,
    depth: int = 0,
) -> Decimal | None:
    resolution = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="native").resolve(
        symbol=symbol,
        target_date=target_date,
        visited=visited,
        depth=depth,
    )
    return resolution.price


def _build_incoming_prices(
    symbol: str,
    df: pd.DataFrame,
    price_resolver: PriceResolver,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.sort_values("date").iterrows():
        target_date = pd.Timestamp(row["date"]).date()
        price = price_resolver.resolve(symbol=symbol, target_date=target_date).price
        if price is None:
            continue

        rows.append({"Date": target_date, "Price": float(price)})

    if not rows:
        return pd.DataFrame(columns=["Date", "Price"])

    return pd.DataFrame(rows, columns=["Date", "Price"])


def generate_protocol_lp_price_files(chain: str) -> list[Path]:
    """
    Builds protocol token price files from non-AAVE protocol-underlying exports.

    args:
        chain: Chain identifier used for protocol-underlying file discovery.

    returns:
        List of updated price CSV paths in data/prices/lp_prices/<chain>.
    """
    token_metadata = _load_token_metadata(chain=chain)
    protocol_rows = _load_protocol_rows(chain=chain)
    symbol_family = build_symbol_family_map(token_metadata=token_metadata)
    ctx = PricingContext(
        chain=chain,
        protocol_rows=protocol_rows,
        symbol_protocol=build_symbol_protocol_map(token_metadata=token_metadata),
        protocol_derived_symbols=set(protocol_rows.keys()),
        symbol_family=symbol_family,
        symbol_metadata=_build_symbol_metadata(token_metadata=token_metadata),
    )
    price_resolver = PriceResolver(ctx=ctx, prices_folder=PRICES_FOLDER, mode="native")

    updated_files: list[Path] = []
    for symbol, df in sorted(ctx.protocol_rows.items(), key=lambda item: item[0]):
        incoming = _build_incoming_prices(
            symbol=symbol,
            df=df,
            price_resolver=price_resolver,
        )
        if incoming.empty:
            continue

        output_path = get_lp_price_file_path(
            chain=chain,
            symbol=symbol,
            prices_folder=PRICES_FOLDER,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_price_csv(file_path=output_path)
        merged = merge_price_frames(existing=existing, incoming=incoming)
        save_price_csv(file_path=output_path, frame=merged)
        updated_files.append(output_path)

    return updated_files
