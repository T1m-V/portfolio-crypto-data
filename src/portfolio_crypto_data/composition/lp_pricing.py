from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from portfolio_core import (
    PortfolioContext,
    active_context,
    load_price_csv,
    merge_price_frames,
    save_price_csv,
)

from portfolio_crypto_data.composition.core import (
    CompositionContext,
    PriceResolver,
    ProtocolStore,
    SymbolMetadata,
    build_symbol_metadata,
)
from portfolio_crypto_data.shared.token_metadata import load_token_metadata
from portfolio_crypto_data.shared.valuation_routes import build_symbol_protocol_map
from portfolio_crypto_data.symbols import build_symbol_family_map

PricingContext = CompositionContext


def _load_token_metadata(
    *, chain: str, tokens_folder: Path
) -> dict[str, dict[str, object]]:
    return load_token_metadata(chain=chain, tokens_folder=tokens_folder)


def _build_symbol_metadata(
    token_metadata: dict[str, dict[str, object]],
) -> dict[str, SymbolMetadata]:
    return build_symbol_metadata(token_metadata=token_metadata)


def _load_protocol_rows(*, chain: str, protocol_root: Path) -> dict[str, pd.DataFrame]:
    return ProtocolStore.load(
        chain=chain,
        root=protocol_root,
        include_aave=False,
    ).rows


def resolve_symbol_price(
    symbol: str,
    target_date: date,
    ctx: PricingContext,
    visited: set[str] | None = None,
    depth: int = 0,
) -> Decimal | None:
    resolution = PriceResolver(ctx=ctx, mode="native").resolve(
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


def generate_protocol_lp_price_files(
    *,
    chain: str,
    context: PortfolioContext | None = None,
    protocol_root: Path | None = None,
    prices_folder: Path | None = None,
    tokens_folder: Path | None = None,
) -> list[Path]:
    """
    Builds protocol token price files from non-AAVE protocol-underlying exports.

    args:
        chain: Chain identifier used for protocol-underlying file discovery.

    returns:
        List of updated price CSV paths in data/prices/lp_prices/<chain>.
    """
    context = context or active_context()
    protocol_root = protocol_root or context.paths.protocol_underlying_tokens
    prices_folder = prices_folder or context.paths.prices
    tokens_folder = tokens_folder or context.paths.tokens
    token_metadata = _load_token_metadata(chain=chain, tokens_folder=tokens_folder)
    protocol_rows = _load_protocol_rows(chain=chain, protocol_root=protocol_root)
    symbol_family = build_symbol_family_map(token_metadata=token_metadata)
    ctx = PricingContext(
        chain=chain,
        protocol_rows=protocol_rows,
        symbol_protocol=build_symbol_protocol_map(token_metadata=token_metadata),
        protocol_derived_symbols=set(protocol_rows.keys()),
        symbol_family=symbol_family,
        symbol_metadata=_build_symbol_metadata(token_metadata=token_metadata),
        currency_metadata=context.currency_metadata(),
    )
    price_resolver = PriceResolver(ctx=ctx, prices_folder=prices_folder, mode="native")

    updated_files: list[Path] = []
    for symbol, df in sorted(ctx.protocol_rows.items(), key=lambda item: item[0]):
        incoming = _build_incoming_prices(
            symbol=symbol,
            df=df,
            price_resolver=price_resolver,
        )
        if incoming.empty:
            continue

        output_path = prices_folder / "lp_prices" / chain / f"{symbol}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_price_csv(file_path=output_path)
        merged = merge_price_frames(existing=existing, incoming=incoming)
        save_price_csv(file_path=output_path, frame=merged)
        updated_files.append(output_path)

    return updated_files
