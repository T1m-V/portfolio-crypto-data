from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date

from portfolio_core import active_context

from portfolio_crypto_data.accounting import accounting_paths, build_accounting_artifacts
from portfolio_crypto_data.composition.lp_pricing import generate_protocol_lp_price_files
from portfolio_crypto_data.dashboard_artifacts import build_arbitrum_dashboard_artifacts
from portfolio_crypto_data.extraction.evm_reader import retrieve_transactions
from portfolio_crypto_data.pipeline_logging import PipelineLogger
from portfolio_crypto_data.protocols.aave import process_all_aave_tokens
from portfolio_crypto_data.protocols.balancer import process_all_balancer_tokens
from portfolio_crypto_data.protocols.beefy import process_all_beefy_tokens
from portfolio_crypto_data.protocols.curve import process_all_curve_tokens
from portfolio_crypto_data.protocols.liquid_staking import process_all_liquid_staking_tokens
from portfolio_crypto_data.raw_snapshots import generate_raw_snapshots

CHAIN = "arbitrum"
PROTOCOL_PROCESSORS: tuple[tuple[str, Callable[..., None]], ...] = (
    ("beefy", process_all_beefy_tokens),
    ("balancer", process_all_balancer_tokens),
    ("curve", process_all_curve_tokens),
    ("aave", process_all_aave_tokens),
    ("liquid_staking", process_all_liquid_staking_tokens),
)


def _build_snapshots() -> None:
    paths = active_context().paths
    transactions = paths.crypto_transactions / f"{CHAIN}_transactions.csv"
    snapshots = paths.crypto_snapshots / f"{CHAIN}_raw_snapshots.csv"
    principal = accounting_paths(chain=CHAIN)
    if not transactions.exists():
        raise FileNotFoundError(f"missing transaction file: {transactions}")

    generate_raw_snapshots(
        input_csv=transactions,
        output_csv=snapshots,
        chain=CHAIN,
        principal_events_csv=principal.principal_events,
        principal_daily_csv=principal.principal_daily,
    )


def rebuild_derived(*, as_of_date: date | None = None) -> None:
    """Rebuild local Arbitrum artifacts without making network requests."""
    _build_snapshots()
    build_accounting_artifacts(chain=CHAIN, as_of_date=as_of_date or date.today())
    build_arbitrum_dashboard_artifacts(chain=CHAIN)


def update_onchain() -> None:
    """Refresh Arbitrum transactions and every artifact derived from them."""
    logger = PipelineLogger()
    logger.stage_start("transactions")
    asyncio.run(retrieve_transactions(chain=CHAIN))
    logger.stage_end("transactions")

    logger.stage_start("snapshots")
    _build_snapshots()
    logger.stage_end("snapshots")

    logger.stage_start("protocols")
    for name, processor in PROTOCOL_PROCESSORS:
        logger.info(f"[{name}] refresh")
        processor(chain=CHAIN, logger=logger)
    logger.stage_end("protocols")

    logger.stage_start("lp_prices")
    generate_protocol_lp_price_files(chain=CHAIN, context=active_context())
    logger.stage_end("lp_prices")

    logger.stage_start("accounting")
    build_accounting_artifacts(chain=CHAIN, as_of_date=date.today())
    logger.stage_end("accounting")

    logger.stage_start("dashboard")
    build_arbitrum_dashboard_artifacts(chain=CHAIN)
    logger.stage_end("dashboard")
