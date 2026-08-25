from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import date

from portfolio_core import active_context

from portfolio_crypto_data.accounting import accounting_paths, build_accounting_artifacts
from portfolio_crypto_data.chain_config import EvmChainConfig, load_evm_chain_configs
from portfolio_crypto_data.composition.lp_pricing import generate_protocol_lp_price_files
from portfolio_crypto_data.dashboard_artifacts import build_chain_dashboard_artifacts
from portfolio_crypto_data.extraction.evm_reader import retrieve_transactions
from portfolio_crypto_data.pipeline_logging import PipelineLogger
from portfolio_crypto_data.protocols.aave import process_all_aave_tokens
from portfolio_crypto_data.protocols.balancer import process_all_balancer_tokens
from portfolio_crypto_data.protocols.beefy import process_all_beefy_tokens
from portfolio_crypto_data.protocols.curve import process_all_curve_tokens
from portfolio_crypto_data.protocols.liquid_staking import process_all_liquid_staking_tokens
from portfolio_crypto_data.raw_snapshots import generate_raw_snapshots

ProtocolProcessor = Callable[..., None]
PROTOCOL_PROCESSORS: dict[str, ProtocolProcessor] = {
    "beefy": process_all_beefy_tokens,
    "balancer": process_all_balancer_tokens,
    "curve": process_all_curve_tokens,
    "aave": process_all_aave_tokens,
    "liquid_staking": process_all_liquid_staking_tokens,
}


def _configured_chains(
    chain_configs: Iterable[EvmChainConfig] | None,
) -> tuple[EvmChainConfig, ...]:
    configs = tuple(chain_configs) if chain_configs is not None else load_evm_chain_configs()
    if not configs:
        raise ValueError("At least one EVM chain must be configured.")

    chain_names: set[str] = set()
    for config in configs:
        if config.chain in chain_names:
            raise ValueError(f"Chain '{config.chain}' is configured more than once.")
        chain_names.add(config.chain)

        unsupported = sorted(set(config.protocols) - PROTOCOL_PROCESSORS.keys())
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(
                f"Chain '{config.chain}' enables unsupported protocol(s): {joined}."
            )
    return configs


def _build_snapshots(*, config: EvmChainConfig) -> None:
    paths = active_context().paths
    transactions = paths.crypto_transactions / f"{config.chain}_transactions.csv"
    snapshots = paths.crypto_snapshots / f"{config.chain}_raw_snapshots.csv"
    principal = accounting_paths(chain=config.chain)
    if not transactions.exists():
        raise FileNotFoundError(f"missing transaction file: {transactions}")

    generate_raw_snapshots(
        input_csv=transactions,
        output_csv=snapshots,
        chain=config.chain,
        principal_events_csv=principal.principal_events,
        principal_daily_csv=principal.principal_daily,
    )


def _refresh_protocols(*, config: EvmChainConfig, logger: PipelineLogger) -> None:
    for name in config.protocols:
        processor = PROTOCOL_PROCESSORS.get(name)
        if processor is None:
            raise ValueError(
                f"Chain '{config.chain}' enables unsupported protocol '{name}'."
            )
        logger.info(f"[{config.chain}:{name}] refresh")
        processor(chain=config.chain, logger=logger)


def _rebuild_chain(*, config: EvmChainConfig, as_of_date: date) -> None:
    _build_snapshots(config=config)
    build_accounting_artifacts(chain=config.chain, as_of_date=as_of_date)
    build_chain_dashboard_artifacts(chain=config.chain)


def rebuild_derived(
    *,
    as_of_date: date | None = None,
    chain_configs: Iterable[EvmChainConfig] | None = None,
) -> None:
    """Rebuild local artifacts for every configured EVM chain without network requests."""
    target_date = as_of_date or date.today()
    for config in _configured_chains(chain_configs):
        _rebuild_chain(config=config, as_of_date=target_date)


def _update_chain(*, config: EvmChainConfig, logger: PipelineLogger) -> None:
    chain = config.chain
    logger.stage_start(f"{chain}:transactions")
    asyncio.run(retrieve_transactions(config=config))
    logger.stage_end(f"{chain}:transactions")

    logger.stage_start(f"{chain}:snapshots")
    _build_snapshots(config=config)
    logger.stage_end(f"{chain}:snapshots")

    if config.protocols:
        logger.stage_start(f"{chain}:protocols")
        _refresh_protocols(config=config, logger=logger)
        logger.stage_end(f"{chain}:protocols")

        logger.stage_start(f"{chain}:lp_prices")
        generate_protocol_lp_price_files(chain=chain, context=active_context())
        logger.stage_end(f"{chain}:lp_prices")

    logger.stage_start(f"{chain}:accounting")
    build_accounting_artifacts(chain=chain, as_of_date=date.today())
    logger.stage_end(f"{chain}:accounting")

    logger.stage_start(f"{chain}:dashboard")
    build_chain_dashboard_artifacts(chain=chain)
    logger.stage_end(f"{chain}:dashboard")


def update_onchain(
    *,
    chain_configs: Iterable[EvmChainConfig] | None = None,
) -> None:
    """Refresh every configured EVM chain and its derived artifacts."""
    logger = PipelineLogger()
    for config in _configured_chains(chain_configs):
        _update_chain(config=config, logger=logger)
