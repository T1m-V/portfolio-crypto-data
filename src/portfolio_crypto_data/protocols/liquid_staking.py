from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from portfolio_crypto_data.datetime_utils import format_daily_datetime
from portfolio_crypto_data.pipeline_logging import PipelineLogger
from portfolio_crypto_data.protocols.common import (
    load_block_map,
    load_chain_web3,
    load_snapshot_ranges,
    load_tokens,
    resolve_date_window,
    resolve_effective_start_date,
    resolve_protocol_end_date,
    should_skip_date_window,
    write_protocol_history_csv,
)
from portfolio_crypto_data.symbols import sanitize_symbol

RATE_PROVIDER_ABI = [
    {
        "inputs": [],
        "name": "getRate",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]


@dataclass(frozen=True)
class LiquidStakingTokenConfig:
    symbol: str
    underlying_symbol: str
    rate_provider_address: str
    rate_provider_method: str = "getRate"
    rate_scale: int = 10**18


def _load_liquid_staking_configs(chain: str) -> tuple[LiquidStakingTokenConfig, ...]:
    configs: list[LiquidStakingTokenConfig] = []
    for meta in load_tokens(chain=chain).values():
        if sanitize_symbol(meta.get("protocol")).lower() != "liquid_staking":
            continue

        symbol = sanitize_symbol(meta.get("symbol"))
        underlying = sanitize_symbol(meta.get("underlying_symbol")) or sanitize_symbol(
            meta.get("price_source") or meta.get("family")
        )
        rate_provider = str(meta.get("rate_provider_address") or "").strip()
        if not symbol or not underlying or not rate_provider:
            raise ValueError(
                f"Chain '{chain}' has incomplete liquid-staking token metadata."
            )
        try:
            rate_scale = int(meta.get("rate_scale", 10**18))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Chain '{chain}' has an invalid liquid-staking rate scale."
            ) from exc
        configs.append(
            LiquidStakingTokenConfig(
                symbol=symbol,
                underlying_symbol=underlying,
                rate_provider_address=rate_provider,
                rate_provider_method=str(meta.get("rate_provider_method") or "getRate"),
                rate_scale=rate_scale,
            )
        )
    return tuple(sorted(configs, key=lambda config: config.symbol))


def get_liquid_staking_history(
    chain: str,
    symbol: str,
    underlying_symbol: str,
    rate_provider_address: str,
    start_date: str,
    end_date: str,
    rate_provider_method: str = "getRate",
    rate_scale: int = 10**18,
    logger: PipelineLogger | None = None,
) -> None:
    logger = logger or PipelineLogger()
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    rate_provider = w3.eth.contract(
        address=w3.to_checksum_address(rate_provider_address),
        abi=RATE_PROVIDER_ABI,
    )
    history_data: list[dict[str, object]] = []

    current_dt = start_dt
    total_days = max((end_dt.date() - start_dt.date()).days + 1, 1)
    day_index = 0
    while current_dt <= end_dt:
        day_index += 1
        date_str = format_daily_datetime(current_dt)
        block_num = block_map.get(date_str)
        if block_num is None:
            current_dt += timedelta(days=1)
            continue

        logger.protocol_day(
            "liquid_staking",
            symbol,
            date_str=date_str,
            block_number=block_num,
            day_index=day_index,
            total_days=total_days,
        )
        try:
            if len(w3.eth.get_code(rate_provider.address, block_identifier=block_num)) == 0:
                current_dt += timedelta(days=1)
                continue

            rate_raw = getattr(rate_provider.functions, rate_provider_method)().call(
                block_identifier=block_num
            )
            ratio = Decimal(rate_raw) / Decimal(rate_scale)
            row: dict[str, object] = {
                "date": format_daily_datetime(current_dt),
                "block": block_num,
                "lst_balance": 1.0,
                f"asset_{underlying_symbol}": float(ratio),
            }
            history_data.append(row)
        except Exception as e:
            logger.info(f"[liquid_staking] Error on {current_dt.date()} for {symbol}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="liquid_staking",
        chain=chain,
        symbol=symbol,
        history_data=history_data,
        fieldnames=["date", "block", "lst_balance", f"asset_{underlying_symbol}"],
    )
    if output:
        logger.protocol_end("liquid_staking", symbol, output)


def process_all_liquid_staking_tokens(
    chain: str,
    logger: PipelineLogger | None = None,
) -> None:
    logger = logger or PipelineLogger()
    token_ranges = load_snapshot_ranges(chain=chain)
    for config in _load_liquid_staking_configs(chain=chain):
        rng = token_ranges.get(config.symbol)
        if rng is None:
            logger.protocol_skip("liquid_staking", config.symbol, "no snapshot data found")
            continue
        resolved_start_date = resolve_effective_start_date(
            protocol="liquid_staking",
            chain=chain,
            symbol=config.symbol,
            fallback_start_date=format_daily_datetime(rng["start"]),
        )
        end_date = resolve_protocol_end_date(rng)
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            logger.protocol_skip(
                "liquid_staking",
                config.symbol,
                f"start={resolved_start_date} is after end={end_date}",
            )
            continue

        assert resolved_start_date is not None
        logger.protocol_start("liquid_staking", config.symbol, resolved_start_date, end_date)
        get_liquid_staking_history(
            chain=chain,
            symbol=config.symbol,
            underlying_symbol=config.underlying_symbol,
            rate_provider_address=config.rate_provider_address,
            start_date=resolved_start_date,
            end_date=end_date,
            rate_provider_method=config.rate_provider_method,
            rate_scale=config.rate_scale,
            logger=logger,
        )
