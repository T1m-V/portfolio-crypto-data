from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from blockchain_reader.datetime_utils import format_daily_datetime
from blockchain_reader.protocols.common import (
    load_block_map,
    load_chain_web3,
    load_snapshot_ranges,
    resolve_date_window,
    resolve_effective_start_date,
    should_skip_date_window,
    write_protocol_history_csv,
)

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


LIQUID_STAKING_TOKENS: dict[str, list[LiquidStakingTokenConfig]] = {
    "arbitrum": [
        LiquidStakingTokenConfig(
            symbol="wstETH",
            underlying_symbol="ETH",
            rate_provider_address="0xf7c5c26B574063e7b098ed74fAd6779e65E3F836",
        )
    ]
}


def _resolve_fallback_start_date(
    symbol: str,
    token_ranges: dict[str, dict[str, object]],
    block_map: dict[str, int],
) -> str | None:
    rng = token_ranges.get(symbol)
    if rng is not None:
        return format_daily_datetime(rng["start"])
    if block_map:
        return format_daily_datetime(min(block_map.keys()))
    return None


def get_liquid_staking_history(
    chain: str,
    symbol: str,
    underlying_symbol: str,
    rate_provider_address: str,
    start_date: str,
    end_date: str,
    rate_provider_method: str = "getRate",
    rate_scale: int = 10**18,
) -> None:
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    rate_provider = w3.eth.contract(
        address=w3.to_checksum_address(rate_provider_address),
        abi=RATE_PROVIDER_ABI,
    )
    history_data: list[dict[str, object]] = []

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = format_daily_datetime(current_dt)
        block_num = block_map.get(date_str)
        if block_num is None:
            current_dt += timedelta(days=1)
            continue

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
            print(f"[liquid_staking] Error on {current_dt.date()} for {symbol}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="liquid_staking",
        chain=chain,
        symbol=symbol,
        history_data=history_data,
        fieldnames=["date", "block", "lst_balance", f"asset_{underlying_symbol}"],
    )
    if output:
        print(f"[liquid_staking] Saved to {output}")


def process_all_liquid_staking_tokens(chain: str, start_date: str | None = None) -> None:
    chain_configs = LIQUID_STAKING_TOKENS.get(chain, [])
    if not chain_configs:
        return

    try:
        token_ranges = load_snapshot_ranges(chain=chain)
    except FileNotFoundError:
        token_ranges = {}

    block_map = load_block_map(chain=chain)
    for config in chain_configs:
        fallback_start_date = _resolve_fallback_start_date(
            symbol=config.symbol,
            token_ranges=token_ranges,
            block_map=block_map,
        )
        resolved_start_date = resolve_effective_start_date(
            protocol="liquid_staking",
            chain=chain,
            symbol=config.symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = "now"
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            print(
                "[liquid_staking] Skipping "
                f"{config.symbol}: start={resolved_start_date} is after end={end_date}"
            )
            continue

        if resolved_start_date is None:
            print(f"[liquid_staking] Skipping {config.symbol}: no fallback start date found.")
            continue

        print(f"[liquid_staking] Processing {config.symbol} ({resolved_start_date} -> {end_date})")
        get_liquid_staking_history(
            chain=chain,
            symbol=config.symbol,
            underlying_symbol=config.underlying_symbol,
            rate_provider_address=config.rate_provider_address,
            start_date=resolved_start_date,
            end_date=end_date,
            rate_provider_method=config.rate_provider_method,
            rate_scale=config.rate_scale,
        )


if __name__ == "__main__":
    process_all_liquid_staking_tokens(chain="arbitrum")
