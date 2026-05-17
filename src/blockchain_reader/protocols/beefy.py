from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from web3 import Web3

from blockchain_reader.datetime_utils import format_daily_datetime
from blockchain_reader.pipeline_logging import PipelineLogger
from blockchain_reader.protocols.common import (
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

# ==========================================
# ABIS
# ==========================================

BEEFY_VAULT_ABI = [
    {
        "inputs": [],
        "name": "want",
        "outputs": [{"internalType": "contract IERC20", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getPricePerFullShare",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

PAIR_ABI = ERC20_ABI + [
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass(frozen=True)
class PairToken:
    address: str
    reserve: int
    symbol: str
    decimals: int


def _read_pair_token(
    w3: Web3,
    *,
    address: str,
    reserve: int,
    block_number: int,
) -> PairToken:
    token = w3.eth.contract(address=address, abi=ERC20_ABI)
    symbol = token.functions.symbol().call(block_identifier=block_number)
    decimals = int(token.functions.decimals().call(block_identifier=block_number))
    return PairToken(address=address, reserve=reserve, symbol=symbol, decimals=decimals)


def _get_pair_underlying(
    w3: Web3,
    *,
    pair_address: str,
    lp_amount_wei: Decimal,
    block_number: int,
) -> dict[str, Decimal] | None:
    pair = w3.eth.contract(address=pair_address, abi=PAIR_ABI)
    try:
        token0_address = pair.functions.token0().call(block_identifier=block_number)
        token1_address = pair.functions.token1().call(block_identifier=block_number)
        reserves = pair.functions.getReserves().call(block_identifier=block_number)
        total_supply = int(pair.functions.totalSupply().call(block_identifier=block_number))
    except Exception:
        return None

    if total_supply == 0:
        return {}

    share = lp_amount_wei / Decimal(total_supply)
    assets: dict[str, Decimal] = {}
    for token in (
        _read_pair_token(
            w3=w3,
            address=token0_address,
            reserve=int(reserves[0]),
            block_number=block_number,
        ),
        _read_pair_token(
            w3=w3,
            address=token1_address,
            reserve=int(reserves[1]),
            block_number=block_number,
        ),
    ):
        amount = (Decimal(token.reserve) * share) / Decimal(10**token.decimals)
        assets[token.symbol] = assets.get(token.symbol, Decimal("0")) + amount
    return assets


# ==========================================
# CORE BEEFY LOGIC
# ==========================================
def get_beefy_underlying(
    w3: Web3,
    vault_address: str,
    one_unit: int,
    block_number: int,
) -> dict[str, Decimal]:
    """
    Calculates the underlying assets for a given Beefy MooToken balance.
    """
    vault_contract = w3.eth.contract(address=vault_address, abi=BEEFY_VAULT_ABI)

    # 1. Get Price Per Full Share (Ratio of Want Token per MooToken)
    # Beefy PPFS is always scaled to 1e18, regardless of token decimals.
    ppfs = vault_contract.functions.getPricePerFullShare().call(block_identifier=block_number)

    # 2. Get Want Token Address
    want_addr = vault_contract.functions.want().call(block_identifier=block_number)

    # 3. Calculate Underlying Amount in Wei
    # Formula: (MooAmount * PPFS) / 1e18
    underlying_wei = (Decimal(one_unit) * Decimal(ppfs)) / Decimal(10**18)

    pair_assets = _get_pair_underlying(
        w3=w3,
        pair_address=want_addr,
        lp_amount_wei=underlying_wei,
        block_number=block_number,
    )
    if pair_assets is not None:
        return pair_assets

    assets = {}

    # 4. Standard single-token unwrapping
    want_contract = w3.eth.contract(address=want_addr, abi=ERC20_ABI)
    sym = want_contract.functions.symbol().call(block_identifier=block_number)
    dec = want_contract.functions.decimals().call(block_identifier=block_number)

    readable_balance = underlying_wei / Decimal(10**dec)
    assets[sym] = readable_balance

    return assets


# ==========================================
# MAIN LOOP
# ==========================================
def get_beefy_history(
    chain: str,
    vault_address: str,
    start_date: str,
    end_date: str,
    replace_from_date: str | None = None,
    logger: PipelineLogger | None = None,
) -> None:
    logger = logger or PipelineLogger()
    w3 = load_chain_web3(chain=chain)
    start_dt, end_dt = resolve_date_window(start_date=start_date, end_date=end_date)
    block_map = load_block_map(chain=chain)

    history_data = []
    vault = w3.to_checksum_address(vault_address)
    vault_contract = w3.eth.contract(address=vault, abi=BEEFY_VAULT_ABI)

    try:
        vault_decimals = vault_contract.functions.decimals().call()
        vault_symbol = vault_contract.functions.symbol().call()
    except Exception:
        vault_decimals = 18
        vault_symbol = "MOO"

    current_dt = start_dt
    total_days = max((end_dt.date() - start_dt.date()).days + 1, 1)
    day_index = 0
    while current_dt <= end_dt:
        day_index += 1
        date_str = format_daily_datetime(current_dt)
        if date_str not in block_map:
            current_dt += timedelta(days=1)
            continue

        block_num = block_map[date_str]
        logger.protocol_day(
            "beefy",
            vault_symbol,
            date_str=date_str,
            block_number=block_num,
            day_index=day_index,
            total_days=total_days,
        )

        try:
            if len(w3.eth.get_code(vault, block_identifier=block_num)) == 0:
                current_dt += timedelta(days=1)
                continue

            one_unit = 10**vault_decimals
            assets = get_beefy_underlying(w3, vault, one_unit, block_num)

            row = {
                "date": format_daily_datetime(current_dt),
                "block": block_num,
                "moo_balance": 1.0,  # Representing 1 unit of the vault token
            }
            for sym, amt in assets.items():
                row[f"asset_{sym}"] = float(amt)

            history_data.append(row)

        except Exception as e:
            logger.info(f"[beefy] Error on {current_dt.date()} for {vault_symbol}: {e}")

        current_dt += timedelta(days=1)

    output = write_protocol_history_csv(
        protocol="beefy",
        chain=chain,
        symbol=vault_symbol,
        history_data=history_data,
        replace_from_date=replace_from_date,
    )
    if output:
        logger.protocol_end("beefy", vault_symbol, output)


def process_all_beefy_tokens(
    chain: str,
    start_date: str | None = None,
    replace_from_date: str | None = None,
    logger: PipelineLogger | None = None,
) -> None:
    logger = logger or PipelineLogger()
    tokens = load_tokens(chain=chain)
    token_ranges = load_snapshot_ranges(chain=chain)
    for address, info in tokens.items():
        if info.get("protocol") != "beefy":
            continue

        symbol = info.get("symbol", address)
        if symbol not in token_ranges:
            continue

        rng = token_ranges[symbol]
        fallback_start_date = format_daily_datetime(rng["start"])
        resolved_start_date = resolve_effective_start_date(
            protocol="beefy",
            chain=chain,
            symbol=symbol,
            explicit_start_date=start_date,
            fallback_start_date=fallback_start_date,
        )
        end_date = resolve_protocol_end_date(rng)
        if should_skip_date_window(start_date=resolved_start_date, end_date=end_date):
            logger.protocol_skip(
                "beefy",
                symbol,
                f"start={resolved_start_date} is after end={end_date}",
            )
            continue

        if resolved_start_date is None:
            continue

        logger.protocol_start("beefy", symbol, resolved_start_date, end_date)
        get_beefy_history(
            chain=chain,
            vault_address=address,
            start_date=resolved_start_date,
            end_date=end_date,
            replace_from_date=replace_from_date,
            logger=logger,
        )
