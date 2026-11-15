from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import Any

from web3 import Web3

from portfolio_crypto_data.extraction.token_manager import TokenManager

ctx = Context(prec=78, rounding=ROUND_HALF_UP)


# Pre-calculate Keccak hashes to save compute time in loops
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()
APPROVAL_TOPIC = Web3.keccak(text="Approval(address,address,uint256)").hex()
BORROW_TOPIC = Web3.keccak(text="Borrow(address,uint256,uint256,uint256)").hex()
REPAY_BORROW_TOPIC = Web3.keccak(text="RepayBorrow(address,address,uint256,uint256,uint256)").hex()


@dataclass
class AssetMovement:
    """Represents a movement of a native asset or token."""

    symbol: str
    qty: Decimal


@dataclass
class TransactionContext:
    """Holds raw transaction data fetched from Web3."""

    tx: dict[str, Any]
    receipt: dict[str, Any]
    block: dict[str, Any]


@dataclass
class LogResult:
    """Result of processing a transaction log."""

    incoming: AssetMovement | None = None
    outgoing: AssetMovement | None = None
    approval: str | None = None


def _fetch_transaction_data(w3: Web3, tx_hash: str) -> TransactionContext:
    """
    Fetches transaction, receipt, and block data.

    args:
        w3: Web3 instance.
        tx_hash: Hash of the transaction.

    returns:
        Context containing tx, receipt, and block data.
    """
    tx = w3.eth.get_transaction(tx_hash)
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    block = w3.eth.get_block(tx["blockNumber"])
    return TransactionContext(tx=tx, receipt=receipt, block=block)


def _calculate_fee(
    *,
    tx: dict[str, Any],
    receipt: dict[str, Any],
    my_address: str,
    native_symbol: str,
    native_decimals: int,
) -> tuple[Decimal, str]:
    """
    Calculates the transaction fee in native currency.

    args:
        tx: Transaction data.
        receipt: Transaction receipt.
        my_address: User's wallet address.

    returns:
        Tuple of (fee_amount, fee_token_symbol).
    """
    gas_used = Decimal(receipt["gasUsed"])
    gas_price_val = receipt.get("effectiveGasPrice")
    if gas_price_val is None:
        gas_price_val = tx.get("gasPrice", 0)
    gas_price = Decimal(gas_price_val)
    fee_native = (gas_used * gas_price) / (Decimal(10) ** native_decimals)

    from_addr = str(tx.get("from") or "")
    if from_addr.lower() == my_address.lower():
        return fee_native, native_symbol
    return Decimal(0), ""


def _process_native_transfer(
    *,
    tx: dict[str, Any],
    my_address: str,
    native_symbol: str,
    native_decimals: int,
) -> tuple[list[AssetMovement], list[AssetMovement]]:
    """
    Identifies native-asset transfers involving the user.

    args:
        tx: Transaction data.
        my_address: User's wallet address.
        native_symbol: Symbol used by the chain's native asset.
        native_decimals: Decimal precision of the chain's native asset.

    returns:
        Tuple of (incoming_movements, outgoing_movements).
    """
    raw_ins, raw_outs = [], []
    val_int = tx.get("value", 0)
    if val_int > 0:
        val = Decimal(val_int) / (Decimal(10) ** native_decimals)
        to_addr = str(tx.get("to") or "")
        from_addr = str(tx.get("from") or "")
        if to_addr.lower() == my_address.lower():
            raw_ins.append(AssetMovement(symbol=native_symbol, qty=val))
        elif from_addr.lower() == my_address.lower():
            raw_outs.append(AssetMovement(symbol=native_symbol, qty=val))
    return raw_ins, raw_outs


def _process_internal_native_transfer(
    *,
    tx_hash: str,
    internal_native_map: dict[str, Decimal],
    native_symbol: str,
) -> list[AssetMovement]:
    """
    Identifies internal native-asset transfers from the pre-fetched map.

    args:
        tx_hash: Hash of the transaction.
        internal_native_map: Map of transaction hash to internal native-asset value.
        native_symbol: Symbol used by the chain's native asset.

    returns:
        List of internal native-asset movements.
    """
    norm_hash = tx_hash.lower()
    for h, val in internal_native_map.items():
        if h.lower() == norm_hash:
            return [AssetMovement(symbol=native_symbol, qty=Decimal(str(val)))]
    return []


def _process_log_entry(
    log: dict[str, Any], my_address: str, token_manager: TokenManager, fetch_metadata: bool
) -> LogResult:
    """
    Routes a log entry to the appropriate handler.

    args:
        log: Raw log entry.
        my_address: User's wallet address.
        token_manager: Manager for token metadata.
        fetch_metadata: Whether to fetch missing token info.

    returns:
        Result containing movements or approvals.
    """
    if not log.get("topics"):
        return LogResult()

    sig = log["topics"][0].hex().lower()
    if not sig.startswith("0x"):
        sig = "0x" + sig

    transfer_sig = TRANSFER_TOPIC.lower()
    if not transfer_sig.startswith("0x"):
        transfer_sig = "0x" + transfer_sig

    approval_sig = APPROVAL_TOPIC.lower()
    if not approval_sig.startswith("0x"):
        approval_sig = "0x" + approval_sig

    borrow_sig = BORROW_TOPIC.lower()
    if not borrow_sig.startswith("0x"):
        borrow_sig = "0x" + borrow_sig

    repay_sig = REPAY_BORROW_TOPIC.lower()
    if not repay_sig.startswith("0x"):
        repay_sig = "0x" + repay_sig

    if sig == transfer_sig and len(log["topics"]) == 3:
        return _handle_transfer_log(
            log=log,
            my_address=my_address,
            token_manager=token_manager,
            fetch_metadata=fetch_metadata,
        )

    if sig == approval_sig and len(log["topics"]) == 3:
        return _handle_approval_log(
            log=log,
            my_address=my_address,
            token_manager=token_manager,
            fetch_metadata=fetch_metadata,
        )

    if sig == borrow_sig:
        return _handle_compound_borrow_event(
            log=log,
            my_address=my_address,
            token_manager=token_manager,
            fetch_metadata=fetch_metadata,
            borrower_offset=24,
            amount_offset=64,
            is_repay=False,
        )

    if sig == repay_sig:
        return _handle_compound_borrow_event(
            log=log,
            my_address=my_address,
            token_manager=token_manager,
            fetch_metadata=fetch_metadata,
            borrower_offset=88,
            amount_offset=128,
            is_repay=True,
        )

    return LogResult()


def _handle_transfer_log(
    log: dict[str, Any], my_address: str, token_manager: TokenManager, fetch_metadata: bool
) -> LogResult:
    """
    Parses an ERC20 Transfer log.

    args:
        log: Raw log entry.
        my_address: User's wallet address.
        token_manager: Manager for token metadata.
        fetch_metadata: Whether to fetch missing token info.

    returns:
        Result containing the transfer movement.
    """
    token_addr: str = log["address"]
    from_addr: str = "0x" + log["topics"][1].hex()[-40:]
    to_addr: str = "0x" + log["topics"][2].hex()[-40:]

    if my_address.lower() not in (from_addr.lower(), to_addr.lower()):
        return LogResult()

    token_info = token_manager.get_token(address=token_addr, fetch_if_missing=fetch_metadata)
    if not token_info or token_info.get("ignore_transfers"):
        return LogResult()

    raw_amount_int = int(log["data"].hex(), 16)
    decimals = int(token_info["decimals"])
    readable_amount = Decimal(raw_amount_int) / (Decimal(10) ** decimals)
    item = AssetMovement(symbol=token_info["symbol"], qty=readable_amount)

    if to_addr.lower() == my_address.lower():
        return LogResult(incoming=item)
    return LogResult(outgoing=item)


def _handle_approval_log(
    log: dict[str, Any], my_address: str, token_manager: TokenManager, fetch_metadata: bool
) -> LogResult:
    """
    Parses an ERC20 Approval log.

    args:
        log: Raw log entry.
        my_address: User's wallet address.
        token_manager: Manager for token metadata.
        fetch_metadata: Whether to fetch missing token info.

    returns:
        Result containing the approval string.
    """
    owner_addr: str = "0x" + log["topics"][1].hex()[-40:]
    if owner_addr.lower() == my_address.lower():
        token_info = token_manager.get_token(
            address=log["address"], fetch_if_missing=fetch_metadata
        )
        symbol = token_info["symbol"] if token_info else "UNK"
        return LogResult(approval=f"Approve {symbol}")
    return LogResult()


def _handle_compound_borrow_event(
    log: dict[str, Any],
    my_address: str,
    token_manager: TokenManager,
    fetch_metadata: bool,
    *,
    borrower_offset: int,
    amount_offset: int,
    is_repay: bool,
) -> LogResult:
    """
    Parses a Compound v2 / Tectonic Borrow or RepayBorrow log.
    """
    data_hex = log["data"].hex()
    if data_hex.startswith("0x"):
        data_hex = data_hex[2:]
    if len(data_hex) < amount_offset + 64:
        return LogResult()

    borrower = "0x" + data_hex[borrower_offset : borrower_offset + 40]
    if borrower.lower() != my_address.lower():
        return LogResult()

    raw_amount_int = int(data_hex[amount_offset : amount_offset + 64], 16)
    token_info = token_manager.get_token(address=log["address"], fetch_if_missing=fetch_metadata)
    decimals = int(token_info.get("decimals", 18)) if token_info else 18
    symbol = token_info.get("symbol", "UNK") if token_info else "UNK"
    readable_amount = Decimal(raw_amount_int) / (Decimal(10) ** decimals)
    movement = AssetMovement(symbol=symbol, qty=readable_amount)
    return LogResult(outgoing=movement) if is_repay else LogResult(incoming=movement)


def _is_ignored_outgoing_transfer(
    log: dict[str, Any], my_address: str, token_manager: TokenManager
) -> bool:
    if not log.get("topics") or len(log["topics"]) != 3:
        return False
    sig = log["topics"][0].hex().lower()
    if not sig.startswith("0x"):
        sig = "0x" + sig
    transfer_sig = TRANSFER_TOPIC.lower()
    if not transfer_sig.startswith("0x"):
        transfer_sig = "0x" + transfer_sig
    if sig != transfer_sig:
        return False
    from_addr = "0x" + log["topics"][1].hex()[-40:]
    if from_addr.lower() != my_address.lower():
        return False
    token_info = token_manager.get_token(address=log["address"], fetch_if_missing=False)
    return bool(token_info and token_info.get("ignore_transfers"))


def _get_token_movements(
    receipt: dict[str, Any], my_address: str, token_manager: TokenManager, fetch_metadata: bool
) -> tuple[list[AssetMovement], list[AssetMovement], list[str]]:
    """
    Extracts all token movements and approvals from receipt logs.

    args:
        receipt: Transaction receipt.
        my_address: User's wallet address.
        token_manager: Manager for token metadata.
        fetch_metadata: Whether to fetch missing token info.

    returns:
        Tuple of (incoming, outgoing, approvals).
    """
    raw_ins, raw_outs, approvals = [], [], []
    for log in receipt.get("logs", []):
        result = _process_log_entry(
            log=log,
            my_address=my_address,
            token_manager=token_manager,
            fetch_metadata=fetch_metadata,
        )
        if result.incoming:
            raw_ins.append(result.incoming)
        if result.outgoing:
            raw_outs.append(result.outgoing)
        if result.approval:
            approvals.append(result.approval)
    return raw_ins, raw_outs, approvals


def _net_token_movements(
    raw_ins: list[AssetMovement], raw_outs: list[AssetMovement]
) -> tuple[list[AssetMovement], list[AssetMovement]]:
    """
    Consolidates raw movements by netting ins and outs.

    args:
        raw_ins: List of incoming movements.
        raw_outs: List of outgoing movements.

    returns:
        Tuple of (net_incoming, net_outgoing).
    """
    net_movements: dict[str, Decimal] = {}

    for item in raw_ins:
        sym = item.symbol
        net_movements[sym] = net_movements.get(sym, Decimal(0)) + item.qty

    for item in raw_outs:
        sym = item.symbol
        net_movements[sym] = net_movements.get(sym, Decimal(0)) - item.qty

    final_ins = []
    final_outs = []

    for sym, qty in net_movements.items():
        if qty > 0:
            final_ins.append(AssetMovement(symbol=sym, qty=qty))
        elif qty < 0:
            final_outs.append(AssetMovement(symbol=sym, qty=abs(qty)))

    return final_ins, final_outs


def _fmt_qty(val: Decimal) -> str:
    """
    Formats a decimal quantity for display.

    args:
        val: The decimal value.

    returns:
        Formatted string without trailing zeros.
    """
    s = "{:.18f}".format(val)
    return s.rstrip("0").rstrip(".") if "." in s else s


def _classify_and_format_transaction(
    tx_hash: str,
    date_str: str,
    final_ins: list[AssetMovement],
    final_outs: list[AssetMovement],
    fee_val: Decimal,
    fee_token: str,
    approvals: list[str],
    had_ignored_out: bool = False,
    transaction_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Determines transaction type and formats final output.

    args:
        tx_hash: Transaction hash.
        date_str: Formatted date string.
        final_ins: Net incoming movements.
        final_outs: Net outgoing movements.
        fee_val: Transaction fee amount.
        fee_token: Fee token symbol.
        approvals: List of approval strings.
        had_ignored_out: Whether an ignored token was sent out.
        transaction_overrides: Optional per-hash overrides for manual corrections.

    returns:
        Dictionary of transaction details or None if ignored.
    """
    if transaction_overrides:
        override = transaction_overrides.get(tx_hash.lower())
        if override:
            if override.get("ignore"):
                final_ins = []
                final_outs = []
            if "outs" in override:
                final_outs = [
                    AssetMovement(symbol=x["symbol"], qty=Decimal(str(x["qty"])))
                    for x in override["outs"]
                ]
            if "ins" in override:
                final_ins = [
                    AssetMovement(symbol=x["symbol"], qty=Decimal(str(x["qty"])))
                    for x in override["ins"]
                ]

    qtys_in = [x.qty for x in final_ins]
    sum_in = sum(qtys_in, Decimal(0))
    has_in = bool(final_ins and sum_in > 0)

    qtys_out = [x.qty for x in final_outs]
    sum_out = sum(qtys_out, Decimal(0))
    has_out = bool(final_outs and sum_out > 0)

    tx_type = "Interaction"
    if has_in and has_out:
        tx_type = "Swap"
    elif has_in:
        tx_type = "Reward" if had_ignored_out else "Receive"
    elif has_out:
        tx_type = "Send"
    elif approvals:
        tx_type = ", ".join(approvals)

    if fee_val == 0 and not has_in and not has_out and not approvals:
        return None

    return {
        "TX Hash": tx_hash,
        "Date": date_str,
        "Qty in": ", ".join([_fmt_qty(val=q) for q in qtys_in]),
        "Token in": ", ".join([x.symbol for x in final_ins]),
        "Qty out": ", ".join([_fmt_qty(val=q) for q in qtys_out]),
        "Token out": ", ".join([x.symbol for x in final_outs]),
        "Type": tx_type,
        "Fee": _fmt_qty(val=fee_val),
        "Fee Token": fee_token,
    }


def analyze_transaction(
    *,
    tx_hash: str,
    w3: Web3,
    my_address: str,
    token_manager: TokenManager,
    internal_native_map: dict[str, Decimal],
    fetch_metadata: bool,
    native_symbol: str,
    native_decimals: int,
    transaction_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Orchestrates the analysis of a single transaction.

    args:
        tx_hash: Transaction hash.
        w3: Web3 instance.
        my_address: User's wallet address.
        token_manager: Manager for token metadata.
        internal_native_map: Map of internal native-asset transfers.
        fetch_metadata: Whether to fetch missing token info.
        native_symbol: Symbol used by the chain's native asset.
        native_decimals: Decimal precision of the chain's native asset.

    returns:
        Dictionary of transaction details, or None when the transaction has no
        relevant movements. Analysis errors propagate to the ingestion batch.
    """
    data = _fetch_transaction_data(w3=w3, tx_hash=tx_hash)
    tx, receipt, block = data.tx, data.receipt, data.block

    date_str = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc).strftime(
        "%d/%m/%Y %H:%M:%S"
    )
    fee_val, fee_token = _calculate_fee(
        tx=tx,
        receipt=receipt,
        my_address=my_address,
        native_symbol=native_symbol,
        native_decimals=native_decimals,
    )

    raw_ins, raw_outs = _process_native_transfer(
        tx=tx,
        my_address=my_address,
        native_symbol=native_symbol,
        native_decimals=native_decimals,
    )
    raw_ins.extend(
        _process_internal_native_transfer(
            tx_hash=tx_hash,
            internal_native_map=internal_native_map,
            native_symbol=native_symbol,
        )
    )

    log_ins, log_outs, approvals = _get_token_movements(
        receipt=receipt,
        my_address=my_address,
        token_manager=token_manager,
        fetch_metadata=fetch_metadata,
    )
    raw_ins.extend(log_ins)
    raw_outs.extend(log_outs)

    had_ignored_out = any(
        _is_ignored_outgoing_transfer(log=log, my_address=my_address, token_manager=token_manager)
        for log in receipt.get("logs", [])
    )

    final_ins, final_outs = _net_token_movements(raw_ins=raw_ins, raw_outs=raw_outs)

    return _classify_and_format_transaction(
        tx_hash=tx_hash,
        date_str=date_str,
        final_ins=final_ins,
        final_outs=final_outs,
        fee_val=fee_val,
        fee_token=fee_token,
        approvals=approvals,
        had_ignored_out=had_ignored_out,
        transaction_overrides=transaction_overrides,
    )
