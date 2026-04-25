from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from blockchain_reader.datetime_utils import (
    format_daily_datetime,
    parse_transaction_datetime_series,
)
from blockchain_reader.raw_snapshots import CryptoTracker, TxEntry
from blockchain_reader.symbols import sanitize_symbol
from file_paths import BLOCKCHAIN_SNAPSHOT_FOLDER, BLOCKCHAIN_TRANSACTIONS_FOLDER

MAX_INVALID_DATE_RATIO = 0.1
INTEREST_TYPES = {"interest", "fixed term interest", "interest additional"}
CARD_TYPES_TO_IGNORE = {
    "credit card withdrawal credit",
    "manual repayment",
    "manual sell order",
}
IGNORED_TYPES = {
    "transfer in",
    "transfer out",
    "locking term deposit",
    "unlocking term deposit",
}
INPUT_RECEIVE_TYPES = {
    "assimilation",
    "top up crypto",
}
INPUT_REWARD_TYPES = {
    "bonus",
    "dividend",
    "referral bonus",
}
INPUT_OUTPUT_SWAP_TYPES = {
    "exchange",
}
SNAPSHOT_COLUMNS = ["Date", "Coin", "Quantity", "Principal Invested"]
LIQUIDATION_REVIEW_COLUMNS = [
    "Date / Time (UTC)",
    "Transaction",
    "Type",
    "Input Amount",
    "Input Currency",
    "Output Amount",
    "Output Currency",
    "USD Equivalent",
    "Details",
]
MANUAL_REPAYMENT_PAIR_WINDOW = pd.Timedelta(hours=6)
MANUAL_REPAYMENT_USD_TOLERANCE = Decimal("0.05")
MANUAL_REPAYMENT_USD_TOLERANCE_BY_TOKEN = {
    "USDC": Decimal("0.60"),
}
LIQUIDATION_REVIEW_FILENAME = "nexo_liquidation_only_review.csv"
ActionHandler = Callable[[pd.Series], "NormalizedAction"]


@dataclass
class RewardInstruction:
    entry: TxEntry
    allocations: list[tuple[str | None, float]]


@dataclass
class NormalizedAction:
    action: str
    ins: list[TxEntry]
    outs: list[TxEntry]
    rewards: list[RewardInstruction]
    principal_overrides: dict[str, float] | None = None
    principal_additions: dict[str, float] | None = None


@dataclass
class ManualRepaymentLeg:
    idx: int
    date: pd.Timestamp
    usd_equivalent: Decimal
    matching_usd_equivalent: Decimal
    entry: TxEntry


@dataclass
class ManualRepaymentPair:
    manual_sell: ManualRepaymentLeg
    manual_repayment: ManualRepaymentLeg


class NexoTransactionNormalizer:
    CARD_DEBT_STABLES = frozenset({"USD", "USDX", "xUSD"})
    CANONICAL_DEBT_TOKEN = "USD"

    def __init__(
        self,
        known_symbols: set[str],
        refund_purchase_tokens: dict[str, list[tuple[pd.Timestamp, str]]] | None = None,
        debit_mode_purchase_indices: set[int] | None = None,
        eur_mode_cashback_reversal_indices: set[int] | None = None,
    ):
        self.symbol_lookup = {symbol.upper(): symbol for symbol in known_symbols}
        self.refund_purchase_tokens = refund_purchase_tokens or {}
        self.debit_mode_purchase_indices = debit_mode_purchase_indices or set()
        self.eur_mode_cashback_reversal_indices = eur_mode_cashback_reversal_indices or set()
        self.handlers: dict[str, ActionHandler] = self._build_handlers()

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> "NexoTransactionNormalizer":
        symbols: set[str] = set()
        refund_purchase_tokens: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        debit_mode_purchase_indices = cls._build_debit_mode_purchase_indices(frame=frame)
        eur_mode_cashback_reversal_indices = cls._build_eur_mode_cashback_reversal_indices(
            frame=frame
        )
        for column in ("Input Currency", "Output Currency"):
            if column not in frame.columns:
                continue
            for raw_symbol in frame[column].dropna():
                normalized = sanitize_symbol(raw_symbol)
                if normalized and normalized != "-":
                    symbols.add(cls._canonicalize_stable_token(symbol=normalized))

        if {"Type", "Details", "Input Currency", "Date"}.issubset(frame.columns):
            purchase_rows = frame[
                frame["Type"].fillna("").str.strip().str.lower() == "nexo card purchase"
            ]
            for _, row in purchase_rows.iterrows():
                if int(row.name) in debit_mode_purchase_indices:
                    continue
                if cls._is_rejected_card_row(row=row):
                    continue
                details_key = cls._card_merchant_key(details=row.get("Details"))
                if not details_key:
                    continue
                input_symbol = sanitize_symbol(row.get("Input Currency"))
                if not input_symbol:
                    continue
                input_symbol = cls._canonicalize_stable_token(symbol=input_symbol)
                date_val = pd.to_datetime(row.get("Date"), errors="coerce")
                if pd.isna(date_val):
                    continue
                if details_key not in refund_purchase_tokens:
                    refund_purchase_tokens[details_key] = []
                refund_purchase_tokens[details_key].append((date_val, input_symbol))

        symbols.add(cls.CANONICAL_DEBT_TOKEN)
        return cls(
            known_symbols=symbols,
            refund_purchase_tokens=refund_purchase_tokens,
            debit_mode_purchase_indices=debit_mode_purchase_indices,
            eur_mode_cashback_reversal_indices=eur_mode_cashback_reversal_indices,
        )

    def normalize_row(self, row: pd.Series) -> NormalizedAction:
        tx_type = str(row.get("Type") or "").strip().lower()
        if self._should_ignore_row(row=row, tx_type=tx_type):
            return NormalizedAction(action="skip", ins=[], outs=[], rewards=[])

        handler = self.handlers.get(tx_type)
        if handler is None:
            raise ValueError(f"Unsupported NEXO transaction type: {tx_type or '<empty>'}")
        return handler(row)

    def _build_handlers(self) -> dict[str, ActionHandler]:
        return {
            **dict.fromkeys(CARD_TYPES_TO_IGNORE, self._handle_skip),
            **dict.fromkeys(INPUT_RECEIVE_TYPES, self._handle_positive_input_receive),
            **dict.fromkeys(INPUT_REWARD_TYPES, self._handle_free_input_reward),
            **dict.fromkeys(INPUT_OUTPUT_SWAP_TYPES, self._handle_input_output_swap),
            **dict.fromkeys(INTEREST_TYPES, self._handle_interest),
            "deposit to exchange": self._handle_skip,
            "exchange deposited on": self._handle_exchange_deposited_on,
            "nexo card purchase": self._handle_nexo_card_purchase,
            "nexo card refund": self._handle_nexo_card_refund,
            "nexo card cashback reversal": self._handle_nexo_card_cashback_reversal,
            "exchange liquidation": self._handle_exchange_liquidation,
            "exchange to withdraw": self._handle_exchange_to_withdraw,
            "credit card fiatx exchange to withdraw": self._handle_exchange_to_withdraw,
            "withdraw exchanged": self._handle_skip,
            "exchange credit": self._handle_exchange_credit,
            "nexo card transaction fee": self._handle_standard_withdrawal,
            "loan withdrawal": self._handle_standard_withdrawal,
            "withdrawal": self._handle_standard_withdrawal,
            "deposit over repayment": self._handle_deposit_over_repayment,
            "cashback": self._handle_cashback,
            "exchange cashback": self._handle_free_input_reward,
        }

    @staticmethod
    def _handle_skip(row: pd.Series) -> NormalizedAction:
        del row
        return NormalizedAction(action="skip", ins=[], outs=[], rewards=[])

    def _handle_exchange_deposited_on(self, row: pd.Series) -> NormalizedAction:
        return self._build_receive_action(ins=self._normalize_exchange_deposited_on_ins(row=row))

    def _handle_nexo_card_purchase(self, row: pd.Series) -> NormalizedAction:
        if self._is_rejected_card_row(row=row):
            return self._handle_skip(row)
        if int(row.name) in self.debit_mode_purchase_indices:
            return self._handle_skip(row)

        outs = self._normalize_withdrawal_outs(row=row)
        overrides = self._build_eur_principal_overrides_for_outs(row=row, outs=outs)
        return NormalizedAction(
            action="send" if outs else "skip",
            ins=[],
            outs=outs,
            rewards=[],
            principal_overrides=overrides,
        )

    def _handle_nexo_card_refund(self, row: pd.Series) -> NormalizedAction:
        if self._is_rejected_card_row(row=row) or self._is_eur_mode_card_refund(row=row):
            return self._handle_skip(row)

        refund_entry = self._normalize_card_refund_receive(row=row)
        ins = [refund_entry] if refund_entry else []
        overrides = self._build_eur_principal_overrides_for_ins(row=row, ins=ins)
        return NormalizedAction(
            action="receive" if refund_entry else "skip",
            ins=ins,
            outs=[],
            rewards=[],
            principal_overrides=overrides,
        )

    def _handle_nexo_card_cashback_reversal(self, row: pd.Series) -> NormalizedAction:
        if self._is_rejected_card_row(row=row):
            return self._handle_skip(row)
        if int(row.name) in self.eur_mode_cashback_reversal_indices:
            return self._handle_skip(row)

        outs = self._normalize_card_cashback_reversal_outs(row=row)
        overrides = self._build_eur_principal_overrides_for_outs(row=row, outs=outs)
        eur_notional = self._extract_eur_notional(row=row)
        additions = {"NEXO": eur_notional} if eur_notional and eur_notional > 0 else None
        return NormalizedAction(
            action="send" if outs else "skip",
            ins=[],
            outs=outs,
            rewards=[],
            principal_overrides=overrides,
            principal_additions=additions,
        )

    def _handle_exchange_liquidation(self, row: pd.Series) -> NormalizedAction:
        ins, outs = self._normalize_exchange_liquidation_swap(row=row)
        return self._build_swap_action(ins=ins, outs=outs)

    def _handle_exchange_to_withdraw(self, row: pd.Series) -> NormalizedAction:
        return self._build_send_action(outs=self._normalize_exchange_to_withdraw_outs(row=row))

    def _handle_exchange_credit(self, row: pd.Series) -> NormalizedAction:
        if self._is_card_loan_withdrawal(row=row):
            return self._handle_skip(row)
        ins, outs = self._normalize_input_output_swap(row=row)
        return self._build_swap_action(ins=ins, outs=outs)

    def _handle_standard_withdrawal(self, row: pd.Series) -> NormalizedAction:
        return self._build_send_action(outs=self._normalize_withdrawal_outs(row=row))

    def _handle_deposit_over_repayment(self, row: pd.Series) -> NormalizedAction:
        return self._build_send_action(outs=self._normalize_positive_input_outs(row=row))

    def _handle_cashback(self, row: pd.Series) -> NormalizedAction:
        return self._build_reward_action(
            row=row,
            allocations=[(None, 0.5), (self.CANONICAL_DEBT_TOKEN, 0.5)],
        )

    def _handle_free_input_reward(self, row: pd.Series) -> NormalizedAction:
        return self._build_reward_action(row=row, allocations=[(None, 1.0)])

    def _handle_interest(self, row: pd.Series) -> NormalizedAction:
        negative_stable_interest_out = self._normalize_negative_stable_interest_outs(row=row)
        if negative_stable_interest_out:
            return NormalizedAction(
                action="send",
                ins=[],
                outs=negative_stable_interest_out,
                rewards=[],
                principal_overrides={"USD": 0.0},
            )

        reward = self._extract_positive_input_entry(row=row)
        if reward is None:
            return self._handle_skip(row)

        source_coin = self._infer_interest_source_coin(row=row, fallback=reward.token)
        return self._build_reward_action(
            row=row,
            allocations=[(source_coin, 0.75), (reward.token, 0.25)],
        )

    def _handle_positive_input_receive(self, row: pd.Series) -> NormalizedAction:
        return self._build_receive_action(ins=self._normalize_positive_input_ins(row=row))

    def _handle_input_output_swap(self, row: pd.Series) -> NormalizedAction:
        ins, outs = self._normalize_input_output_swap(row=row)
        return self._build_swap_action(ins=ins, outs=outs)

    def _build_reward_action(
        self,
        *,
        row: pd.Series,
        allocations: list[tuple[str | None, float]],
    ) -> NormalizedAction:
        reward = self._extract_positive_input_entry(row=row)
        if reward is None:
            return self._handle_skip(row)

        return NormalizedAction(
            action="reward",
            ins=[],
            outs=[],
            rewards=[RewardInstruction(entry=reward, allocations=allocations)],
        )

    @staticmethod
    def _should_ignore_row(*, row: pd.Series, tx_type: str) -> bool:
        if tx_type in IGNORED_TYPES:
            return True

        details = str(row.get("Details") or "").strip().lower()
        # Internal NEXO wallet hops are non-economic moves and should not affect snapshots.
        if re.search(r"transfer from .*wallet to .*wallet", details):
            return True

        return False

    @staticmethod
    def _build_receive_action(*, ins: list[TxEntry]) -> NormalizedAction:
        return NormalizedAction(
            action="receive" if ins else "skip",
            ins=ins,
            outs=[],
            rewards=[],
        )

    @staticmethod
    def _build_send_action(*, outs: list[TxEntry]) -> NormalizedAction:
        return NormalizedAction(
            action="send" if outs else "skip",
            ins=[],
            outs=outs,
            rewards=[],
        )

    @staticmethod
    def _build_swap_action(*, ins: list[TxEntry], outs: list[TxEntry]) -> NormalizedAction:
        return NormalizedAction(
            action="swap" if ins and outs else "skip",
            ins=ins,
            outs=outs,
            rewards=[],
        )

    def _normalize_positive_input_ins(self, row: pd.Series) -> list[TxEntry]:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount"))
        if not input_symbol or input_amount <= 0:
            return []
        return [TxEntry(token=input_symbol, quantity=input_amount)]

    def _normalize_positive_input_outs(self, row: pd.Series) -> list[TxEntry]:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount"))
        if not input_symbol or input_amount <= 0:
            return []
        return [TxEntry(token=input_symbol, quantity=input_amount)]

    def _normalize_input_output_swap(self, row: pd.Series) -> tuple[list[TxEntry], list[TxEntry]]:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = self._parse_amount(value=row.get("Output Amount")).copy_abs()

        if not input_symbol or input_amount <= 0:
            return [], []
        if not output_symbol or output_amount <= 0:
            return [], []
        return [TxEntry(token=output_symbol, quantity=output_amount)], [
            TxEntry(token=input_symbol, quantity=input_amount)
        ]

    def _normalize_exchange_liquidation_swap(
        self, row: pd.Series
    ) -> tuple[list[TxEntry], list[TxEntry]]:
        """
        Normalizes exchange liquidation as conversion into debt token (typically xUSD/USDX).

        NEXO commonly exports both legs as positive. For repayment logic we interpret:
        - input leg as source sold away (out)
        - output leg as destination bought (in)
        """
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = self._parse_amount(value=row.get("Output Amount")).copy_abs()

        outs: list[TxEntry] = []
        ins: list[TxEntry] = []
        if input_symbol and input_amount > 0:
            outs.append(TxEntry(token=input_symbol, quantity=input_amount))
        if output_symbol and output_amount > 0:
            ins.append(TxEntry(token=output_symbol, quantity=output_amount))
        return ins, outs

    def _normalize_card_refund_receive(self, row: pd.Series) -> TxEntry | None:
        """
        Normalizes card refunds as debt-token receive using USD-equivalent quantity.

        This models refunds as debt reduction in the card debt bucket rather than EUR holdings.
        """
        quantity = self._parse_usd_equivalent(value=row.get("USD Equivalent"))
        if quantity <= 0:
            quantity = self._infer_refund_quantity_from_stable_leg(row=row)
        if quantity <= 0:
            return None

        debt_token = self._infer_refund_debt_token(row=row)
        if not debt_token:
            return None
        return TxEntry(token=debt_token, quantity=quantity)

    def _infer_refund_quantity_from_stable_leg(self, row: pd.Series) -> Decimal:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount")).copy_abs()
        if input_symbol in self.CARD_DEBT_STABLES and input_amount > 0:
            return input_amount

        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        output_amount = self._parse_amount(value=row.get("Output Amount")).copy_abs()
        if output_symbol in self.CARD_DEBT_STABLES and output_amount > 0:
            return output_amount

        return Decimal(0)

    def _infer_refund_debt_token(self, row: pd.Series) -> str:
        details_key = self._card_merchant_key(details=row.get("Details"))
        refund_date = pd.to_datetime(row.get("Date"), errors="coerce")

        if details_key and not pd.isna(refund_date):
            candidates = self.refund_purchase_tokens.get(details_key, [])
            for purchase_date, purchase_token in reversed(candidates):
                if purchase_date <= refund_date:
                    return purchase_token

        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        if input_symbol in self.CARD_DEBT_STABLES:
            return input_symbol

        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        if output_symbol in self.CARD_DEBT_STABLES:
            return output_symbol

        return self.CANONICAL_DEBT_TOKEN

    def _normalize_exchange_to_withdraw_outs(self, row: pd.Series) -> list[TxEntry]:
        """
        Normalizes exchange-to-withdraw as a pure send of the source asset.

        In NEXO exports this often appears as EURX -> EUR, but economically it is a
        withdrawal flow out of the portfolio and should reduce only the source token.
        """
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount")).copy_abs()
        if not input_symbol or input_amount <= 0:
            return []
        return [TxEntry(token=input_symbol, quantity=input_amount)]

    def _normalize_exchange_deposited_on_ins(self, row: pd.Series) -> list[TxEntry]:
        """
        Normalizes exchange-deposited-on as a pure receive of the destination asset.

        For EUR top-ups, this books the EURX credit from the absolute EUR input amount
        and ignores the mirrored EUR leg. This keeps credits stable even when NEXO
        exports a zero EURX output amount.
        """
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount"))
        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        output_amount = self._parse_amount(value=row.get("Output Amount")).copy_abs()

        if output_symbol and input_symbol == "EUR" and input_amount < 0:
            return [TxEntry(token=output_symbol, quantity=input_amount.copy_abs())]
        if not output_symbol or output_amount <= 0:
            return []
        return [TxEntry(token=output_symbol, quantity=output_amount)]

    def _normalize_card_cashback_reversal_outs(self, row: pd.Series) -> list[TxEntry]:
        quantity = self._parse_usd_equivalent(value=row.get("USD Equivalent"))
        if quantity <= 0:
            quantity = self._infer_refund_quantity_from_stable_leg(row=row)
        if quantity <= 0:
            return []

        debt_token = self._infer_refund_debt_token(row=row)
        if not debt_token:
            return []
        return [TxEntry(token=debt_token, quantity=quantity)]

    def _normalize_negative_stable_interest_outs(self, row: pd.Series) -> list[TxEntry]:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        if input_symbol not in self.CARD_DEBT_STABLES:
            return []

        input_amount = self._parse_amount(value=row.get("Input Amount"))
        if input_amount >= 0:
            return []

        return [TxEntry(token="USD", quantity=input_amount.copy_abs())]

    def _build_eur_principal_overrides_for_outs(
        self, *, row: pd.Series, outs: list[TxEntry]
    ) -> dict[str, float] | None:
        eur_amount = self._extract_eur_notional(row=row)
        if eur_amount is None:
            return None
        return {entry.token: -eur_amount for entry in outs}

    def _build_eur_principal_overrides_for_ins(
        self, *, row: pd.Series, ins: list[TxEntry]
    ) -> dict[str, float] | None:
        eur_amount = self._extract_eur_notional(row=row)
        if eur_amount is None:
            return None
        return {entry.token: eur_amount for entry in ins}

    def _extract_eur_notional(self, *, row: pd.Series) -> float | None:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = self._parse_amount(value=row.get("Output Amount")).copy_abs()

        if output_symbol == "EUR" and output_amount > 0:
            return float(output_amount)
        if input_symbol == "EUR" and input_amount > 0:
            return float(input_amount)
        return None

    @classmethod
    def _card_merchant_key(cls, details: object) -> str:
        raw = str(details or "").strip()
        cleaned = re.sub(r"^(approved|rejected)\s*/\s*", "", raw, flags=re.IGNORECASE).strip()
        if not cleaned:
            return ""
        head = cleaned.split("|")[0].strip()
        head = re.sub(r"\s+", " ", head).lower()
        if head in {"refund", "approved", "rejected"}:
            return ""
        return head

    @staticmethod
    def _is_card_loan_withdrawal(*, row: pd.Series) -> bool:
        details = str(row.get("Details") or "").strip().lower()
        return "nexo card loan withdrawal" in details

    @staticmethod
    def _is_rejected_card_row(*, row: pd.Series) -> bool:
        details = str(row.get("Details") or "").strip().lower()
        return details.startswith("rejected")

    @classmethod
    def _is_eur_mode_card_refund(cls, *, row: pd.Series) -> bool:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        return (
            input_symbol == "EUR"
            and output_symbol == "EUR"
            and input_amount > 0
            and output_amount > 0
        )

    @classmethod
    def _build_debit_mode_purchase_indices(cls, *, frame: pd.DataFrame) -> set[int]:
        required_columns = {
            "Type",
            "Date",
            "Input Amount",
            "Input Currency",
            "Output Amount",
            "Output Currency",
            "Details",
        }
        if not required_columns.issubset(frame.columns):
            return set()

        fiatx_counts: dict[tuple[pd.Timestamp, Decimal], int] = {}
        withdraw_counts: dict[tuple[pd.Timestamp, Decimal], int] = {}

        for _, row in frame.iterrows():
            tx_type = str(row.get("Type") or "").strip().lower()
            if tx_type == "credit card fiatx exchange to withdraw":
                eur_amount = cls._extract_fiatx_withdraw_eur_amount(row=row)
                if eur_amount <= 0:
                    continue
                date_val = pd.to_datetime(row.get("Date"), errors="coerce")
                if pd.isna(date_val):
                    continue
                key = (date_val, eur_amount)
                fiatx_counts[key] = fiatx_counts.get(key, 0) + 1
            elif tx_type == "withdraw exchanged":
                eur_amount = cls._extract_withdraw_exchanged_eur_amount(row=row)
                if eur_amount <= 0:
                    continue
                date_val = pd.to_datetime(row.get("Date"), errors="coerce")
                if pd.isna(date_val):
                    continue
                key = (date_val, eur_amount)
                withdraw_counts[key] = withdraw_counts.get(key, 0) + 1

        available_counts = {
            key: min(fiatx_counts.get(key, 0), withdraw_counts.get(key, 0))
            for key in set(fiatx_counts) | set(withdraw_counts)
        }

        debit_mode_indices: set[int] = set()
        purchase_rows = frame[
            frame["Type"].fillna("").str.strip().str.lower() == "nexo card purchase"
        ]
        for idx, row in purchase_rows.iterrows():
            if cls._is_rejected_card_row(row=row):
                continue
            eur_amount = cls._extract_purchase_eur_amount(row=row)
            if eur_amount <= 0:
                continue
            date_val = pd.to_datetime(row.get("Date"), errors="coerce")
            if pd.isna(date_val):
                continue
            key = (date_val, eur_amount)
            if available_counts.get(key, 0) <= 0:
                continue
            debit_mode_indices.add(idx)
            available_counts[key] -= 1

        return debit_mode_indices

    @classmethod
    def _build_eur_mode_cashback_reversal_indices(cls, *, frame: pd.DataFrame) -> set[int]:
        required_columns = {
            "Type",
            "Date",
            "Input Currency",
            "Input Amount",
            "Output Currency",
            "Output Amount",
        }
        if not required_columns.issubset(frame.columns):
            return set()

        deposit_bundle_dates: set[pd.Timestamp] = set()
        for date_val, same_time_rows in frame.groupby("Date", sort=False):
            if pd.isna(date_val):
                continue

            has_deposit_to_exchange = False
            has_exchange_deposited_on = False
            for _, row in same_time_rows.iterrows():
                tx_type = str(row.get("Type") or "").strip().lower()
                if tx_type == "deposit to exchange" and cls._is_eur_to_eurx_deposit_to_exchange(
                    row=row
                ):
                    has_deposit_to_exchange = True
                elif (
                    tx_type == "exchange deposited on"
                    and cls._is_eur_to_eurx_exchange_deposited_on(row=row)
                ):
                    has_exchange_deposited_on = True

            if has_deposit_to_exchange and has_exchange_deposited_on:
                deposit_bundle_dates.add(date_val)

        if not deposit_bundle_dates:
            return set()

        reversal_indices: set[int] = set()
        reversal_rows = frame[
            frame["Type"].fillna("").str.strip().str.lower() == "nexo card cashback reversal"
        ]
        for idx, row in reversal_rows.iterrows():
            if cls._is_rejected_card_row(row=row):
                continue
            date_val = pd.to_datetime(row.get("Date"), errors="coerce")
            if pd.isna(date_val) or date_val not in deposit_bundle_dates:
                continue
            if not cls._row_has_positive_eur_notional(row=row):
                continue
            reversal_indices.add(idx)

        return reversal_indices

    @classmethod
    def _extract_purchase_eur_amount(cls, *, row: pd.Series) -> Decimal:
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        if output_symbol == "EUR" and output_amount > 0:
            return output_amount
        return Decimal(0)

    @classmethod
    def _extract_fiatx_withdraw_eur_amount(cls, *, row: pd.Series) -> Decimal:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        if input_symbol == "EURX" and output_symbol == "EUR" and output_amount > 0:
            return output_amount
        if input_symbol == "EURX" and input_amount > 0:
            return input_amount
        return Decimal(0)

    @classmethod
    def _extract_withdraw_exchanged_eur_amount(cls, *, row: pd.Series) -> Decimal:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount"))
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        if input_symbol == "EUR" and input_amount < 0:
            return input_amount.copy_abs()
        if output_symbol == "EUR" and output_amount > 0:
            return output_amount
        return Decimal(0)

    @classmethod
    def _is_eur_to_eurx_deposit_to_exchange(cls, *, row: pd.Series) -> bool:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount"))
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        return (
            input_symbol == "EUR"
            and output_symbol == "EURX"
            and input_amount > 0
            and output_amount > 0
        )

    @classmethod
    def _is_eur_to_eurx_exchange_deposited_on(cls, *, row: pd.Series) -> bool:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount"))
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        return (
            input_symbol == "EUR"
            and output_symbol == "EURX"
            and input_amount < 0
            and output_amount > 0
        )

    @classmethod
    def _row_has_positive_eur_notional(cls, *, row: pd.Series) -> bool:
        input_symbol = cls._parse_symbol(value=row.get("Input Currency"))
        output_symbol = cls._parse_symbol(value=row.get("Output Currency"))
        input_amount = cls._parse_amount(value=row.get("Input Amount")).copy_abs()
        output_amount = cls._parse_amount(value=row.get("Output Amount")).copy_abs()
        return (output_symbol == "EUR" and output_amount > 0) or (
            input_symbol == "EUR" and input_amount > 0
        )

    def _normalize_withdrawal_outs(self, row: pd.Series) -> list[TxEntry]:
        """
        Normalizes NEXO withdrawal rows as pure outflows.

        NEXO often mirrors withdrawal rows with a positive output leg for the same asset.
        For withdrawal semantics we should only keep negative legs as sends.
        """
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        output_symbol = self._parse_symbol(value=row.get("Output Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount"))
        output_amount = self._parse_amount(value=row.get("Output Amount"))

        out_by_symbol: dict[str, Decimal] = {}
        if input_symbol and input_amount < 0:
            out_by_symbol[input_symbol] = out_by_symbol.get(input_symbol, Decimal(0)) + (
                input_amount.copy_abs()
            )
        if output_symbol and output_amount < 0:
            out_by_symbol[output_symbol] = out_by_symbol.get(output_symbol, Decimal(0)) + (
                output_amount.copy_abs()
            )

        return [
            TxEntry(token=symbol, quantity=amount)
            for symbol, amount in sorted(out_by_symbol.items())
            if amount > 0
        ]

    def _extract_positive_input_entry(self, row: pd.Series) -> TxEntry | None:
        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        input_amount = self._parse_amount(value=row.get("Input Amount"))
        if not input_symbol or input_amount <= 0:
            return None
        return TxEntry(token=input_symbol, quantity=input_amount)

    def _infer_interest_source_coin(self, row: pd.Series, fallback: str) -> str:
        details = str(row.get("Details") or "")
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", details)
        for token in reversed(tokens):
            normalized = sanitize_symbol(token)
            if not normalized:
                continue
            mapped = self.symbol_lookup.get(normalized.upper())
            if mapped:
                return self._canonicalize_stable_token(symbol=mapped)
            if self._is_upper_symbol_token(token=token):
                return self._canonicalize_stable_token(symbol=normalized)

        input_symbol = self._parse_symbol(value=row.get("Input Currency"))
        if input_symbol:
            return input_symbol
        return fallback

    @staticmethod
    def _is_upper_symbol_token(*, token: str) -> bool:
        cleaned = token.strip()
        if not (2 <= len(cleaned) <= 12):
            return False
        if cleaned != cleaned.upper():
            return False
        return any(ch.isalpha() for ch in cleaned)

    @staticmethod
    def _parse_symbol(value: object) -> str:
        normalized = sanitize_symbol(value)
        if not normalized or normalized == "-":
            return ""
        return NexoTransactionNormalizer._canonicalize_stable_token(symbol=normalized)

    @classmethod
    def _canonicalize_stable_token(cls, *, symbol: str) -> str:
        if symbol.upper() in {"USD", "USDX", "XUSD"}:
            return cls.CANONICAL_DEBT_TOKEN
        return symbol

    @staticmethod
    def _parse_amount(value: object) -> Decimal:
        text = str(value or "").strip()
        if not text:
            return Decimal(0)
        try:
            return Decimal(text)
        except Exception:
            return Decimal(0)

    @staticmethod
    def _parse_usd_equivalent(value: object) -> Decimal:
        text = str(value or "").strip().replace("$", "").replace(",", "")
        if not text:
            return Decimal(0)
        try:
            return Decimal(text)
        except Exception:
            return Decimal(0)


def _extract_manual_sell_leg(
    *, normalizer: NexoTransactionNormalizer, idx: int, row: pd.Series
) -> ManualRepaymentLeg | None:
    input_symbol = normalizer._parse_symbol(value=row.get("Input Currency"))
    input_amount = normalizer._parse_amount(value=row.get("Input Amount"))
    output_symbol = normalizer._parse_symbol(value=row.get("Output Currency"))
    output_amount = normalizer._parse_amount(value=row.get("Output Amount"))
    usd_equivalent = normalizer._parse_usd_equivalent(value=row.get("USD Equivalent"))
    if usd_equivalent <= 0:
        return None

    if input_symbol and input_amount < 0:
        entry = TxEntry(token=input_symbol, quantity=input_amount.copy_abs())
    elif output_symbol and output_amount < 0:
        entry = TxEntry(token=output_symbol, quantity=output_amount.copy_abs())
    else:
        return None

    date = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.isna(date):
        return None
    return ManualRepaymentLeg(
        idx=idx,
        date=date,
        usd_equivalent=usd_equivalent,
        matching_usd_equivalent=usd_equivalent,
        entry=entry,
    )


def _extract_manual_repayment_leg(
    *, normalizer: NexoTransactionNormalizer, idx: int, row: pd.Series
) -> ManualRepaymentLeg | None:
    input_symbol = normalizer._parse_symbol(value=row.get("Input Currency"))
    input_amount = normalizer._parse_amount(value=row.get("Input Amount"))
    output_symbol = normalizer._parse_symbol(value=row.get("Output Currency"))
    output_amount = normalizer._parse_amount(value=row.get("Output Amount"))
    usd_equivalent = normalizer._parse_usd_equivalent(value=row.get("USD Equivalent"))
    if usd_equivalent <= 0:
        return None
    matching_usd_equivalent = usd_equivalent

    fee_currency = normalizer._parse_symbol(value=row.get("Fee Currency"))
    fee_amount = normalizer._parse_amount(value=row.get("Fee"))
    if fee_amount > 0 and fee_currency == NexoTransactionNormalizer.CANONICAL_DEBT_TOKEN:
        # Manual repayments can include a USD fee that belongs in the bundle match key.
        matching_usd_equivalent += fee_amount

    if input_symbol and input_amount > 0:
        entry = TxEntry(token=input_symbol, quantity=input_amount)
    elif output_symbol and output_amount > 0:
        entry = TxEntry(token=output_symbol, quantity=output_amount)
    else:
        return None

    date = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.isna(date):
        return None
    return ManualRepaymentLeg(
        idx=idx,
        date=date,
        usd_equivalent=usd_equivalent,
        matching_usd_equivalent=matching_usd_equivalent,
        entry=entry,
    )


def _build_manual_repayment_pairs(
    *, frame: pd.DataFrame, normalizer: NexoTransactionNormalizer
) -> list[ManualRepaymentPair]:
    manual_sells: list[ManualRepaymentLeg] = []
    manual_repayments: list[ManualRepaymentLeg] = []

    for idx, row in frame.iterrows():
        tx_type = str(row.get("Type") or "").strip().lower()
        if tx_type == "manual sell order":
            leg = _extract_manual_sell_leg(normalizer=normalizer, idx=idx, row=row)
            if leg is not None:
                manual_sells.append(leg)
        elif tx_type == "manual repayment":
            leg = _extract_manual_repayment_leg(normalizer=normalizer, idx=idx, row=row)
            if leg is not None:
                manual_repayments.append(leg)

    manual_sells = sorted(manual_sells, key=lambda leg: (leg.date, leg.idx))
    manual_repayments = sorted(manual_repayments, key=lambda leg: (leg.date, leg.idx))

    used_repayment_indices: set[int] = set()
    pairs: list[ManualRepaymentPair] = []
    for sell_leg in manual_sells:
        usd_tolerance = MANUAL_REPAYMENT_USD_TOLERANCE_BY_TOKEN.get(
            sell_leg.entry.token,
            MANUAL_REPAYMENT_USD_TOLERANCE,
        )
        candidates: list[tuple[pd.Timedelta, Decimal, ManualRepaymentLeg]] = []
        for repayment_leg in manual_repayments:
            if repayment_leg.idx in used_repayment_indices:
                continue
            time_delta = abs(repayment_leg.date - sell_leg.date)
            if time_delta > MANUAL_REPAYMENT_PAIR_WINDOW:
                continue
            usd_delta = (repayment_leg.matching_usd_equivalent - sell_leg.usd_equivalent).copy_abs()
            if usd_delta > usd_tolerance:
                continue
            candidates.append((time_delta, usd_delta, repayment_leg))

        if not candidates:
            continue

        _, _, matched_repayment = min(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2].date,
                candidate[2].idx,
            ),
        )
        used_repayment_indices.add(matched_repayment.idx)
        pairs.append(
            ManualRepaymentPair(
                manual_sell=sell_leg,
                manual_repayment=matched_repayment,
            )
        )
    return pairs


def _build_manual_repayment_actions(
    *,
    frame: pd.DataFrame,
    normalizer: NexoTransactionNormalizer,
) -> tuple[dict[int, NormalizedAction], set[int], list[dict[str, object]]]:
    pairs = _build_manual_repayment_pairs(frame=frame, normalizer=normalizer)

    pair_actions_by_row_idx: dict[int, NormalizedAction] = {}
    liquidation_consumed_indices: set[int] = set()

    liquidation_rows: list[tuple[int, pd.Timestamp, Decimal, str, str]] = []
    for idx, row in frame.iterrows():
        tx_type = str(row.get("Type") or "").strip().lower()
        if tx_type != "exchange liquidation":
            continue
        date = pd.to_datetime(row.get("Date"), errors="coerce")
        if pd.isna(date):
            continue
        usd_equivalent = normalizer._parse_usd_equivalent(value=row.get("USD Equivalent"))
        ins, outs = normalizer._normalize_exchange_liquidation_swap(row=row)
        in_token = ins[0].token if ins else ""
        out_token = outs[0].token if outs else ""
        liquidation_rows.append((idx, date, usd_equivalent, in_token, out_token))

    for pair in pairs:
        pair_actions_by_row_idx[pair.manual_repayment.idx] = NormalizedAction(
            action="swap",
            ins=[pair.manual_repayment.entry],
            outs=[pair.manual_sell.entry],
            rewards=[],
        )

        candidates: list[tuple[pd.Timedelta, Decimal, int]] = []
        for liq_idx, liq_date, liq_usd, liq_in_token, liq_out_token in liquidation_rows:
            if liq_idx in liquidation_consumed_indices:
                continue
            if liq_in_token != pair.manual_repayment.entry.token:
                continue
            if liq_out_token != pair.manual_sell.entry.token:
                continue

            time_delta = abs(liq_date - pair.manual_repayment.date)
            if time_delta > MANUAL_REPAYMENT_PAIR_WINDOW:
                continue
            usd_delta = (liq_usd - pair.manual_repayment.usd_equivalent).copy_abs()
            if usd_delta > MANUAL_REPAYMENT_USD_TOLERANCE:
                continue
            candidates.append((time_delta, usd_delta, liq_idx))

        if not candidates:
            continue

        _, _, matched_liq_idx = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
        )
        liquidation_consumed_indices.add(matched_liq_idx)

    review_rows: list[dict[str, object]] = []
    for idx, row in frame.iterrows():
        tx_type = str(row.get("Type") or "").strip().lower()
        if tx_type != "exchange liquidation":
            continue
        if idx in liquidation_consumed_indices:
            continue
        review_rows.append({column: row.get(column, "") for column in LIQUIDATION_REVIEW_COLUMNS})

    review_rows = sorted(
        review_rows,
        key=lambda item: (
            str(item.get("Date / Time (UTC)", "")),
            str(item.get("Transaction", "")),
        ),
    )

    return pair_actions_by_row_idx, liquidation_consumed_indices, review_rows


def _save_liquidation_review(*, review_rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not review_rows:
        pd.DataFrame(columns=LIQUIDATION_REVIEW_COLUMNS).to_csv(output_path, index=False)
        return
    frame = pd.DataFrame(review_rows)
    frame = frame[LIQUIDATION_REVIEW_COLUMNS]
    frame.to_csv(output_path, index=False)


def _save_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        pd.DataFrame(columns=SNAPSHOT_COLUMNS).to_csv(output_path, index=False)
        return

    frame = pd.DataFrame(history)
    if frame.empty:
        pd.DataFrame(columns=SNAPSHOT_COLUMNS).to_csv(output_path, index=False)
        return

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame["Date"] = frame["Date"].map(format_daily_datetime)
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["Principal Invested"] = pd.to_numeric(frame["Principal Invested"], errors="coerce")
    frame = frame.dropna(subset=["Quantity", "Principal Invested"])
    frame = frame.sort_values(by=["Date", "Coin"], ascending=[True, True])
    frame = frame[SNAPSHOT_COLUMNS]
    frame.to_csv(output_path, index=False)


def _load_nexo_transaction_exports(input_csv: Path) -> pd.DataFrame:
    transaction_folder = input_csv if input_csv.is_dir() else input_csv.parent
    csv_paths = sorted(path for path in transaction_folder.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No NEXO transaction CSV files found in {transaction_folder}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, dtype=str)
        frame["__source_file"] = csv_path.name
        frame["__source_row"] = range(len(frame))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def _apply_generic_action(
    *,
    tracker: CryptoTracker,
    action: NormalizedAction,
    date: pd.Timestamp,
    touched_coins: set[str],
) -> None:
    overrides = action.principal_overrides or {}
    additions = action.principal_additions or {}

    if action.action == "swap":
        tracker._process_swap(
            ins=action.ins,
            outs=action.outs,
            date=date,
            touched_coins=touched_coins,
        )
        return

    if action.action == "buy":
        if len(action.ins) != 1 or len(action.outs) != 1:
            return
        entry_in = action.ins[0]
        entry_out = action.outs[0]
        asset_in = tracker.fetch_asset(entry_in.token)
        asset_in.buy(
            amount_bought=entry_in.quantity,
            fiat_spent=entry_out.quantity,
            currency=entry_out.token,
            date=date,
        )
        touched_coins.add(asset_in.coin)
        return

    if action.action == "sell":
        if len(action.ins) != 1 or len(action.outs) != 1:
            return
        entry_in = action.ins[0]
        entry_out = action.outs[0]
        asset_out = tracker.fetch_asset(entry_out.token)
        asset_out.sell(
            amount_sold=entry_out.quantity,
            fiat_received=entry_in.quantity,
            currency=entry_in.token,
            date=date,
        )
        touched_coins.add(asset_out.coin)
        return

    if action.action == "receive":
        for entry in action.ins:
            asset = tracker.fetch_asset(entry.token)
            if entry.token in overrides:
                asset.quantity += entry.quantity
                asset.adjust_principal(overrides[entry.token])
            else:
                asset.receive(amount_received=entry.quantity, date=date)
            touched_coins.add(asset.coin)
        return

    if action.action == "send":
        for entry in action.outs:
            asset = tracker.fetch_asset(entry.token)
            if entry.token in overrides:
                asset.quantity -= entry.quantity
                asset.adjust_principal(overrides[entry.token])
            else:
                asset.send(amount_sent=entry.quantity, date=date)
            touched_coins.add(asset.coin)

    for token, principal_delta in additions.items():
        if principal_delta == 0:
            continue
        asset = tracker.fetch_asset(token)
        asset.adjust_principal(principal_delta)
        touched_coins.add(asset.coin)


def generate_nexo_raw_snapshots(input_csv: Path, output_csv: Path) -> None:
    frame = _load_nexo_transaction_exports(input_csv=input_csv)
    parsed_dates = parse_transaction_datetime_series(frame["Date / Time (UTC)"])
    invalid_date_count = int(parsed_dates.isna().sum())
    total_rows = len(frame)
    if total_rows > 0 and (invalid_date_count / total_rows) > MAX_INVALID_DATE_RATIO:
        raise ValueError(
            f"Aborting snapshot generation: invalid dates={invalid_date_count}/{total_rows} "
            f"({invalid_date_count / total_rows:.1%})."
        )
    if invalid_date_count:
        print(f"[nexo_snapshots] Dropping {invalid_date_count} rows with invalid Date values.")

    frame["Date"] = parsed_dates
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values(
        by=["Date", "__source_file", "__source_row"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    normalizer = NexoTransactionNormalizer.from_dataframe(frame=frame)
    pair_actions_by_row_idx, _, liquidation_review_rows = _build_manual_repayment_actions(
        frame=frame,
        normalizer=normalizer,
    )
    liquidation_review_path = output_csv.with_name(LIQUIDATION_REVIEW_FILENAME)
    _save_liquidation_review(
        review_rows=liquidation_review_rows,
        output_path=liquidation_review_path,
    )

    tracker = CryptoTracker(chain="nexo", token_metadata={})

    for idx, row in frame.iterrows():
        date = row["Date"]
        tx_type = str(row.get("Type") or "").strip().lower()
        if idx in pair_actions_by_row_idx:
            action = pair_actions_by_row_idx[idx]
        elif tx_type == "exchange liquidation":
            action = NormalizedAction(action="skip", ins=[], outs=[], rewards=[])
        else:
            action = normalizer.normalize_row(row=row)
        if action.action == "skip":
            continue

        touched_coins: set[str] = set()
        if action.action == "reward":
            for reward in action.rewards:
                tracker.apply_reward_with_allocations(
                    reward_token=reward.entry.token,
                    reward_quantity=reward.entry.quantity,
                    date=date,
                    allocations=reward.allocations,
                    touched_coins=touched_coins,
                )
        else:
            _apply_generic_action(
                tracker=tracker,
                action=action,
                date=date,
                touched_coins=touched_coins,
            )

        if touched_coins:
            tracker._update_snapshots(touched_coins=touched_coins, date=date)

    _save_history(history=tracker.history, output_path=output_csv)
    print(f"Portfolio snapshots successfully saved to {output_csv}")
    print(
        "[nexo_snapshots] Wrote exchange-liquidation review list "
        f"({len(liquidation_review_rows)} rows) to {liquidation_review_path}"
    )


if __name__ == "__main__":
    generate_nexo_raw_snapshots(
        input_csv=BLOCKCHAIN_TRANSACTIONS_FOLDER / "cex" / "nexo",
        output_csv=BLOCKCHAIN_SNAPSHOT_FOLDER / "cex" / "nexo" / "nexo_raw_snapshots.csv",
    )
