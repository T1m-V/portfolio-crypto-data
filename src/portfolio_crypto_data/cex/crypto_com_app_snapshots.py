from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from portfolio_crypto_data.cex.actions import (
    NormalizedAction,
    RewardInstruction,
    apply_normalized_action,
)
from portfolio_crypto_data.cex.export_loader import (
    CexExportSpec,
    load_cex_exports,
    prepare_cex_dates,
)
from portfolio_crypto_data.cex.snapshot_io import save_cex_snapshot_history
from portfolio_crypto_data.raw_snapshots import PortfolioLedger, TransactionApplier, TxEntry
from portfolio_crypto_data.symbols import sanitize_symbol

APP_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

REQUIRED_APP_COLUMNS = {
    "Timestamp (UTC)",
    "Transaction Description",
    "Currency",
    "Amount",
    "To Currency",
    "To Amount",
    "Native Currency",
    "Native Amount",
    "Native Amount (in USD)",
    "Transaction Kind",
}
OPTIONAL_APP_COLUMNS = {"Transaction Hash"}
FINGERPRINT_COLUMNS = (
    "Timestamp (UTC)",
    "Transaction Description",
    "Currency",
    "Amount",
    "To Currency",
    "To Amount",
    "Native Currency",
    "Native Amount",
    "Native Amount (in USD)",
    "Transaction Kind",
)
CRYPTO_COM_APP_EXPORT_SPEC = CexExportSpec(
    provider_name="Crypto.com App",
    required_columns=frozenset(REQUIRED_APP_COLUMNS),
    optional_columns=tuple(sorted(OPTIONAL_APP_COLUMNS)),
    fingerprint_columns=FINGERPRINT_COLUMNS,
    identity_column="Transaction Hash",
)

REWARD_KINDS = {
    "crypto_earn_interest_paid",
    "finance.lockup.dpos_non_compound_interest.crypto_wallet",
    "mco_stake_reward",
    "referral_gift",
    "reimbursement",
    "supercharger_reward_to_app_credited",
    "finance.lockup.dpos_compound_interest.crypto_wallet",
    "referral_bonus",
    "referral_card_cashback",
    "rewards_platform_deposit_credited",
    "transfer_cashback",
}
RECEIVE_KINDS = {
    "admin_wallet_credited",
    "crypto_deposit",
    "exchange_to_crypto_transfer",
}
SEND_KINDS = {
    "card_top_up",
    "crypto_viban_exchange",
    "transfer.p2p_transfer.crypto_wallet.crypto_wallet.debit",
    "crypto_to_exchange_transfer",
    "crypto_withdrawal",
}
INTERNAL_SEND_KINDS = {
    "crypto_earn_program_created",
    "lockup_lock",
    "lockup_upgrade",
    "supercharger_deposit",
    "finance.lockup.dpos_lock.crypto_wallet",
    "finance.lockup.dpos_lock_upgrade.crypto_wallet",
}
INTERNAL_RECEIVE_KINDS = {
    "crypto_earn_program_withdrawn",
    "lockup_unlock",
    "supercharger_withdrawal",
}
REVERSAL_KINDS = {"card_cashback_reverted", "reimbursement_reverted"}
DUST_MATCH_WINDOW = pd.Timedelta(seconds=1)
DUST_KINDS = {
    "dust_conversion_credited",
    "dust_conversion_debited",
}
SUPPORTED_APP_KINDS = {
    *REWARD_KINDS,
    *RECEIVE_KINDS,
    *SEND_KINDS,
    *INTERNAL_SEND_KINDS,
    *INTERNAL_RECEIVE_KINDS,
    *REVERSAL_KINDS,
    *DUST_KINDS,
    "crypto_transfer",
    "viban_purchase",
    "crypto_exchange",
}


class CryptoComAppTransactionNormalizer:
    def normalize_row(self, row: pd.Series) -> NormalizedAction:
        kind = self._kind(row)
        if kind in REWARD_KINDS:
            reward = self._positive_primary(row)
            return NormalizedAction(
                action="reward",
                rewards=[RewardInstruction(entry=reward, allocations=[(None, 1.0)])],
            )
        if kind in RECEIVE_KINDS:
            return NormalizedAction(action="receive", ins=[self._positive_primary(row)])
        if kind in SEND_KINDS:
            return NormalizedAction(action="send", outs=[self._negative_primary(row)])
        if kind in INTERNAL_SEND_KINDS:
            self._negative_primary(row)
            return NormalizedAction(action="skip")
        if kind in INTERNAL_RECEIVE_KINDS:
            self._positive_primary(row)
            return NormalizedAction(action="skip")
        if kind == "crypto_transfer":
            entry = self._entry(symbol=row.get("Currency"), amount=row.get("Amount"))
            if entry.quantity > 0:
                return NormalizedAction(action="receive", ins=[entry])
            entry.quantity = entry.quantity.copy_abs()
            return NormalizedAction(action="send", outs=[entry])
        if kind == "viban_purchase":
            self._negative_primary(row)
            return NormalizedAction(action="receive", ins=[self._positive_to(row)])
        if kind in REVERSAL_KINDS:
            entry = self._negative_primary(row)
            return NormalizedAction(
                action="send",
                outs=[entry],
                principal_overrides={entry.token: 0.0},
            )
        if kind == "crypto_exchange":
            return NormalizedAction(
                action="swap",
                ins=[self._positive_to(row)],
                outs=[self._negative_primary(row)],
            )
        if kind in DUST_KINDS:
            raise ValueError("Dust conversion legs must be normalized as a matched pair.")
        raise ValueError(f"Unsupported Crypto.com App transaction kind: {kind or '<empty>'}")

    def build_dust_action(
        self,
        *,
        credited: pd.Series,
        debited: list[pd.Series],
    ) -> NormalizedAction:
        if self._kind(credited) != "dust_conversion_credited":
            raise ValueError("Invalid credited dust-conversion leg.")
        if not debited or any(self._kind(row) != "dust_conversion_debited" for row in debited):
            raise ValueError("Invalid debited dust-conversion leg.")
        return NormalizedAction(
            action="swap",
            ins=[self._positive_primary(credited)],
            outs=[self._negative_primary(row) for row in debited],
        )

    @staticmethod
    def _kind(row: pd.Series) -> str:
        value = row.get("Transaction Kind")
        return "" if pd.isna(value) else str(value).strip().lower()

    @classmethod
    def _positive_primary(cls, row: pd.Series) -> TxEntry:
        return cls._entry(
            symbol=row.get("Currency"),
            amount=row.get("Amount"),
            expected_sign=1,
        )

    @classmethod
    def _negative_primary(cls, row: pd.Series) -> TxEntry:
        return cls._entry(
            symbol=row.get("Currency"),
            amount=row.get("Amount"),
            expected_sign=-1,
        )

    @classmethod
    def _positive_to(cls, row: pd.Series) -> TxEntry:
        return cls._entry(
            symbol=row.get("To Currency"),
            amount=row.get("To Amount"),
            expected_sign=1,
        )

    @staticmethod
    def _entry(*, symbol: object, amount: object, expected_sign: int | None = None) -> TxEntry:
        token = sanitize_symbol(symbol)
        if not token:
            raise ValueError("Crypto.com App transaction is missing an asset symbol.")

        text = "" if pd.isna(amount) else str(amount).strip()
        try:
            quantity = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Invalid Crypto.com App amount.") from exc

        if not quantity.is_finite():
            raise ValueError("Invalid Crypto.com App amount.")
        if quantity == 0 or (expected_sign is not None and (quantity > 0) != (expected_sign > 0)):
            direction = (
                "nonzero"
                if expected_sign is None
                else ("positive" if expected_sign > 0 else "negative")
            )
            raise ValueError(f"Expected a {direction} Crypto.com App amount for {token}.")
        return TxEntry(
            token=token, quantity=quantity if expected_sign is None else quantity.copy_abs()
        )


def _load_crypto_com_app_exports(input_csv: Path) -> pd.DataFrame:
    return load_cex_exports(input_csv=input_csv, spec=CRYPTO_COM_APP_EXPORT_SPEC)


def _build_dust_actions(
    *,
    frame: pd.DataFrame,
    normalizer: CryptoComAppTransactionNormalizer,
) -> tuple[dict[int, NormalizedAction], set[int]]:
    dust = frame[frame["Transaction Kind"].str.lower().isin(DUST_KINDS)]
    actions: dict[int, NormalizedAction] = {}
    consumed: set[int] = set()
    if dust.empty:
        return actions, consumed

    # App exports may record a multi-asset conversion's debits one second
    # apart from its single CRO credit. Require a unique matching credit.
    kinds = dust["Transaction Kind"].str.lower()
    credits = dust[kinds == "dust_conversion_credited"]
    debits = dust[kinds == "dust_conversion_debited"]
    matched: dict[int, list[int]] = {int(idx): [] for idx in credits.index}
    for debit_idx, debit in debits.iterrows():
        candidates = credits[
            credits["Transaction Description"].eq(debit["Transaction Description"])
            & credits["Date"].sub(debit["Date"]).abs().le(DUST_MATCH_WINDOW)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Unmatched Crypto.com App dust conversion: expected one matching credit."
            )
        matched[int(candidates.index[0])].append(int(debit_idx))

    for credit_idx, debit_indices in matched.items():
        if not debit_indices:
            raise ValueError("Unmatched Crypto.com App dust conversion: credit has no debits.")
        indices = {credit_idx, *debit_indices}
        actions[min(indices)] = normalizer.build_dust_action(
            credited=frame.loc[credit_idx],
            debited=[frame.loc[idx] for idx in debit_indices],
        )
        consumed.update(indices)
    return actions, consumed


def generate_crypto_com_app_raw_snapshots(input_csv: Path, output_csv: Path) -> None:
    frame = _load_crypto_com_app_exports(input_csv=input_csv)
    parsed_dates = pd.to_datetime(
        frame["Timestamp (UTC)"],
        format=APP_TIMESTAMP_FORMAT,
        errors="coerce",
    )
    frame = prepare_cex_dates(
        frame=frame,
        parsed_dates=parsed_dates,
        provider_name="crypto_com_app_snapshots",
        source_column="Timestamp (UTC)",
    )

    kinds = set(frame["Transaction Kind"].str.strip().str.lower())
    unsupported = sorted(kinds.difference(SUPPORTED_APP_KINDS))
    if unsupported:
        raise ValueError(
            "Unsupported Crypto.com App transaction kind(s): " + ", ".join(unsupported)
        )

    normalizer = CryptoComAppTransactionNormalizer()
    dust_actions, consumed_dust_rows = _build_dust_actions(
        frame=frame,
        normalizer=normalizer,
    )
    ledger = PortfolioLedger(chain="crypto_com_app", token_metadata={})
    applier = TransactionApplier(ledger=ledger)

    for idx, row in frame.iterrows():
        if idx in consumed_dust_rows and idx not in dust_actions:
            continue
        action = dust_actions.get(idx) or normalizer.normalize_row(row=row)
        touched_coins = apply_normalized_action(
            ledger=ledger,
            applier=applier,
            action=action,
            date=row["Date"],
        )
        if touched_coins:
            ledger.update_snapshots(touched_coins=touched_coins, date_value=row["Date"])

    save_cex_snapshot_history(history=ledger.history, output_csv=output_csv)
    print(f"Crypto.com App portfolio snapshots successfully saved to {output_csv}")
