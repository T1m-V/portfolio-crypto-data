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
    "finance.lockup.dpos_compound_interest.crypto_wallet",
    "referral_bonus",
    "referral_card_cashback",
    "rewards_platform_deposit_credited",
    "transfer_cashback",
}
RECEIVE_KINDS = {
    "crypto_deposit",
    "exchange_to_crypto_transfer",
}
SEND_KINDS = {
    "crypto_to_exchange_transfer",
    "crypto_transfer",
    "crypto_withdrawal",
}
SKIPPED_KINDS = {
    "finance.lockup.dpos_lock.crypto_wallet",
    "finance.lockup.dpos_lock_upgrade.crypto_wallet",
}
DUST_KINDS = {
    "dust_conversion_credited",
    "dust_conversion_debited",
}
SUPPORTED_APP_KINDS = {
    *REWARD_KINDS,
    *RECEIVE_KINDS,
    *SEND_KINDS,
    *SKIPPED_KINDS,
    *DUST_KINDS,
    "card_cashback_reverted",
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
        if kind in SKIPPED_KINDS:
            self._negative_primary(row)
            return NormalizedAction(action="skip")
        if kind == "card_cashback_reverted":
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
        debited: pd.Series,
    ) -> NormalizedAction:
        if self._kind(credited) != "dust_conversion_credited":
            raise ValueError("Invalid credited dust-conversion leg.")
        if self._kind(debited) != "dust_conversion_debited":
            raise ValueError("Invalid debited dust-conversion leg.")
        return NormalizedAction(
            action="swap",
            ins=[self._positive_primary(credited)],
            outs=[self._negative_primary(debited)],
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
    def _entry(*, symbol: object, amount: object, expected_sign: int) -> TxEntry:
        token = sanitize_symbol(symbol)
        if not token:
            raise ValueError("Crypto.com App transaction is missing an asset symbol.")

        text = "" if pd.isna(amount) else str(amount).strip()
        try:
            quantity = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Invalid Crypto.com App amount.") from exc

        if quantity == 0 or (quantity > 0) != (expected_sign > 0):
            direction = "positive" if expected_sign > 0 else "negative"
            raise ValueError(
                f"Expected a {direction} Crypto.com App amount for {token}."
            )
        return TxEntry(token=token, quantity=quantity.copy_abs())


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

    group_columns = ["Date", "Transaction Description"]
    for _, group in dust.groupby(group_columns, sort=False, dropna=False):
        kinds = group["Transaction Kind"].str.lower()
        credited = group[kinds == "dust_conversion_credited"]
        debited = group[kinds == "dust_conversion_debited"]
        if len(credited) != 1 or len(debited) != 1 or len(group) != 2:
            raise ValueError(
                "Unmatched Crypto.com App dust conversion: "
                f"credited legs={len(credited)}, debited legs={len(debited)}."
            )

        credited_idx = int(credited.index[0])
        debited_idx = int(debited.index[0])
        action_idx = min(credited_idx, debited_idx)
        actions[action_idx] = normalizer.build_dust_action(
            credited=credited.iloc[0],
            debited=debited.iloc[0],
        )
        consumed.update({credited_idx, debited_idx})
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
