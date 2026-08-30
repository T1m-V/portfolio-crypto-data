from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from portfolio_crypto_data.raw_snapshots import PortfolioLedger, TransactionApplier, TxEntry


@dataclass
class RewardInstruction:
    entry: TxEntry
    allocations: list[tuple[str | None, float]]


@dataclass
class NormalizedAction:
    action: str
    ins: list[TxEntry] = field(default_factory=list)
    outs: list[TxEntry] = field(default_factory=list)
    rewards: list[RewardInstruction] = field(default_factory=list)
    principal_overrides: dict[str, float] | None = None
    principal_additions: dict[str, float] | None = None


def apply_normalized_action(
    *,
    ledger: PortfolioLedger,
    applier: TransactionApplier,
    action: NormalizedAction,
    date: pd.Timestamp,
) -> set[str]:
    touched_coins: set[str] = set()
    if action.action == "skip":
        return touched_coins

    if action.action == "reward":
        for reward in action.rewards:
            applier.apply_reward_with_allocations(
                reward_token=reward.entry.token,
                reward_quantity=reward.entry.quantity,
                date_value=date,
                allocations=reward.allocations,
                touched_coins=touched_coins,
            )
        return touched_coins

    overrides = action.principal_overrides or {}
    additions = action.principal_additions or {}

    if action.action == "swap":
        if overrides:
            for entry in action.ins:
                asset = ledger.fetch_asset(entry.token)
                asset.quantity += entry.quantity
                ledger.adjust_principal(
                    asset=asset,
                    amount_eur=overrides.get(entry.token, 0.0),
                    date_value=date,
                    action="cex_swap_in_override",
                )
                touched_coins.add(asset.coin)
            for entry in action.outs:
                asset = ledger.fetch_asset(entry.token)
                asset.quantity -= entry.quantity
                ledger.adjust_principal(
                    asset=asset,
                    amount_eur=overrides.get(entry.token, 0.0),
                    date_value=date,
                    action="cex_swap_out_override",
                )
                touched_coins.add(asset.coin)
        else:
            applier.apply_swap(
                ins=action.ins,
                outs=action.outs,
                date_value=date,
                touched_coins=touched_coins,
            )
        return touched_coins

    if action.action == "receive":
        for entry in action.ins:
            asset = ledger.fetch_asset(entry.token)
            if entry.token in overrides:
                asset.quantity += entry.quantity
                ledger.adjust_principal(
                    asset=asset,
                    amount_eur=overrides[entry.token],
                    date_value=date,
                    action="cex_receive_override",
                )
            else:
                applier.receive(
                    asset=asset,
                    amount_received=entry.quantity,
                    date_value=date,
                )
            touched_coins.add(asset.coin)
        return touched_coins

    if action.action == "send":
        for entry in action.outs:
            asset = ledger.fetch_asset(entry.token)
            if entry.token in overrides:
                asset.quantity -= entry.quantity
                ledger.adjust_principal(
                    asset=asset,
                    amount_eur=overrides[entry.token],
                    date_value=date,
                    action="cex_send_override",
                )
            else:
                applier.send(
                    asset=asset,
                    amount_sent=entry.quantity,
                    date_value=date,
                )
            touched_coins.add(asset.coin)
    else:
        raise ValueError(f"Unsupported normalized CEX action: {action.action}")

    for token, principal_delta in additions.items():
        if principal_delta == 0:
            continue
        asset = ledger.fetch_asset(token)
        ledger.adjust_principal(
            asset=asset,
            amount_eur=principal_delta,
            date_value=date,
            action="cex_principal_addition",
        )
        touched_coins.add(asset.coin)
    return touched_coins
