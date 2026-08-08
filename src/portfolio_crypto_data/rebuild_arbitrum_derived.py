from __future__ import annotations

import argparse
import sys
from typing import Any

from portfolio_core import active_context

from portfolio_crypto_data.accounting import accounting_paths, build_accounting_artifacts
from portfolio_crypto_data.dashboard_artifacts import build_arbitrum_dashboard_artifacts
from portfolio_crypto_data.raw_snapshots import generate_raw_snapshots

CHAIN = "arbitrum"


def rebuild_arbitrum_derived(*, as_of_date: str = "2026-05-09") -> dict[str, Any]:
    paths = active_context().paths
    transaction_path = paths.crypto_transactions / f"{CHAIN}_transactions.csv"
    snapshot_path = paths.crypto_snapshots / f"{CHAIN}_raw_snapshots.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(f"missing transaction file: {transaction_path}")

    principal_paths = accounting_paths(chain=CHAIN)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    principal_paths.principal_events.parent.mkdir(parents=True, exist_ok=True)

    generate_raw_snapshots(
        input_csv=transaction_path,
        output_csv=snapshot_path,
        chain=CHAIN,
        principal_events_csv=principal_paths.principal_events,
        principal_daily_csv=principal_paths.principal_daily,
    )
    accounting_result = build_accounting_artifacts(chain=CHAIN, as_of_date=as_of_date)
    dashboard_paths = build_arbitrum_dashboard_artifacts(chain=CHAIN)
    return {
        "chain": CHAIN,
        "as_of_date": as_of_date,
        "raw_snapshots": str(snapshot_path),
        "principal_events": str(principal_paths.principal_events),
        "principal_daily": str(principal_paths.principal_daily),
        "accounting_rows": accounting_result.rows_written,
        "accounting_errors": accounting_result.errors,
        "dashboard": str(dashboard_paths.asset_daily.parent),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild derived Arbitrum dashboard data.")
    parser.add_argument("--as-of-date", default="2026-05-09")
    args = parser.parse_args(argv)
    try:
        result = rebuild_arbitrum_derived(as_of_date=args.as_of_date)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Rebuilt {result['chain']} derived artifacts through {result['as_of_date']}")
    print(f"raw_snapshots: {result['raw_snapshots']}")
    print(f"principal_daily: {result['principal_daily']}")
    print(f"dashboard: {result['dashboard']}")
    if result["accounting_errors"]:
        print("accounting errors:")
        for error in result["accounting_errors"]:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
