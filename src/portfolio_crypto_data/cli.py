from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

from portfolio_core import PortfolioContext, mutation_session, validate_data_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-crypto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--chain", type=str, default=None, help="Specific chain to update or rebuild"
    )
    parser.add_argument("command", choices=["update", "rebuild"])
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse existing snapshots and accounting history when rebuilding",
    )
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        default=None,
        help="Start date (YYYY-MM-DD) for incremental accounting rebuild",
    )
    return parser


def _refresh_cex_exports(
    *,
    label: str,
    input_path: Path,
    output_path: Path,
    generator: Callable[[Path, Path], None],
) -> None:
    csv_files = list(input_path.glob("*.csv")) if input_path.exists() else []
    if not csv_files:
        print(f"Skipping {label} refresh; no CSV exports found in {input_path}")
        return
    input_files = [*csv_files, *input_path.glob("*.json")]
    if output_path.exists() and output_path.stat().st_mtime >= max(
        path.stat().st_mtime for path in input_files
    ):
        print(f"Skipping {label} refresh; {output_path.name} is up to date")
        return
    generator(input_path, output_path)


def _refresh_nexo(*, context: PortfolioContext) -> None:
    from portfolio_crypto_data.cex.nexo_snapshots import generate_nexo_raw_snapshots

    _refresh_cex_exports(
        label="NEXO",
        input_path=context.paths.crypto_transactions / "cex" / "nexo",
        output_path=(
            context.paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
        ),
        generator=generate_nexo_raw_snapshots,
    )


def _refresh_crypto_com_app(*, context: PortfolioContext) -> None:
    from portfolio_crypto_data.cex.crypto_com_app_snapshots import (
        generate_crypto_com_app_raw_snapshots,
    )

    _refresh_cex_exports(
        label="Crypto.com App",
        input_path=context.paths.crypto_transactions / "cex" / "crypto_com_app",
        output_path=(
            context.paths.crypto_snapshots
            / "cex"
            / "crypto_com_app"
            / "crypto_com_app_raw_snapshots.csv"
        ),
        generator=generate_crypto_com_app_raw_snapshots,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = PortfolioContext.load(args.data_dir)
    validate_data_workspace(context.paths.root)
    try:
        with context.activate(), mutation_session(paths=context.paths):
            from portfolio_crypto_data.chain_config import load_evm_chain_configs
            from portfolio_crypto_data.update import rebuild_derived, update_onchain

            chain_configs = None
            if args.chain:
                chain_configs = tuple(
                    c for c in load_evm_chain_configs() if c.chain == args.chain.strip().lower()
                )
                if not chain_configs:
                    raise ValueError(f"Chain '{args.chain}' not found or disabled in config.")

            if args.command == "update":
                if chain_configs is not None:
                    update_onchain(chain_configs=chain_configs)
                else:
                    update_onchain()
                if not args.chain:
                    _refresh_nexo(context=context)
                    _refresh_crypto_com_app(context=context)
            else:
                rebuild_derived(
                    chain_configs=chain_configs,
                    incremental=args.incremental,
                    from_date=args.from_date,
                )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
