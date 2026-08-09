from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from portfolio_core import PortfolioContext, mutation_session, validate_data_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-crypto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("command", choices=["update", "rebuild"])
    return parser


def _refresh_nexo(*, context: PortfolioContext) -> None:
    from portfolio_crypto_data.cex.nexo_snapshots import generate_nexo_raw_snapshots

    nexo_input = context.paths.crypto_transactions / "cex" / "nexo"
    if nexo_input.exists() and any(nexo_input.glob("*.csv")):
        generate_nexo_raw_snapshots(
            input_csv=nexo_input,
            output_csv=(
                context.paths.crypto_snapshots / "cex" / "nexo" / "nexo_raw_snapshots.csv"
            ),
        )
    else:
        print(f"Skipping Nexo refresh; no CSV exports found in {nexo_input}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = PortfolioContext.from_root(args.data_dir)
    validate_data_workspace(context.paths.root)
    try:
        with context.activate(), mutation_session(
            paths=context.paths,
            component=f"crypto-{args.command}",
            version=version("portfolio-crypto-data"),
        ):
            from portfolio_crypto_data.update import rebuild_derived, update_onchain

            if args.command == "update":
                update_onchain()
                _refresh_nexo(context=context)
            else:
                rebuild_derived()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
