from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from portfolio_core import PortfolioContext, mutation_session, validate_data_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-crypto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("command", choices=["update", "rebuild"])
    return parser


def _update(*, argv: list[str], context: PortfolioContext) -> int:
    from portfolio_crypto_data.cex.nexo_snapshots import generate_nexo_raw_snapshots
    from portfolio_crypto_data.update import main as update_evm

    result = update_evm(argv)
    if result:
        return result

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
    return 0


def main(argv: list[str] | None = None) -> int:
    args, remaining = _parser().parse_known_args(argv)
    context = PortfolioContext.from_root(args.data_dir)
    validate_data_workspace(context.paths.root)
    with context.activate(), mutation_session(
        paths=context.paths,
        component=f"crypto-{args.command}",
        version=version("portfolio-crypto-data"),
    ):
        if args.command == "update":
            return _update(argv=remaining, context=context)

        from portfolio_crypto_data.rebuild_arbitrum_derived import main as rebuild

        return rebuild(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
