from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from portfolio_core import PortfolioContext, mutation_session, validate_data_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-crypto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("command", choices=["update", "rebuild"])
    return parser


def _refresh_cex_exports(
    *,
    label: str,
    input_path: Path,
    output_path: Path,
    generator: Callable[[Path, Path], None],
) -> None:
    if input_path.exists() and any(input_path.glob("*.csv")):
        generator(input_path, output_path)
    else:
        print(f"Skipping {label} refresh; no CSV exports found in {input_path}")


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
    context = PortfolioContext.from_root(args.data_dir)
    validate_data_workspace(context.paths.root)
    try:
        with context.activate(), mutation_session(paths=context.paths):
            from portfolio_crypto_data.update import rebuild_derived, update_onchain

            if args.command == "update":
                update_onchain()
                _refresh_nexo(context=context)
                _refresh_crypto_com_app(context=context)
            else:
                rebuild_derived()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
