from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio-crypto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("command", choices=["update", "rebuild"])
    return parser


def _update(argv: list[str]) -> int:
    from portfolio_core import BLOCKCHAIN_SNAPSHOT_FOLDER, BLOCKCHAIN_TRANSACTIONS_FOLDER

    from portfolio_crypto_data.cex.nexo_snapshots import generate_nexo_raw_snapshots
    from portfolio_crypto_data.update import main as update_evm

    result = update_evm(argv)
    if result:
        return result

    nexo_input = BLOCKCHAIN_TRANSACTIONS_FOLDER / "cex" / "nexo"
    if nexo_input.exists() and any(nexo_input.glob("*.csv")):
        generate_nexo_raw_snapshots(
            input_csv=nexo_input,
            output_csv=(
                BLOCKCHAIN_SNAPSHOT_FOLDER / "cex" / "nexo" / "nexo_raw_snapshots.csv"
            ),
        )
    else:
        print(f"Skipping Nexo refresh; no CSV exports found in {nexo_input}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args, remaining = _parser().parse_known_args(argv)
    os.environ["PORTFOLIO_DATA_DIR"] = str(args.data_dir.resolve())
    if args.command == "update":
        return _update(remaining)

    from portfolio_crypto_data.rebuild_arbitrum_derived import main as rebuild

    return rebuild(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
