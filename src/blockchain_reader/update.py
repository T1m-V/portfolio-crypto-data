from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Literal

import pandas as pd

from blockchain_reader.accounting import accounting_paths, build_accounting_artifacts
from blockchain_reader.composition.lp_pricing import generate_protocol_lp_price_files
from blockchain_reader.dashboard_artifacts import (
    artifact_paths as dashboard_artifact_paths,
)
from blockchain_reader.dashboard_artifacts import (
    build_arbitrum_dashboard_artifacts,
)
from blockchain_reader.datetime_utils import (
    TRANSACTION_DATETIME_FORMAT,
    format_daily_datetime,
    parse_daily_datetime,
    parse_transaction_datetime_series,
)
from blockchain_reader.extraction.evm_reader import retrieve_transactions
from blockchain_reader.pipeline_logging import PipelineLogger
from blockchain_reader.protocols.aave import process_all_aave_tokens
from blockchain_reader.protocols.aura import process_all_aura_tokens
from blockchain_reader.protocols.balancer import process_all_balancer_tokens
from blockchain_reader.protocols.beefy import process_all_beefy_tokens
from blockchain_reader.protocols.curve import process_all_curve_tokens
from blockchain_reader.protocols.liquid_staking import process_all_liquid_staking_tokens
from blockchain_reader.raw_snapshots import generate_raw_snapshots
from file_paths import (
    BLOCKCHAIN_SNAPSHOT_FOLDER,
    BLOCKCHAIN_TRANSACTIONS_FOLDER,
    LP_PRICES_FOLDER,
    PROTOCOL_UNDERLYING_TOKEN_FOLDER,
)

PROTOCOL_PROCESSORS: tuple[tuple[str, Callable[..., object]], ...] = (
    ("beefy", process_all_beefy_tokens),
    ("balancer", process_all_balancer_tokens),
    ("aura", process_all_aura_tokens),
    ("curve", process_all_curve_tokens),
    ("aave", process_all_aave_tokens),
    ("liquid_staking", process_all_liquid_staking_tokens),
)

StageName = Literal[
    "transactions",
    "snapshots",
    "protocols",
    "lp_prices",
    "accounting",
    "dashboard",
]
STAGE_ORDER: tuple[StageName, ...] = (
    "transactions",
    "snapshots",
    "protocols",
    "lp_prices",
    "accounting",
    "dashboard",
)


@dataclass(frozen=True)
class StageResult:
    name: str
    start_date: str | None
    end_date: str | None
    files: list[Path] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineRunReport:
    chain: str
    from_date: str | None
    to_date: str | None
    stages: list[StageResult]

    @property
    def errors(self) -> list[str]:
        return [error for stage in self.stages for error in stage.errors]

    @property
    def warnings(self) -> list[str]:
        return [warning for stage in self.stages for warning in stage.warnings]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise RuntimeError("\n".join(self.errors))


def _transaction_path(chain: str) -> Path:
    return BLOCKCHAIN_TRANSACTIONS_FOLDER / f"{chain}_transactions.csv"


def _raw_snapshot_path(chain: str) -> Path:
    return BLOCKCHAIN_SNAPSHOT_FOLDER / f"{chain}_raw_snapshots.csv"


def _stage_index(stage: StageName) -> int:
    return STAGE_ORDER.index(stage)


def _stage_slice(from_stage: StageName, to_stage: StageName) -> tuple[StageName, ...]:
    start = _stage_index(from_stage)
    end = _stage_index(to_stage)
    if start > end:
        raise ValueError(
            f"from_stage must be before or equal to to_stage: {from_stage} > {to_stage}"
        )
    return STAGE_ORDER[start : end + 1]


def _delete_file(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def _clear_protocol_outputs(chain: str) -> None:
    root = PROTOCOL_UNDERLYING_TOKEN_FOLDER
    if not root.exists():
        return
    for protocol_dir in root.iterdir():
        if not protocol_dir.is_dir():
            continue
        for path in protocol_dir.glob(f"{chain}_*.csv"):
            _delete_file(path)


def _clear_lp_price_outputs(chain: str) -> None:
    root = LP_PRICES_FOLDER / chain
    if not root.exists():
        return
    for path in root.glob("*.csv"):
        _delete_file(path)


def _clear_accounting_outputs(chain: str) -> None:
    paths = accounting_paths(chain=chain)
    for path in (
        paths.principal_events,
        paths.principal_daily,
        paths.source_base_daily,
        paths.base_daily,
        paths.issues,
    ):
        _delete_file(path)


def _clear_dashboard_outputs(chain: str) -> None:
    paths = dashboard_artifact_paths(chain=chain)
    for path in (
        paths.asset_daily,
        paths.timeseries_daily,
        paths.composition_daily,
        paths.source_daily,
        paths.transactions_dashboard,
        paths.assets,
    ):
        _delete_file(path)
    _delete_file(paths.asset_daily.parent / "data_quality.csv")


def _clear_selected_derived_outputs(
    *,
    chain: str,
    selected_stages: tuple[StageName, ...],
    skip_lp_prices: bool,
) -> None:
    if "snapshots" in selected_stages:
        _delete_file(_raw_snapshot_path(chain))
    if "protocols" in selected_stages:
        _clear_protocol_outputs(chain=chain)
    if "lp_prices" in selected_stages and not skip_lp_prices:
        _clear_lp_price_outputs(chain=chain)
    if "accounting" in selected_stages:
        _clear_accounting_outputs(chain=chain)
    if "dashboard" in selected_stages:
        _clear_dashboard_outputs(chain=chain)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def _latest_transaction_date(path: Path) -> date | None:
    frame = _read_csv(path)
    if frame.empty or "Date" not in frame.columns:
        return None
    parsed = parse_transaction_datetime_series(frame["Date"]).dropna()
    if parsed.empty:
        return None
    return pd.Timestamp(parsed.max()).date()


def _latest_daily_date(path: Path) -> date | None:
    frame = _read_csv(path)
    if frame.empty or "Date" not in frame.columns:
        return None
    parsed = pd.to_datetime(frame["Date"].map(parse_daily_datetime), errors="coerce").dropna()
    if parsed.empty:
        return None
    return pd.Timestamp(parsed.max()).date()


def infer_update_from_date(chain: str) -> date | None:
    latest_dates = [
        value
        for value in (
            _latest_transaction_date(_transaction_path(chain)),
            _latest_daily_date(_raw_snapshot_path(chain)),
        )
        if value is not None
    ]
    if not latest_dates:
        return None
    return min(latest_dates) - timedelta(days=1)


def _normalize_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_daily_datetime(value)
    if parsed is None:
        raise ValueError(f"Invalid date: {value}")
    return parsed.date()


def _format_transaction_bound(value: date | None) -> str | None:
    if value is None:
        return None
    return datetime.combine(value, datetime.min.time()).strftime(TRANSACTION_DATETIME_FORMAT)


def _format_daily_bound(value: date | None) -> str | None:
    if value is None:
        return None
    return format_daily_datetime(value)


def _count_rows(path: Path) -> int:
    frame = _read_csv(path)
    return 0 if frame.empty else len(frame)


def _merge_generated_daily_rows(
    *,
    existing_path: Path,
    generated_path: Path,
    from_date: date | None,
) -> int:
    generated = _read_csv(generated_path)
    if generated.empty or from_date is None:
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(existing_path, index=False)
        return len(generated)

    existing = _read_csv(existing_path)
    if existing.empty:
        merged = generated
    else:
        existing_dates = pd.to_datetime(
            existing["Date"].map(parse_daily_datetime),
            errors="coerce",
        ).dt.date
        generated_dates = pd.to_datetime(
            generated["Date"].map(parse_daily_datetime),
            errors="coerce",
        ).dt.date
        before = existing[existing_dates < from_date].copy()
        replacement = generated[generated_dates >= from_date].copy()
        merged = pd.concat([before, replacement], ignore_index=True, sort=False)

    merged.to_csv(existing_path, index=False)
    return len(merged)


def _run_transactions_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    _ = replace_derived, logger
    path = _transaction_path(chain)
    try:
        asyncio.run(
            retrieve_transactions(
                chain=chain,
                start_date=_format_transaction_bound(from_date),
                end_date=_format_transaction_bound(to_date),
            )
        )
    except Exception as exc:
        return StageResult(
            name="transactions",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            files=[path],
            errors=[str(exc)],
        )
    return StageResult(
        name="transactions",
        start_date=_format_daily_bound(from_date),
        end_date=_format_daily_bound(to_date),
        files=[path],
        rows={"transactions": _count_rows(path)},
    )


def _run_snapshots_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    _ = replace_derived, logger
    transactions_path = _transaction_path(chain)
    output_path = _raw_snapshot_path(chain)
    principal_paths = accounting_paths(chain=chain)
    if not transactions_path.exists():
        return StageResult(
            name="snapshots",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            files=[output_path],
            errors=[f"missing transaction file: {transactions_path}"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    principal_paths.principal_events.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        suffix=".csv",
        prefix=f"{chain}_raw_snapshots_",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    with NamedTemporaryFile(
        suffix=".csv",
        prefix=f"{chain}_principal_events_",
        dir=principal_paths.principal_events.parent,
        delete=False,
    ) as temp_file:
        temp_principal_events = Path(temp_file.name)
    with NamedTemporaryFile(
        suffix=".csv",
        prefix=f"{chain}_principal_daily_",
        dir=principal_paths.principal_daily.parent,
        delete=False,
    ) as temp_file:
        temp_principal_daily = Path(temp_file.name)

    try:
        generate_raw_snapshots(
            input_csv=transactions_path,
            output_csv=temp_path,
            chain=chain,
            principal_events_csv=temp_principal_events,
            principal_daily_csv=temp_principal_daily,
        )
        rows = _merge_generated_daily_rows(
            existing_path=output_path,
            generated_path=temp_path,
            from_date=from_date,
        )
        temp_principal_events.replace(principal_paths.principal_events)
        temp_principal_daily.replace(principal_paths.principal_daily)
    except Exception as exc:
        return StageResult(
            name="snapshots",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            files=[output_path],
            errors=[str(exc)],
        )
    finally:
        for path in (temp_path, temp_principal_events, temp_principal_daily):
            if path.exists():
                path.unlink()

    return StageResult(
        name="snapshots",
        start_date=_format_daily_bound(from_date),
        end_date=_format_daily_bound(to_date),
        files=[output_path, principal_paths.principal_events, principal_paths.principal_daily],
        rows={
            "raw_snapshots": rows,
            "principal_events": _count_rows(principal_paths.principal_events),
            "principal_daily": _count_rows(principal_paths.principal_daily),
        },
    )


def _run_protocols_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    logger = logger or PipelineLogger()
    warnings: list[str] = []
    errors: list[str] = []
    start_date = _format_daily_bound(from_date)
    replace_from_date = start_date if replace_derived else None
    for protocol_name, processor in PROTOCOL_PROCESSORS:
        try:
            processor(
                chain=chain,
                start_date=start_date,
                replace_from_date=replace_from_date,
                logger=logger,
            )
        except Exception as exc:
            errors.append(f"{protocol_name}: {exc}")

    return StageResult(
        name="protocols",
        start_date=start_date,
        end_date=_format_daily_bound(to_date),
        warnings=warnings,
        errors=errors,
    )


def _run_lp_prices_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    _ = replace_derived, logger
    try:
        files = generate_protocol_lp_price_files(chain=chain)
    except Exception as exc:
        return StageResult(
            name="lp_prices",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            errors=[str(exc)],
        )
    return StageResult(
        name="lp_prices",
        start_date=_format_daily_bound(from_date),
        end_date=_format_daily_bound(to_date),
        files=list(files),
        rows={"files": len(files)},
    )


def _run_accounting_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    _ = replace_derived, logger
    build_accounting_artifacts(chain=chain, as_of_date=to_date)
    paths = accounting_paths(chain=chain)
    issues = _read_csv(paths.issues)
    errors = []
    if not issues.empty:
        errors = [
            f"{row['Date']} {row['Source']}->{row['BaseCoin']}: {row['Reason']}"
            for _, row in issues.iterrows()
        ]
    return StageResult(
        name="accounting",
        start_date=_format_daily_bound(from_date),
        end_date=_format_daily_bound(to_date),
        files=[
            paths.principal_events,
            paths.principal_daily,
            paths.source_base_daily,
            paths.base_daily,
            paths.issues,
        ],
        rows={
            "principal_events": _count_rows(paths.principal_events),
            "principal_daily": _count_rows(paths.principal_daily),
            "source_base_daily": _count_rows(paths.source_base_daily),
            "base_daily": _count_rows(paths.base_daily),
            "issues": _count_rows(paths.issues),
        },
        errors=errors,
    )


def _run_dashboard_stage(
    *,
    chain: str,
    from_date: date | None,
    to_date: date | None,
    replace_derived: bool = False,
    logger: PipelineLogger | None = None,
) -> StageResult:
    _ = replace_derived, logger
    if chain != "arbitrum":
        return StageResult(
            name="dashboard",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            warnings=[f"dashboard artifacts are only configured for arbitrum, skipped {chain}"],
        )

    try:
        paths = build_arbitrum_dashboard_artifacts(chain=chain)
    except Exception as exc:
        return StageResult(
            name="dashboard",
            start_date=_format_daily_bound(from_date),
            end_date=_format_daily_bound(to_date),
            errors=[str(exc)],
        )

    return StageResult(
        name="dashboard",
        start_date=_format_daily_bound(from_date),
        end_date=_format_daily_bound(to_date),
        files=[
            paths.asset_daily,
            paths.timeseries_daily,
            paths.composition_daily,
            paths.source_daily,
            paths.transactions_dashboard,
            paths.assets,
        ],
    )


STAGE_RUNNERS: dict[StageName, Callable[..., StageResult]] = {
    "transactions": _run_transactions_stage,
    "snapshots": _run_snapshots_stage,
    "protocols": _run_protocols_stage,
    "lp_prices": _run_lp_prices_stage,
    "accounting": _run_accounting_stage,
    "dashboard": _run_dashboard_stage,
}


def update_blockchain_data(
    chain: str = "arbitrum",
    from_date: str | date | datetime | None = None,
    to_date: str | date | datetime | None = None,
    overwrite_from_date: bool = False,
    update_prices: bool = False,
    from_stage: StageName = "transactions",
    to_stage: StageName = "dashboard",
    skip_lp_prices: bool = False,
    replace_derived: bool = False,
) -> PipelineRunReport:
    """
    Updates blockchain data and derived artifacts in dependency order.

    args:
        chain: Chain identifier.
        from_date: Optional update start date. Defaults to the latest known
            transaction/snapshot date minus one day.
        to_date: Optional inclusive update end date. Defaults to today.
        overwrite_from_date: Allows callers to intentionally replace derived rows from
            the selected start date. Raw transaction rows before that date are never
            deleted by this command.
        update_prices: Reserved for explicit direct-price refreshes; off by default.
        from_stage: First pipeline stage to run.
        to_stage: Last pipeline stage to run.
        skip_lp_prices: Skip LP price CSV regeneration if the selected range includes it.
        replace_derived: Replace selected derived outputs. With no from_date this clears
            selected chain-derived outputs before running.

    returns:
        Structured report describing stage outputs and blocking issues.
    """
    selected_stages = _stage_slice(from_stage=from_stage, to_stage=to_stage)
    effective_replace_derived = replace_derived or overwrite_from_date
    resolved_from = (
        _normalize_date(from_date) if from_date is not None else infer_update_from_date(chain)
    )
    resolved_to = _normalize_date(to_date) or date.today()
    warnings = []
    if update_prices:
        warnings.append(
            "direct price fetching is not part of this command yet; using existing price files"
        )
    if "transactions" not in selected_stages:
        warnings.append(f"transaction gathering skipped because from_stage={from_stage}")
    if skip_lp_prices and "lp_prices" in selected_stages:
        warnings.append("LP price regeneration skipped by request")
    if not effective_replace_derived and from_date is not None:
        warnings.append(
            "raw transactions before from_date are preserved; "
            "derived rows are rebuilt from from_date"
        )

    logger = PipelineLogger()
    if effective_replace_derived and resolved_from is None:
        _clear_selected_derived_outputs(
            chain=chain,
            selected_stages=selected_stages,
            skip_lp_prices=skip_lp_prices,
        )

    stages: list[StageResult] = []
    if warnings:
        stages.append(
            StageResult(
                name="preflight",
                start_date=_format_daily_bound(resolved_from),
                end_date=_format_daily_bound(resolved_to),
                warnings=warnings,
            )
        )

    for stage_name in selected_stages:
        if stage_name == "lp_prices" and skip_lp_prices:
            continue
        runner = STAGE_RUNNERS[stage_name]
        logger.stage_start(stage_name)
        stage = runner(
            chain=chain,
            from_date=resolved_from,
            to_date=resolved_to,
            replace_derived=effective_replace_derived,
            logger=logger,
        )
        stages.append(stage)
        logger.stage_end(stage_name, errors=bool(stage.errors))
        if stage.errors:
            break

    return PipelineRunReport(
        chain=chain,
        from_date=_format_daily_bound(resolved_from),
        to_date=_format_daily_bound(resolved_to),
        stages=stages,
    )


def _print_report(report: PipelineRunReport) -> None:
    print(f"Blockchain update: {report.chain}")
    print(f"Window: {report.from_date or '<initial>'} -> {report.to_date or '<today>'}")
    for stage in report.stages:
        status = "ERROR" if stage.errors else "OK"
        print(f"- {stage.name}: {status}")
        if stage.rows:
            row_summary = ", ".join(f"{key}={value}" for key, value in stage.rows.items())
            print(f"  rows: {row_summary}")
        if stage.files:
            print(f"  files: {len(stage.files)}")
        for warning in stage.warnings:
            print(f"  warning: {warning}")
        for error in stage.errors:
            print(f"  error: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update blockchain data and derived artifacts.")
    parser.add_argument("--chain", default="arbitrum")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--overwrite-from-date", action="store_true")
    parser.add_argument("--update-prices", action="store_true")
    parser.add_argument("--from-stage", choices=STAGE_ORDER, default="transactions")
    parser.add_argument("--to-stage", choices=STAGE_ORDER, default="dashboard")
    parser.add_argument("--skip-lp-prices", action="store_true")
    parser.add_argument("--replace-derived", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = update_blockchain_data(
            chain=args.chain,
            from_date=args.from_date,
            to_date=args.to_date,
            overwrite_from_date=args.overwrite_from_date,
            update_prices=args.update_prices,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            skip_lp_prices=args.skip_lp_prices,
            replace_derived=args.replace_derived,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
