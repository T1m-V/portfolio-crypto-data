from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import blockchain_reader.accounting as accounting
import blockchain_reader.rebuild_arbitrum_derived as rebuild
import blockchain_reader.update as update
from blockchain_reader.dashboard_artifacts import ArbitrumDashboardArtifactPaths


def _patch_update_paths(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "transactions": tmp_path / "transactions",
        "snapshots": tmp_path / "snapshots",
        "accounting": tmp_path / "accounting",
        "dashboard": tmp_path / "dashboard",
        "protocols": tmp_path / "protocol_underlying_tokens",
        "lp_prices": tmp_path / "prices" / "lp_prices",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(update, "BLOCKCHAIN_TRANSACTIONS_FOLDER", paths["transactions"])
    monkeypatch.setattr(update, "BLOCKCHAIN_SNAPSHOT_FOLDER", paths["snapshots"])
    monkeypatch.setattr(update, "PROTOCOL_UNDERLYING_TOKEN_FOLDER", paths["protocols"])
    monkeypatch.setattr(update, "LP_PRICES_FOLDER", paths["lp_prices"])
    monkeypatch.setattr(accounting, "BLOCKCHAIN_ACCOUNTING_FOLDER", paths["accounting"])

    def fake_dashboard_paths(chain: str = "arbitrum") -> ArbitrumDashboardArtifactPaths:
        root = paths["dashboard"] / chain
        return ArbitrumDashboardArtifactPaths(
            asset_daily=root / "asset_daily.csv",
            timeseries_daily=root / "timeseries_daily.csv",
            composition_daily=root / "composition_daily.csv",
            source_daily=root / "source_daily.csv",
            transactions_dashboard=root / "transactions_dashboard.csv",
            assets=root / "assets.csv",
        )

    monkeypatch.setattr(update, "dashboard_artifact_paths", fake_dashboard_paths)
    return paths


def test_infer_update_from_date_uses_oldest_latest_transaction_or_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_update_paths(monkeypatch, tmp_path)
    pd.DataFrame(
        {
            "TX Hash": ["a", "b"],
            "Date": ["05/01/2025 10:00:00", "10/01/2025 10:00:00"],
        }
    ).to_csv(paths["transactions"] / "arbitrum_transactions.csv", index=False)
    pd.DataFrame(
        {
            "Date": ["2025-01-07 00:00:00", "2025-01-08 00:00:00"],
            "Coin": ["ETH", "ETH"],
            "Quantity": [1, 2],
        }
    ).to_csv(paths["snapshots"] / "arbitrum_raw_snapshots.csv", index=False)

    assert update.infer_update_from_date("arbitrum") == date(2025, 1, 7)


def test_merge_generated_daily_rows_replaces_only_from_update_date(tmp_path) -> None:
    existing_path = tmp_path / "arbitrum_raw_snapshots.csv"
    generated_path = tmp_path / "rebuilt.csv"
    pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "Coin": ["ETH", "ETH", "ETH"],
            "Quantity": [1, 2, 3],
        }
    ).to_csv(existing_path, index=False)
    pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "Coin": ["ETH", "ETH", "ETH", "ETH"],
            "Quantity": [10, 20, 30, 40],
        }
    ).to_csv(generated_path, index=False)

    rows = update._merge_generated_daily_rows(
        existing_path=existing_path,
        generated_path=generated_path,
        from_date=date(2025, 1, 3),
    )

    result = pd.read_csv(existing_path)
    assert rows == 4
    assert result["Quantity"].tolist() == [1, 2, 30, 40]


def test_update_blockchain_data_runs_stages_in_dependency_order(monkeypatch, tmp_path) -> None:
    paths = _patch_update_paths(monkeypatch, tmp_path)
    calls: list[str] = []

    async def fake_retrieve_transactions(chain: str, start_date: str | None, end_date: str | None):
        calls.append(f"transactions:{start_date}:{end_date}")
        pd.DataFrame(
            {
                "TX Hash": ["h1"],
                "Date": ["02/01/2025 10:00:00"],
            }
        ).to_csv(paths["transactions"] / f"{chain}_transactions.csv", index=False)

    def fake_generate_raw_snapshots(
        input_csv: Path,
        output_csv: Path,
        chain: str,
        principal_events_csv: Path | None = None,
        principal_daily_csv: Path | None = None,
    ) -> None:
        calls.append("snapshots")
        pd.DataFrame(
            {
                "Date": ["2025-01-02"],
                "Coin": ["ETH"],
                "Quantity": [1],
                "Principal Invested": [100],
            }
        ).to_csv(output_csv, index=False)
        if principal_events_csv is not None:
            pd.DataFrame(
                columns=[
                    "Date",
                    "TX Hash",
                    "Action",
                    "Source",
                    "BaseCoin",
                    "PrincipalDeltaEUR",
                    "PrincipalBalanceEUR",
                ]
            ).to_csv(principal_events_csv, index=False)
        if principal_daily_csv is not None:
            pd.DataFrame(
                {
                    "Date": ["2025-01-02"],
                    "Coin": ["ETH"],
                    "PrincipalInvestedEUR": [100],
                }
            ).to_csv(principal_daily_csv, index=False)

    def fake_protocol(name: str):
        def _run(
            chain: str,
            start_date: str | None = None,
            replace_from_date: str | None = None,
            logger=None,
        ) -> None:
            calls.append(f"protocol:{name}:{start_date}:{replace_from_date}:{logger is not None}")

        return _run

    def fake_lp_prices(chain: str) -> list[Path]:
        calls.append("lp_prices")
        return [tmp_path / "ETH.csv"]

    def fake_build_accounting_artifacts(chain: str, as_of_date=None) -> object:
        calls.append(f"accounting:{as_of_date}")
        paths_for_chain = accounting.accounting_paths(chain)
        paths_for_chain.source_base_daily.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Date": ["2025-01-02"]}).to_csv(
            paths_for_chain.source_base_daily,
            index=False,
        )
        pd.DataFrame({"Date": ["2025-01-02"]}).to_csv(paths_for_chain.base_daily, index=False)
        pd.DataFrame(columns=["Date", "Source", "BaseCoin", "Reason"]).to_csv(
            paths_for_chain.issues,
            index=False,
        )
        return object()

    def fake_dashboard(chain: str) -> ArbitrumDashboardArtifactPaths:
        calls.append("dashboard")
        root = paths["dashboard"] / chain
        return ArbitrumDashboardArtifactPaths(
            asset_daily=root / "asset_daily.csv",
            timeseries_daily=root / "timeseries_daily.csv",
            composition_daily=root / "composition_daily.csv",
            source_daily=root / "source_daily.csv",
            transactions_dashboard=root / "transactions_dashboard.csv",
            assets=root / "assets.csv",
        )

    monkeypatch.setattr(update, "retrieve_transactions", fake_retrieve_transactions)
    monkeypatch.setattr(update, "generate_raw_snapshots", fake_generate_raw_snapshots)
    monkeypatch.setattr(
        update,
        "PROTOCOL_PROCESSORS",
        (("beefy", fake_protocol("beefy")), ("aave", fake_protocol("aave"))),
    )
    monkeypatch.setattr(update, "generate_protocol_lp_price_files", fake_lp_prices)
    monkeypatch.setattr(update, "build_accounting_artifacts", fake_build_accounting_artifacts)
    monkeypatch.setattr(update, "build_arbitrum_dashboard_artifacts", fake_dashboard)

    report = update.update_blockchain_data(
        chain="arbitrum",
        from_date="2025-01-02",
        to_date="2025-01-03",
        replace_derived=True,
    )

    assert report.errors == []
    assert [stage.name for stage in report.stages] == [
        "transactions",
        "snapshots",
        "protocols",
        "lp_prices",
        "accounting",
        "dashboard",
    ]
    assert calls == [
        "transactions:02/01/2025 00:00:00:03/01/2025 00:00:00",
        "snapshots",
        "protocol:beefy:2025-01-02 00:00:00:2025-01-02 00:00:00:True",
        "protocol:aave:2025-01-02 00:00:00:2025-01-02 00:00:00:True",
        "lp_prices",
        "accounting:2025-01-03",
        "dashboard",
    ]


def test_rebuild_arbitrum_derived_runs_only_snapshot_accounting_and_dashboard(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _patch_update_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(rebuild, "BLOCKCHAIN_TRANSACTIONS_FOLDER", paths["transactions"])
    monkeypatch.setattr(rebuild, "BLOCKCHAIN_SNAPSHOT_FOLDER", paths["snapshots"])
    monkeypatch.setattr(accounting, "BLOCKCHAIN_ACCOUNTING_FOLDER", paths["accounting"])
    calls: list[str] = []
    pd.DataFrame({"Date": ["01/01/2025 10:00:00"]}).to_csv(
        paths["transactions"] / "arbitrum_transactions.csv",
        index=False,
    )

    def fake_generate_raw_snapshots(
        input_csv: Path,
        output_csv: Path,
        chain: str,
        principal_events_csv: Path | None = None,
        principal_daily_csv: Path | None = None,
    ) -> None:
        calls.append(f"snapshots:{input_csv.name}:{chain}")
        pd.DataFrame(
            {
                "Date": ["2025-01-01"],
                "Coin": ["ETH"],
                "Quantity": [1],
                "Principal Invested": [0],
            }
        ).to_csv(output_csv, index=False)
        pd.DataFrame(columns=["Date"]).to_csv(principal_events_csv, index=False)
        pd.DataFrame(
            {"Date": ["2025-01-01"], "Coin": ["ETH"], "PrincipalInvestedEUR": [1000]}
        ).to_csv(principal_daily_csv, index=False)

    def fake_build_accounting_artifacts(chain: str, as_of_date=None) -> object:
        calls.append(f"accounting:{chain}:{as_of_date}")
        return SimpleNamespace(rows_written={"issues": 0}, errors=[])

    def fake_dashboard(chain: str) -> ArbitrumDashboardArtifactPaths:
        calls.append(f"dashboard:{chain}")
        root = paths["dashboard"] / chain
        return ArbitrumDashboardArtifactPaths(
            asset_daily=root / "asset_daily.csv",
            timeseries_daily=root / "timeseries_daily.csv",
            composition_daily=root / "composition_daily.csv",
            source_daily=root / "source_daily.csv",
            transactions_dashboard=root / "transactions_dashboard.csv",
            assets=root / "assets.csv",
        )

    monkeypatch.setattr(rebuild, "generate_raw_snapshots", fake_generate_raw_snapshots)
    monkeypatch.setattr(rebuild, "build_accounting_artifacts", fake_build_accounting_artifacts)
    monkeypatch.setattr(rebuild, "build_arbitrum_dashboard_artifacts", fake_dashboard)

    result = rebuild.rebuild_arbitrum_derived(as_of_date="2026-05-09")

    assert calls == [
        "snapshots:arbitrum_transactions.csv:arbitrum",
        "accounting:arbitrum:2026-05-09",
        "dashboard:arbitrum",
    ]
    assert result["chain"] == "arbitrum"


def test_update_blockchain_data_from_snapshots_skips_transactions(monkeypatch, tmp_path) -> None:
    _patch_update_paths(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_runner(name: str):
        def _run(
            *,
            chain: str,
            from_date: date | None,
            to_date: date | None,
            replace_derived: bool = False,
            logger=None,
        ) -> update.StageResult:
            calls.append(f"{name}:{replace_derived}:{logger is not None}")
            return update.StageResult(
                name=name,
                start_date=update._format_daily_bound(from_date),
                end_date=update._format_daily_bound(to_date),
            )

        return _run

    for stage in update.STAGE_ORDER:
        monkeypatch.setitem(update.STAGE_RUNNERS, stage, fake_runner(stage))

    report = update.update_blockchain_data(
        chain="arbitrum",
        from_date="2025-01-02",
        to_date="2025-01-03",
        from_stage="snapshots",
        replace_derived=True,
    )

    assert report.errors == []
    assert calls == [
        "snapshots:True:True",
        "protocols:True:True",
        "lp_prices:True:True",
        "accounting:True:True",
        "dashboard:True:True",
    ]


def test_update_blockchain_data_protocols_can_skip_lp_prices(monkeypatch, tmp_path) -> None:
    _patch_update_paths(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_runner(name: str):
        def _run(
            *,
            chain: str,
            from_date: date | None,
            to_date: date | None,
            replace_derived: bool = False,
            logger=None,
        ) -> update.StageResult:
            calls.append(name)
            return update.StageResult(
                name=name,
                start_date=update._format_daily_bound(from_date),
                end_date=update._format_daily_bound(to_date),
            )

        return _run

    for stage in update.STAGE_ORDER:
        monkeypatch.setitem(update.STAGE_RUNNERS, stage, fake_runner(stage))

    report = update.update_blockchain_data(
        chain="arbitrum",
        from_date="2025-01-02",
        to_date="2025-01-03",
        from_stage="protocols",
        skip_lp_prices=True,
    )

    assert report.errors == []
    assert calls == ["protocols", "accounting", "dashboard"]


def test_update_blockchain_data_to_accounting_stops_before_dashboard(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_update_paths(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_runner(name: str):
        def _run(
            *,
            chain: str,
            from_date: date | None,
            to_date: date | None,
            replace_derived: bool = False,
            logger=None,
        ) -> update.StageResult:
            calls.append(name)
            return update.StageResult(
                name=name,
                start_date=update._format_daily_bound(from_date),
                end_date=update._format_daily_bound(to_date),
            )

        return _run

    for stage in update.STAGE_ORDER:
        monkeypatch.setitem(update.STAGE_RUNNERS, stage, fake_runner(stage))

    report = update.update_blockchain_data(
        chain="arbitrum",
        from_date="2025-01-02",
        to_date="2025-01-03",
        from_stage="snapshots",
        to_stage="accounting",
    )

    assert report.errors == []
    assert calls == ["snapshots", "protocols", "lp_prices", "accounting"]


def test_update_blockchain_data_rejects_invalid_stage_order() -> None:
    try:
        update.update_blockchain_data(from_stage="dashboard", to_stage="snapshots")
    except ValueError as exc:
        assert "from_stage must be before or equal to to_stage" in str(exc)
    else:
        raise AssertionError("Expected invalid stage ordering to fail")


def test_replace_derived_clears_only_selected_chain_outputs(monkeypatch, tmp_path) -> None:
    paths = _patch_update_paths(monkeypatch, tmp_path)
    protocol_file = paths["protocols"] / "beefy" / "arbitrum_old.csv"
    other_chain_protocol = paths["protocols"] / "beefy" / "polygon_old.csv"
    lp_file = paths["lp_prices"] / "arbitrum" / "LP.csv"
    tx_file = paths["transactions"] / "arbitrum_transactions.csv"
    for path in (protocol_file, other_chain_protocol, lp_file, tx_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Date,Value\n2025-01-01,1\n", encoding="utf-8")

    def fake_runner(
        *,
        chain: str,
        from_date: date | None,
        to_date: date | None,
        replace_derived: bool = False,
        logger=None,
    ) -> update.StageResult:
        return update.StageResult(
            name="protocols",
            start_date=update._format_daily_bound(from_date),
            end_date=update._format_daily_bound(to_date),
        )

    monkeypatch.setitem(update.STAGE_RUNNERS, "protocols", fake_runner)
    monkeypatch.setattr(update, "infer_update_from_date", lambda chain: None)

    report = update.update_blockchain_data(
        chain="arbitrum",
        from_date=None,
        to_date="2025-01-03",
        from_stage="protocols",
        to_stage="protocols",
        replace_derived=True,
    )

    assert report.errors == []
    assert not protocol_file.exists()
    assert other_chain_protocol.exists()
    assert lp_file.exists()
    assert tx_file.exists()
