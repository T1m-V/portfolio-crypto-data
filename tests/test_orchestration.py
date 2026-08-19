from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from portfolio_crypto_data import cli, update


def test_update_runs_one_fixed_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def retrieve(*, chain: str) -> None:
        calls.append(f"transactions:{chain}")

    monkeypatch.setattr(update, "retrieve_transactions", retrieve)
    monkeypatch.setattr(update, "_build_snapshots", lambda: calls.append("snapshots"))
    monkeypatch.setattr(
        update,
        "PROTOCOL_PROCESSORS",
        (("one", lambda **_: calls.append("protocol:one")),),
    )
    monkeypatch.setattr(
        update,
        "generate_protocol_lp_price_files",
        lambda **_: calls.append("lp_prices"),
    )
    monkeypatch.setattr(
        update,
        "build_accounting_artifacts",
        lambda **_: calls.append("accounting"),
    )
    monkeypatch.setattr(
        update,
        "build_arbitrum_dashboard_artifacts",
        lambda **_: calls.append("dashboard"),
    )

    update.update_onchain()

    assert calls == [
        "transactions:arbitrum",
        "snapshots",
        "protocol:one",
        "lp_prices",
        "accounting",
        "dashboard",
    ]


def test_rebuild_is_local_and_uses_requested_date(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(update, "_build_snapshots", lambda: calls.append("snapshots"))
    monkeypatch.setattr(
        update,
        "build_accounting_artifacts",
        lambda **kwargs: calls.append(kwargs["as_of_date"]),
    )
    monkeypatch.setattr(
        update,
        "build_arbitrum_dashboard_artifacts",
        lambda **_: calls.append("dashboard"),
    )

    update.rebuild_derived(as_of_date=date(2026, 8, 8))

    assert calls == ["snapshots", date(2026, 8, 8), "dashboard"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [("update", ["update", "nexo"]), ("rebuild", ["rebuild"])],
)
def test_cli_preserves_dashboard_command_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    expected: list[str],
) -> None:
    calls: list[str] = []
    (tmp_path / "portfolio.toml").write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(update, "update_onchain", lambda: calls.append("update"))
    monkeypatch.setattr(update, "rebuild_derived", lambda: calls.append("rebuild"))
    monkeypatch.setattr(cli, "_refresh_nexo", lambda **_: calls.append("nexo"))

    result = cli.main(["--data-dir", str(tmp_path), command])

    assert result == 0
    assert calls == expected
    assert not (tmp_path / "runtime" / "mutation.lock").exists()


def test_rebuild_requires_transactions() -> None:
    with pytest.raises(FileNotFoundError, match="missing transaction file"):
        update.rebuild_derived()
