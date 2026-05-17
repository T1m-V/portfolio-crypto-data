from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_blockchain_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Forces blockchain-reader tests to use temporary folders by default.
    Individual tests can still override with explicit patch(...) contexts.
    """
    prices_root = tmp_path / "prices"
    protocol_root = tmp_path / "protocol_underlying_tokens"
    snapshots_root = tmp_path / "snapshots"
    tokens_root = tmp_path / "tokens"

    prices_root.mkdir(parents=True, exist_ok=True)
    protocol_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.mkdir(parents=True, exist_ok=True)
    tokens_root.mkdir(parents=True, exist_ok=True)

    import blockchain_reader.accounting as accounting
    from blockchain_reader.composition import lp_pricing
    from price_history.price_data_utils import save_price_csv as _save_price_csv

    monkeypatch.setattr(lp_pricing, "PRICES_FOLDER", prices_root)
    monkeypatch.setattr(lp_pricing, "PROTOCOL_UNDERLYING_TOKEN_FOLDER", protocol_root)
    monkeypatch.setattr(lp_pricing, "TOKENS_FOLDER", tokens_root)

    monkeypatch.setattr(accounting, "PRICES_FOLDER", prices_root)
    monkeypatch.setattr(accounting, "PROTOCOL_UNDERLYING_TOKEN_FOLDER", protocol_root)
    monkeypatch.setattr(accounting, "BLOCKCHAIN_SNAPSHOT_FOLDER", snapshots_root)
    monkeypatch.setattr(accounting, "TOKENS_FOLDER", tokens_root)
    monkeypatch.setattr(accounting, "BLOCKCHAIN_ACCOUNTING_FOLDER", tmp_path / "accounting")

    repo_data_root = (Path(__file__).resolve().parents[2] / "data").resolve()

    def guarded_save_price_csv(*, file_path: Path, frame) -> None:
        resolved = Path(file_path).resolve()
        if resolved == repo_data_root or repo_data_root in resolved.parents:
            raise AssertionError(f"Test attempted to write into repository data folder: {resolved}")
        _save_price_csv(file_path=file_path, frame=frame)

    monkeypatch.setattr(lp_pricing, "save_price_csv", guarded_save_price_csv)
