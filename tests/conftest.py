from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from portfolio_core import PortfolioContext
from portfolio_core import save_price_csv as _save_price_csv


@pytest.fixture(autouse=True)
def isolate_portfolio_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[PortfolioContext]:
    """Run every crypto test inside an explicit temporary portfolio context."""
    context = PortfolioContext.from_root(tmp_path, load_secrets=False)
    for directory in (
        context.paths.prices,
        context.paths.protocol_underlying_tokens,
        context.paths.crypto_snapshots,
        context.paths.tokens,
        context.paths.accounting,
        context.paths.dashboard_artifacts,
        context.paths.crypto_transactions,
        context.paths.block_map,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _save_price_csv(
        file_path=context.paths.direct_price("USD_EUR"),
        frame=pd.DataFrame({"Date": ["2000-01-01"], "Price": [1.0]}),
    )

    repo_data_root = (Path(__file__).resolve().parent.parent / "data").resolve()

    def guarded_save_price_csv(*, file_path: Path, frame) -> None:
        resolved = Path(file_path).resolve()
        if resolved == repo_data_root or repo_data_root in resolved.parents:
            raise AssertionError(f"Test attempted to write into repository data folder: {resolved}")
        _save_price_csv(file_path=file_path, frame=frame)

    from portfolio_crypto_data.composition import lp_pricing

    monkeypatch.setattr(lp_pricing, "save_price_csv", guarded_save_price_csv)
    with context.activate():
        yield context
