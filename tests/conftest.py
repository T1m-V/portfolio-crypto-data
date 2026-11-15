from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from portfolio_core import PortfolioContext


@pytest.fixture(autouse=True)
def isolate_portfolio_io(
    tmp_path: Path,
) -> Iterator[PortfolioContext]:
    """Run every crypto test inside an explicit temporary portfolio context."""
    context = PortfolioContext.load(tmp_path, load_secrets=False)
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

    from portfolio_core import save_price_csv

    save_price_csv(
        file_path=context.paths.direct_price("USD_EUR"),
        frame=pd.DataFrame({"Date": ["2000-01-01"], "Price": [1.0]}),
    )

    with context.activate():
        yield context
