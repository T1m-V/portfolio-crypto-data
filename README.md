# portfolio-crypto-data

## End-user installation

The dashboard and data workspace install the immutable `v0.3.0` package from GitHub. A source
checkout is not required. To install the CLI by itself:

```powershell
uv tool install --no-sources "portfolio-crypto-data @ git+https://github.com/T1m-V/portfolio-crypto-data.git@v0.3.0"
portfolio-crypto --help
```

## Developer setup

Clone `portfolio-core` next to this repository. `[tool.uv.sources]` overrides the released core
dependency with `../portfolio-core` in editable mode for this checkout:

```powershell
cd C:\Users\timvo\source\portfolio\portfolio-crypto-data
uv sync --frozen
uv run python -c "from pathlib import Path; import portfolio_core; print(Path(portfolio_core.__file__).resolve())"
uv run ruff check src tests
uv run python -m pytest
uv build
```

The printed core path should be inside the sibling `portfolio-core` checkout. Python edits there
are visible immediately; rerun `uv lock` and `uv sync` only after dependency metadata changes.

Loader commands operate on an explicit data workspace:

```powershell
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data update
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data rebuild
```

The update command runs the EVM pipeline and refreshes custodial snapshots when supported
transaction exports are present. Place NEXO exports in `crypto/transactions/cex/nexo/` and
Crypto.com App Token Wallet exports in `crypto/transactions/cex/crypto_com_app/`. Crypto.com App
and Crypto.com Exchange are separate entities; Exchange exports are not ingested yet. All writes
are rooted in the explicitly selected external data workspace.

Crypto.com App snapshots include crypto held in Earn, staking, and Supercharger; transfers
between these products and the app wallet do not change total holdings or principal. Rewards
increase holdings without adding invested principal, and reward reversals undo only quantity.
Fiat purchases add the received crypto; fiat sales, card top-ups, and outgoing peer transfers
remove crypto without creating a fiat or recipient balance. Incoming and outgoing legacy
`crypto_transfer` rows follow the sign of the exported amount.

Dust conversions combine one credit with one or more debits sharing its description within
one second. Missing or ambiguous matches and unknown transaction types stop ingestion.
