# AGENTS.md

## Purpose

`portfolio-crypto-data` owns on-chain and custodial crypto ingestion, transaction analysis,
protocol decomposition, economic-principal accounting, snapshot generation, LP pricing, and the
derived artifacts consumed by the dashboard. It exposes the `portfolio-crypto` CLI.

## Repository Role

| Repository | Relationship to this package |
| --- | --- |
| `portfolio-core` | The only first-party dependency. It owns settings, workspace paths, metadata, and shared price/forex behavior. |
| `portfolio-dashboard` | Imports read-oriented dashboard artifacts and symbol helpers, and invokes the CLI for refresh jobs. |
| `portfolio-data` | Stores chain configuration, transactions, protocol histories, accounting state, prices, and generated dashboard artifacts. Its lockfile pins this package. |
| `portfolio-market-data` | Sibling loader. Neither loader may depend on the other; common behavior belongs in core. |

## Package Map

- `extraction`: EVM readers, token management, and raw transaction analysis.
- `protocols`: Aave, Aura, Balancer, Beefy, Curve, and liquid-staking decomposition.
- `composition`: recursive exposure expansion and LP valuation.
- `accounting.py`, `principal_ledger.py`, `raw_snapshots.py`: balances and economic-principal
  state.
- `cex`: Nexo snapshot processing.
- `dashboard_artifacts.py`: read contract used by `portfolio-dashboard`.
- `nexo_dashboard.py`: read-only Nexo projection used by `portfolio-dashboard`.
- `update.py` and `cli.py`: orchestration/process boundary.

Supported commands:

```powershell
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data update
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data rebuild
```

## Refactoring Policy

- Favor a clear accounting model over compatibility with former monorepo modules or call shapes.
- Update all call sites and tests immediately when changing signatures or types.
- Do not restore a `blockchain_reader` namespace, accept obsolete argument aliases, or retain old
  implementations beside replacements.
- Keep shared filesystem and price rules in `portfolio-core`; do not create another path layer here.
- Keep loader mutation behind the CLI or explicit orchestration functions. Dashboard read code may
  import artifacts, but must not assemble a second crypto pipeline.
- Create one `PortfolioContext` at the CLI boundary. Internal pipeline code may use its task-local
  activation within that call tree, but must not cache a data root at module import time.
- Prefer explicit transformations and small accounting operations over compatibility branches.
- When persisted formats change, update all readers/writers and provide a one-way migration if
  existing user data needs it. Avoid indefinite dual-format support.

Cross-repository contracts may break when coordinated. The important requirement is that the
dashboard, data workspace, tests, documentation, package versions, and tags move together.

## Data Contracts

- Transaction exports use this exact column order:
  `TX Hash, Date, Qty in, Token in, Qty out, Token out, Type, Fee, Fee Token`.
- User-facing transaction timestamps are `DD/MM/YYYY HH:MM:SS`.
- Protocol history timestamps are `YYYY-MM-DD`.
- Chain configuration is private at `config/chains.json`.
- Crypto runtime state is rooted under `crypto/`; dashboard artifacts remain a deliberate read
  contract with `portfolio-dashboard`.
- Direct prices are read from the canonical core-managed `prices/` layout.

Preserve accounting meaning and deterministic output. Avoid unrelated regeneration or formatting
churn in user data.

## External Effects

EVM/RPC calls, protocol refreshes, raw snapshot generation, and rebuilds can be slow and mutate
private state. Do not run `update`, `rebuild`, extraction modules, or pipeline entry points unless
the user explicitly requests the operation. Unit tests must use mocked clients and temporary
directories.

Never expose RPC credentials, API keys, wallet-specific configuration, or private transaction
content in logs, fixtures, commits, or review output.

Both CLI commands acquire the core-owned workspace mutation lock and publish a run manifest.
Persisted outputs must use atomic replacement so dashboard readers never observe partial files.

## Release Coordination

- A core upgrade requires a new core tag, an updated dependency/source, and a regenerated lockfile.
- A crypto release requires a new tag followed by source/lock updates in `portfolio-dashboard` and
  `portfolio-data`.
- Never move a published tag; publish a new package version.

## Development

```powershell
uv sync
uv run ruff check src tests
uv run python -m pytest
uv build
```

The actual virtual environment belongs in uv's centralized cache, not in OneDrive or Git.
