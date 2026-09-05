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

## EVM chain configuration

The EVM pipeline processes every enabled entry in the workspace's private
`config/chains.json`. Chain-specific behavior belongs in configuration and token metadata rather
than Python constants. For example:

```json
{
  "arbitrum": {
    "enabled": true,
    "chain_id": "42161",
    "my_address": "0x...",
    "rpc_url": "https://...",
    "api_url": "https://api.etherscan.io/v2/api",
    "api_key": "...",
    "native_symbol": "ETH",
    "native_decimals": 18,
    "explorer_include_chain_id": true,
    "protocols": ["beefy", "balancer", "curve", "aave", "liquid_staking"]
  },
  "another_evm_chain": {
    "enabled": true,
    "chain_id": "1234",
    "my_address": "0x...",
    "rpc_url": "https://...",
    "api_url": "https://explorer.example/api",
    "api_key": "...",
    "native_symbol": "COIN",
    "native_decimals": 18,
    "explorer_include_chain_id": false,
    "protocols": []
  }
}
```

`explorer_include_chain_id` selects between the Etherscan V2 shape and chain-specific compatible
endpoints. `protocols` is the per-chain capability list; an empty list still runs transaction,
snapshot, accounting, and dashboard stages.

Aave wrapper metadata must identify its underlying price asset with `price_source` or `family`.
Nonstandard debt-wrapper names must also use `position_type: "debt"`; standard `variableDebt` and
`stableDebt` names remain recognizable. Liquid-staking metadata uses:

```json
{
  "symbol": "wrappedLST",
  "protocol": "liquid_staking",
  "underlying_symbol": "ETH",
  "rate_provider_address": "0x...",
  "rate_provider_method": "getRate",
  "rate_scale": 1000000000000000000
}
```

The method and scale are optional and default to `getRate` and `10^18`. Generated files remain
separate per chain, using the chain name in transaction, snapshot, protocol, accounting, and
dashboard paths.

Crypto.com App snapshots include crypto held in Earn, staking, and Supercharger; transfers
between these products and the app wallet do not change total holdings or principal. Rewards
increase holdings without adding invested principal, and reward reversals undo only quantity.
Fiat purchases add the received crypto; fiat sales, card top-ups, and outgoing peer transfers
remove crypto without creating a fiat or recipient balance. Incoming and outgoing legacy
`crypto_transfer` rows follow the sign of the exported amount.

Dust conversions combine one credit with one or more debits sharing its description within
one second. Missing or ambiguous matches and unknown transaction types stop ingestion.
