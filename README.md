# portfolio-crypto-data

## End-user installation

The dashboard and data workspace install the immutable `v0.2.0` package from GitHub. A source
checkout is not required. To install the CLI by itself:

```powershell
uv tool install "portfolio-crypto-data @ git+https://github.com/T1m-V/portfolio-crypto-data.git@v0.2.0"
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

The update command runs the EVM pipeline and refreshes Nexo snapshots when transaction exports
are present. All writes are rooted in the explicitly selected external data workspace.
