# portfolio-crypto-data

```powershell
uv sync
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data update
uv run portfolio-crypto --data-dir C:\path\to\portfolio-data rebuild
```

The update command runs the EVM pipeline and refreshes Nexo snapshots when transaction exports
are present. All writes are rooted in the explicitly selected external data workspace.
