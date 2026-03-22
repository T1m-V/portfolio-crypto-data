import json
import threading
from pathlib import Path
from typing import Any

from web3 import Web3

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]


class TokenManager:
    """
    Manages token metadata with caching and thread safety.

    args:
        token_path: Path to the JSON cache file.
        w3: Web3 instance.
    """

    def __init__(self, token_path: Path, w3: Web3, flush_every: int = 25):
        self.path = token_path
        self.w3 = w3
        self.lock = threading.Lock()  # Mutex for thread safety
        self.flush_every = flush_every
        self.pending_writes = 0
        self.cache: dict[str, Any] = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        """
        Loads the token cache from disk.

        returns:
            Dictionary of cached token data.
        """
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Error reading token DB: {e}")
        return {"native": {"symbol": "ETH", "decimals": 18, "resolved": True}}

    def _save_cache(self) -> None:
        """
        Saves the current cache to disk. Must be called inside a lock.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            print(f"[!] Error saving token DB: {e}")

    def _store_token(self, address: str, token_data: dict[str, Any]) -> dict[str, Any]:
        self.cache[address] = token_data
        self.pending_writes += 1
        if self.pending_writes >= self.flush_every:
            self._save_cache()
            self.pending_writes = 0
        return token_data

    def flush(self) -> None:
        with self.lock:
            if self.pending_writes > 0:
                self._save_cache()
                self.pending_writes = 0

    def get_token(self, address: str, fetch_if_missing: bool = False) -> dict[str, Any] | None:
        """
        Retrieves token metadata, optionally fetching from chain.

        args:
            address: Contract address of the token.
            fetch_if_missing: Whether to query the chain if not cached.

        returns:
            Dictionary of token metadata or None.
        """
        addr_lower = address.lower()

        # 1. Fast Read (Lock-free optimization)
        if addr_lower in self.cache:
            return self.cache[addr_lower]

        # 2. If we aren't allowed to fetch, return None immediately
        if not fetch_if_missing:
            return None

        # 3. Fetch from Chain (Thread-Safe Section)
        with self.lock:
            if addr_lower in self.cache:
                return self.cache[addr_lower]

            try:
                checksum_addr = Web3.to_checksum_address(address)
                contract = self.w3.eth.contract(address=checksum_addr, abi=ERC20_ABI)
                symbol = contract.functions.symbol().call()
                decimals = contract.functions.decimals().call()

                token_data = {"symbol": symbol, "decimals": decimals, "resolved": True}
                return self._store_token(address=addr_lower, token_data=token_data)

            except Exception:
                # Fallback for errors (e.g., non-standard ERC20 or network blip)
                fallback = {"symbol": f"UNK-{address[:4]}", "decimals": 18, "resolved": False}
                return self._store_token(address=addr_lower, token_data=fallback)
