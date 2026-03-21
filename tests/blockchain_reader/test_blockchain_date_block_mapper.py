from blockchain_reader.date_block_mapper import get_block_by_timestamp


class FakeEth:
    def __init__(self, ts_by_block: dict[int, int]):
        self.ts_by_block = ts_by_block

    def get_block(self, block_identifier: int) -> dict[str, int]:
        return {"timestamp": self.ts_by_block[block_identifier]}


class FakeWeb3:
    def __init__(self, ts_by_block: dict[int, int]):
        self.eth = FakeEth(ts_by_block)


def test_get_block_by_timestamp_exact_match() -> None:
    w3 = FakeWeb3({0: 0, 1: 10, 2: 20, 3: 30, 4: 40})
    block = get_block_by_timestamp(w3=w3, target_ts=20, low=0, high=4)
    assert block == 2


def test_get_block_by_timestamp_returns_first_block_after_target() -> None:
    w3 = FakeWeb3({0: 0, 1: 10, 2: 20, 3: 30, 4: 40})
    block = get_block_by_timestamp(w3=w3, target_ts=25, low=0, high=4)
    assert block == 3
