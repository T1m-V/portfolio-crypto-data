from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineLogger:
    progress_interval: int = 25
    enabled: bool = True

    def info(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def stage_start(self, stage: str) -> None:
        self.info(f"[pipeline] Starting {stage}")

    def stage_end(self, stage: str, *, errors: bool = False) -> None:
        status = "failed" if errors else "done"
        self.info(f"[pipeline] {stage} {status}")

    def protocol_start(self, protocol: str, symbol: str, start_date: str, end_date: str) -> None:
        self.info(f"[{protocol}] Processing {symbol} ({start_date} -> {end_date})")

    def protocol_end(self, protocol: str, symbol: str, output: object | None = None) -> None:
        suffix = f": {output}" if output else ""
        self.info(f"[{protocol}] Finished {symbol}{suffix}")

    def protocol_skip(self, protocol: str, symbol: str, reason: str) -> None:
        self.info(f"[{protocol}] Skipping {symbol}: {reason}")

    def protocol_day(
        self,
        protocol: str,
        symbol: str,
        *,
        date_str: str,
        block_number: int,
        day_index: int,
        total_days: int,
    ) -> None:
        if day_index == 1 or day_index == total_days or day_index % self.progress_interval == 0:
            self.info(
                f"[{protocol}] {symbol}: day {day_index}/{total_days} "
                f"{date_str} block={block_number}"
            )


NULL_LOGGER = PipelineLogger(enabled=False)
