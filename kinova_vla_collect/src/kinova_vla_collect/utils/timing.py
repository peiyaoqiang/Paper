from __future__ import annotations

import time
from dataclasses import dataclass


def now_ns() -> int:
    return time.time_ns()


def monotonic_ns() -> int:
    return time.monotonic_ns()


@dataclass
class RateLimiter:
    hz: float

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ValueError("RateLimiter hz must be positive")
        self.period_s = 1.0 / self.hz
        self._next_time = time.monotonic()

    def sleep(self) -> None:
        self._next_time += self.period_s
        remaining = self._next_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self._next_time = time.monotonic()
