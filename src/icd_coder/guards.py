"""The two limits every model call must respect: wall-clock time and money."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Deadline:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._ends_at = clock() + seconds

    def remaining(self) -> float:
        return max(0.0, self._ends_at - self._clock())

    def expired(self) -> bool:
        return self.remaining() == 0.0


class Budget:
    """Spend shared by every worker thread, in US dollars."""

    def __init__(self, limit_usd: float) -> None:
        self._limit = limit_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def can_afford(self, worst_case_usd: float) -> bool:
        with self._lock:
            return self._spent + worst_case_usd <= self._limit

    def charge(self, usd: float) -> None:
        with self._lock:
            self._spent += usd


class FailureStreak:
    """Trips after too many provider failures in a row, e.g. an exhausted key."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._count = 0
        self._lock = threading.Lock()

    def record(self, *, succeeded: bool) -> None:
        with self._lock:
            self._count = 0 if succeeded else self._count + 1

    def tripped(self) -> bool:
        with self._lock:
            return self._count >= self._limit
