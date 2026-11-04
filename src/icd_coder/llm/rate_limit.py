"""Client-side pacing for providers with a tokens-per-minute limit."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class Reservation:
    sent_at: float
    tokens: int


class TokenRateLimiter:
    """A sliding one-minute window of tokens, shared by every thread using one model."""

    def __init__(self, tokens_per_minute: int) -> None:
        self._limit = tokens_per_minute
        self._window: deque[Reservation] = deque()
        self._condition = threading.Condition()

    def reserve(self, tokens: int) -> Reservation:
        """Block until `tokens` fit in the window, then claim them."""
        with self._condition:
            while True:
                now = time.monotonic()
                while self._window and self._window[0].sent_at <= now - WINDOW_SECONDS:
                    self._window.popleft()
                in_use = sum(item.tokens for item in self._window)
                if not self._window or in_use + tokens <= self._limit:
                    reservation = Reservation(now, tokens)
                    self._window.append(reservation)
                    return reservation
                self._condition.wait(self._window[0].sent_at + WINDOW_SECONDS - now)

    def settle(self, reservation: Reservation, tokens: int) -> None:
        """Replace an estimate with the tokens the provider actually counted."""
        with self._condition:
            reservation.tokens = tokens
            self._condition.notify_all()
