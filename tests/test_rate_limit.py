import threading
import time

from icd_coder.llm.rate_limit import TokenRateLimiter


def test_reservations_within_the_limit_do_not_wait() -> None:
    limiter = TokenRateLimiter(1_000)
    started = time.monotonic()
    for _ in range(4):
        limiter.reserve(250)
    assert time.monotonic() - started < 0.1


def test_an_oversized_request_passes_when_the_window_is_empty() -> None:
    limiter = TokenRateLimiter(100)
    reservation = limiter.reserve(5_000)
    assert reservation.tokens == 5_000


def test_a_full_window_blocks_the_next_request() -> None:
    limiter = TokenRateLimiter(100)
    limiter.reserve(100)
    finished = threading.Event()
    waiter = threading.Thread(target=lambda: (limiter.reserve(10), finished.set()), daemon=True)
    waiter.start()
    assert not finished.wait(0.2)


def test_settling_records_the_actual_tokens() -> None:
    limiter = TokenRateLimiter(100)
    reservation = limiter.reserve(90)
    limiter.settle(reservation, 20)
    started = time.monotonic()
    limiter.reserve(70)
    assert time.monotonic() - started < 0.1
