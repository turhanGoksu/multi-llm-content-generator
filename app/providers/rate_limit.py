"""Per-provider client-side rate limiting."""

import time
from collections import deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0

# Cooldown applied after a 429 that did not say how long to wait.
DEFAULT_COOLDOWN_SECONDS = 10.0


class ProviderRateLimiter:
    """Sliding-window requests-per-minute limit, plus a cooldown after a 429.

    Never sleeps: `try_acquire` either reserves a slot immediately or reports
    how long until one frees up, so callers can fail fast instead of blocking
    a request. Safe without a lock because it never awaits, so no other
    coroutine can run between its check and its update.
    """

    def __init__(
        self, requests_per_minute: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rpm = requests_per_minute
        self._clock = clock
        self._sent_at: deque[float] = deque()
        self._cooldown_until = 0.0

    def try_acquire(self) -> float | None:
        """Reserve a request slot now.

        Returns None if the request may be sent, or the number of seconds
        until it could be sent (the slot is not reserved in that case).
        """
        now = self._clock()
        if now < self._cooldown_until:
            return self._cooldown_until - now
        while self._sent_at and now - self._sent_at[0] >= WINDOW_SECONDS:
            self._sent_at.popleft()
        if len(self._sent_at) >= self._rpm:
            return WINDOW_SECONDS - (now - self._sent_at[0])
        self._sent_at.append(now)
        return None

    def start_cooldown(self, seconds: float | None) -> None:
        """Block requests for `seconds` (or the default) after a provider 429."""
        duration = seconds if seconds is not None else DEFAULT_COOLDOWN_SECONDS
        self._cooldown_until = max(self._cooldown_until, self._clock() + duration)
