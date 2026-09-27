from app.providers.rate_limit import (
    DEFAULT_COOLDOWN_SECONDS,
    WINDOW_SECONDS,
    ProviderRateLimiter,
)


def test_allows_up_to_rpm_then_reports_wait(fake_clock) -> None:
    limiter = ProviderRateLimiter(2, clock=fake_clock)
    assert limiter.try_acquire() is None
    assert limiter.try_acquire() is None
    assert limiter.try_acquire() == WINDOW_SECONDS


def test_slot_frees_when_oldest_request_leaves_window(fake_clock) -> None:
    limiter = ProviderRateLimiter(1, clock=fake_clock)
    limiter.try_acquire()
    fake_clock.now = WINDOW_SECONDS - 1
    assert limiter.try_acquire() == 1
    fake_clock.now = WINDOW_SECONDS
    assert limiter.try_acquire() is None


def test_refused_request_does_not_consume_a_slot(fake_clock) -> None:
    limiter = ProviderRateLimiter(1, clock=fake_clock)
    limiter.try_acquire()
    for _ in range(5):
        limiter.try_acquire()
    fake_clock.now = WINDOW_SECONDS
    assert limiter.try_acquire() is None


def test_cooldown_blocks_until_it_expires(fake_clock) -> None:
    limiter = ProviderRateLimiter(100, clock=fake_clock)
    limiter.start_cooldown(40)
    fake_clock.now = 30
    assert limiter.try_acquire() == 10
    fake_clock.now = 40
    assert limiter.try_acquire() is None


def test_cooldown_without_hint_uses_default(fake_clock) -> None:
    limiter = ProviderRateLimiter(100, clock=fake_clock)
    limiter.start_cooldown(None)
    assert limiter.try_acquire() == DEFAULT_COOLDOWN_SECONDS


def test_shorter_cooldown_does_not_shorten_existing_one(fake_clock) -> None:
    limiter = ProviderRateLimiter(100, clock=fake_clock)
    limiter.start_cooldown(40)
    limiter.start_cooldown(5)
    assert limiter.try_acquire() == 40
