import pytest

from tests.helpers import FakeClock


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
