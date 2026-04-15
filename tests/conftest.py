"""Shared pytest fixtures for MEDAL tests."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_X(rng):
    """200 samples, 20 features — small enough for fast CPU tests."""
    return rng.standard_normal((200, 20)).astype(np.float32)


@pytest.fixture
def small_Z(rng):
    """Synthetic 2-D teacher embedding for 200 samples."""
    return rng.standard_normal((200, 2)).astype(np.float32)
