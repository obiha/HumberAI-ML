"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from latedeliveries.data import make_orders
from latedeliveries.pipeline import split_data


@pytest.fixture
def orders_frame():
    """A small, deterministic sample of orders for fast tests."""
    return make_orders(n=300, seed=7)


@pytest.fixture
def split_frame(orders_frame):
    """A train/test split of orders_frame."""
    return split_data(orders_frame)
