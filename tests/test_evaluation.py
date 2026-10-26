"""Tests for the cost rule and the threshold search.

Each example here is worked out by hand first: a voucher always costs $4,
and every late order that is not flagged costs $15, on top of any voucher.
"""

from __future__ import annotations

import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_of_a_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


@pytest.mark.parametrize(
    "y_true, flagged, expected",
    [
        # Nobody flagged, nobody late: no vouchers, no refunds.
        ([0, 0, 0], [False, False, False], 0),
        # Everybody flagged, nobody late: three vouchers, no refunds.
        ([0, 0, 0], [True, True, True], 3 * 4),
        # Everybody flagged and everybody late: vouchers only, no refunds.
        ([1, 1, 1], [True, True, True], 3 * 4),
        # Nobody flagged, two of three late: two missed refunds.
        ([1, 1, 0], [False, False, False], 2 * 15),
    ],
)
def test_order_cost_parametrized(y_true, flagged, expected):
    assert order_cost(y_true, flagged) == expected


def test_threshold_flag_nobody_is_cheapest():
    """When nothing is actually late, the cheapest policy flags nobody."""
    y_true = [0, 0, 0, 0, 0]
    y_proba = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold, table = choose_threshold(y_true, y_proba)
    best_cost = min(row["cost"] for row in table)
    assert best_cost == 0
    assert order_cost(y_true, [p >= threshold for p in y_proba]) == best_cost


def test_threshold_flag_everybody_is_cheapest():
    """When everything is late, flagging everybody beats missing refunds."""
    y_true = [1, 1, 1, 1, 1]
    y_proba = [0.1, 0.3, 0.5, 0.7, 0.9]
    threshold, table = choose_threshold(y_true, y_proba)
    flagged = [p >= threshold for p in y_proba]
    assert all(flagged)
    assert order_cost(y_true, flagged) == 5 * 4
