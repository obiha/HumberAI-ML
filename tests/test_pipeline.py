"""Tests for the feature lists, the split, and build_pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET,
    build_pipeline,
    split_data,
)


def test_feature_columns_exclude_leaky_columns():
    """order_id, prep_minutes and customer_rating must never be features."""
    for column in ["order_id", "prep_minutes", "customer_rating"]:
        assert column in EXCLUDED_COLUMNS
        assert column not in FEATURE_COLUMNS
    assert TARGET not in FEATURE_COLUMNS


def test_split_is_stratified_and_repeatable(orders_frame):
    """The late rate is preserved across the split, and the split is seeded."""
    x_train, x_test, y_train, y_test = split_data(orders_frame)
    assert abs(y_train.mean() - y_test.mean()) < 0.1

    x_train_2, x_test_2, y_train_2, y_test_2 = split_data(orders_frame)
    pd.testing.assert_frame_equal(x_train, x_train_2)
    pd.testing.assert_series_equal(y_train, y_train_2)


def test_predicted_probabilities_are_valid(split_frame):
    """predict_proba returns one valid probability per row, per class."""
    x_train, x_test, y_train, y_test = split_frame
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_missing_value_and_unseen_category(split_frame):
    """The fitted pipeline predicts for a missing number and an unseen category."""
    x_train, x_test, y_train, y_test = split_frame
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(x_train, y_train)

    odd_row = x_test.iloc[[0]].copy()
    odd_row["courier_months"] = np.nan
    odd_row["weather"] = "hail"

    proba = pipeline.predict_proba(odd_row)
    assert proba.shape == (1, 2)
    assert np.all((proba >= 0) & (proba <= 1))
