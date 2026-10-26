"""Tests that a saved pipeline predicts the same after it is reloaded."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline


def test_saved_model_predicts_same_after_reload(split_frame, tmp_path):
    x_train, x_test, y_train, y_test = split_frame
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(x_train, y_train)
    original_proba = pipeline.predict_proba(x_test)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    reloaded_proba = reloaded.predict_proba(x_test)

    assert np.array_equal(original_proba, reloaded_proba)
