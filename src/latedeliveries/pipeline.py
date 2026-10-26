"""Feature lists, the train/test split, and the modelling pipeline.

Only columns whose value is known at the moment an order is placed are used
as features. ``prep_minutes`` is set while the kitchen is working on the
order, after it has been placed, and ``customer_rating`` is given by the
customer after the order has already arrived (the simulation even derives it
from ``late``), so both are excluded as leakage. ``order_id`` is only an
identifier.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 7

TARGET = "late"

# Known at the moment the order is placed.
NUMERIC_FEATURES = ["distance_km", "items", "hour", "courier_months"]
CATEGORICAL_FEATURES = ["weather", "day_type", "vehicle"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Left out on purpose: order_id is an identifier, prep_minutes is only known
# once the kitchen has finished preparing the order, and customer_rating is
# only known after the order has already arrived.
EXCLUDED_COLUMNS = ["order_id", "prep_minutes", "customer_rating"]


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a stratified 75/25 split of the feature frame and target."""
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET]
    return train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Return a Pipeline that preprocesses FEATURE_COLUMNS and fits model."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
