"""Compare models, tune, choose a threshold, score the test set once, and save.

Running ``python -m latedeliveries.train`` recreates reports/cv_results.csv,
reports/metrics.json, models/late_orders.joblib and models/model_info.json
from nothing.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from .data import make_orders
from .evaluation import choose_threshold, classification_metrics, policy_costs
from .pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "logistic_regression_balanced": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
}

# Chosen after comparing CANDIDATES on the training-only CV results (see
# reports/cv_results.csv and the README): logistic regression with balanced
# class weight gives recall high enough to catch late orders, at an ROC-AUC
# close to the best model, which matters more than accuracy under the
# scenario's cost rule.
CHOSEN_MODEL_NAME = "logistic_regression_balanced"

PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"],
}


def compare_models(x_train, y_train) -> pd.DataFrame:
    """Run 5-fold stratified CV for every candidate and summarise the scores."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in CANDIDATES.items():
        pipeline = build_pipeline(model)
        # The dummy baseline never predicts the positive class, which makes
        # precision/F1 undefined on some folds; that is expected, not a bug.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)


def tune_chosen_model(x_train, y_train) -> GridSearchCV:
    """Grid-search the chosen model's hyperparameters, scoring by ROC-AUC.

    ROC-AUC is used because the final decision is a threshold chosen from
    costs, not a fixed 0.5 cut-off, so the search should favour a model that
    ranks orders well across all thresholds rather than one tuned for
    accuracy at a single cut-off.
    """
    pipeline = build_pipeline(CANDIDATES[CHOSEN_MODEL_NAME])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, PARAM_GRID, scoring="roc_auc", cv=cv, refit=True, n_jobs=-1
    )
    search.fit(x_train, y_train)
    return search


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frame = make_orders()
    x_train, x_test, y_train, y_test = split_data(frame)

    cv_results = compare_models(x_train, y_train)
    cv_results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)

    search = tune_chosen_model(x_train, y_train)
    best_pipeline = search.best_estimator_

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        best_pipeline, x_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold, cost_table = choose_threshold(y_train, oof_proba)

    best_pipeline.fit(x_train, y_train)
    test_proba = best_pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    test_metrics = classification_metrics(y_test, test_pred, test_proba)
    costs = policy_costs(y_test, test_proba, threshold)

    untuned_row = cv_results.loc[
        cv_results["model"] == CHOSEN_MODEL_NAME, ["roc_auc_mean", "roc_auc_std"]
    ].iloc[0]

    metrics = {
        "threshold": threshold,
        "cost_table": cost_table,
        "test_metrics": test_metrics,
        "policy_costs": costs,
        "best_params": search.best_params_,
        "best_cv_roc_auc": search.best_score_,
        "untuned_cv_roc_auc_mean": float(untuned_row["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": float(untuned_row["roc_auc_std"]),
        "chosen_model": CHOSEN_MODEL_NAME,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    joblib.dump(best_pipeline, MODELS_DIR / "late_orders.joblib")
    model_info = {
        "sklearn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "best_params": search.best_params_,
        "chosen_model": CHOSEN_MODEL_NAME,
        "test_metrics": test_metrics,
        "policy_costs": costs,
    }
    (MODELS_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2))

    print(cv_results)
    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)
    print(
        "Untuned CV ROC-AUC:",
        untuned_row["roc_auc_mean"],
        "+/-",
        untuned_row["roc_auc_std"],
    )
    print("Chosen threshold:", threshold)
    print("Test metrics:", test_metrics)
    print("Policy costs:", costs)


if __name__ == "__main__":
    main()
