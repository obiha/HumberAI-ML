"""The cost rule, the threshold search, and test-set metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4
LATE_COST = 15

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def order_cost(y_true, flagged) -> float:
    """Return $4 per flagged order plus $15 per late order left unflagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    missed_late = np.sum((y_true == 1) & ~flagged)
    return VOUCHER_COST * np.sum(flagged) + LATE_COST * missed_late


def cost_at_threshold(y_true, y_proba, threshold: float) -> float:
    """Return the order_cost of flagging every order at or above threshold."""
    flagged = np.asarray(y_proba) >= threshold
    return order_cost(y_true, flagged)


def choose_threshold(
    y_true, y_proba, thresholds=THRESHOLD_GRID
) -> tuple[float, list[dict]]:
    """Return the threshold with the lowest cost and the full cost table."""
    table = [
        {"threshold": float(t), "cost": float(cost_at_threshold(y_true, y_proba, t))}
        for t in thresholds
    ]
    best = min(table, key=lambda row: row["cost"])
    return best["threshold"], table


def policy_costs(y_true, y_proba, threshold: float) -> dict[str, float]:
    """Return the cost of flagging nobody, flagging everybody, and the model."""
    y_true = np.asarray(y_true)
    flagged_model = np.asarray(y_proba) >= threshold
    return {
        "flag_nobody": float(order_cost(y_true, np.zeros_like(y_true, dtype=bool))),
        "flag_everybody": float(order_cost(y_true, np.ones_like(y_true, dtype=bool))),
        "model": float(order_cost(y_true, flagged_model)),
    }


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Return confusion matrix and precision/recall/F1/ROC-AUC at a threshold."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "confusion_matrix": cm.tolist(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
