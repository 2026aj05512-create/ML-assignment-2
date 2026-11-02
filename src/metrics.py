"""Evaluation metrics shared by train.py and the Streamlit app.

The problem has six classes, so precision, recall and F1 are macro-averaged
(every class counts equally, which matters because the classes are not the
same size). AUC is one-vs-rest, also macro-averaged.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _auc_ovr(y_true, y_proba, classes):
    """One-vs-rest macro AUC, or NaN when it cannot be computed.

    AUC needs at least one positive row for every class, so a small uploaded
    file that is missing a class cannot be scored.
    """
    try:
        return roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        )
    except ValueError:
        return float("nan")


def evaluate(y_true, y_pred, y_proba, classes):
    """Return the six assignment metrics as a dict keyed by METRIC_NAMES."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": _auc_ovr(y_true, y_proba, np.asarray(classes)),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
