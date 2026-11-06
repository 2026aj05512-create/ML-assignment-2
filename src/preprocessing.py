"""Preprocessing used by every model.

The only transformation applied to the data is feature scaling with a
StandardScaler (zero mean, unit variance per feature). No new features are
created: the 36 spectral values are already numeric, complete and on the same
0-255 scale, and the EDA showed no missing values or duplicates to clean.

The scaler is fitted on the TRAINING data only and then reused unchanged on
the test data, so no information from the test set leaks into training.
It is part of every model pipeline (see src/models.py) and is also saved on
its own as model/scaler.joblib, which is the standalone preprocessing artifact.
"""

from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler

from .data_loader import TARGET_COL, feature_names

SCALER_PATH = Path(__file__).resolve().parent.parent / "model" / "scaler.joblib"


def split_features_target(df):
    """Return (X, y): the 36 feature columns and the land-cover target."""
    return df[feature_names()], df[TARGET_COL]


def make_scaler():
    """A fresh, unfitted StandardScaler."""
    return StandardScaler()


def fit_and_save_scaler(X_train):
    """Fit the scaler on the training features and save it to model/scaler.joblib."""
    scaler = make_scaler().fit(X_train)
    SCALER_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_scaler():
    """Load the saved, already fitted scaler."""
    return joblib.load(SCALER_PATH)
