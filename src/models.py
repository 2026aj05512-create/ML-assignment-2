"""The classifiers and where their saved files live.

Every model is a scikit-learn Pipeline: StandardScaler followed by the
classifier. Saving the whole pipeline means the scaler fitted on the training
data is stored together with the model, so the app can score raw test rows
without refitting anything.

No hyper-parameter search is done. The dataset documentation asks that
cross-validation is not used, so sensible fixed settings are used instead.

To add a model: write a small factory function, add it to _REGISTRY below,
then run train.py again. The app picks it up automatically.
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SEED = 42


def _logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=SEED)


# display name -> (file name, classifier factory)
_REGISTRY = {
    "Logistic Regression": ("logistic_regression.joblib", _logistic_regression),
}

MODEL_NAMES = list(_REGISTRY)


def build_model(name):
    """Return an untrained Pipeline (scaler + classifier) for the given name."""
    _, factory = _REGISTRY[name]
    return Pipeline([("scaler", StandardScaler()), ("classifier", factory())])


def model_path(name):
    """Location of the saved pipeline for the given model name."""
    return MODEL_DIR / _REGISTRY[name][0]
