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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SEED = 42


def _logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=SEED)


def _decision_tree():
    # Default settings: the tree grows until every leaf is pure. random_state
    # fixes the tie-breaking between equally good splits so results repeat.
    return DecisionTreeClassifier(random_state=SEED)


def _knn():
    # k = 5 is the usual starting point. KNN is distance based, which is why
    # the scaler in the pipeline matters for this model.
    return KNeighborsClassifier(n_neighbors=5)


def _naive_bayes():
    # Gaussian variant: the features are continuous pixel intensities, so each
    # feature is modelled as a normal distribution within every class.
    return GaussianNB()


def _svm():
    # RBF kernel with scikit-learn's default C and gamma. SVMs are sensitive to
    # feature scale, so the scaler in the pipeline matters here too.
    # probability=True is needed so that AUC can be calculated.
    return SVC(kernel="rbf", probability=True, random_state=SEED)


# display name -> (file name, classifier factory)
_REGISTRY = {
    "Logistic Regression": ("logistic_regression.joblib", _logistic_regression),
    "Decision Tree": ("decision_tree.joblib", _decision_tree),
    "KNN": ("knn.joblib", _knn),
    "Naive Bayes": ("naive_bayes.joblib", _naive_bayes),
    "Support Vector Machine": ("svm.joblib", _svm),
}

MODEL_NAMES = list(_REGISTRY)


def build_model(name):
    """Return an untrained Pipeline (scaler + classifier) for the given name."""
    _, factory = _REGISTRY[name]
    return Pipeline([("scaler", StandardScaler()), ("classifier", factory())])


def model_path(name):
    """Location of the saved pipeline for the given model name."""
    return MODEL_DIR / _REGISTRY[name][0]
