"""All the machine-learning code for the Landsat land-cover project, in one file.

Sections: data loading, EDA, preprocessing, the five models, evaluation metrics, training.

Run from the project root:

    python src/core.py

That prints a short EDA summary, writes test_data.csv, trains the five models once on the
training split, saves them to model/, and prints the test-set results. app.py imports the
loading, model and metric helpers from here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
SEED = 42


# =============================================================================
# 1. DATA LOADING
# The raw files (data/sat.trn, data/sat.tst) are space separated with no header:
# 36 spectral values followed by the class of the central pixel. Each row is a 3x3
# pixel neighbourhood, 4 spectral bands per pixel (9 x 4 = 36 features).
# =============================================================================

TARGET_COL = "land_cover"

# Pixels are stored left to right, top to bottom (see data/sat.doc).
PIXEL_POSITIONS = [
    "top_left", "top_mid", "top_right",
    "mid_left", "centre", "mid_right",
    "bottom_left", "bottom_mid", "bottom_right",
]

# Label 6 ("mixture class") exists in the documentation but has no rows in the data.
CLASS_NAMES = {
    1: "red soil",
    2: "cotton crop",
    3: "grey soil",
    4: "damp grey soil",
    5: "soil with vegetation stubble",
    7: "very damp grey soil",
}


def feature_names():
    """The 36 feature names, e.g. 'centre_band3'."""
    return [f"{pos}_band{band}" for pos in PIXEL_POSITIONS for band in range(1, 5)]


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=feature_names() + [TARGET_COL])


def load_train():
    """Official training split (4435 rows)."""
    return _read(DATA_DIR / "sat.trn")


def load_test():
    """Official test split (2000 rows)."""
    return _read(DATA_DIR / "sat.tst")


def split_features_target(df):
    """Return (X, y): the 36 feature columns and the land-cover target."""
    return df[feature_names()], df[TARGET_COL]


def export_test_csv():
    """Write the official test split to test_data.csv (the file uploaded in the app)."""
    test = load_test()
    test.to_csv(TEST_CSV, index=False)
    print(f"Wrote {TEST_CSV.name}: {test.shape[0]} rows, {test.shape[1]} columns")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def eda():
    """Print a short summary of the data."""
    train, test = load_train(), load_test()
    X = train[feature_names()]

    print("== EDA ==")
    print(f"Train: {train.shape}   Test: {test.shape}")
    print(f"Missing values  train/test: {int(train.isna().sum().sum())}/{int(test.isna().sum().sum())}")
    print(f"Duplicate rows  train/test: {int(train.duplicated().sum())}/{int(test.duplicated().sum())}")
    print(f"Feature values range from {X.values.min()} to {X.values.max()}")

    counts = pd.DataFrame({
        "class": pd.Series(CLASS_NAMES),
        "train": train[TARGET_COL].value_counts().sort_index(),
        "test": test[TARGET_COL].value_counts().sort_index(),
    })
    counts["train %"] = (100 * counts["train"] / counts["train"].sum()).round(1)
    counts["test %"] = (100 * counts["test"] / counts["test"].sum()).round(1)
    print("\nClass distribution:")
    print(counts.to_string())

    corr = X.corr().values
    pairs = corr[np.triu_indices_from(corr, k=1)]
    print(f"\nHighest correlation between two features: {pairs.max():.3f}")
    print(f"Share of feature pairs with correlation above 0.9: {100 * (pairs > 0.9).mean():.1f}%")

    centre = [c for c in feature_names() if c.startswith("centre_")]
    profile = train.groupby(TARGET_COL)[centre].mean().round(1)
    profile.index = [f"{i} {CLASS_NAMES[i]}" for i in profile.index]
    print("\nMean centre-pixel value per class:")
    print(profile.to_string())


# =============================================================================
# 3. PREPROCESSING
# The only step is standard scaling (mean 0, standard deviation 1 per feature), fitted on
# the training data only. No new features are created: all 36 inputs are already numeric,
# complete and on the same 0-255 scale. Scaling matters for Logistic Regression (accuracy
# falls from 83.95% to 79.45% without it) and for SVM; it makes little difference to the
# Decision Tree, KNN and Naive Bayes, but every model uses it so they are treated alike.
# The scaler is part of each saved pipeline, so the app scales uploaded data automatically.
# =============================================================================

def build_pipeline(classifier):
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


# =============================================================================
# 4. MODELS
# Fixed, sensible settings. The dataset documentation asks for a single train/test run
# with no cross-validation, so there is no hyper-parameter search.
# =============================================================================

def logistic_regression():
    # A linear model that gives a probability for each class.
    return LogisticRegression(max_iter=1000, random_state=SEED)


def decision_tree():
    # Defaults: grows until every leaf is pure (so it fits the training data perfectly).
    return DecisionTreeClassifier(random_state=SEED)


def knn():
    # Majority vote of the 5 closest training rows.
    return KNeighborsClassifier(n_neighbors=5)


def naive_bayes():
    # Gaussian variant: each feature is modelled as a normal distribution within each class.
    return GaussianNB()


def svm():
    # RBF kernel, default C and gamma. probability=True is needed to calculate AUC.
    return SVC(kernel="rbf", probability=True, random_state=SEED)


# display name -> (saved file name, function that creates the classifier)
MODELS = {
    "Logistic Regression": ("logistic_regression.joblib", logistic_regression),
    "Decision Tree": ("decision_tree.joblib", decision_tree),
    "KNN": ("knn.joblib", knn),
    "Naive Bayes": ("naive_bayes.joblib", naive_bayes),
    "Support Vector Machine": ("svm.joblib", svm),
}
MODEL_NAMES = list(MODELS)


def model_path(name, model_dir=MODEL_DIR):
    return Path(model_dir) / MODELS[name][0]


# =============================================================================
# 5. EVALUATION METRICS
# Six classes, so Precision, Recall and F1 are macro-averaged (every class counts equally)
# and AUC is one-vs-rest, macro-averaged.
# =============================================================================

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _auc_ovr(y_true, y_proba, classes):
    # AUC needs at least one row of every class, so a small uploaded file can give NaN.
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
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


# =============================================================================
# 6. TRAINING
# Each model is trained once on the training split, saved, then evaluated once on the
# official test split.
# =============================================================================

def train_all(model_dir=MODEL_DIR):
    """Train and save every model, and return the test-set metrics as a DataFrame."""
    X_train, y_train = split_features_target(load_train())
    X_test, y_test = split_features_target(load_test())
    Path(model_dir).mkdir(exist_ok=True)

    rows = []
    for name, (_, make_classifier) in MODELS.items():
        print(f"Training {name} ...")
        model = build_pipeline(make_classifier())
        model.fit(X_train, y_train)
        joblib.dump(model, model_path(name, model_dir))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        rows.append({"ML Model Name": name, **evaluate(y_test, y_pred, y_proba, model.classes_)})

    return pd.DataFrame(rows, columns=["ML Model Name"] + METRIC_NAMES)


def main():
    eda()
    print()
    export_test_csv()
    print()
    results = train_all()
    print("\nTest-set results")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
