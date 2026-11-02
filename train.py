"""Train the five classifiers once on the training split and save them.

Each model is then evaluated once on the official test split and the numbers
are written to results/metrics_comparison.csv.

Run from the project root:  python train.py
"""

from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import TARGET_COL, feature_names, load_test, load_train
from src.metrics import METRIC_NAMES, evaluate
from src.models import MODEL_DIR, MODEL_NAMES, build_model, model_path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    train, test = load_train(), load_test()
    features = feature_names()
    X_train, y_train = train[features], train[TARGET_COL]
    X_test, y_test = test[features], test[TARGET_COL]

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name in MODEL_NAMES:
        print(f"Training {name} ...")
        model = build_model(name)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path(name))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        scores = evaluate(y_test, y_pred, y_proba, model.classes_)
        rows.append({"ML Model Name": name, **scores})

    results = pd.DataFrame(rows, columns=["ML Model Name"] + METRIC_NAMES)
    results.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)

    print("\nTest-set results")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
