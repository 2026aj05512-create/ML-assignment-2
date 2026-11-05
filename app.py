"""Streamlit app: score the saved Landsat classifiers on an uploaded test CSV.

The models are loaded from model/ and are never retrained here. The uploaded
file only needs the 36 feature columns plus the `land_cover` target column.
"""

import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import CLASS_NAMES, TARGET_COL, feature_names
from src.metrics import METRIC_NAMES, evaluate
from src.models import MODEL_NAMES, model_path

BUNDLED_TEST_FILE = Path(__file__).resolve().parent / "test_data.csv"
FEATURES = feature_names()

st.set_page_config(
    page_title="Landsat Land-Cover Classifier",
    page_icon=":satellite:",
    layout="wide",
)


# ---------------------------------------------------------------- helpers ---
@st.cache_resource
def load_model(name):
    """Load a saved pipeline (scaler + classifier) from model/."""
    return joblib.load(model_path(name))


@st.cache_data
def score_model(name, data):
    """Predict on `data` with a saved model and return everything the UI needs."""
    model = load_model(name)
    X, y = data[FEATURES], data[TARGET_COL]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "classes": list(model.classes_),
        "y_pred": y_pred,
        "metrics": evaluate(y, y_pred, y_proba, model.classes_),
    }


def read_input(uploaded_file, use_bundled):
    """Return (dataframe, source label), or (None, None) if nothing was chosen."""
    if use_bundled:
        return pd.read_csv(BUNDLED_TEST_FILE), "bundled test_data.csv"
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    return None, None


def check_input(data, known_classes):
    """Return a list of problems with the uploaded data (empty if it is fine)."""
    problems = []
    missing = [c for c in FEATURES + [TARGET_COL] if c not in data.columns]
    if missing:
        shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        problems.append(f"Missing {len(missing)} required column(s): {shown}")
        return problems
    if data[FEATURES + [TARGET_COL]].isna().any().any():
        problems.append("The file contains empty cells in the feature or target columns.")
        return problems
    unknown = sorted(set(data[TARGET_COL]) - set(known_classes))
    if unknown:
        problems.append(
            f"Unknown values in `{TARGET_COL}`: {unknown}. "
            f"Expected labels are {sorted(known_classes)}."
        )
    return problems


def class_label(code):
    return f"{code} - {CLASS_NAMES.get(code, 'unknown')}"


def plot_confusion_matrix(y_true, y_pred, classes):
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    labels = [class_label(c) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def plot_metric_comparison(comparison):
    """Grouped bar chart: one group per model, one bar per metric.

    Drawn with matplotlib so it is a fixed image: it cannot be zoomed or
    dragged by scrolling over it.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    comparison[METRIC_NAMES].plot(
        kind="bar", ax=ax, width=0.82, color=sns.color_palette("colorblind", len(METRIC_NAMES))
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_xticklabels([textwrap.fill(name, width=14) for name in comparison.index], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(METRIC_NAMES), loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    return fig


def report_table(y_true, y_pred, classes):
    report = classification_report(
        y_true, y_pred, labels=classes,
        target_names=[class_label(c) for c in classes],
        output_dict=True, zero_division=0,
    )
    table = pd.DataFrame(report).T
    # sklearn stores a single number for "accuracy", which leaves support empty.
    table.loc["accuracy", "support"] = len(y_true)
    table["support"] = table["support"].astype(int)
    return table


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Model", MODEL_NAMES)

    st.subheader("Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    use_bundled = st.checkbox("Use the bundled test_data.csv instead")
    if BUNDLED_TEST_FILE.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=BUNDLED_TEST_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    st.caption(
        f"The CSV needs the 36 feature columns and a `{TARGET_COL}` column. "
        "Models are pre-trained and are not retrained on your upload."
    )

# ------------------------------------------------------------------- main ---
st.title("Landsat Satellite Land-Cover Classifier")
st.write(
    "Each row is a 3x3 pixel neighbourhood from a satellite image (4 spectral bands x "
    "9 pixels = 36 features). The models predict the land-cover class of the central pixel."
)

data, source = read_input(uploaded, use_bundled)

if data is None:
    st.info("Upload a test CSV in the sidebar, or tick the bundled test data option, to begin.")
    with st.expander("Expected file format and classes", expanded=True):
        st.write(f"Columns: {', '.join(FEATURES[:4])}, ... , {FEATURES[-1]}, `{TARGET_COL}`")
        st.table(pd.DataFrame({"Label": list(CLASS_NAMES), "Class": list(CLASS_NAMES.values())}))
    st.stop()

classes = load_model(model_name).classes_
problems = check_input(data, classes)
if problems:
    for message in problems:
        st.error(message)
    st.stop()

st.caption(f"Scoring **{len(data):,} rows** from {source}")

# The comparison tab only makes sense once there is more than one model.
show_comparison = len(MODEL_NAMES) > 1
tab_labels = [f"{model_name} results"] + (["Compare all models"] if show_comparison else [])
tabs = st.tabs(tab_labels)
tab_selected = tabs[0]

with tab_selected:
    result = score_model(model_name, data)

    columns = st.columns(len(METRIC_NAMES))
    for column, metric in zip(columns, METRIC_NAMES):
        value = result["metrics"][metric]
        column.metric(metric, "n/a" if pd.isna(value) else f"{value:.4f}")
    if pd.isna(result["metrics"]["AUC"]):
        st.caption("AUC needs every class to appear at least once in the uploaded file.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion_matrix(data[TARGET_COL], result["y_pred"], result["classes"]))
    with right:
        st.subheader("Classification report")
        st.dataframe(
            report_table(data[TARGET_COL], result["y_pred"], result["classes"]).style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}
            ),
            width="stretch",
        )

if show_comparison:
    with tabs[1]:
        st.subheader(f"All {len(MODEL_NAMES)} models on this data")
        comparison = pd.DataFrame(
            [{"ML Model Name": name, **score_model(name, data)["metrics"]} for name in MODEL_NAMES]
        ).set_index("ML Model Name")
        st.dataframe(
            comparison.style.format("{:.4f}", na_rep="n/a").highlight_max(
                axis=0, color="#cfe8cf"
            ),
            width="stretch",
        )
        st.caption("Green marks the best score in each column.")
        st.pyplot(plot_metric_comparison(comparison))
        st.caption("The chart's y-axis starts at 0.5 so the differences between models are visible.")
