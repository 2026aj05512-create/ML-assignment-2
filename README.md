# Landsat Satellite Land-Cover Classification

ML Assignment 2 - M.Tech (AIML), BITS Pilani WILP

Five classification models trained on the UCI Statlog (Landsat Satellite) dataset,
compared on six evaluation metrics, and served through a Streamlit web app.

> Work in progress. Sections below are filled in as the project is built.

## a. Problem statement

Given the multi-spectral readings of a 3x3 pixel neighbourhood in a satellite image,
predict the land-cover class of the central pixel (for example red soil, cotton crop or
grey soil). This is a **multi-class classification** problem with six classes.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian
Naive Bayes and Support Vector Machine) are trained on the same data, compared on six
metrics, and exposed through a Streamlit app where a test CSV can be uploaded and scored
with any of the saved models.

## b. Dataset description

**Source:** UCI Machine Learning Repository - *Statlog (Landsat Satellite)*
(https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite).
Generated from Landsat Multi-Spectral Scanner imagery.

| Property | Value |
|---|---|
| Task | Multi-class classification (6 classes) |
| Instances | 6,435 in total: 4,435 training + 2,000 test (official split) |
| Features | 36 numeric (4 spectral bands x 9 pixels of a 3x3 neighbourhood) |
| Feature range | 8-bit pixel intensity (0-255); observed 27-157 |
| Missing values | None |
| Duplicate rows | None |
| Target | Land-cover class of the central pixel |

**Classes** (train / test rows):

| Label | Class | Train | Test |
|---|---|---|---|
| 1 | red soil | 1,072 | 461 |
| 2 | cotton crop | 479 | 224 |
| 3 | grey soil | 961 | 397 |
| 4 | damp grey soil | 415 | 211 |
| 5 | soil with vegetation stubble | 470 | 237 |
| 7 | very damp grey soil | 1,038 | 470 |

Label 6 ("mixture class") exists in the original documentation but has no examples in the
data, so only six labels appear.

**Feature naming:** each feature is called `<pixel position>_band<1-4>`, for example
`centre_band3`. Pixel positions run left to right, top to bottom: `top_left`, `top_mid`,
`top_right`, `mid_left`, `centre`, `mid_right`, `bottom_left`, `bottom_mid`,
`bottom_right`.

**Evaluation protocol:** the dataset documentation asks that cross-validation is not
used, so models are trained once on `data/sat.trn` and evaluated once on the official test
split (`data/sat.tst`, also provided as `test_data.csv`).

Exploration notes are in `notebooks/01_eda.ipynb`.

## c. GitHub repository link

_To be added._

## d. Models used

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner for this dataset** | |

## How the metrics are calculated

All six metrics are computed on the official test split. Because there are six classes,
Precision, Recall and F1 are **macro-averaged** (each class counts equally), AUC is
**one-vs-rest macro-averaged**, and MCC is the multi-class version. Every model is a
scikit-learn pipeline of `StandardScaler` followed by the classifier, with fixed
settings (no hyper-parameter search, in line with the dataset's no-cross-validation note).

## How to run

```
pip install -r requirements.txt
python prepare_data.py     # writes test_data.csv from data/sat.tst
python train.py            # trains the 5 models into model/ and writes results/metrics_comparison.csv
streamlit run app.py       # launches the app
```

The saved models in `model/` were trained with the scikit-learn version pinned in
`requirements.txt`. Loading them with a different version can fail, so keep the pin in sync.

## Project structure

```
ML-assignment-2/
|-- app.py                  Streamlit app
|-- train.py                Trains and saves the five models, writes the results table
|-- prepare_data.py         Writes test_data.csv from the official test split
|-- requirements.txt
|-- README.md
|-- test_data.csv           Test data used in the experiments (upload this in the app)
|-- data/                   Raw dataset files (sat.trn, sat.tst, sat.doc)
|-- model/                  Saved trained pipelines (.joblib), one per classifier
|-- results/                metrics_comparison.csv written by train.py
|-- notebooks/              Exploratory analysis notebook
|-- src/                    data_loader.py, models.py, metrics.py
|-- .streamlit/config.toml  App theme
```
