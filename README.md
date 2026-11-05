# Landsat Satellite Land-Cover Classification

ML Assignment 2 - M.Tech (AIML), BITS Pilani WILP. Classification models compared on a
satellite-imagery dataset and deployed as a Streamlit web app.

## a. Problem statement

Land-cover maps (which parts of a region are bare soil, crops, damp ground and so on) are
normally built by combining satellite imagery with ground surveys. Surveys are slow and
costly, so being able to label land automatically from the imagery itself is valuable for
tasks such as agricultural monitoring and land management.

This project tackles that as a supervised classification problem. A Landsat satellite
records every pixel of an image in four spectral bands. For a 3x3 neighbourhood of pixels
(9 pixels x 4 bands = 36 numbers), the goal is to predict the **land-cover class of the
central pixel**. The neighbouring pixels are included because they give spatial context
around the pixel being labelled.

- **Target:** `land_cover`, one of six classes (red soil, cotton crop, grey soil, damp grey
  soil, soil with vegetation stubble, very damp grey soil).
- **Problem type:** multi-class classification.
- **Class balance:** moderately imbalanced. In the training data the largest class
  (red soil) is 24.2% of the rows and the smallest (damp grey soil) is 9.4%.
- **Objective:** train five different classifiers on the same data, evaluate each on a
  held-out test set using six metrics, compare them, and make them usable through a
  Streamlit app where a test CSV can be uploaded and scored.

## b. Dataset description

**Source:** UCI Machine Learning Repository, *Statlog (Landsat Satellite)*
(https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite). The data comes from
Landsat Multi-Spectral Scanner imagery of a small 82 x 100 pixel area, with each pixel
covering roughly 80 m x 80 m on the ground. Class labels come from an on-site visit.

### Dataset overview

| Property | Value |
|---|---|
| Task | Multi-class classification (6 classes) |
| Total instances | 6,435 |
| Training instances | 4,435 (`data/sat.trn`, official split) |
| Test instances | 2,000 (`data/sat.tst`, official split, also saved as `test_data.csv`) |
| Input features | 36 numeric |
| Target | `land_cover` (class of the central pixel) |
| Feature range | 8-bit pixel intensity (0-255); observed values 27 to 157 |
| Missing values | None |
| Duplicate rows | None |

### Classes

| Label | Class | Train rows | Train % | Test rows | Test % |
|---|---|---|---|---|---|
| 1 | red soil | 1,072 | 24.2 | 461 | 23.0 |
| 2 | cotton crop | 479 | 10.8 | 224 | 11.2 |
| 3 | grey soil | 961 | 21.7 | 397 | 19.8 |
| 4 | damp grey soil | 415 | 9.4 | 211 | 10.6 |
| 5 | soil with vegetation stubble | 470 | 10.6 | 237 | 11.8 |
| 7 | very damp grey soil | 1,038 | 23.4 | 470 | 23.5 |

Label 6 ("mixture class") appears in the original documentation but has no examples in the
data, so only six labels are present. The test set has nearly the same class proportions as
the training set (within about two percentage points).

### Features

Each row is one 3x3 pixel neighbourhood. Every pixel contributes four spectral band
values (two visible and two near-infrared bands, labelled `band1` to `band4`), giving
9 x 4 = 36 features. Features are named `<pixel position>_band<n>`, for example
`centre_band3`.

| Pixel position in the 3x3 grid | Feature names |
|---|---|
| top_left, top_mid, top_right | `top_left_band1` ... `top_right_band4` (12 features) |
| mid_left, centre, mid_right | `mid_left_band1` ... `mid_right_band4` (12 features) |
| bottom_left, bottom_mid, bottom_right | `bottom_left_band1` ... `bottom_right_band4` (12 features) |

The four `centre_band1` to `centre_band4` features describe the pixel that is actually
being classified. The other 32 describe its surroundings.

Findings from the exploratory analysis (`notebooks/01_eda.ipynb`):

- The classes have clearly different average centre-pixel profiles. For example cotton crop
  has a very low band 2 and the highest band 4.
- Damp grey soil and very damp grey soil have similar profiles, which makes them the
  hardest pair to separate.
- Neighbouring pixels in the same band are strongly correlated (highest pairwise
  correlation 0.959; 5.1% of feature pairs are above 0.9), so the features are far from
  independent.

### Data preprocessing

1. **Official split used as given.** The dataset already comes with a training file and a
   test file, so the data is not re-split.
2. **No cleaning needed.** There are no missing values and no duplicate rows.
3. **Feature scaling.** A `StandardScaler` is fitted on the training data only and then
   applied to the test data. It is part of each model's pipeline, so the saved models
   scale new data automatically and nothing is refitted in the app.
4. **Labels kept as they are** (1, 2, 3, 4, 5, 7), with human-readable names added for the
   app and the reports.
5. **No cross-validation.** The dataset documentation asks for a single train/test run,
   so each model is trained once on the training file and evaluated once on the test file.
   No hyper-parameter search is done.
6. **Reproducibility.** `random_state=42` is used wherever a model has randomness.

## c. GitHub repository link

https://github.com/2026aj05512-create/ML-assignment-2

Live Streamlit app: _link to be added after deployment._

### Main project files

| File or folder | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `train.py` | Trains the five models once on the training split, saves them, and writes the results table |
| `prepare_data.py` | Writes `test_data.csv` from the official test split |
| `test_data.csv` | Test data used in the experiments; upload this file in the app |
| `src/data_loader.py` | Loads the raw dataset files and names the columns |
| `src/models.py` | Defines the five classifiers and where each saved model lives |
| `src/metrics.py` | Calculates the six evaluation metrics (shared by `train.py` and the app) |
| `model/` | Saved trained pipelines (`.joblib`), one per classifier |
| `results/metrics_comparison.csv` | Test-set metrics for all models, written by `train.py` |
| `data/` | Raw dataset files (`sat.trn`, `sat.tst`) and the original documentation (`sat.doc`) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `.streamlit/config.toml` | App theme |
| `requirements.txt` | Python dependencies |

### What the app does

- Upload a test CSV, or tick the option to use the bundled `test_data.csv`.
- Choose a model from a dropdown. The saved model is loaded, not retrained.
- Shows Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
- Shows the confusion matrix and the per-class classification report.
- A second tab compares all five models on the uploaded data and highlights the best
  score for each metric.
- Checks the uploaded file and shows a clear message if columns are missing, cells are
  empty or the labels are not recognised.

The uploaded CSV must contain the 36 feature columns and the `land_cover` column.

## d. Models used and evaluation metrics

All five models were trained on the same training split and evaluated on the same test
split. Each one is a scikit-learn pipeline: a `StandardScaler` followed by the classifier.

| # | Model | How it works | Settings |
|---|---|---|---|
| 1 | Logistic Regression | A linear model that outputs a probability for each class | `max_iter=1000`, other settings default |
| 2 | Decision Tree | Splits the data with a sequence of feature thresholds | Defaults (Gini, no depth limit); the tree grows to depth 21 with 382 leaves |
| 3 | K-Nearest Neighbours | Predicts the majority class among the closest training rows | `n_neighbors=5`, Euclidean distance |
| 4 | Naive Bayes | Applies Bayes' rule assuming features are independent and normally distributed within each class | Gaussian variant, defaults |
| 5 | Support Vector Machine | Finds a maximum-margin boundary using a kernel | RBF kernel, `C=1.0`, `gamma="scale"`, `probability=True` so AUC can be computed |

### Evaluation metrics

There are six classes, so the metrics are averaged across classes as follows.

| Metric | Meaning | Averaging |
|---|---|---|
| Accuracy | Share of test rows predicted correctly | - |
| AUC | Area under the ROC curve: how well predicted probabilities rank the true class above the others | One-vs-rest, macro-averaged |
| Precision | Of the rows predicted as a class, the share that truly belong to it | Macro-averaged |
| Recall | Of the rows that truly belong to a class, the share the model found | Macro-averaged |
| F1 | Harmonic mean of precision and recall | Macro-averaged |
| MCC | Matthews correlation coefficient: agreement between predicted and actual labels, from -1 to 1, which stays informative when classes are different sizes | Multi-class form |

Macro-averaging gives every class equal weight, which matters because the classes are not
the same size.

### Comparison table

Results on the 2,000-row test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8395 | 0.9755 | 0.8116 | 0.7938 | 0.7971 | 0.8032 |
| Decision Tree | 0.8505 | 0.9002 | 0.8331 | 0.8306 | 0.8317 | 0.8164 |
| KNN | 0.9045 | 0.9786 | 0.8951 | 0.8886 | 0.8916 | 0.8827 |
| Naive Bayes | 0.7965 | 0.9551 | 0.7864 | 0.7877 | 0.7833 | 0.7531 |
| Support Vector Machine | 0.8955 | 0.9850 | 0.8852 | 0.8724 | 0.8769 | 0.8718 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner for this dataset** | |

## How to run

```
pip install -r requirements.txt
python prepare_data.py     # writes test_data.csv from data/sat.tst
python train.py            # trains the 5 models into model/ and writes results/metrics_comparison.csv
streamlit run app.py       # launches the app
```

The saved models in `model/` were trained with the scikit-learn version pinned in
`requirements.txt`. Loading them with a different version can fail, so keep the pin in
sync.

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch with `app.py` as
the entry point. Dependencies come from `requirements.txt`.
