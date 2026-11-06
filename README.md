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

Findings from the exploratory analysis (the `eda()` function in `src/core.py`, printed
when you run it):

- The classes have clearly different average centre-pixel profiles. For example cotton crop
  has a very low band 2 and the highest band 4.
- Damp grey soil and very damp grey soil have similar profiles, which makes them the
  hardest pair to separate.
- Neighbouring pixels in the same band are strongly correlated (highest pairwise
  correlation 0.959; 10.5% of feature pairs, 66 of 630, are above 0.9), so the features are far from
  independent.

### Data preprocessing

1. **Official split used as given.** The dataset already comes with a training file and a
   test file, so the data is not re-split.
2. **No cleaning needed.** There are no missing values and no duplicate rows.
3. **Feature scaling.** A `StandardScaler` is fitted on the training data only and then
   applied to the test data. It is part of each model's pipeline, so the saved models
   scale uploaded data automatically and nothing is refitted in the app. Scaling clearly
   helps Logistic Regression (test accuracy 83.95% with it, 79.45% without) and also SVM
   (89.55% against 88.60%); it makes almost no difference to the Decision Tree, KNN and
   Naive Bayes, but all five use it so they are treated alike. No new features are
   created: all 36 inputs are already numeric, complete and on the same 0-255 scale, so
   scaling is the only transformation needed.
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
| `src/core.py` | All the machine-learning code in one file: data loading, EDA, preprocessing, the five models, the evaluation metrics, and training. Run it with `python src/core.py` |
| `test_data.csv` | Test data used in the experiments (the official test split); upload this file in the app |
| `model/` | Saved trained pipelines (`.joblib`), one per classifier. Each includes the fitted scaler |
| `data/` | Raw dataset files (`sat.trn`, `sat.tst`) and the original documentation (`sat.doc`) |
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
The code for each model is in section 4 of `src/core.py`, and training is in section 6.

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
| Logistic Regression | Reaches 83.95% accuracy, with 87.6% on the training data, so the gap is small and the model is not overfitting. Its probability ranking is strong (AUC 0.9755, third best), but its hard predictions are weaker. It is excellent on red soil (F1 0.98) and cotton crop (0.94), yet it struggles with damp grey soil (F1 0.38, recall 0.30): 103 of the 211 damp grey rows are labelled very damp grey soil. A single linear boundary cannot separate classes whose spectral profiles overlap like these. |
| Decision Tree | Scores 85.05% accuracy, slightly above Logistic Regression, but it memorises the training data: 100% training accuracy against 85% on the test set, with a tree 21 levels deep and 382 leaves. Its AUC of 0.9002 is the lowest of all five models, because a fully grown tree gives almost all-or-nothing class probabilities, which ranks rows poorly. Damp grey soil is again the weak class (F1 0.55). Limiting the depth would likely reduce the overfitting. |
| KNN | The best accuracy (90.45%) and the best Precision (0.8951), Recall (0.8886), F1 (0.8916) and MCC (0.8827), with a second-best AUC of 0.9786. The gap between training accuracy (93.9%) and test accuracy (90.5%) is small, so it generalises well. It is also the best model on the hardest class, damp grey soil (F1 0.71, 145 of 211 rows correct). After scaling, pixels of the same land-cover class sit close together, which suits a nearest-neighbour vote. |
| Naive Bayes | The weakest model on accuracy (79.65%), F1 (0.7833) and MCC (0.7531). Training accuracy (79.8%) is almost the same as test accuracy (79.7%), so it underfits rather than overfits: its assumptions are too simple for this data. The features are strongly correlated (neighbouring pixels up to 0.959), which breaks the independence assumption Naive Bayes relies on. It struggles most with soil with vegetation stubble (F1 0.65) and damp grey soil (F1 0.57); 93 very damp grey rows were labelled damp grey. Its AUC (0.9551) is still better than the Decision Tree's, so its probabilities rank reasonably even when its hard labels are wrong. |
| Support Vector Machine | Second on accuracy (89.55%) and best on AUC (0.9850), meaning it ranks the correct class highest more consistently than any other model. Training accuracy is 91.0% against 89.6% on test, so it generalises well, and it uses 1,432 of the 4,435 training rows as support vectors. It is very strong on red soil (F1 0.99), cotton crop (0.97) and grey soil (0.92), but its recall on damp grey soil is only 0.58 (F1 0.64), which is where it falls behind KNN. |
| **Overall winner for this dataset** | **KNN.** It is best on five of the six metrics (Accuracy, Precision, Recall, F1, MCC) and is also best on the hardest class. SVM is a close second: it has the highest AUC, and its accuracy is only 0.9 percentage points lower (18 of the 2,000 test rows). Because the dataset documentation asks for a single train/test run instead of cross-validation, differences this small should be read with caution, but both models are clearly ahead of Logistic Regression, the Decision Tree and Naive Bayes. Damp grey soil is the hardest class for every model, and it is mostly confused with very damp grey soil and grey soil. |

## How to run

```
pip install -r requirements.txt
python src/core.py         # prints the EDA, writes test_data.csv, trains the 5 models into model/
streamlit run app.py       # launches the app
```

The saved models in `model/` were trained with the scikit-learn version pinned in
`requirements.txt`. Loading them with a different version can fail, so keep the pin in
sync.

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch with `app.py` as
the entry point. Dependencies come from `requirements.txt`.
