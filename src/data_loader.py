"""Helpers for loading the UCI Statlog (Landsat Satellite) dataset.

The raw files (data/sat.trn and data/sat.tst) are space separated, have no
header row, and hold 36 spectral values followed by the class label.

Each row describes a 3x3 pixel neighbourhood from a satellite image. For each
of the 9 pixels, 4 spectral band values are stored, so 9 x 4 = 36 features.
The label is the land-cover class of the CENTRAL pixel.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "sat.trn"
TEST_FILE = DATA_DIR / "sat.tst"

TARGET_COL = "land_cover"

# Pixels are stored left-to-right, top-to-bottom (see data/sat.doc).
PIXEL_POSITIONS = [
    "top_left", "top_mid", "top_right",
    "mid_left", "centre", "mid_right",
    "bottom_left", "bottom_mid", "bottom_right",
]
N_BANDS = 4

# Class 6 ("mixture class") exists in the original documentation but has no
# examples in this dataset, so only six labels appear.
CLASS_NAMES = {
    1: "red soil",
    2: "cotton crop",
    3: "grey soil",
    4: "damp grey soil",
    5: "soil with vegetation stubble",
    7: "very damp grey soil",
}


def feature_names():
    """Return the 36 feature names, e.g. 'centre_band3'."""
    return [
        f"{position}_band{band}"
        for position in PIXEL_POSITIONS
        for band in range(1, N_BANDS + 1)
    ]


def _read(path):
    columns = feature_names() + [TARGET_COL]
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def load_train():
    """Official training split (4435 rows)."""
    return _read(TRAIN_FILE)


def load_test():
    """Official test split (2000 rows)."""
    return _read(TEST_FILE)
