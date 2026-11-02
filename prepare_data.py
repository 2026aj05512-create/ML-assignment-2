"""Write the official Landsat test split to test_data.csv.

test_data.csv has a header row, the 36 feature columns and the target column.
It is the file you upload in the Streamlit app.

Run from the project root:  python prepare_data.py
"""

from pathlib import Path

from src.data_loader import load_test

OUTPUT = Path(__file__).resolve().parent / "test_data.csv"


def main():
    test = load_test()
    test.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.name}: {test.shape[0]} rows, {test.shape[1]} columns")


if __name__ == "__main__":
    main()
