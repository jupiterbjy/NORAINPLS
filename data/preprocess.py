"""
Preprocessing code
"""

import pathlib

import pandas as pd


# Path defining
ROOT = pathlib.Path(__file__).parent

RAW_TRAIN_DATA_PATH = ROOT / "raw" / "training"
RAW_GENERALITY_DATA_PATH = ROOT / "raw" / "generality"

TRAIN_DATA_PATH = ROOT / "training"
GENERALITY_DATA_PATH = ROOT / "generality"

# make sure subdir exists
TRAIN_DATA_PATH.mkdir(exist_ok=True)
GENERALITY_DATA_PATH.mkdir(exist_ok=True)
(TRAIN_DATA_PATH / "x").mkdir(exist_ok=True)
(TRAIN_DATA_PATH / "y").mkdir(exist_ok=True)
(GENERALITY_DATA_PATH / "x").mkdir(exist_ok=True)
(GENERALITY_DATA_PATH / "y").mkdir(exist_ok=True)


WHITELIST_COLUMNS = [
    "기온(°C)",
    "강수량(mm)",
    "풍속(m/s)",
    "습도(%)",
    "현지기압(hPa)",
    "해면기압(hPa)",
    "전운량(10분위)",
    "중하층운량(10분위)",
    "지면온도(°C)",
]


def _preprocess_raw(csv_path):
    """Loads single csv file, and yields dataframe chunks.
    Each chunk is one continuous sections, thus each are
    split from sensor anomaly positions."""

    data = pd.read_csv(csv_path)
    data = data[WHITELIST_COLUMNS]
    data = data.fillna(0)

    # rain amount is something we want to predict, remove it out of X and make it as Y
    x = data.drop("강수량(mm)", axis=1)
    y = data["강수량(mm)"]

    # offset x's data and y's data by 1 hour
    return x[:-1], y[1:]


def _convert_csv(input_path: pathlib.Path, dest_path: pathlib.Path):
    """Find all csv files and preprocess"""

    for file in input_path.iterdir():
        if file.suffix != ".csv":
            continue

        print("- Processing", file)

        # preprocess, then save without header & index
        x, y = _preprocess_raw(file)
        x.to_csv(dest_path / "x" / file.name, header=False, index=False)
        y.to_csv(dest_path / "y" / file.name, header=False, index=False)


if __name__ == "__main__":
    print("Preprocessing Generality data")
    _convert_csv(RAW_GENERALITY_DATA_PATH, GENERALITY_DATA_PATH)

    print("Preprocessing Training data")
    _convert_csv(RAW_TRAIN_DATA_PATH, TRAIN_DATA_PATH)
