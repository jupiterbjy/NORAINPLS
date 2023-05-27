"""
Preprocessing code
"""

import pathlib
from os import PathLike

import pandas as pd
import numpy as np

BLACKLIST_COLUMNS = 0, 1, 2, 4, 6, 8, 10, 12, 14, 18
ALLOWED_COLUMNS = list(set(range(19)) - set(BLACKLIST_COLUMNS))

RAW_DATA_PATH = pathlib.Path("data/raw")
PREPROCESSED_DATA_PATH = pathlib.Path("data/preprocessed")

# make sure x, y dir exists
X_PATH = (PREPROCESSED_DATA_PATH / "x")
Y_PATH = (PREPROCESSED_DATA_PATH / "y")
X_PATH.mkdir(exist_ok=True)
Y_PATH.mkdir(exist_ok=True)


def preprocess_raw_gen(csv_path):
    """Loads single csv file, and yields dataframe chunks.
    Each chunk is one continuous sections, thus each are
    split from sensor anomaly positions."""

    data = pd.read_csv(csv_path)
    data = data.iloc[:, ALLOWED_COLUMNS]
    data = data.fillna(0)

    x = data.drop("강수량(mm)", axis=1)
    y = data["강수량(mm)"]

    # Not attempting 0~1 normalization now
    # Instead, just subtracting 1000 from pressure data
    x["현지기압(hPa)"] = x["현지기압(hPa)"].subtract(1000)
    x["해면기압(hPa)"] = x["해면기압(hPa)"].subtract(1000)

    return x, y


def preprocess_all():
    """Find all csv files and preprocess"""

    for file in RAW_DATA_PATH.iterdir():
        if file.suffix != ".csv":
            continue

        # preprocess, then save without header & index
        x, y = preprocess_raw_gen(file)
        x.to_csv(X_PATH / file.name, header=False, index=False)
        y.to_csv(Y_PATH / file.name, header=False, index=False)


if __name__ == '__main__':
    preprocess_all()
