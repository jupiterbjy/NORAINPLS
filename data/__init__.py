"""
Dataset definition
"""

import pathlib
import numpy as np
import pandas as pd


_PREPROCESSED_DATA_PATH = pathlib.Path("data/preprocessed")

# make sure x, y dir exists
_X_PATH = (_PREPROCESSED_DATA_PATH / "x")
_Y_PATH = (_PREPROCESSED_DATA_PATH / "y")


def _load_preprocessed():
    """Load preprocessed data"""
    xs = []
    ys = []

    # for each year fetch data and insert into xs / ys
    for x_file, y_file in zip(_X_PATH.iterdir(), _Y_PATH.iterdir()):

        # just in case make sure both matches and is .csv
        if not (x_file.name == y_file.name and x_file.suffix == y_file.suffix == ".csv"):
            print(f"! File {x_file.name} is not a training data, skipping")
            continue

        print(f"- Loading {x_file.name}")
        # otherwise add to list
        xs.append(np.array(pd.read_csv(x_file)))
        ys.append(np.array(pd.read_csv(y_file)))

    print(f"Fetched {len(xs)} datasets")
    return xs, ys


# create singleton
class _Dataset:
    xs, ys = _load_preprocessed()

    def __iter__(self):
        return ((x, y) for x, y in zip(self.xs, self.ys))

    def __len__(self):
        len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


dataset = _Dataset()
