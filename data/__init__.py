"""
Dataset definition
"""

import pathlib
import numpy as np
import pandas as pd


_PREPROCESSED_DATA_PATH = pathlib.Path("data/training")

ROOT = pathlib.Path(__file__).parent

TRAIN_DATA_PATH = ROOT / "training"
GENERALITY_DATA_PATH = ROOT / "generality"


def _load_csv(parent_dir: pathlib.Path):
    """Load training data"""

    print(f"Loading {parent_dir.name} Dataset")

    # fetch path to order it
    x_paths = list((parent_dir / "x").iterdir())
    x_paths.sort(key=lambda p: p.stem)

    y_paths = [parent_dir / "y" / x_path.name for x_path in x_paths]

    xs = []
    ys = []

    # for each year fetch data and insert into xs / ys
    for x_file, y_file in zip(x_paths, y_paths):

        # just in case make sure both matches and is .csv
        if not (x_file.name == y_file.name and x_file.suffix == y_file.suffix == ".csv"):
            print(f"! File {x_file.name} is not a training data, skipping")
            continue

        print(f"- Loading {x_file.name}")
        # otherwise add to list
        xs.append(np.array(pd.read_csv(x_file)))
        ys.append(np.array(pd.read_csv(y_file)))

    return xs, ys


class _Dataset:
    def __init__(self, parent_dir: pathlib.Path, divide_factor=0.2):

        xs_per_yr, ys_per_yr = _load_csv(parent_dir)

        self.xs = np.concatenate(xs_per_yr)
        self.ys = np.concatenate(ys_per_yr)

        # change Y to 0-1
        self.ys = np.array([1.0 if rain != 0.0 else 0.0 for rain in self.ys])

        self.div = int((1 - divide_factor) * len(self))

    def separate_data(self):
        """Divide into training_x, training_y, test_x, test_y"""

        return self.xs[:self.div], self.ys[:self.div], self.xs[self.div:], self.ys[self.div:]

    def __iter__(self):
        return ((x, y) for x, y in zip(self.xs, self.ys))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


training_set = _Dataset(TRAIN_DATA_PATH)
generality_set = _Dataset(GENERALITY_DATA_PATH)
