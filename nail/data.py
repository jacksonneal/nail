import json
from os.path import isdir, join
from typing import List

import pandas as pd
from numerapi import NumerAPI

from .config import settings

FEATURES_FILE = "features.json"
TRAINING_FILE = "train.parquet"

DATA_TYPE_COL = "data_type"
ERA_COL = "era"
TARGET_COL = "target_cyrus_v4_20"


def load_data():
    if isdir(join(settings.data_dir, settings.data_version)):
        print(f"{settings.data_version} already downloaded, skipping")
        return

    napi = NumerAPI()
    datasets = [d for d in napi.list_datasets() if d.startswith(settings.data_version)]
    for d in datasets:
        napi.download_dataset(d, dest_path=join(settings.data_dir, d))


def features() -> List[str]:
    with open(join(settings.version_data_dir, FEATURES_FILE), "r") as f:
        feature_metadata = json.load(f)
    return feature_metadata["feature_sets"][settings.feature_set]


def columns() -> List[str]:
    return features() + [ERA_COL, DATA_TYPE_COL, TARGET_COL]


def training_data() -> pd.DataFrame:
    return pd.read_parquet(
        join(settings.version_data_dir, TRAINING_FILE), columns=columns()
    )


if __name__ == "__main__":
    load_data()
    print(features())
    print(len(training_data()))
