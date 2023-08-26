import json
from os.path import isdir, join
from typing import List

from numerapi import NumerAPI
from pandas import DataFrame, read_parquet

from .config import settings

FEATURES_FILE = "features.json"
TRAINING_FILE = "train.parquet"
VALIDATION_FILE = "validation.parquet"
LIVE_FILE = "live.parquet"

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


def read_features() -> List[str]:
    with open(join(settings.version_data_dir, FEATURES_FILE), "r") as f:
        feature_metadata = json.load(f)
    return feature_metadata["feature_sets"][settings.feature_set]


def read_columns() -> List[str]:
    return read_features() + [ERA_COL, DATA_TYPE_COL, TARGET_COL]


def read_data(file_name: str) -> DataFrame:
    return read_parquet(
        join(settings.version_data_dir, file_name), columns=read_columns()
    )


def read_training_data() -> DataFrame:
    return read_data(TRAINING_FILE)


def read_validation_data() -> DataFrame:
    return read_data(VALIDATION_FILE)


def read_live_data() -> DataFrame:
    return read_data(LIVE_FILE)


if __name__ == "__main__":
    load_data()
