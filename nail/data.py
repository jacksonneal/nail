import json
from os import remove
from os.path import isfile, join

# from pandas import DataFrame, read_parquet
from dask.dataframe import read_parquet
from numerapi import NumerAPI

from .config import settings

FEATURES_FILE = "features.json"
TRAINING_FILE = "train.parquet"
VALIDATION_FILE = "validation.parquet"
LIVE_FILE = "live.parquet"

DATA_TYPE_COL = "data_type"
ERA_COL = "era"
TARGET_COL = "target"


def download_data():
    napi = NumerAPI()
    cur_round = napi.get_current_round()
    if not isfile(join(settings.version_data_dir, f"round-{cur_round}-{LIVE_FILE}")):
        napi.download_dataset(
            join(settings.data_version, LIVE_FILE),
            join(settings.version_data_dir, f"round-{cur_round}-{LIVE_FILE}"),
        )
        if isfile(join(settings.version_data_dir, LIVE_FILE)):
            remove(join(settings.version_data_dir, LIVE_FILE))

    datasets = [d for d in napi.list_datasets() if d.startswith(settings.data_version)]
    for d in datasets:
        napi.download_dataset(d, join(settings.data_dir, d))


def read_features() -> list[str]:
    with open(join(settings.version_data_dir, FEATURES_FILE)) as f:
        feature_metadata = json.load(f)
    return feature_metadata["feature_sets"][settings.feature_set]


def read_columns() -> list[str]:
    return read_features() + [ERA_COL, DATA_TYPE_COL, TARGET_COL]


def read_data(file_name: str):
    return read_parquet(
        join(settings.version_data_dir, file_name),
        columns=read_columns(),
        split_row_groups="adaptive",
    )


def read_partitioned_training_data():
    return read_data(f"partitioned-{TRAINING_FILE}")


def read_training_data():
    return read_data(TRAINING_FILE)


def read_validation_data():
    return read_data(VALIDATION_FILE)


def read_live_data():
    return read_data(LIVE_FILE)


if __name__ == "__main__":
    download_data()
