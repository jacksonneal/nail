from os.path import isdir, join

from numerapi import NumerAPI

from .config import settings


def load_data(version: str = "v4.1"):
    napi = NumerAPI()
    datasets = [
        x
        for x in napi.list_datasets()
        if x.startswith(version) and not isdir(join(settings.data_dir, x))
    ]
    for x in datasets:
        napi.download_dataset(x, dest_path=join(settings.data_dir, x))


if __name__ == "__main__":
    load_data()
