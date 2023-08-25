from numerapi import NumerAPI

from .config import settings


def load_data():
    napi = NumerAPI()

    # Let's see what files are available for download in the latest v4.1 dataset
    print([f for f in napi.list_datasets() if f.startswith("v4.1")])

    # Download the training data
    napi.download_dataset(
        "v4.1/train.parquet", dest_path=f"{settings.data_dir}/v4.1/train.parquet"
    )
