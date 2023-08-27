from os.path import join

from dask.dataframe import to_parquet

from .config import settings
from .data import TRAINING_FILE, read_training_data


def do_repartition():
    df = read_training_data()
    df = df.repartition(partition_size="100MB")
    to_parquet(df, join(settings.version_data_dir, f"partitioned-{TRAINING_FILE}"))


if __name__ == "__main__":
    do_repartition()
