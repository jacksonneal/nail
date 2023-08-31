import dask.distributed
import xgboost as xgb

from .data import (
    ERA_COL,
    TARGET_COL,
    read_features,
    read_partitioned_training_data,
    read_training_data,
)


def train():
    cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=4)
    client = dask.distributed.Client(cluster)
    print(client)

    training_data = read_training_data()

    # every_4th_era = training_data[ERA_COL].unique()[::4].compute()
    # training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]

    X = training_data[read_features()]
    y = training_data[TARGET_COL]

    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    output = xgb.dask.train(
        client,
        {"verbosity": 2, "tree_method": "gpu_hist", "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )


if __name__ == "__main__":
    train()
