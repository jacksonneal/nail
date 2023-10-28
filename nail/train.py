from os import *
from os.path import join

import xgboost as xgb

from .config import settings
from .data import TARGET_COL, read_features, read_training_data

MODEL_NAME = "model"


def train():
    training_data = read_training_data()
    features = read_features()
    training_data_matrix = xgb.DMatrix(
        training_data[features],
        label=training_data[TARGET_COL],
    )

    bst = xgb.train(
        {"tree_method": settings.xgb_tree_method, "verbosity": 2},
        training_data_matrix,
    )
    bst.save_model(join(settings.models_dir, f"{MODEL_NAME}.json"))


if __name__ == "__main__":
    train()
