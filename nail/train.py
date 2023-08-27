import gc
from os.path import join

import lightgbm as lgb

from .config import settings
from .data import TARGET_COL, read_features, read_training_data, read_validation_data

MODEL_NAME = "model"


def train():
    training_data = read_training_data()
    validation_data = read_validation_data()
    features = read_features()

    training_dataset = lgb.Dataset(
        training_data[features], label=training_data[TARGET_COL], free_raw_data=True
    )
    validation_dataset = lgb.Dataset(
        validation_data[features], label=validation_data[TARGET_COL], free_raw_data=True
    )
    gc.collect()

    params = {"force_col_wise": True}
    booster = lgb.train(
        params,
        training_dataset,
        valid_sets=[validation_dataset],
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
    )

    booster.save_model(join(settings.models_dir, f"{MODEL_NAME}.txt"))


if __name__ == "__main__":
    train()
