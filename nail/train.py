from os.path import join

from xgboost import XGBRegressor

from .config import settings
from .data import TARGET_COL, read_features, read_training_data

MODEL_NAME = "model"


def train():
    training_data = read_training_data()
    features = read_features()

    reg = XGBRegressor(tree_method="gpu_hist")
    model = reg.fit(training_data[features], training_data[TARGET_COL])
    model.save_model(join(settings.models_dir, f"{MODEL_NAME}.json"))


if __name__ == "__main__":
    train()
