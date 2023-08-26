from os.path import isfile, join
from typing import Any, Optional

from lightgbm import LGBMRegressor
from pandas import read_pickle

from .config import settings

MODEL_NAME = "lgbm_model"

params = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 2**5,
    "colsample_bytree": 0.1,
}


def load_model(model_name: str) -> Optional[Any]:
    model_filepath = join(settings.models_dir, f"{model_name}.pkl")
    if not isfile(model_filepath):
        return None
    return read_pickle(model_filepath)


def get_model() -> LGBMRegressor:
    return load_model(MODEL_NAME) or LGBMRegressor(**params)
