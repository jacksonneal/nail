from os.path import join

import cloudpickle
import lightgbm as lgb
from pandas import DataFrame, Series

from nail.train import MODEL_NAME

from .config import settings
from .data import read_features


def predict_outer():
    features = read_features()

    model = lgb.Booster(model_file=join(settings.models_dir, f"{MODEL_NAME}.txt"))

    # Define your prediction pipeline as a function that takes an era of features
    # as input and outputs your predictions for that era
    def predict(live_features: DataFrame) -> DataFrame:
        live_predictions = model.predict(live_features[features])
        submission = Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    p = cloudpickle.dumps(predict)
    with open("predict.pkl", "wb") as f:
        f.write(p)


if __name__ == "__main__":
    predict_outer()
