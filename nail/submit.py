from os.path import join

import cloudpickle
from pandas import DataFrame, Series
from xgboost import Booster, DMatrix

from nail.data import read_features

from .config import settings
from .train import MODEL_NAME


def submit():
    features = read_features()

    booster = Booster()
    booster.load_model(join(settings.models_dir, f"{MODEL_NAME}.json"))

    def predict(live_features: DataFrame) -> DataFrame:
        live_feature_matrix = DMatrix(live_features[features])

        predictions = booster.predict(live_feature_matrix)
        prediction_series = Series(predictions, index=live_features.index)

        return prediction_series.to_frame("prediction")

    p = cloudpickle.dumps(predict)
    with open("predict.pkl", "wb") as f:
        f.write(p)


if __name__ == "__main__":
    submit()
