from os.path import join

from pandas import DataFrame, Series
from xgboost import Booster, DMatrix

from .config import settings
from .data import read_features, read_live_data
from .train import MODEL_NAME


def predict(live_features: DataFrame) -> DataFrame:
    live_feature_matrix = DMatrix(live_features)

    booster = Booster()
    booster.load_model(join(settings.models_dir, f"{MODEL_NAME}.json"))
    predictions = booster.predict(live_feature_matrix)

    prediction_series = Series(predictions, index=live_features.index)
    return prediction_series.to_frame("prediction")


if __name__ == "__main__":
    live_data = read_live_data()
    features = read_features()
    predict(live_data[features])
