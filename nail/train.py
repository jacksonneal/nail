from nail.data import TARGET_COL, read_training_data

from .model import get_model


def train():
    model = get_model()
    training_data = read_training_data()

    model.fit(
        training_data.filter(like="feature_", axis="columns"), training_data[TARGET_COL]
    )


if __name__ == "__main__":
    train()
