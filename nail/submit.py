import cloudpickle

from .predict import predict


def submit():
    p = cloudpickle.dumps(predict)
    with open("predict.pkl", "wb") as f:
        f.write(p)


if __name__ == "__main__":
    submit()
