import torch

from .data import load_data

if __name__ == "__main__":
    print(torch.cuda.is_available())
    # load_data()
