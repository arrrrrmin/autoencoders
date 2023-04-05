from pathlib import Path

from torchvision.transforms import functional as ptv_f
from torchvision.datasets import MNIST


if __name__ == "__main__":
    data_dir = "./dataset/"
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    dataset = MNIST(data_dir, download=True)
    print(ptv_f.pil_to_tensor(dataset[0][0]).shape)
    print("Done.")
