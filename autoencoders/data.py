import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./dataset/MNIST", batch_size=32, train_size=None
    ):
        super().__init__()
        self.data_dir = data_dir
        # Input images are padded in x and y (28 + 2 + 2, 28 + 2 + 2) = (32, 32)
        self.transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])

        self.dims = (1, 32, 32)
        self.num_classes = 10
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = os.cpu_count()

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def _get_mnist(
        self, train: bool, transform: transforms.Compose, download: bool = False
    ):
        return MNIST(self.data_dir, train, transform, download=download)

    def prepare_data(self):
        self._get_mnist(train=True, download=True, transform=self.transform)
        self._get_mnist(train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = self._get_mnist(train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = self._split_train_val(mnist_full)

        if stage == "test" or stage is None:
            self.mnist_test = self._get_mnist(train=False, transform=self.transform)

    def _split_train_val(self, mnist_full) -> (MNIST, MNIST):
        filter_mask = torch.zeros(len(mnist_full), dtype=torch.int)
        split_idx = torch.randperm(
            len(mnist_full), generator=torch.Generator().manual_seed(42)
        )
        bootstrap_size = self.train_size if self.train_size is not None else 55000
        filter_mask.scatter_(0, split_idx[:bootstrap_size], 1)

        mnist_train = Subset(mnist_full, filter_mask.nonzero().squeeze())
        mnist_val = Subset(mnist_full, split_idx[55000:])

        return mnist_train, mnist_val

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
