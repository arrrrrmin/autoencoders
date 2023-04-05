from functools import reduce

import torch
from torch import nn, Tensor


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (1, 32, 32),
        latent_dim: int = 20,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        in_units = reduce(lambda a, b: a * b, self.input_shape)
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_units, self.latent_dim),
            nn.ReLU(),
        )

    def forward(self, inputs) -> Tensor:
        return self.layer(inputs)


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim: int = 20, output_shape: tuple = (1, 32, 32)):
        super().__init__()

        self.output_shape = output_shape
        self.latent_dim = latent_dim

        out_units = reduce(lambda a, b: a * b, self.output_shape)
        self.layer = nn.Sequential(
            nn.Linear(self.latent_dim, out_units),
            nn.Sigmoid(),
        )

    def forward(self, inputs) -> Tensor:
        outputs = self.layer(inputs)
        outputs = outputs.view(-1, *self.output_shape)
        return outputs


class SimpleBlottleneck(nn.Module):
    def __init__(self, latent_dim: int):
        super(SimpleBlottleneck, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, encoded) -> (Tensor, Tensor):
        return encoded, self._loss()

    @staticmethod
    def _loss() -> Tensor:
        return torch.tensor(0.0)
