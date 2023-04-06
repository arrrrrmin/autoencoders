import itertools
from functools import reduce

import torch
from torch import nn


class DeepEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (1, 32, 32),
        num_layers: int = 3,
        latent_dim: int = 20,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.Sequential:
        units = self._get_units()
        layers = []
        units_a, units_b = itertools.tee(units)
        next(units_b, None)
        for in_units, out_units in zip(units_a, units_b):
            layers += [
                nn.Sequential(
                    nn.Linear(in_units, out_units, bias=False),
                    nn.BatchNorm1d(out_units),
                    nn.ReLU(True),
                )
            ]
        layers += [nn.Linear(units[-1], self.latent_dim)]
        return nn.Sequential(*layers)

    def _get_units(self) -> list[int]:
        in_units = reduce(lambda a, b: a * b, self.input_shape)
        shrinkage = int(pow(in_units // self.latent_dim, 1 / self.num_layers))
        units = [in_units // (shrinkage**i) for i in range(self.num_layers)]
        return units

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = self.layers(inputs)
        return outputs


class DeepDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 20,
        num_layers: int = 3,
        output_shape: tuple = (1, 32, 32),
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_shape = output_shape
        self.layers = self._build_layers()

    def _build_layers(self):
        units = self._get_units()
        layers = []
        units_a, units_b = itertools.tee(units)
        next(units_b, None)
        for in_units, out_units in zip(units_a, units_b):
            layers += [
                nn.Sequential(
                    nn.Linear(in_units, out_units, bias=False),
                    nn.BatchNorm1d(out_units),
                    nn.ReLU(True),
                )
            ]
        layers[-1] = nn.Sequential(nn.Linear(units[-2], units[-1]), nn.Sigmoid())
        return nn.Sequential(*layers)

    def _get_units(self):
        final_units = reduce(lambda a, b: a * b, self.output_shape)
        shrinkage = int(pow(final_units // self.latent_dim, 1 / self.num_layers))
        units = [final_units // (shrinkage**i) for i in range(self.num_layers)]
        units.reverse()
        units = [self.latent_dim] + units
        return units

    def forward(self, inputs):
        outputs = self.layers(inputs)
        outputs = outputs.view(-1, *self.output_shape)
        return outputs


# For the bottleneck, we'll use the same as the simple bottleneck
