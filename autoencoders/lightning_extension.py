from typing import Optional

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from skimage.util import random_noise
from torch import Tensor
from torch import nn
from torch.nn import MSELoss, BCELoss

from autoencoders.models.sparse import SparseBottleneck


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.Module,
        decoder: nn.Module,
        lr: Optional[float] = 0.01,
        noise_ratio: Optional[float] = None,
        recon_loss: Optional[str] = "bce",
        prevent_tb_logging: Optional[bool] = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.prevent_tb_logging = prevent_tb_logging

        self.lr = lr
        self.noise_ratio = noise_ratio or 0.0
        self.criterion_recon = None
        if recon_loss.lower() == "bce":
            self.criterion_recon = BCELoss(reduction="none")
        elif recon_loss.lower() == "rmse":
            self.criterion_recon = MSELoss(reduction="mean")

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training and self.noise_ratio > 0:
            inputs = self._apply_noise(inputs)
        encoded = self.encoder(inputs)
        latent_code, _ = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)
        return decoded

    def training_step(self, inputs: Tensor, batch_idx: int) -> Tensor:
        inputs, _ = inputs
        loss, bottleneck_loss, recon_loss = self._get_losses(inputs)

        self.log("train/recon", recon_loss)
        self.log("train/bottleneck", bottleneck_loss)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, inputs: Tensor, batch_idx: int) -> None:
        self._evaluate(inputs, batch_idx, mode="val")

    def test_step(self, inputs: Tensor, batch_idx: int) -> None:
        self._evaluate(inputs, batch_idx, mode="test")

    def _evaluate(self, inputs: Tensor, batch_idx: int, mode: str) -> None:
        (inputs, _) = inputs
        if mode == "val" and batch_idx == 0:
            self._log_generate_images(inputs, mode)

        loss, bottleneck_loss, recon_loss = self._get_losses(inputs)

        self.log(f"{mode}/recon", recon_loss)
        self.log(f"{mode}/bottleneck", bottleneck_loss)
        self.log(f"{mode}/loss", loss)

    def _plot_images(self, title: str, images: Tensor):
        images = images.squeeze().clone().detach().cpu().numpy()
        print(f"{title} at {self.global_step}")
        for i in range(images.shape[0]):
            columns = images.shape[0] // 8
            plt.subplot(columns, images.shape[0] // columns, i + 1)
            plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap="binary")
        plt.show(block=False)

    def _log_generate_images(self, inputs: Tensor, mode: str):
        if self.noise_ratio > 0.0:
            inputs = self._apply_noise(inputs)
        outputs = self(inputs)
        comparison = torch.cat([inputs, outputs], dim=2)
        if self.prevent_tb_logging:
            self._plot_images(f"{mode}/reconstructions", comparison)
        else:
            self.logger.experiment.add_images(  # noqa
                f"{mode}/reconstructions", comparison, self.global_step
            )
        if isinstance(self.bottleneck, SparseBottleneck):
            latent_influence = self.decoder(torch.eye(20).to(self.device)).detach().cpu()
            if self.prevent_tb_logging:
                self._plot_images(f"{mode}/latent-influence", latent_influence)
            else:
                self.logger.experiment.add_images(  # noqa
                    f"{mode}/latent-influence", latent_influence, self.global_step
                )

    def _get_losses(self, inputs: Tensor) -> (Tensor, Tensor, Tensor):
        encoded = self.encoder(inputs)
        latent_code, bottleneck_loss = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        # In case we want to use binary cross entropy loss
        if isinstance(self.criterion_recon, nn.BCELoss):
            recon_loss = self.criterion_recon(decoded.squeeze(), inputs.squeeze())
            recon_loss = recon_loss.mean(0).sum()
        # Else we assume mean squared error loss is unsed (we additionally
        # take the square root for absolute MSE)
        else:
            recon_loss = torch.sqrt(self.criterion_recon(decoded, inputs))
        loss = recon_loss + bottleneck_loss
        return loss, bottleneck_loss, recon_loss

    def _apply_noise(self, inputs: Tensor) -> Tensor:
        noised_input = torch.tensor(
            random_noise(
                inputs.detach().cpu(),
                mode="s&p",
                amount=self.noise_ratio,
                clip=True,
            )
        )
        return noised_input.to(self.device)
