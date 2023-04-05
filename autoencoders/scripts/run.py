import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import loggers

from autoencoders.data import MNISTDataModule
from autoencoders.lightning_extension import AutoEncoder
from autoencoders.models.deep import DeepEncoder, DeepDecoder
from autoencoders.models.simple import SimpleEncoder, SimpleDecoder, SimpleBlottleneck
from autoencoders.models.sparse import SparseBottleneck


def _train(
    ae: AutoEncoder,
    logger_module: loggers.Logger,
    datamodule: pl.LightningDataModule,
    epochs: Optional[int] = 60,
):
    trainer = pl.Trainer(
        max_epochs=epochs, deterministic=True, logger=logger_module, detect_anomaly=True
    )
    trainer.fit(ae, datamodule=datamodule)
    checkpoint_path = trainer.checkpoint_callback.last_model_path  # noqa
    return checkpoint_path


def build_logger(model_type: str, datamodule_name: str, task: Optional[str] = None):
    log_dir = os.path.normpath(os.path.join(os.getcwd(), "logs", datamodule_name))
    task = task if task else "featureExtraction"
    experiment_name = f"{model_type}_{task}"
    return loggers.tensorboard.TensorBoardLogger(log_dir, experiment_name)


class Runner:
    def __init__(
        self,
        batch_size: Optional[int] = 32,
        training_size: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        pl.seed_everything(seed)
        self.batch_size = batch_size
        self.training_size = training_size
        self.dataset = MNISTDataModule("./dataset/", batch_size, training_size)
        self.dataset.prepare_data()
        self.dataset.setup("fit")

    def train_simple_autoencoder(
        self,
        lr: Optional[float] = 0.001,
        epochs: Optional[int] = 20,
    ):
        input_shape = (1, 32, 32)
        latent_dim = 20
        encoder = SimpleEncoder(input_shape, latent_dim)
        decoder = SimpleDecoder(latent_dim, input_shape)
        bottleneck = SimpleBlottleneck(latent_dim)
        model = AutoEncoder(encoder, bottleneck, decoder, lr=lr)
        logger = build_logger("simpleautoencoder", datamodule_name="MNIST", task="reconstruction")
        _train(model, logger, self.dataset, epochs)

    def train_deep_autoencoder(
        self,
        depth: Optional[int] = 3,
        lr: Optional[float] = 0.001,
        epochs: Optional[int] = 20,
    ):
        input_shape = (1, 32, 32)
        latent_dim = 20
        encoder = DeepEncoder(input_shape, depth, latent_dim)
        decoder = DeepDecoder(latent_dim, depth, input_shape)
        bottleneck = SimpleBlottleneck(latent_dim)
        model = AutoEncoder(encoder, bottleneck, decoder, lr=lr)
        logger = build_logger("deepautoencoder", datamodule_name="MNIST", task="reconstruction")
        _train(model, logger, self.dataset, epochs)

    def train_denoising_autoencoder(
        self,
        depth: Optional[int] = 3,
        lr: Optional[float] = 0.001,
        epochs: Optional[int] = 20,
        noise_ratio: Optional[float] = 0.25,
    ):
        input_shape = (1, 32, 32)
        latent_dim = 20
        encoder = DeepEncoder(input_shape, depth, latent_dim)
        decoder = DeepDecoder(latent_dim, depth, input_shape)
        bottleneck = SimpleBlottleneck(latent_dim)
        model = AutoEncoder(encoder, bottleneck, decoder, lr=lr, noise_ratio=noise_ratio)
        logger = build_logger("denoisingautoencoder", datamodule_name="MNIST", task="reconstruction")
        _train(model, logger, self.dataset, epochs)

    def train_sparse_autoencoder(
        self,
        depth: Optional[int] = 3,
        lr: Optional[float] = 0.001,
        epochs: Optional[int] = 20,
        sparsity_loss: Optional[str] = "kld",
    ):
        input_shape = (1, 32, 32)
        latent_dim = 20
        encoder = DeepEncoder(input_shape, depth, latent_dim)
        decoder = DeepDecoder(latent_dim, depth, input_shape)
        bottleneck = SparseBottleneck(latent_dim, sparsity_loss)
        model = AutoEncoder(encoder, bottleneck, decoder, lr=lr)
        logger = build_logger("sparseautoencoder", datamodule_name="MNIST", task="reconstruction")
        _train(model, logger, self.dataset, epochs)


if __name__ == "__main__":
    # Runner is just a simple wrapper for different configurations of auto encoders.
    r = Runner(batch_size=32, training_size=None)
    # r.train_simple_autoencoder(0.001, 20)
    # r.train_deep_autoencoder(3, 0.001, 20)
    r.train_denoising_autoencoder(3, 0.001, 20, 0.05)
    # r.train_sparse_autoencoder(3, 0.001, 20, "kld")
