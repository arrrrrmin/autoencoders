import torch
from torch import nn, Tensor


# For encoding and decoding layers we'll use DeepEncoder and DeepDecoder

# The basic idea is to force sparsity to exlicitly correlate certain image properties
# to certain areas in the latent code. This is done by performing another activation
# on the latent code and compute a penalty for this activation to be sparse.
# If done correctly the latent code can be changed an the output image generated by
# the decoder is reactiving on this change (e.g. reduce bright lighting in images).

# On Kullback Liebler Divergence, please note that this distance between probability
# distributions assumes (in our case latent code) the distributions are non negative.
# This is important to choose the correct activation function on the encoder.

# Here is an example why:
# >>> a_lc = torch.tanh(torch.rand((4,8,)) + torch.randint(low=-1, high=1, size=(4,8,))).mean(0)
# >>> a_lc
# tensor([-0.0651, -0.1393, -0.3940,  0.0504, -0.2356,  0.0768, -0.1498,  0.4374])
# >>> sparsity = 0.25
# >>> torch.log(sparsity / a_lc)
# tensor([    nan,     nan,     nan,  1.6015,     nan,  1.1798,     nan, -0.5594])

# Already the first operation on the simulated latent code (e.g. tanh ) yields nan
# values. Which breaks the loss. Sigmoid outputs values in (0,1) range, which keeps
# use save from these errors.

# Much deeper details on how to enforce a sparsity constraint in the latent code
# been explored by [L. Zhang and Y. Lu](https://ieeexplore.ieee.org/document/7280364)

# Literature suggests that the same can be achived by using a log loss on the sigmoid
# activations of the bottleneck layer. Below is a simple simulation how lower average
# activations recieve lower penalty:
# >>> activations = torch.sigmoid((torch.rand(4, 8))
# >>> logloss = torch.log(1 + torch.pow(activations.mean(0), 2)).sum()
# >>> logloss
# tensor(2.5719)
# >>> activations -= 0.001
# >>> torch.log(1 + torch.pow(activations.mean(0), 2)).sum()
# tensor(2.5648)
# >>> activations -= 0.005
# >>> torch.log(1 + torch.pow(activations.mean(0), 2)).sum()
# tensor(2.5292)


class SparseBottleneck(nn.Module):
    def __init__(
        self,
        latent_dim: int = 20,
        loss_type: str = "kld",
        sparsity: float = 0.25,
        beta: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.sparsity = sparsity
        self.beta = beta

    def forward(self, encoded: Tensor) -> (Tensor, Tensor):
        latent_code = torch.sigmoid(encoded)
        sparsity_loss = self._loss(latent_code)
        return latent_code, sparsity_loss

    def _loss(self, latent_code: Tensor) -> Tensor:
        l = None
        if self.loss_type == "kld":
            l = self._kld_loss(latent_code)
        elif self.loss_type == "log":
            l = self._log_loss(latent_code)
        return l

    def _kld_loss(self, latent_code: Tensor) -> Tensor:
        average_activation = torch.mean(latent_code, dim=0)
        kl_div = self.sparsity * torch.log(self.sparsity / average_activation) + (1 - self.sparsity) * torch.log(
            (1 - self.sparsity) / (1 - average_activation)
        )
        kl_div = torch.sum(kl_div)
        kl_div *= self.beta
        return kl_div

    @staticmethod
    def _log_loss(latent_code: Tensor) -> Tensor:
        average_activation = torch.mean(latent_code, dim=0)
        return torch.log(1 + torch.pow(average_activation, 2)).sum()
