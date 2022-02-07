import torch
from stylegan2_torch.equalized_lr import EqualLeakyReLU
from torch import nn
from torch.functional import Tensor


class Normalize(nn.Module):
    """
    Normalize latent vector for each sample
    """

    def forward(self, input: Tensor) -> Tensor:
        # input: (N, style_dim)
        # Normalize z in each sample to N(0,1)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Sequential):
    """
    Mapping network from sampling space (z) to disentangled latent space (w)
    """

    def __init__(self, latent_dim: int, n_mlp: int, lr_mlp_mult: float):
        super().__init__(
            Normalize(),
            *[
                EqualLeakyReLU(
                    latent_dim,
                    latent_dim,
                    lr_mult=lr_mlp_mult,
                )
                for _ in range(n_mlp)
            ]
        )
