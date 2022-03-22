import random
from typing import Callable, Dict, List, Literal, TypeVar, cast

import torch
from torch import nn
from torch.functional import Tensor

C = TypeVar("C", bound=Callable)
T = TypeVar("T")

Resolution = Literal[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

default_channels: Dict[Resolution, int] = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 512,
    128: 256,
    256: 128,
    512: 64,
    1024: 32,
}


def make_kernel(
    k: List[int],
    factor: int = 1,
) -> Tensor:
    """
    Creates 2D kernel from 1D kernel, compensating for zero-padded upsampling factor
    """
    kernel = torch.tensor(k, dtype=torch.float32)

    kernel = kernel[None, :] * kernel[:, None]

    kernel /= kernel.sum()
    kernel *= factor ** 2

    return kernel


def accumulate(
    model1: nn.Module,
    model2: nn.Module,
    decay: float = 0.5 ** (32 / (10 * 1000)),
) -> None:
    """
    Accumulate parameters of model2 onto model1 using EMA
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def make_noise(batch: int, latent_dim: int, n_noise: int, device: str):
    """
    Makes a random, normally distributed latent vector.
    """

    return torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)


def mixing_noise(batch: int, latent_dim: int, prob: float, device: str):
    """
    Makes a random, normally distributed latent vector. Returns a pair if mixing regularization.
    """
    return make_noise(batch, latent_dim, 2 if random.random() < prob else 1, device)


def proxy(f: C) -> C:
    """
    Proxy function signature map for `Module.__call__` type hint.
    """
    return cast(C, lambda self, *x, **y: super(self.__class__, self).__call__(*x, **y))
