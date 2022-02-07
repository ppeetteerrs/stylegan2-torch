__version__ = "0.1.0"

from typing import Dict, Literal

from stylegan2_torch.discriminator import Discriminator
from stylegan2_torch.equalized_lr import Blur, EqualConv2d, EqualLeakyReLU, EqualLinear
from stylegan2_torch.generator import Generator

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

__all__ = [
    "Discriminator",
    "Generator",
    "Resolution",
    "default_channels",
    "Blur",
    "EqualConv2d",
    "EqualLeakyReLU",
    "EqualLinear",
]
