from contextlib import suppress
from importlib import metadata

from stylegan2_torch.discriminator import Discriminator
from stylegan2_torch.equalized_lr import (Blur, EqualConv2d, EqualLeakyReLU,
                                          EqualLinear)
from stylegan2_torch.generator import Generator
from stylegan2_torch.utils import Resolution, default_channels

__author__ = "Peter Yuen"
__email__ = "ppeetteerrsx@gmail.com"
__version__ = "test"
with suppress(Exception):
    __version__ = metadata.version("stylegan2_torch")


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
