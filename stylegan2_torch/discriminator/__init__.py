import math
from typing import Any, Dict, List, Literal, Tuple, overload

import torch
from stylegan2_torch.discriminator.blocks import ConvBlock, ResBlock
from stylegan2_torch.equalized_lr import EqualLeakyReLU, EqualLinear
from stylegan2_torch.utils import Resolution, default_channels, proxy
from torch import nn
from torch.functional import Tensor


class Discriminator(nn.Module):
    """
    Discriminator module
    """

    def __init__(
        self,
        resolution: Resolution,
        channels: Dict[Resolution, int] = default_channels,
        blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        super().__init__()

        # FromRGB followed by ResBlock
        self.n_layers = int(math.log(resolution, 2))

        self.blocks = nn.Sequential(
            ConvBlock(1, channels[resolution], 1),
            *[
                ResBlock(channels[2**i], channels[2 ** (i - 1)], blur_kernel)
                for i in range(self.n_layers, 2, -1)
            ],
        )

        # Minibatch std settings
        self.stddev_group = 4
        self.stddev_feat = 1

        # Final layers
        self.final_conv = ConvBlock(channels[4] + 1, channels[4], 3)
        self.final_relu = EqualLeakyReLU(channels[4] * 4 * 4, channels[4])
        self.final_linear = EqualLinear(channels[4], 1)

    @overload
    def forward(
        self, input: Tensor, *, return_features: Literal[False] = False
    ) -> Tensor:
        ...

    @overload
    def forward(
        self, input: Tensor, *, return_features: Literal[True]
    ) -> Tuple[Tensor, Tensor]:
        ...

    def forward(self, input: Tensor, *, return_features: bool = False):
        # Downsampling blocks
        out: Tensor = self.blocks(input)

        # Minibatch stddev layer in Progressive GAN https://www.youtube.com/watch?v=V1qQXb9KcDY
        # Purpose is to provide variational information to the discriminator to prevent mode collapse
        # Other layers do not cross sample boundaries
        batch, channel, height, width = out.shape
        n_groups = min(batch, self.stddev_group)
        stddev = out.view(
            n_groups, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(n_groups, 1, height, width)
        out = torch.cat([out, stddev], 1)

        # Final layers
        out = self.final_conv(out)
        features = self.final_relu(out.view(batch, -1))
        out = self.final_linear(features)

        if return_features:
            return out, features
        else:
            return out

    __call__ = proxy(forward)
