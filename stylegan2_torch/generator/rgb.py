import math
from typing import List, Optional

import torch
from stylegan2_torch.equalized_lr import EqualLinear
from stylegan2_torch.generator.conv_block import group_conv, mod
from stylegan2_torch.op.upfirdn2d import upfirdn2d
from stylegan2_torch.utils import make_kernel
from torch import nn
from torch.functional import Tensor
from torch.nn.parameter import Parameter


class Upsample(nn.Module):
    """
    Upsampling + apply blurring FIR filter
    """

    def __init__(self, blur_kernel: List[int], factor: int):
        super().__init__()

        self.factor = factor

        # Factor to compensate for averaging with zeros
        self.kernel: Tensor
        self.register_buffer("kernel", make_kernel(blur_kernel, self.factor))

        # Since upsampling by factor means there is factor - 1 pad1 already built-in
        """
        UPSAMPLE CASE

           kernel: [kkkkk]................[kkkkk] (k_w = 5)
        upsampled:     [x---x---x---x---x---x---] (in_w = 6, up_x = 4)
           padded: [ppppx---x---x---x---x---x---] (pad0 = 4, pad1 = 0)
           output:   [oooooooooooooooooooooooo]   (out_w = 24)
        Hence, pad0 + pad1 = k_w - 1
               pad0 - pad1 = up_x - 1


        DOWNSAMPLE CASE
        
           kernel: [kkkkk]...............[kkkkk] (k_w = 5)
            input:   [xxxxxxxxxxxxxxxxxxxxxxxx]  (in_w = 24)
           padded: [ppxxxxxxxxxxxxxxxxxxxxxxxxp] (pad0 = 2, pad1 = 1)
           output:   [o-o-o-o-o-o-o-o-o-o-o-o]   (out_w = 12)
        Since last (factor - 1) elements are discarded anyway,
        they don't need to be padded
        Hence, pad0 + pad1 = k_w - 1 - (factor - 1)
               pad0 - pad1 = 0 or 1
        """
        p = len(blur_kernel) - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input: Tensor) -> Tensor:
        return upfirdn2d(input,
                         self.kernel,
                         up=self.factor,
                         down=1,
                         pad=self.pad)


class ToRGB(nn.Module):

    def __init__(
        self,
        in_channel: int,
        latent_dim: int,
        up: int,
        blur_kernel: List[int],
    ):
        super().__init__()

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        # Trainable parameters
        self.weight = Parameter(torch.randn(1, 1, in_channel, 1, 1))
        self.scale = 1 / math.sqrt(in_channel)
        self.bias = Parameter(torch.zeros(1, 1, 1, 1))

        if up > 1:
            self.upsample = Upsample(blur_kernel, up)

    def forward(self,
                input: Tensor,
                w: Tensor,
                prev_output: Optional[Tensor] = None) -> Tensor:
        batch, in_channel, _, _ = input.shape

        # Get style vectors (N, 1, C_in, 1, 1)
        style = self.affine(w).view(batch, 1, in_channel, 1, 1)

        # Modulate weights with equalized learning rate (N, C_out, C_in, K_h, K_w)
        weight = mod(self.scale * self.weight, style)

        # Perform convolution and add bias
        out = group_conv(input, weight) + self.bias

        if prev_output is not None:
            out = out + self.upsample(prev_output)

        return out
