import math
from typing import List

import torch
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_conv_gradfix import conv2d

from stylegan2_torch.op.fused_act import fused_leaky_relu
from stylegan2_torch.op.upfirdn2d import upfirdn2d
from stylegan2_torch.utils import make_kernel, proxy


class EqualConv2d(nn.Module):
    """
    Conv2d with equalized learning rate
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        # Equalized Learning Rate
        self.weight = Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        # std = gain / sqrt(fan_in)
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)
        self.stride = stride
        self.padding = padding
        self.bias = Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return conv2d(
            input=input,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

    __call__ = proxy(forward)


class EqualLinear(nn.Module):
    """
    Linear with equalized learning rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias_init: int = 0,
        lr_mult: float = 1,
    ):
        super().__init__()

        # Equalized Learning Rate
        self.weight = Parameter(torch.randn(out_dim, in_dim).div_(lr_mult))

        self.bias = Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.scale = (1 / math.sqrt(in_dim)) * lr_mult

        self.lr_mult = lr_mult

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mult)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

    __call__ = proxy(forward)


class EqualLeakyReLU(nn.Module):
    """
    Leaky ReLU with equalized learning rate
    """

    def __init__(self, in_dim: int, out_dim: int, lr_mult: float = 1):
        super().__init__()

        # Equalized Learning Rate
        self.weight = Parameter(torch.randn(out_dim, in_dim).div_(lr_mult))

        self.bias = Parameter(torch.zeros(out_dim))

        self.scale = (1 / math.sqrt(in_dim)) * lr_mult

        self.lr_mult = lr_mult

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.weight * self.scale)
        return fused_leaky_relu(out, self.bias * self.lr_mult)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

    __call__ = proxy(forward)


class Blur(nn.Module):
    def __init__(self, blur_kernel: List[int], factor: int, kernel_size: int):
        """
        Apply blurring FIR filter (before / after) a (downsampling / upsampling) op

        Case 1: Upsample (factor > 0)
            Applied after a transpose convolution of stride U and kernel size K

        Args:
            input (Tensor): (N, C, (H - 1) * U + K - 1 + 1, (W - 1) * U + K - 1 + 1)
            blur_kernel (Tensor): FIR filter
            factor (int, optional): U. Defaults to 2.
            kernel_size (int, optional): K. Defaults to 3.

        Returns:
            Tensor: (N, C, H * U, W * U)


        Case 2: Downsample (factor < 0)
            Applied before a convolution of stride U and kernel size K

        Args:
            input (Tensor): (N, C, H, W)
            blur_kernel (Tensor): FIR filter
            factor (int, optional): U. Defaults to 2.
            kernel_size (int, optional): K. Defaults to 3.

        Returns:
            Tensor: (N, C, H - (U + 1) + K - 1, H  - (U + 1) + K - 1)
        """
        super().__init__()

        if factor > 0:
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
        else:
            p = (len(blur_kernel) - abs(factor)) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

        # Factor to compensate for averaging with zeros if upsampling
        self.kernel: Tensor
        self.register_buffer(
            "kernel", make_kernel(blur_kernel, factor if factor > 0 else 1)
        )
        self.pad = (pad0, pad1)

    def forward(self, input: Tensor) -> Tensor:
        return upfirdn2d(input, self.kernel, pad=self.pad)

    __call__ = proxy(forward)
