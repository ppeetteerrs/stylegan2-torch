import math
from typing import List

from stylegan2_torch.equalized_lr import Blur, EqualConv2d
from stylegan2_torch.op.fused_act import FusedLeakyReLU
from stylegan2_torch.utils import proxy
from torch import nn
from torch.functional import Tensor


class ConvBlock(nn.Sequential):
    """
    Convolution in feature space

    EqualConv2d: 2D convolution with equalized learning rate
    FusedLeakyReLU: LeakyReLU with a bias added before activation
    """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int):
        super().__init__(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
            FusedLeakyReLU(out_channel, bias=True),
        )

    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class DownConvBlock(nn.Sequential):
    """
    Downsampling convolution in feature space

    Blur: Gaussian filter as low-pass filter for anti-aliasing + adjust tensor shape to preserve downsampled tensor shape
    EqualConv2d: 2D (downsampling) convolution with equalized learning rate
    FusedLeakyReLU: LeakyReLU with a bias added before activation
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        down: int,
        blur_kernel: List[int],
    ):
        super().__init__(
            Blur(blur_kernel, -down, kernel_size),
            EqualConv2d(
                in_channel, out_channel, kernel_size, padding=0, stride=down, bias=False
            ),
            FusedLeakyReLU(out_channel, bias=True),
        )

    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class RGBDown(nn.Sequential):
    """
    Downsampling convolution in RGB space, hence no need nonlinearity

    Blur: Gaussian filter as low-pass filter for anti-aliasing + adjust tensor shape to preserve downsampled tensor shape
    EqualConv2d: 2D (downsampling) convolution with equalized learning rate
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        down: int,
        blur_kernel: List[int],
    ):

        super().__init__(
            Blur(blur_kernel, -down, kernel_size),
            EqualConv2d(
                in_channel, out_channel, kernel_size, padding=0, stride=down, bias=False
            ),
        )

    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class ResBlock(nn.Module):
    """
    Residual block

    ConvBlock + DownConvBlock: Convolution + downsampling
    RGBDown: Skip connection from higher (double) resolution RGB image
    """

    def __init__(self, in_channel: int, out_channel: int, blur_kernel: List[int]):
        super().__init__()

        self.conv = ConvBlock(in_channel, in_channel, 3)
        self.down_conv = DownConvBlock(
            in_channel, out_channel, 3, down=2, blur_kernel=blur_kernel
        )
        self.skip = RGBDown(in_channel, out_channel, 1, down=2, blur_kernel=blur_kernel)

    def forward(self, input: Tensor) -> Tensor:
        out = self.conv(input)
        out = self.down_conv(out)
        skip = self.skip(input)
        # sqrt 2 to adhere to equalized learning rate philosophy
        # (i.e. preserve variance in forward pass not initialization)
        return (out + skip) / math.sqrt(2)

    __call__ = proxy(forward)
