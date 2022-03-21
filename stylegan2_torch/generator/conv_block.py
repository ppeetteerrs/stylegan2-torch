import math
from typing import List, Optional

import torch
from stylegan2_torch.equalized_lr import Blur, EqualLinear
from stylegan2_torch.op.fused_act import FusedLeakyReLU
from stylegan2_torch.utils import proxy
from torch import nn
from torch.functional import Tensor
from torch.nn.parameter import Parameter
from torch_conv_gradfix import conv2d, conv_transpose2d


def mod(weight: Tensor, style: Tensor) -> Tensor:
    """
    Modulate convolution weights with style vector
    (styling = scale each input feature map before convolution)

    Args:
        weight (Tensor): (1, C_out, C_in, K_h, K_w)
        style (Tensor): (N, 1, C_in, 1, 1)

    Returns:
        Tensor: (N, C_out, C_in, K_h, K_w)
    """
    return weight * style


def demod(weight: Tensor) -> Tensor:
    """
    Demodulate convolution weights
    (normalization = statistically restore output feature map to unit s.d.)

    Args:
        weight (Tensor): (N, C_out, C_in, K_h, K_w)

    Returns:
        Tensor: (N, C_out, C_in, K_h, K_w)
    """
    batch, out_channel, _, _, _ = weight.shape
    demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(
        batch, out_channel, 1, 1, 1
    )
    return weight * demod


def group_conv(input: Tensor, weight: Tensor) -> Tensor:
    """
    Efficiently perform modulated convolution
    (i.e. grouped convolution)

    Args:
        input (Tensor): (N, C_in, H, W)
        weight (Tensor): (N, C_out, C_in, K, K)

    Returns:
        Tensor: (N, C, H + K - 1, W + K - 1)
    """
    batch, in_channel, height, width = input.shape
    _, out_channel, _, k_h, k_w = weight.shape

    weight = weight.view(batch * out_channel, in_channel, k_h, k_w)
    input = input.view(1, batch * in_channel, height, width)

    out = conv2d(input=input, weight=weight, padding=k_h // 2, groups=batch)
    return out.view(batch, out_channel, height, width)


class AddNoise(nn.Module):
    """
    Inject white noise scaled by a learnable scalar (same noise for whole batch)
    """

    def __init__(self):
        super().__init__()

        # Trainable parameters
        self.weight = Parameter(torch.zeros(1))

    def forward(self, input: Tensor, noise: Optional[Tensor]) -> Tensor:
        if noise is None:
            batch, _, height, width = input.shape
            noise = input.new_empty(batch, 1, height, width).normal_()

        return input + self.weight * noise

    __call__ = proxy(forward)


class ModConvBlock(nn.Module):
    """
    Modulated convolution block

    disentangled latent vector (w) => affine transformation => style vector
    style vector => modulate + demodulate convolution weights => new conv weights
    new conv weights & input features => group convolution => output features
    output features => add noise & leaky ReLU => final output features
    """

    def __init__(
        self, in_channel: int, out_channel: int, kernel_size: int, latent_dim: int
    ):
        super().__init__()

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        # Trainable parameters
        self.weight = Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        # Noise and Leaky ReLU
        self.add_noise = AddNoise()
        self.leaky_relu = FusedLeakyReLU(out_channel)

    def forward(self, input: Tensor, w: Tensor, noise: Optional[Tensor]) -> Tensor:
        batch, in_channel, _, _ = input.shape

        # Get style vectors (N, 1, C_in, 1, 1)
        style = self.affine(w).view(batch, 1, in_channel, 1, 1)

        # Modulate weights with equalized learning rate (N, C_out, C_in, K_h, K_w)
        weight = mod(self.scale * self.weight, style)

        # Demodulate weights
        weight = demod(weight)

        # Perform convolution
        out = group_conv(input, weight)

        # Add noise
        out = self.add_noise(out, noise)

        # Add learnable bias and activate
        return self.leaky_relu(out)

    __call__ = proxy(forward)


def group_conv_up(input: Tensor, weight: Tensor, up: int = 2) -> Tensor:
    """
    Efficiently perform upsampling + modulated convolution
    (i.e. grouped transpose convolution)

    Args:
        input (Tensor): (N, C_in, H, W)
        weight (Tensor): (N, C_out, C_in, K, K)
        up (int, optional): U. Defaults to 2.

    Returns:
        Tensor: (N, C, (H - 1) * U + K - 1 + 1, (W - 1) * U + K - 1 + 1)
    """
    batch, in_channel, height, width = input.shape
    _, out_channel, _, k_h, k_w = weight.shape

    weight = weight.transpose(1, 2).reshape(batch * in_channel, out_channel, k_h, k_w)
    input = input.view(1, batch * in_channel, height, width)
    out = conv_transpose2d(
        input=input, weight=weight, stride=up, padding=0, groups=batch
    )
    _, _, out_h, out_w = out.shape
    return out.view(batch, out_channel, out_h, out_w)


class UpModConvBlock(nn.Module):
    """
    Modulated convolution block with upsampling

    disentangled latent vector (w) => affine transformation => style vector
    style vector => modulate + demodulate convolution weights => new conv weights
    new conv weights & input features => group convolution and upsampling => output features
    output features => add noise & leaky ReLU => final output features
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        latent_dim: int,
        up: int,
        blur_kernel: List[int],
    ):
        super().__init__()

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        # Trainable parameters
        self.weight = Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        # Blurring kernel
        self.up = up
        self.blur = Blur(blur_kernel, up, kernel_size)

        # Noise and Leaky ReLU
        self.add_noise = AddNoise()
        self.leaky_relu = FusedLeakyReLU(out_channel)

    def forward(self, input: Tensor, w: Tensor, noise: Optional[Tensor]) -> Tensor:
        batch, in_channel, _, _ = input.shape

        # Get style vectors (N, 1, C_in, 1, 1)
        style = self.affine(w).view(batch, 1, in_channel, 1, 1)

        # Modulate weights with equalized learning rate (N, C_out, C_in, K_h, K_w)
        weight = mod(self.scale * self.weight, style)

        # Demodulate weights
        weight = demod(weight)

        # Reshape to use group convolution
        out = group_conv_up(input, weight, self.up)

        # Apply blurring filter for anti-aliasing (linear operation so order doesn't matter?)
        out = self.blur(out)

        # Add noise
        out = self.add_noise(out, noise)

        # Add learnable bias and activate
        return self.leaky_relu(out)

    __call__ = proxy(forward)
