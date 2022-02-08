import os
from collections import abc
from distutils.util import strtobool
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load

# Load Pytorch extension
module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d_",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
    verbose=strtobool(os.environ.get("STYLEGAN2_BUILD_VERBOSE", "f")),
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx: Any,
        grad_output: Tensor,
        kernel: Tensor,
        grad_kernel: Tensor,
        up: Tuple[int, int],
        down: Tuple[int, int],
        pad: Tuple[int, int, int, int],
        g_pad: Tuple[int, int, int, int],
        in_size: Tuple[int, int, int, int],
        out_size: Tuple[int, int],
    ) -> Tensor:
        # Destructuring
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, 1, out_size[0], out_size[1])

        # Gradient equals to applying sampling in reverse order with flipped kernel
        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        # Save kernel for double derivative
        ctx.save_for_backward(kernel)

        # Context caching
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(
        ctx: Any, gradgrad_input: Tensor
    ) -> Tuple[Optional[Tensor], None, None, None, None, None, None, None, None]:
        # Load saved kernel
        (kernel,) = ctx.saved_tensors

        # Only compute gradient if requested
        gradgrad_out = None
        if ctx.needs_input_grad[0]:
            gradgrad_input = gradgrad_input.reshape(
                -1, 1, ctx.in_size[2], ctx.in_size[3]
            )
            gradgrad_out = upfirdn2d_op.upfirdn2d(
                gradgrad_input.contiguous(),
                kernel,
                ctx.up_x,
                ctx.up_y,
                ctx.down_x,
                ctx.down_y,
                ctx.pad_x0,
                ctx.pad_x1,
                ctx.pad_y0,
                ctx.pad_y1,
            ).view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        kernel: Tensor,
        up: Tuple[int, int],
        down: Tuple[int, int],
        pad: Tuple[int, int, int, int],
    ) -> Tensor:
        # Destructuring
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        _, channel, in_h, in_w = input.shape

        # Cache input shape for backward function
        ctx.in_size = input.shape

        # Reduce input to 3 dimensions
        input = input.reshape(-1, 1, in_h, in_w)

        # Cache kernel and flipped kernel for backward function
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        # Calculate output size
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

        # Calculate gradient padding
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        # Context caching
        ctx.out_size = (out_h, out_w)
        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)
        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        # Apply kernel
        out = upfirdn2d_op.upfirdn2d(
            input,
            kernel,
            up_x,
            up_y,
            down_x,
            down_y,
            pad_x0,
            pad_x1,
            pad_y0,
            pad_y1,
        )

        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[Optional[Tensor], None, None, None, None]:

        # Load saved kernel
        kernel, grad_kernel = ctx.saved_tensors

        # Only compute gradient if requested
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn2dBackward.apply(
                grad_output.contiguous(),
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn2d(
    input: Tensor,
    kernel: Tensor,
    up: Union[int, Tuple[int, int]] = 1,
    down: Union[int, Tuple[int, int]] = 1,
    pad: Union[Tuple[int, int], Tuple[int, int, int, int]] = (0, 0),
) -> Tensor:

    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    out = UpFirDn2d.apply(input, kernel, up, down, pad)

    return out
