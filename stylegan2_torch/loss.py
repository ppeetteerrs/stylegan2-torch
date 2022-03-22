import math
from typing import Literal, Tuple, Union

import torch
from torch import autograd
from torch.functional import Tensor
from torch.nn import functional as F
from torch_conv_gradfix import no_weight_grad


def d_loss(
    real_pred: Tensor,
    fake_pred: Tensor,
) -> Tensor:
    """
    Calculates the discriminator loss.
    (equivalent to adversarial loss in original GAN paper).

    loss = softplus(-f(x)) + softplus(f(x))

    Args:
        real_pred (Tensor): Predicted scores for real images
        fake_pred (Tensor): Predicted scores for fake images

    Returns:
        Tensor: Discriminator loss
    """

    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_reg_loss(real_pred: Tensor, real_img: Tensor) -> Tensor:
    """
    Calculates the discriminator R_1 loss.

    Note:
        The loss function was first proposed in [https://arxiv.org/pdf/1801.04406.pdf](https://arxiv.org/pdf/1801.04406.pdf).
        This regularization term penalizes the discriminator from producing a gradient orthogonal to the true data manifold
        (i.e. Expected gradient w.r.t. real image distribution should be zero). This means that:

        1. Discriminator score cannot improve once generator reaches true data distribution (because discriminator gives same expected score if inputs are from sample distribution, based on this regularization term)
        2. Near Nash equilibrium, discriminator is encouraged to minimize the gradient magnitude (because adversarial loss cannot improve, see 1)

        Points 1 and 2 are sort of chicken-and-egg in nature but the idea is to help converge to the Nash equilibrium.
    """

    # Gradients w.r.t. convolution weights are not needed since only gradients w.r.t. input images are propagated
    with no_weight_grad():
        # create_graph = true because we still need to use this gradient to perform backpropagation
        # real_pred.sum() is needed to obtain a scalar, but does not affect gradients (since each sample independently contributes to output)
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_loss(fake_pred: Tensor) -> Tensor:
    """
    Calculates the generator loss.

    Args:
        fake_pred (Tensor): Predicted scores for fake images

    Returns:
        Tensor: Generator loss
    """
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_reg_loss(
    fake_img: Tensor,
    latents: Tensor,
    mean_path_length: Union[Tensor, Literal[0]],
    decay: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Calculates Generator path length regularization loss.

    Args:
        fake_img (Tensor): Generated images (N, C, H, W)
        latents (Tensor): W+ latent vectors (N, P, 512), P = number of style vectors
        mean_path_length (Union[Tensor, Literal[0]]): Current accumulated mean path length (dynamic `a`)
        decay (float, optional): Decay in accumulating `a`. Defaults to 0.01.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Path loss, mean path, path length

    Note:
        This loss function was first introduced in StyleGAN2.
        The idea is that fixed-sized steps in W results in fixed-magnitude change in image.

        **Key Intuition**: minimizing $\mathbb{E}_{\mathbf{w},\mathbf{y}~N(0,1)}(||\mathbf{J^T_{\mathbf{w}}\mathbf{y}}||_2 - a)^2$ is equivalent to scaling $W+$ equally in each dimension.

        Reason:

        1. Do SVD on $\mathbf{J^T_{\mathbf{w}}} = U \\bar{\Sigma} V^T$
        2. $U$ and $V$ are orthogonal and hence irrelevant (since orthogonal matrices simply rotates the vector, but $\mathbf{y}$ is N(0,1), it is still the same distribution after rotation)
        3. $\\bar{\Sigma}$ has $L$ non-zero singular values representing scaling factor in $L$ dimensions
        4. Loss is minimized when $\\bar{\Sigma}$ has identical singular values equal $\\frac{a}{\sqrt{L}}$ (because high-dimensional normal distributions have norm centered around $\sqrt{L}$)

    Info:
        Implementation:

        1. $a$ is set dynamically using the moving average of the path_lengths (sort of like searching for the appropriate scaling factor in an non-agressive manner).
        2. As explained in paper's Appendix B, ideal weight for path regularization is $\gamma_{pl} = \\frac{\ln 2}{r^2(\ln r - \ln 2)}$. This is achieved by setting `pl_weight`, then in the code, the loss is first scaled by $r^2$ (i.e. height * width) in `noise` then by `n_layers` in `path_lengths` by taken mean over the `n_layers` style vectors. Resulting is equivalent as saying that idea `pl_weight` is 2. See [here](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py).
        3. `path_batch_shrink` controls the fraction of batch size to use to reduce memory footprint of regularization. Since it is done without freeing the memory of the existing batch.
        4. Identity $\mathbf{J^T_{\mathbf{w}}} \mathbf{y} = \\nabla (g(\mathbf{w}) \mathbf{y})$

    """

    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    (grad,) = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths
