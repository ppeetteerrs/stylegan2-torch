from typing import Dict, List, Literal, Optional, Tuple, overload

from stylegan2_torch import Resolution, default_channels
from torch import nn
from torch.functional import Tensor


class ConstantInput(nn.Module):

    def __init__(self, channels: int, size: Resolution):
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...


class Generator(nn.Module):

    def __init__(self,
                 resolution: Resolution,
                 latent_dim: int = 512,
                 n_mlp: int = 8,
                 lr_mlp_mult: float = 0.01,
                 channels: Dict[Resolution, int] = default_channels,
                 blur_kernel: List[int] = [1, 3, 3, 1]):
        ...

    def make_noise(self) -> List[Tensor]:
        ...

    def mean_latent(self, n_sample: int) -> Tensor:
        ...

    @overload
    def __call__(self,
                 input: List[Tensor],
                 *,
                 return_latents: Literal[False] = False,
                 input_type: Literal["z", "w", "w_plus"] = "z",
                 trunc_option: Optional[Tuple[float, Tensor]] = None,
                 mix_index: Optional[int] = None,
                 noises: Optional[List[Optional[Tensor]]] = None) -> Tensor:
        ...

    @overload
    def __call__(
        self,
        input: List[Tensor],
        *,
        return_latents: Literal[True],
        input_type: Literal["z", "w", "w_plus"] = "z",
        trunc_option: Optional[Tuple[float, Tensor]] = None,
        mix_index: Optional[int] = None,
        noises: Optional[List[Optional[Tensor]]] = None
    ) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def forward(self,
                input: List[Tensor],
                *,
                return_latents: Literal[False] = False,
                input_type: Literal["z", "w", "w_plus"] = "z",
                trunc_option: Optional[Tuple[float, Tensor]] = None,
                mix_index: Optional[int] = None,
                noises: Optional[List[Optional[Tensor]]] = None) -> Tensor:
        ...

    @overload
    def forward(
        self,
        input: List[Tensor],
        *,
        return_latents: Literal[True],
        input_type: Literal["z", "w", "w_plus"] = "z",
        trunc_option: Optional[Tuple[float, Tensor]] = None,
        mix_index: Optional[int] = None,
        noises: Optional[List[Optional[Tensor]]] = None
    ) -> Tuple[Tensor, Tensor]:
        ...
