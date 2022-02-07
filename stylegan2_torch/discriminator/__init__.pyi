from typing import Dict, List, Literal, Tuple, overload

from stylegan2_torch import Resolution, default_channels
from torch import nn
from torch.functional import Tensor


class Discriminator(nn.Module):

    def __init__(self,
                 resolution: Resolution,
                 channels: Dict[Resolution, int] = default_channels,
                 blur_kernel: List[int] = [1, 3, 3, 1]):
        ...

    @overload
    def __call__(self,
                 input: Tensor,
                 *,
                 return_features: Literal[False] = False) -> Tensor:
        ...

    @overload
    def __call__(self, input: Tensor, *,
                 return_features: Literal[True]) -> Tuple[Tensor, Tensor]:
        ...

    @overload
    def forward(self,
                input: Tensor,
                *,
                return_features: Literal[False] = False) -> Tensor:
        ...

    @overload
    def forward(self, input: Tensor, *,
                return_features: Literal[True]) -> Tuple[Tensor, Tensor]:
        ...
