# StyleGAN2 Pytorch - Typed, Commented, Installable :)

![action](https://img.shields.io/github/workflow/status/ppeetteerrs/stylegan2-torch/build?logo=githubactions&logoColor=white)
[![pypi](https://img.shields.io/pypi/v/stylegan2-torch.svg)](https://pypi.python.org/pypi/stylegan2-torch)
[![anaconda](https://img.shields.io/conda/vn/ppeetteerrs/stylegan2-torch?logo=anaconda)](https://anaconda.org/ppeetteerrs/stylegan2-torch)
![platform](https://img.shields.io/conda/pn/ppeetteerrs/stylegan2-torch?label=platform&color=blueviolet)
[![codecov](https://img.shields.io/codecov/c/github/ppeetteerrs/stylegan2-torch?label=codecov&logo=codecov)](https://app.codecov.io/gh/ppeetteerrs/stylegan2-torch)
[![docs](https://img.shields.io/github/deployments/ppeetteerrs/stylegan2-torch/github-pages?label=docs&logo=readthedocs)](https://ppeetteerrs.github.io/stylegan2-torch)

This implementation is adapted from [here](https://github.com/rosinality/stylegan2-pytorch). This implementation seems more stable and editable than the over-engineered official implementation.

The focus of this repository is simplicity and readability. If there are any bugs / issues, please kindly let me know or submit a pull request!

Refer to [my blog post](https://ppeetteerrsx.com/post/cuda/stylegan_cuda_kernels/) for an explanation on the custom CUDA kernels. The profiling code to optimize the custom operations is [here](https://github.com/ppeetteerrs/pytorch-cuda-kernels).

## Installation

```bash
pip install stylegan2-torch
```

## Training Tips

1. Use a multi-GPU setup. An RTX 3090 can handle batch size of up to 8 at 1024 resolution. Based on experience, batch size of 8 works but 16 or 32 should be safer.
2. Use LMDB dataset + SSD storage + multiple dataloader workers (and a big enough prefetch factor to cache at least one batch ahead). You never know how much time you waste on dataloading until you optimize it. For me, that shorted the training time by 30% (more time-saving than the custom CUDA kernels).

## Known Issues

Pytorch is known to cause random reboots when using non-deterministic algorithms. Set `torch.use_deterministic_algorithms(True)` if you encounter that.

## To Dos / Won't Dos
1. Tidy up `conv2d_gradfix.py` and `fused_act.py`. These were just copied over from the original repo so they are still ugly and untidy.
2. Provide pretrained model conversion method (not that hard tbh, just go map the state_dict keys).
3. Clean up util functions to aid training loop design.