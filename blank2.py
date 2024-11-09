# Code eager softmax in PyTorch and Triton

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Eager of softmax"""
    x_max = x.max(dim=1, keepdim=True)[0]
    print(f"x_max: {x_max}")


sample = torch.tensor(
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device="cuda"
)

# use torch functional for softmax
ref_out = F.softmax(sample, dim=1)
