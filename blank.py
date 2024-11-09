# Code eager softmax in PyTorch and Triton

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Eager of softmax"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]



@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    rows,
    num_cols,
    block_size: tl.constexpr,
    num_warps,
):
    # setup input pointers
    row_index = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_index * stride_input_row
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets
    
    # move to SRAM
    row = tl.load(input_ptrs, mask=col_offsets < num_cols, other=-float("inf"))
    
    # softmax itself
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator
    
    # write back to HBM
    output_row_ptr = output_ptr + row_index * stride_output_row
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=col_offsets < num_cols)
    

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton implementation of softmax, fwd pass only"""
    rows, cols = x.shape
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    block_size = triton.next_power_of_2(cols)
    
    num_warps = 4 # 32 threads per warp
    if block_size >= 2047: # 2048 is a power of 2
        num_warps = 8
    if block_size >= 4095: # 4096 is a power of 2
        num_warps = 16
    
    grid = (rows,)
    
    # allocate our output buffer
    sm_out = torch.empty_like(x)
    
    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        rows,
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    
    return sm_out
    

sample = torch.tensor(
    [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device="cuda"
)

# use torch functional for softmax
ref_out = F.softmax(sample, dim=1)
print(f"ref_out: {ref_out}")

eager_out = naive_softmax(sample)
print(f"eager_out: {eager_out}")

triton_out = softmax(sample)
print(f"triton_out: {triton_out}")
