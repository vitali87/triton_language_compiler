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

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Online softmax, 2.5x faster than naive"""
    row_count, col_count = x.shape
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    output = torch.empty_like(x)
    for r in range(row_count):
        row_max = 0 # m
        normalizer = 0 # l
        for c in range(col_count):
            curr = x[r, c]
            prev_row_max = row_max
            row_max = max(row_max, curr)
            if row_max > prev_row_max:
                print(f"updated row_max from {prev_row_max} to {row_max}, row {r}")
            normalizer = normalizer * torch.exp(prev_row_max - row_max) + torch.exp(curr - row_max)
        output[r, :] = torch.exp(x[r, :] - row_max) / normalizer
    return output


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

import time

# use torch functional for softmax
time_start = time.perf_counter()
ref_out = F.softmax(sample, dim=1)
time_end = time.perf_counter()
print(f"ref_out: {ref_out}")
print(f"ref_out time: {time_end - time_start}")

time_start = time.perf_counter()
eager_out = naive_softmax(sample)
time_end = time.perf_counter()
print(f"eager_out: {eager_out}")
print(f"eager_out time: {time_end - time_start}")

time_start = time.perf_counter()
triton_out = softmax(sample)
time_end = time.perf_counter()
print(f"triton_out: {triton_out}")
print(f"triton_out time: {time_end - time_start}")

time_start = time.perf_counter()
online_out = online_softmax(sample)
time_end = time.perf_counter()
print(f"online_out: {online_out}")
print(f"online_out time: {time_end - time_start}")
