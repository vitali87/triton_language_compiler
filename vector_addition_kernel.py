# Triton kernel for vector addition

import triton
import triton.language as tl
import torch

def ceil_div(x:int, y:int) -> int:
    """
Let me explain the `ceil_div` function. This is a common utility function used in parallel programming, particularly when dividing work into blocks or grids.

The function performs "ceiling division" - meaning it divides two numbers and rounds up to the nearest integer. Here's how it works:

1. Input:
   - `x`: dividend (the number being divided)
   - `y`: divisor (the number we're dividing by)

2. The formula `(x + y - 1) // y`:
   - First adds `(y - 1)` to `x`
   - Then performs integer division (`//`) by `y`
   - This effectively rounds up the division result

Example:
```python
ceil_div(10, 3) = (10 + 3 - 1) // 3 = 12 // 3 = 4
# Regular division: 10 / 3 = 3.33...
# ceil_div rounds this up to 4
```

In your code's context, this is being used to calculate `grid_size` from `num_elems` and `block_size`. This is a common pattern in GPU programming where you need to ensure you have enough blocks to cover all elements, even if the number of elements isn't perfectly divisible by the block size.

For instance, if you have:
- 1000 elements (`num_elems`)
- block size of 128
- Regular division: 1000 ÷ 128 = 7.8125
- `ceil_div(1000, 128)` = 8 blocks

This ensures you have enough blocks to process all elements, even though the last block won't be fully utilized.
    """
    return (x + y - 1) // y


@triton.jit
def kernel_vector_addition(a_ptr, 
                           b_ptr, 
                           output_ptr, 
                           num_elems:tl.constexpr, 
                           block_size:tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size # 0 * 2 = 0, 1 * 2 = 2, 2 * 2 = 4,
    thread_offset = block_start + tl.arange(0, block_size)
    mask = thread_offset < num_elems
    
    a_pointer = tl.load(a_ptr + thread_offset, mask=mask)
    b_pointer = tl.load(b_ptr + thread_offset, mask=mask)
    output_pointer = tl.load(output_ptr + thread_offset, mask=mask)

    output_pointer = a_pointer + b_pointer
    tl.store(output_ptr + thread_offset, output_pointer, mask=mask)


def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output_buffer = torch.empty_like(a)
    assert a.is_cuda and b.is_cuda
    num_elems = a.numel()
    assert num_elems == b.numel()
    
    # block_size = 128
    block_size = 1024 # trying a bigger size
    grid_size = ceil_div(num_elems, block_size)
    num_warps = 8
    grid = (grid_size, )
    k2 = kernel_vector_addition[grid](a, b, output_buffer, num_elems, block_size=block_size, num_warps=num_warps)
    return output_buffer

def verify_numerical_fidelity():
    torch.manual_seed(2020) # set seed for reproducibility, seed on both CPU and GPU
    vector_size = 8192
    a = torch.randn(vector_size).cuda()
    b = torch.randn(vector_size).cuda()
    output = vector_addition(a, b)
    fidelity = torch.allclose(output, a + b)
    print(f"Fidelity: {fidelity}")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2 ** i for i in range(10, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-addition-performance",
        args={}
    )
)

def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_addition(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    verify_numerical_fidelity()
    benchmark.run(show_plots=True, print_data=True, save_path="./vector_addition_performance")