# import torch
# import torch.nn.functional as F
# import triton
# import triton.language as tl
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def naive_softmax(x: torch.Tensor) -> torch.Tensor:
#     """eager model softmax"""
#     x_max = x.max(dim=1)[0]
#     safe_x = x - x_max[:, None]
#     numerator = torch.exp(safe_x)
#     denominator = numerator.sum(dim=1)
#     return numerator / denominator[:, None]


# @triton.jit
# def _softmax_fwd_kernel(output_ptr: torch.Tensor,
#                         stride_output_raw: int,
#                         input_ptr:torch.Tensor,
#                         stride_input_raw: int,
#                         num_cols: int,
#                         block_size: tl.constexpr):
#     # setup input pointers
#     row_index = tl.program_id(0)
    
#     raw_start_ptr = input_ptr + row_index * stride_input_raw
#     col_offsets = tl.arange(0,block_size)
#     input_ptrs = raw_start_ptr + col_offsets
    
#     mask=col_offsets < num_cols
    
#     # move to SRAM
#     row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
#     # softmax itself
#     safe_row = row - tl.max(row, axis=0)
#     numerator = tl.exp(safe_row)
#     denominator = tl.sum(numerator, axis=0)
#     sm_out = numerator / denominator
    
#     # write back to HBM
#     output_row_ptr = output_ptr + row_index * stride_output_raw
#     output_pointers = output_row_ptr + col_offsets
#     tl.store(output_pointers, sm_out, mask=mask)
    
    


# def softmax(x: torch.Tensor) -> torch.Tensor:
#     """Triton impl of Softmax, fwd pass only"""
#     rows, cols = x.shape
#     assert x.dim() == 2, "softmax expects a 2D tensor for now"
    
#     block_size = triton.next_power_of_2(cols)
#     num_warps = 4
#     if block_size < 2047:
#         num_warps = 8
#     if block_size < 4095:
#         num_warps = 16
        
#     # define our grid
#     grid = (rows,)
    
#     # allocate our output buffer
#     sm_output = torch.empty_like(x)
    
#     _softmax_fwd_kernel[grid](
#         sm_output, 
#         sm_output.stride(0), 
#         x, 
#         x.stride(0), 
#         cols,
#         block_size,
#         num_warps=num_warps
#     )


# sample = torch.tensor(
#     [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device=device
# )

# print(sample.dim())
# ref_out = F.softmax(sample, dim=1)
# print(f"ref_out: {ref_out}")

# eager_out = naive_softmax(sample)
# print(f"eager_out: {eager_out}")

# triton_out = softmax(sample)
# print(f"triton_out: {triton_out}")

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# First, explicitly check CUDA availability and print device info
if not torch.cuda.is_available():
    raise RuntimeError("This code requires CUDA GPU")

device = torch.device("cuda")
print(f"Using device: {device}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """eager model softmax"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]

@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    input_ptr,
    stride_input_row,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute the input/output pointers for this row
    row_start_ptr = input_ptr + row_idx * stride_input_row
    
    # Create offsets for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Create mask for valid columns
    mask = col_offsets < num_cols
    
    # Load input row with mask
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Store the results
    output_row_ptr = output_ptr + row_idx * stride_input_row
    tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton implementation of Softmax forward pass"""
    # Input validation
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 2, "Input must be a 2D tensor"
    
    # Extract dimensions
    rows, cols = x.shape
    
    # Compute block size
    BLOCK_SIZE = triton.next_power_of_2(cols)
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Enqueue kernel
    grid = (rows,)
    
    _softmax_fwd_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        stride_input_row=x.stride(0),
        num_cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

if __name__ == "__main__":
    # Create sample input
    sample = torch.tensor(
        [[1, 2, 3, 4, 5], 
         [5, 4, 3, 2, 1]], 
        dtype=torch.float32, 
        device=device
    )

    print("\nInput shape:", sample.shape)
    print("Input device:", sample.device)

    # Compute reference output using PyTorch
    ref_out = F.softmax(sample, dim=1)
    print("\nReference output:")
    print(ref_out)

    # Compute output using naive implementation
    eager_out = naive_softmax(sample)
    print("\nNaive implementation output:")
    print(eager_out)

    # Compute output using Triton implementation
    triton_out = softmax(sample)
    print("\nTriton implementation output:")
    print(triton_out)

    # Verify results
    print("\nMax absolute difference from reference:")
    print("Naive implementation:", torch.max(torch.abs(ref_out - eager_out)).item())
    print("Triton implementation:", torch.max(torch.abs(ref_out - triton_out)).item())