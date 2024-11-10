import torch

M = 27
N = 24
K = 12

A = torch.randn(M, K)
B = torch.randn(K, N)
# we want M N K to be divisible by 3 to have evenly sized tiles

output = torch.zeros(M, N)

block_M = M // 3
block_N = N // 3
block_K = K // 3

total_reads = 0
total_writes = 0
for start_M in range(0, M, block_M):
    stop_M = start_M + block_M
    for start_N in range(0, N, block_N):
        stop_N = start_N + block_N
        accumulator = torch.zeros(block_M, block_N)
        for start_K in range(0, K, block_K):
            stop_K = start_K + block_K
            tileA = A[start_M:stop_M, start_K:stop_K]
            tileB = B[start_K:stop_K, start_N:stop_N]
            total_reads += tileA.numel() + tileB.numel()
            accumulator += tileA @ tileB
        output[start_M:stop_M, start_N:stop_N] = accumulator
        total_writes += accumulator.numel()

print(output)
print(A @ B)
torch.allclose(output, A @ B)
print(f"Total reads: {total_reads}")
print(f"Total writes: {total_writes}")
