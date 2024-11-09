BLOCK = 512

# This is a GPU kernel in Numba.
# Different instances of this
# function may run in parallel.
@jit
def add(X, Y, Z, N):
   # In Numba/CUDA, each kernel 
   # instance itself uses an SIMT execution
   # model, where instructions are executed in
   # parallel for different values of threadIdx
   tid = threadIdx.x
   bid = blockIdx.x
   # scalar index
   idx = bid * BLOCK + tid
   if id < N:
     # There is no pointer in Numba.
     # Z,X,Y are dense tensors
     Z[idx] = X[idx] + Y[idx]


...
grid = (ceil_div(N, BLOCK),)
block = (BLOCK,)
add[grid, block](x, y, z, x.shape[0])