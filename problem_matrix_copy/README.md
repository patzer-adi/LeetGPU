# Matrix Copy — CUDA Solution

**Problem:** LeetGPU · Matrix Copy (Easy)

---

## What does this program do?

Copies every element of an N×N matrix `A` into matrix `B` on the GPU:

```
B[i][j] = A[i][j]   for all 0 ≤ i, j < N
```

This is purely a **memory bandwidth** exercise — no arithmetic, no shared
memory, no synchronisation. The GPU's performance here is limited entirely by
how fast it can read and write global memory.

---

## Memory Layout

Both matrices are stored as **flat, row-major 1D arrays** of length N×N:

```
2D index (row, col)  ←→  flat index  row * N + col

Matrix:               Flat array:
 A[0][0]  A[0][1]     [A[0][0], A[0][1], A[1][0], A[1][1], ...]
 A[1][0]  A[1][1]      index 0   index 1   index 2   index 3
```

Because the layout is already flat and contiguous, a single thread ID `tid`
maps directly to one element — no 2D index calculation needed at all.

---

## Files

| File | Purpose |
|---|---|
| `solution.cu` | CUDA kernel + `solve()` entry point + optional test harness |
| `Makefile` | Builds the shared library and/or the test binary |
| `README.md` | This file |

---

## Code Walkthrough — Line by Line

### The Kernel

```cuda
__global__ void copy_matrix_kernel(const float* __restrict__ A,
                                    float*       __restrict__ B,
                                    int N)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (unsigned int)(N * N))
        B[tid] = A[tid];
}
```

| Line | What it does |
|---|---|
| `__global__` | Marks this as a GPU kernel — callable from CPU, runs on GPU |
| `__restrict__` | Promises the compiler that `A` and `B` don't overlap in memory, enabling better load/store scheduling |
| `blockIdx.x * blockDim.x + threadIdx.x` | Standard 1D thread index formula — every thread gets a unique `tid` |
| `if (tid < N * N)` | Bounds check — the last block may have threads beyond the array end |
| `B[tid] = A[tid]` | The entire computation: one load from `A`, one store to `B` |

### The `solve` Function

```cuda
extern "C" void solve(const float* A, float* B, int N)
{
    int total           = N * N;        // total elements to copy
    int threadsPerBlock = 256;          // 256 = safe, fast default for all GPUs
    int blocksPerGrid   = (total + threadsPerBlock - 1) / threadsPerBlock;

    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();            // wait for GPU to finish before returning
}
```

**Why 256 threads per block?**  
256 is a multiple of the warp size (32), fits within all GPU register limits,
and gives the scheduler enough warps per SM to hide memory latency.  
For a pure copy kernel like this, anywhere from 128–512 works equally well.

**Why `(total + threadsPerBlock - 1) / threadsPerBlock`?**  
This is integer ceiling division. Example: if total = 1000 and
threadsPerBlock = 256, you need `ceil(1000/256) = 4` blocks (which launch
1024 threads). The extra 24 threads are blocked by the `if (tid < N*N)` guard.

---

## Was the Original Code Correct?

**Yes — the logic was completely correct.** The original would produce the right
answer on the judge. The only two things changed for cleanliness:

| Item | Original | Improved |
|---|---|---|
| `__restrict__` | absent | added — signals no pointer aliasing |
| `uint` type | `uint tid` | `unsigned int tid` — more portable, no implicit typedef needed |

These are style/performance micro-improvements, not correctness fixes.

---

## Why is This Purely Memory-Bound?

Each thread does:
- **1 global memory read** — `A[tid]`
- **1 global memory write** — `B[tid]`
- **0 arithmetic operations**

The GPU's compute units sit idle. Performance is determined entirely by
**global memory bandwidth**. On an RTX 2080 (~448 GB/s), copying a 4096×4096
float matrix (64 MB) takes approximately:

```
64 MB read + 64 MB write = 128 MB total traffic
128 MB / 448 GB/s ≈ 0.28 ms
```

This is why the judge uses N=4096 for the perf test — it's large enough to
saturate the memory bus.

---

## Building

> Requires: NVIDIA CUDA Toolkit ≥ 11.x, `nvcc` on PATH.

### Adjust GPU architecture

Edit `CUDA_ARCH` in the `Makefile`:

| GPU | `CUDA_ARCH` |
|---|---|
| RTX 2070 / 2080 | `sm_75` |
| RTX 3080 / 3090 | `sm_86` |
| RTX 4090 | `sm_89` |
| A100 | `sm_80` |

### Build the shared library (for the judge)

```bash
make lib
# produces: matrix_copy.so
```

### Build and run tests

```bash
make test
```

Expected output:

```
Test Example 1 (2x2)        N=2      PASS ✓
Test Example 2 (3x3)        N=3      PASS ✓
Test Stress (4096x4096)     N=4096   PASS ✓
```

### Clean

```bash
make clean
```

---

## Example Walkthrough

### Example 1 — N=2

```
A (flat): [1.0, 2.0, 3.0, 4.0]

tid=0: B[0] = A[0] = 1.0   → row 0, col 0
tid=1: B[1] = A[1] = 2.0   → row 0, col 1
tid=2: B[2] = A[2] = 3.0   → row 1, col 0
tid=3: B[3] = A[3] = 4.0   → row 1, col 1

B (flat): [1.0, 2.0, 3.0, 4.0] ✓
```

All 4 threads run in a **single warp** (warp size = 32). The GPU launches one
block of 256 threads; 252 threads hit the `if` guard and do nothing.

---

## Complexity

| | Value |
|---|---|
| Time complexity | O(N²) — one operation per element, fully parallel |
| Threads launched | N² (padded to next multiple of 256) |
| Memory read | N² × 4 bytes |
| Memory write | N² × 4 bytes |
| Arithmetic ops | 0 |
| Shared memory | None |
| Bottleneck | Global memory bandwidth |
