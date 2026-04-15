# LeetGPU – Matrix Addition (Easy)

A CUDA solution for the **Matrix Addition** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `matrix_addition.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [Project Structure](#project-structure)
6. [Requirements](#requirements)
7. [Building & Running](#building--running)
8. [VSCode Integration](#vscode-integration)
9. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Given two square matrices `A` and `B` of identical dimensions **N × N**, containing
32-bit floating-point numbers stored in row-major format, compute their element-wise
sum and store it in output matrix `C`:

```
C[i][j] = A[i][j] + B[i][j]   for all i, j in [0, N)
```

Because the matrices are square and stored flat in memory, every element has a single
linear index `i` where `0 ≤ i < N*N`. The operation reduces to:

```
C[i] = A[i] + B[i]
```

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
A = [[1.0, 2.0],      B = [[5.0, 6.0],      C = [[ 6.0,  8.0],
     [3.0, 4.0]]           [7.0, 8.0]]            [10.0, 12.0]]
```

### Example 2

```
A = [[1.5, 2.5, 3.5],     B = [[0.5, 0.5, 0.5],     C = [[ 2.0,  3.0,  4.0],
     [4.5, 5.5, 6.5],          [0.5, 0.5, 0.5],           [ 5.0,  6.0,  7.0],
     [7.5, 8.5, 9.5]]          [0.5, 0.5, 0.5]]            [ 8.0,  9.0, 10.0]]
```

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N         | 1 – 4096 |

Both matrices are always **N × N** (square).  
Performance is evaluated at **N = 4096** (16,777,216 elements per matrix).

---

## How the Solution Works

### File: `matrix_addition.cu`

#### 1. Flattening the 2-D problem to 1-D

Because element-wise addition only requires that matching indices are added —
with no dependency on row or column position — the N×N matrix is treated as
a flat array of `N*N` elements. This avoids the overhead of 2-D indexing.

#### 2. Thread-to-element mapping

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Each thread gets one unique linear index `i`. Threads are launched in a single
1-D grid, and each one adds one pair of elements:

```
Thread 0  →  C[0]  = A[0]  + B[0]
Thread 1  →  C[1]  = A[1]  + B[1]
...
Thread i  →  C[i]  = A[i]  + B[i]
```

#### 3. Boundary guard

```cpp
if (i < N*N)
```

The total number of elements `N*N` may not be a multiple of 256, so the last
block can contain threads that exceed the valid range. This guard ensures those
threads do nothing.

#### 4. The addition

```cpp
C[i] = A[i] + B[i];
```

One read from `A`, one read from `B`, one write to `C` — all at the same
address `i`. This is the simplest possible memory pattern: fully coalesced
reads and writes, since consecutive threads access consecutive addresses.

#### 5. Grid and block dimensions

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N * N + 255) / 256;
```

256 threads per block is a reliable default — it occupies a full set of 8 warps,
which is enough to hide memory latency on the GTX 750 Ti without excessive
register pressure. The ceiling division ensures every element gets a thread.

#### 6. Memory access pattern

| Access | Pattern | Notes |
|--------|---------|-------|
| Read `A[i]` | Threads in a warp access `A[i], A[i+1], ..., A[i+31]` — perfectly **coalesced** | Max bandwidth |
| Read `B[i]` | Same — fully **coalesced** | Max bandwidth |
| Write `C[i]` | Same — fully **coalesced** | Max bandwidth |

Matrix addition is a **memory-bandwidth-bound** problem: the GPU spends almost
all its time moving data, not computing. The 1-D coalesced access pattern
extracts the maximum available bandwidth.

#### 7. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Blocks the host until all GPU threads have finished before `solve()` returns.

---

## Project Structure

```
matrix_addition/
├── matrix_addition.cu         # ← your CUDA solution (unchanged)
├── main.cpp                   # local test harness (fixed examples + randomised tests)
├── Makefile                   # build rules for nvcc + g++-10, sm_50
├── .vscode/
│   ├── c_cpp_properties.json  # IntelliSense config
│   └── tasks.json             # Build / Build & Run / Clean tasks
└── README.md
```

---

## Requirements

| Tool | Version |
|------|---------|
| CUDA Toolkit | 11.x (last release with full sm_50 support) |
| `nvcc` | ships with CUDA Toolkit |
| `g++` | 10 (`g++-10`) |
| GPU | NVIDIA GTX 750 Ti (Maxwell, Compute Capability 5.0) |
| OS | Linux (Ubuntu 20.04 / 22.04 recommended) |

> **CUDA 12 users:** add `-allow-unsupported-compiler` to `NVCCFLAGS` in the Makefile.

Install on Ubuntu:

```bash
sudo apt install g++-10
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

---

## Building & Running

```bash
cd matrix_addition

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 (2x2)                    N=   2  ... PASS
Example 2 (3x3)                    N=   3  ... PASS
1x1                                N=   1  ... PASS
15x15 (non-power-of-2)             N=  15  ... PASS
16x16                              N=  16  ... PASS
32x32                              N=  32  ... PASS
128x128                            N= 128  ... PASS
256x256                            N= 256  ... PASS
1024x1024                          N=1024  ... PASS
4096x4096 (benchmark size)         N=4096  ... PASS

10 / 10 tests passed.
```

---

## VSCode Integration

1. Open the `matrix_addition/` folder in VSCode.
2. Recommended extensions:
   - **C/C++** (`ms-vscode.cpptools`)
   - **Makefile Tools** (`ms-vscode.makefile-tools`)
   - **CUDA C++** (optional, `.cu` syntax highlighting)
3. Keyboard shortcuts:
   - `Ctrl+Shift+B` → **Build**
   - `Ctrl+Shift+P` → **Tasks: Run Task** → **Build & Run**

---

## Uploading to GitHub

```bash
cd matrix_addition
git init
git add .
git commit -m "Initial commit: LeetGPU Matrix Addition (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
