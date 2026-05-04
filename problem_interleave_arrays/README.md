# LeetGPU – Interleave Arrays (Easy)

A CUDA solution for the **Interleave Arrays** problem on LeetGPU.  
All GPU work is in `interleave_arrays.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [Memory Access Pattern — The Key Detail](#memory-access-pattern--the-key-detail)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)
8. [Building & Running](#building--running)
9. [VSCode Integration](#vscode-integration)
10. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Given two input arrays **A** and **B**, each of length **N**, produce a single
output array of length **2N** where elements from A and B alternate:

```
output = [A[0], B[0], A[1], B[1], A[2], B[2], ..., A[N-1], B[N-1]]
```

```
A:      a₀  a₁  a₂  a₃
B:      b₀  b₁  b₂  b₃
             ↓
output: a₀ b₀ a₁ b₁ a₂ b₂ a₃ b₃
```

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
A      = [1.0,  2.0,  3.0]
B      = [4.0,  5.0,  6.0]
output = [1.0,  4.0,  2.0,  5.0,  3.0,  6.0]
```

### Example 2

```
A      = [10.0, 20.0]
B      = [30.0, 40.0]
output = [10.0, 30.0, 20.0, 40.0]
```

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N | 1 – 50,000,000 |

Performance is evaluated at **N = 25,000,000** (output array = 200 MB).

---

## How the Solution Works

### File: `interleave_arrays.cu`

#### 1. Thread-to-element mapping

One thread per index `i` in [0, N), each responsible for placing one pair of
values into the output:

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
    output[i * 2]     = A[i];   // even output slot  → from A
    output[i * 2 + 1] = B[i];   // odd  output slot  → from B
}
```

Thread `i` writes to `output[2i]` and `output[2i+1]` — the two output slots
that correspond to position `i` in the interleaved sequence.

#### 2. Index formula

The key insight is the mapping between the input index `i` and the output index:

```
A[i]  →  output[2*i]       (stride-2 scatter, even positions)
B[i]  →  output[2*i + 1]   (stride-2 scatter, odd  positions)
```

Because each thread handles exactly one `i`, no two threads ever write to the
same output address — there are no data races.

#### 3. Grid configuration

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

Grid is sized for N (the number of input pairs), not 2N. The `if (i < N)` guard
handles the last partial block.

#### 4. `unsigned int` for the index

The thread index is stored as `unsigned int` rather than the typical `int`.
For N up to 50,000,000 both types work fine (INT_MAX ≈ 2.1 billion), but
`unsigned int` makes it explicit that the index is never negative and avoids
signed overflow warnings when computing `i * 2 + 1` near large N values.

---

## Memory Access Pattern — The Key Detail

This is the most interesting aspect of this kernel from a GPU architecture
perspective.

### Reads from A and B — coalesced ✓

Consecutive threads read consecutive elements:
```
Thread 0 reads A[0],  Thread 1 reads A[1],  ...  Thread 31 reads A[31]
Thread 0 reads B[0],  Thread 1 reads B[1],  ...  Thread 31 reads B[31]
```
Both input arrays are accessed with stride-1, which is ideal for memory
coalescing — a full cache line is loaded per warp per array.

### Writes to output — stride-2, partially coalesced ✗/✓

Consecutive threads write to every other output slot:
```
Thread 0 writes output[0] and output[1]
Thread 1 writes output[2] and output[3]
Thread 2 writes output[4] and output[5]
...
```

Within a warp, the 32 threads write to 64 consecutive output slots
(`output[0]` through `output[63]`). Even though each thread writes two
separate addresses, the entire range is contiguous in memory — the hardware
can service this as two coalesced transactions per warp rather than 64
separate ones. Performance is close to ideal despite the stride-2 appearance.

---

## Project Structure

```
interleave_arrays/
├── interleave_arrays.cu       # ← your CUDA solution (unchanged)
├── main.cpp                   # test harness (fixed examples + random + edge cases)
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
cd interleave_arrays

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [1,2,3] / [4,5,6]               N=        3  ... PASS
Example 2 [10,20] / [30,40]               N=        2  ... PASS
N=1 (single element pair)                  N=        1  ... PASS
N=255 (non-power-of-2)                     N=      255  ... PASS
N=256 (one full block)                     N=      256  ... PASS
N=257 (one block + 1)                      N=      257  ... PASS
N=10,000                                   N=    10000  ... PASS
N=1,000,000                                N=  1000000  ... PASS
N=25,000,000 (benchmark size)              N= 25000000  ... PASS

9 / 9 tests passed.
```

---

## VSCode Integration

1. Open the `interleave_arrays/` folder in VSCode.
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
cd interleave_arrays
git init
git add .
git commit -m "Initial commit: LeetGPU Interleave Arrays (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
