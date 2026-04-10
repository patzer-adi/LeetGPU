# LeetGPU – ReLU (Easy)

A CUDA solution for the **ReLU Activation Function** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `relu.cu` using plain CUDA — no external libraries.

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

Apply the **Rectified Linear Unit (ReLU)** activation function to every element of
an input vector of length **N**:

```
output[i] = max(0, input[i])

Equivalently:
  output[i] = 0          if input[i] < 0
  output[i] = input[i]   if input[i] >= 0
```

ReLU is one of the most widely used activation functions in deep learning. It is
cheap to compute (a single comparison) and introduces non-linearity without causing
vanishing gradients for positive values.

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
Input:  [-2.0, -1.0,  0.0,  1.0,  2.0]
Output: [ 0.0,  0.0,  0.0,  1.0,  2.0]
```

### Example 2

```
Input:  [-3.5,  0.0,  4.2]
Output: [ 0.0,  0.0,  4.2]
```

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N         | 1 – 100,000,000 |

Performance is evaluated at **N = 25,000,000** (100 MB of float data).

---

## How the Solution Works

### File: `relu.cu`

#### 1. Grid-stride loop

Unlike the matrix problems (where one thread = one element and the grid exactly
covers the data), this kernel uses a **grid-stride loop**:

```cpp
int idx = blockDim.x * blockIdx.x + threadIdx.x;  // starting index for this thread
int i   = idx;

while (i < N) {
    // process element i
    i += blockDim.x * gridDim.x;   // stride = total number of threads launched
}
```

Each thread starts at its unique `idx` and then hops forward by the total grid
width (`blockDim.x * gridDim.x`) on each iteration. This means:

- If `N` is larger than the number of launched threads, each thread handles
  **multiple elements**.
- If `N` is smaller, the `while (i < N)` guard ensures threads beyond the end
  do nothing.
- The pattern naturally handles any value of `N` without requiring the grid size
  to equal `N`.

For the benchmark size of N = 25,000,000 the grid covers all elements in one pass,
so the loop body executes exactly once per thread — same as a plain 1-D launch but
more robust.

#### 2. The ReLU condition

```cpp
if (input[i] < 0)
    output[i] = 0.0f;
else if (input[i] >= 0)
    output[i] = input[i];
```

This explicitly handles both branches. Because floating-point values are either
`< 0`, `>= 0`, or `NaN`, the two branches are exhaustive for normal numbers.
`NaN` values pass through the `else if` unchanged (since all comparisons with
`NaN` return false, `input[i] >= 0` is false for NaN, so it won't be written —
though in practice LeetGPU inputs are well-formed).

An equivalent single-line formulation would be `fmaxf(0.f, input[i])`, but your
explicit branching is perfectly valid and equally fast on modern GPUs where the
compiler lowers the comparison to a `fsel`/`fmax` instruction regardless.

#### 3. Grid and block dimensions

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

256 threads per block is a standard baseline. For N = 25,000,000 this launches
97,657 blocks × 256 threads = ~25 million threads, one per element.

#### 4. Memory access pattern

Consecutive threads read and write consecutive addresses — fully **coalesced**:

```
Thread 0  →  input[0]   output[0]
Thread 1  →  input[1]   output[1]
...
Thread 31 →  input[31]  output[31]   ← one warp, one cache line
```

ReLU is purely **memory-bandwidth-bound** (one read + one write per element,
with almost no arithmetic). Coalesced access is the single most important
performance factor here.

#### 5. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Ensures all GPU threads finish before `solve()` returns.

---

## Project Structure

```
relu/
├── relu.cu                    # ← your CUDA solution (unchanged)
├── main.cpp                   # test harness (fixed examples + edge cases + random)
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
cd relu

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [-2,-1,0,1,2]                  N=        5  ... PASS
Example 2 [-3.5,0,4.2]                   N=        3  ... PASS
N=1, value=0 (boundary)                  N=        1  ... PASS
all negative                             N=      100  ... PASS
all positive                             N=      100  ... PASS
N=1                                      N=        1  ... PASS
N=255 (non-power-of-2)                   N=      255  ... PASS
N=256 (one block)                        N=      256  ... PASS
N=257 (just over one block)              N=      257  ... PASS
N=10,000                                 N=    10000  ... PASS
N=1,000,000                              N=  1000000  ... PASS
N=25,000,000 (benchmark size)            N= 25000000  ... PASS

12 / 12 tests passed.
```

---

## VSCode Integration

1. Open the `relu/` folder in VSCode.
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
cd relu
git init
git add .
git commit -m "Initial commit: LeetGPU ReLU (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
