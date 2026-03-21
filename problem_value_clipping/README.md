# LeetGPU – Value Clipping (Easy)

A CUDA solution for the **Value Clipping** problem on LeetGPU.  
All GPU work is in `value_clipping.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [Branch Analysis](#branch-analysis)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)
8. [Building & Running](#building--running)
9. [VSCode Integration](#vscode-integration)
10. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Apply **value clipping** element-wise to an input vector of length **N**.
Each element is clamped to the closed interval **[lo, hi]**:

```
         ┌  lo        if x < lo
clip(x) =│  x         if lo ≤ x ≤ hi
         └  hi        if x > hi
```

Clipping is used in machine learning for:
- **Activation stabilisation** — preventing exploding activations
- **Pre-quantisation** — constraining values before mapping to fixed-point
- **Gradient clipping** — capping gradients during backpropagation

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
Input:  [1.5, -2.0, 3.0, 4.5]   lo = 0.0,  hi = 3.5
Output: [1.5,  0.0, 3.0, 3.5]
```

- `1.5`  → in range → pass through as `1.5`
- `-2.0` → below lo → clamp to `0.0`
- `3.0`  → in range → pass through as `3.0`
- `4.5`  → above hi → clamp to `3.5`

### Example 2

```
Input:  [-1.0, 2.0, 5.0]   lo = -0.5,  hi = 2.5
Output: [-0.5, 2.0, 2.5]
```

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N | 1 – 100,000 |
| input[i] | −10⁶ – 10⁶ |
| lo, hi | lo ≤ hi (guaranteed) |

Performance is evaluated at **N = 100,000**.

---

## How the Solution Works

### File: `value_clipping.cu`

#### 1. Thread-to-element mapping

One thread per element, plain 1-D grid:

```cpp
int idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx < N) { ... }
```

The `if (idx < N)` guard handles the last partial block when N is not a
multiple of 256.

#### 2. The clipping logic

```cpp
if (input[idx] > hi)
    output[idx] = hi;
else if (input[idx] < lo)
    output[idx] = lo;
else if (input[idx] <= hi && input[idx] >= lo)
    output[idx] = input[idx];
```

The three branches are evaluated in order:
1. If the value exceeds `hi` → clamp to `hi`
2. If the value is below `lo` → clamp to `lo`
3. If the value is within `[lo, hi]` → copy it unchanged

The third condition `input[idx] <= hi && input[idx] >= lo` is logically
redundant — if neither of the first two branches fired, the value must
already be in range. However, it is explicit and safe: the compiler will
optimise this into a single unconditional store since the condition is
always true at that point.

#### 3. Grid configuration

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

For N = 100,000: 391 blocks × 256 threads = 100,096 threads, with 96
idle threads in the last block.

#### 4. Memory access pattern

Fully **coalesced** reads and writes — consecutive threads access
consecutive float addresses. This kernel is entirely memory-bandwidth-bound;
the clipping arithmetic (two comparisons and a conditional assignment) is
negligible compared to the cost of loading and storing the data.

#### 5. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Blocks the host until all GPU threads finish before `solve()` returns.

---

## Branch Analysis

The kernel has three explicit `if/else if` branches. On a GPU, divergence
within a warp (32 threads) occurs when threads take different paths. For
value clipping:

- If all 32 threads in a warp have values in-range → no divergence
- If some threads need clamping → the warp serialises the branches

In practice, for typical ML weight or activation distributions (roughly
Gaussian centred near zero), most values fall within a reasonable `[lo, hi]`
range, so warp divergence is minimal. For adversarial inputs (e.g. all values
at the extremes), some divergence occurs but the kernel is still
memory-bandwidth-bound, so the impact is small.

An equivalent branchless formulation using `fminf`/`fmaxf` would be:

```cpp
output[idx] = fminf(hi, fmaxf(lo, input[idx]));
```

This avoids all branching and maps to two hardware `fsel` instructions. Your
explicit branch version produces identical results and the compiler may
optimise it to the same code in practice.

---

## Project Structure

```
value_clipping/
├── value_clipping.cu          # ← your CUDA solution (unchanged)
├── main.cpp                   # test harness (examples + boundary cases + random)
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
cd value_clipping

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [1.5,-2,3,4.5]               N=      4  lo= 0.0  hi=3.5  ... PASS
Example 2 [-1,2,5]                      N=      3  lo=-0.5  hi=2.5  ... PASS
x == lo exactly                          N=      1  lo= 0.0  hi=1.0  ... PASS
x == hi exactly                          N=      1  lo= 0.0  hi=1.0  ... PASS
lo == hi (degenerate)                    N=      1  lo= 2.0  hi=2.0  ... PASS
lo == hi, val < lo                       N=      1  lo= 2.0  hi=2.0  ... PASS
lo == hi, val > hi                       N=      1  lo= 2.0  hi=2.0  ... PASS
negative range, clamp up                 N=      1  lo=-3.0  hi=-1.0 ... PASS
negative range, clamp dn                 N=      1  lo=-3.0  hi=-1.0 ... PASS
negative range, pass-thru                N=      1  lo=-3.0  hi=-1.0 ... PASS
N=1                                      N=     1  lo= -1.0  hi= 1.0 ... PASS
N=255 (non-power-of-2)                   N=   255  lo=  0.0  hi=10.0 ... PASS
N=256 (one full block)                   N=   256  lo=  0.0  hi=10.0 ... PASS
N=257 (one block + 1)                    N=   257  lo=  0.0  hi=10.0 ... PASS
N=10,000                                 N= 10000  lo=-100.0 hi=100.0 ... PASS
N=100,000 (benchmark, full range)        N=100000  lo=-1000000.0 hi=1000000.0 ... PASS
N=100,000 (lo==hi==0, all collapse)      N=100000  lo=  0.0  hi= 0.0 ... PASS

17 / 17 tests passed.
```

---

## VSCode Integration

1. Open the `value_clipping/` folder in VSCode.
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
cd value_clipping
git init
git add .
git commit -m "Initial commit: LeetGPU Value Clipping (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
