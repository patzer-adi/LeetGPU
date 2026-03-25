# LeetGPU – Leaky ReLU (Easy)

A CUDA solution for the **Leaky ReLU Activation Function** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `leaky_relu.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [Leaky ReLU vs ReLU](#leaky-relu-vs-relu)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)
8. [Building & Running](#building--running)
9. [VSCode Integration](#vscode-integration)
10. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Apply the **Leaky ReLU** activation function to every element of an input vector of
length **N**:

```
         ┌  x          if x > 0
f(x)  =  │
         └  α · x      if x ≤ 0

where α = 0.01
```

Unlike standard ReLU (which hard-clips all negative values to 0), Leaky ReLU
allows a small, non-zero gradient for negative inputs. This prevents the
"dying ReLU" problem in deep networks where neurons whose inputs are always
negative can stop learning entirely.

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
Input:  x = [ 1.0,  -2.0,  3.0,  -4.0]
Output: y = [ 1.0,  -0.02, 3.0,  -0.04]
```

Verification:
- `1.0  > 0` → passes through unchanged: `1.0`
- `-2.0 ≤ 0` → scaled by 0.01: `0.01 × -2.0 = -0.02`
- `3.0  > 0` → passes through unchanged: `3.0`
- `-4.0 ≤ 0` → scaled by 0.01: `0.01 × -4.0 = -0.04`

### Example 2

```
Input:  x = [-1.5,   0.0,   2.5,  -3.0]
Output: y = [-0.015, 0.0,   2.5,  -0.03]
```

Note that `0.0` satisfies `x ≤ 0` and produces `0.01 × 0.0 = 0.0` — identical
to the standard ReLU output at zero.

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N | 1 – 100,000,000 |
| input[i] | −1000.0 – 1000.0 |

Performance is evaluated at **N = 50,000,000** (200 MB of float data).

---

## How the Solution Works

### File: `leaky_relu.cu`

#### 1. Grid-stride loop

The kernel uses the same **grid-stride loop** pattern as the ReLU solution:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
int i  = id;

while (i < N) {
    // process element i
    i += blockDim.x * gridDim.x;   // hop by total grid width
}
```

Each thread starts at its unique linear index `id` and advances by the full
grid width on each iteration. This handles any value of N — including values
larger than the maximum grid — without changing the launch configuration.

For N = 50,000,000 the grid comfortably covers all elements in one pass
(~195,313 blocks × 256 threads), so the loop body runs exactly once per thread.

#### 2. The Leaky ReLU condition

```cpp
if (input[i] > 0) {
    output[i] = input[i];           // positive: pass through
} else if (input[i] <= 0) {
    output[i] = 0.01f * input[i];   // zero or negative: scale by α
}
```

The two branches cover all real numbers. Note that `0.0` falls into the
`<= 0` branch and gets multiplied by `0.01`, producing `0.0` — which is
mathematically correct and matches the expected output in Example 2.

The `0.01f` constant is a single-precision float literal. Using `0.01`
(double) would promote the multiply to double precision and then truncate back,
which is both slower and unnecessary here.

#### 3. Difference from ReLU

| | ReLU | Leaky ReLU |
|---|---|---|
| `x > 0` | `output = x` | `output = x` |
| `x == 0` | `output = 0` | `output = 0.01 × 0 = 0` |
| `x < 0` | `output = 0` | `output = 0.01 × x` (small negative) |

The only code difference is one line: replacing `output[i] = 0.0f` with
`output[i] = 0.01f * input[i]`. Everything else — grid-stride loop, launch
config, synchronisation — is identical.

#### 4. Grid and block dimensions

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

256 threads per block, one thread per element, ceiling division so no element
is missed.

#### 5. Memory access pattern

Fully **coalesced** reads and writes — consecutive threads access consecutive
addresses. This is critical for a memory-bandwidth-bound kernel like this one,
where the only arithmetic is a single comparison and conditional multiply per
element.

#### 6. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Blocks the host until all GPU threads finish.

---

## Leaky ReLU vs ReLU

Both kernels in this repo share the same structure. Here is a side-by-side
comparison of the active line:

```cpp
// ReLU
output[i] = 0.0f;           // hard zero for negatives

// Leaky ReLU
output[i] = 0.01f * input[i];  // small negative slope for negatives
```

The Leaky ReLU's small slope (`α = 0.01`) means:
- Gradients can still flow back through negative-input neurons during training.
- The output magnitude for large negative inputs is attenuated by 100×
  (e.g., −1000 → −10), keeping activations bounded.

---

## Project Structure

```
leaky_relu/
├── leaky_relu.cu              # ← your CUDA solution (unchanged)
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
cd leaky_relu

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [1,-2,3,-4]                     N=        4  ... PASS
Example 2 [-1.5,0,2.5,-3]                 N=        4  ... PASS
N=1, value=0.0 (exact boundary)           N=        1  ... PASS
all at min (-1000)                         N=      100  ... PASS
all at max (+1000)                         N=      100  ... PASS
N=1                                        N=        1  ... PASS
N=255 (non-power-of-2)                     N=      255  ... PASS
N=256 (one full block)                     N=      256  ... PASS
N=257 (one block + 1)                      N=      257  ... PASS
N=10,000                                   N=    10000  ... PASS
N=1,000,000                                N=  1000000  ... PASS
N=50,000,000 (benchmark size)              N= 50000000  ... PASS

12 / 12 tests passed.
```

---

## VSCode Integration

1. Open the `leaky_relu/` folder in VSCode.
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
cd leaky_relu
git init
git add .
git commit -m "Initial commit: LeetGPU Leaky ReLU (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
