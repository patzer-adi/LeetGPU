# LeetGPU – Sigmoid Linear Unit / SiLU (Easy)

A CUDA solution for the **SiLU Activation Function** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `silu.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [SiLU vs ReLU Family](#silu-vs-relu-family)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)
8. [Building & Running](#building--running)
9. [VSCode Integration](#vscode-integration)
10. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Apply the **SiLU (Sigmoid Linear Unit)** activation function — also called **Swish** —
to every element of an input vector of length **N**:

```
SiLU(x) = x · σ(x)  =  x / (1 + e^(-x))

where σ(x) = 1 / (1 + e^(-x))  is the sigmoid function
```

SiLU was introduced in 2017 and is used in modern architectures such as
EfficientNet, LLaMA, and many transformer variants. Unlike ReLU, it is smooth
everywhere and has a non-monotonic region for small negative values, which
gives neurons more expressive power.

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
Input:  [0.5,        1.0,       -0.5      ]
Output: [0.3112295,  0.7310586, -0.1887705]
```

Verification for `x = 0.5`:
```
σ(0.5) = 1 / (1 + e^(-0.5)) = 1 / (1 + 0.60653) ≈ 0.62246
SiLU(0.5) = 0.5 × 0.62246 ≈ 0.31123
```

### Example 2

```
Input:  [-1.0,        -2.0,        -3.0,        -4.0,        -5.0       ]
Output: [-0.26894143, -0.23840584, -0.14227763, -0.07194484, -0.03346425]
```

Note how large negative inputs produce small negative outputs (approaching 0),
unlike ReLU which clips them hard to 0.

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N | 1 – 10,000 (constraint) |
| input[i] | −100.0 – 100.0 |

Performance is evaluated at **N = 50,000**.

---

## How the Solution Works

### File: `silu.cu`

#### 1. Thread-to-element mapping

The kernel uses a simple 1-D grid — one thread per element, no loop needed
since N ≤ 50,000 fits comfortably within a single grid launch:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < N) {
    output[id] = input[id] / (1 + expf(-input[id]));
}
```

The boundary guard `if (id < N)` handles the last partial block when N is
not a multiple of 256.

#### 2. The SiLU formula

```cpp
output[id] = input[id] / (1 + expf(-input[id]));
```

This directly encodes `x / (1 + e^{-x})`, which is equivalent to `x · σ(x)`.
The two formulations are mathematically identical; the division form avoids a
separate sigmoid computation and expresses it in one expression.

Breaking it down step by step for a single thread:
```
x        = input[id]
exp_neg_x = expf(-x)            // GPU hardware exp, ~20 clock cycles
denom    = 1.0f + exp_neg_x
output   = x / denom
```

`expf` is the single-precision hardware exponential — faster than `exp`
(double) and sufficient for float32 accuracy.

#### 3. Numerical behaviour at the extremes

| Input range | sigmoid(x) ≈ | SiLU(x) ≈ |
|-------------|-------------|-----------|
| x >> 0 (e.g. 100) | ≈ 1.0 | ≈ x (nearly linear) |
| x = 0 | 0.5 | 0.0 |
| x << 0 (e.g. -100) | ≈ 0.0 | ≈ 0.0 (approaches zero from below) |
| x ≈ -1.278 | — | minimum ≈ -0.2785 (the non-monotonic dip) |

For `x = 100`: `expf(-100)` underflows to 0 in float32, so the denominator
becomes exactly 1 and `output = 100 / 1 = 100`. No special-case needed.

For `x = -100`: `expf(100)` is huge (~2.7×10⁴³ in double, but float32 max
is ~3.4×10³⁸, so it saturates to `+Inf`). The denominator becomes `Inf`,
and `output = -100 / Inf = 0`. Again, no special-case needed — IEEE 754
float arithmetic handles this gracefully.

#### 4. Grid and block dimensions

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

For N = 50,000: 196 blocks × 256 threads = 50,176 threads launched, covering
all 50,000 elements with 176 idle threads in the last block.

#### 5. Memory access pattern

Fully **coalesced** reads and writes — consecutive threads access consecutive
addresses. The only arithmetic per thread is a negation, one `expf`, one
addition, and one division. This kernel is therefore both memory-bandwidth-bound
(for large N) and compute-bound at the `expf` call (which is the most expensive
single operation on the critical path).

#### 6. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Ensures all GPU threads finish before `solve()` returns.

---

## SiLU vs ReLU Family

| Activation | Formula | Negative region | Smooth? | Used in |
|------------|---------|----------------|---------|---------|
| ReLU | `max(0, x)` | Hard zero | No (kink at 0) | ResNet, VGG |
| Leaky ReLU | `x > 0 ? x : 0.01x` | Small slope | No (kink at 0) | YOLO, GANs |
| SiLU / Swish | `x · σ(x)` | Smooth near-zero dip | Yes | EfficientNet, LLaMA |

SiLU's key advantages over ReLU-family activations:
- **Smooth everywhere** — the gradient is always defined, no kink at zero
- **Non-monotonic** — has a small dip around x ≈ −1.28, giving neurons
  the ability to suppress slightly negative inputs while still passing large
  ones
- **Self-gating** — the `x · σ(x)` form means the input gates itself
  through its own sigmoid, without any learned parameters

The trade-off is the `expf` call, which costs ~20 GPU clock cycles vs. the
single comparison of ReLU. For most model sizes this is not a bottleneck.

---

## Project Structure

```
silu/
├── silu.cu                    # ← your CUDA solution (unchanged)
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
cd silu

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [0.5,1,-0.5]                    N=        3  ... PASS
Example 2 [-1,-2,-3,-4,-5]                N=        5  ... PASS
SiLU(0) == 0                              N=        1  ... PASS
SiLU(100) ≈ 100                           N=        1  ... PASS
N=1                                        N=        1  ... PASS
N=255 (non-power-of-2)                     N=      255  ... PASS
N=256 (one full block)                     N=      256  ... PASS
N=257 (one block + 1)                      N=      257  ... PASS
N=1000, full range                         N=     1000  ... PASS
N=10,000 (constraint max)                  N=    10000  ... PASS
N=50,000 (benchmark size)                  N=    50000  ... PASS

11 / 11 tests passed.
```

---

## VSCode Integration

1. Open the `silu/` folder in VSCode.
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
cd silu
git init
git add .
git commit -m "Initial commit: LeetGPU SiLU (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
