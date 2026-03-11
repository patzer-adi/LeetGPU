# LeetGPU – Sigmoid Activation (Easy)

A CUDA solution for the **Sigmoid Activation** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `sigmoid.cu` using plain CUDA — no external libraries.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Examples](#examples)
3. [Constraints](#constraints)
4. [How the Solution Works](#how-the-solution-works)
5. [Sigmoid vs SiLU](#sigmoid-vs-silu)
6. [Project Structure](#project-structure)
7. [Requirements](#requirements)
8. [Building & Running](#building--running)
9. [VSCode Integration](#vscode-integration)
10. [Uploading to GitHub](#uploading-to-github)

---

## Problem Statement

Apply the **sigmoid** function element-wise to an input vector `X` of length **N**,
storing the results in output vector `Y`:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

The sigmoid function maps any real number to the open interval **(0, 1)**:
- As x → +∞, sigmoid(x) → 1
- At x = 0, sigmoid(0) = 0.5 exactly
- As x → -∞, sigmoid(x) → 0

It is one of the oldest and most fundamental activation functions in neural networks,
and also appears as a component of gates in LSTMs, GRUs, and attention mechanisms.

The `solve` function signature must not be changed.

---

## Examples

### Example 1

```
X = [0.0,    1.0,    -1.0,   2.0   ]
Y = [0.5000, 0.7311, 0.2689, 0.8808]
```

Verification for `x = 1.0`:
```
e^(-1) ≈ 0.3679
sigmoid(1) = 1 / (1 + 0.3679) = 1 / 1.3679 ≈ 0.7311
```

### Example 2

```
X = [0.5,    -0.5,   3.0,    -3.0  ]
Y = [0.6225, 0.3775, 0.9526, 0.0474]
```

Note the symmetry: `sigmoid(-x) = 1 - sigmoid(x)`, so
`sigmoid(-0.5) = 1 - sigmoid(0.5) = 1 - 0.6225 = 0.3775`.

---

## Constraints

| Parameter | Range |
|-----------|-------|
| N | 1 – 100,000,000 |
| X[i] | finite 32-bit floats |

Performance is evaluated at **N = 50,000,000** (200 MB of float data).

---

## How the Solution Works

### File: `sigmoid.cu`

#### 1. Thread-to-element mapping

A plain 1-D grid — one thread per element:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < N) {
    Y[id] = 1.0f / (1.0f + expf(-X[id]));
}
```

The `if (id < N)` guard handles the last partial block when N is not a multiple
of 256. Threads outside the valid range do nothing.

#### 2. The sigmoid formula

```cpp
Y[id] = 1.0f / (1.0f + expf(-X[id]));
```

Step by step for one thread:

```
x       = X[id]
neg_x   = -x                     // negate
e_neg_x = expf(neg_x)            // hardware exp (~20 cycles on Maxwell)
denom   = 1.0f + e_neg_x
Y[id]   = 1.0f / denom           // rcp instruction
```

`expf` is the single-precision hardware exponential — approximately 1 ULP
accurate and much faster than the double-precision `exp`. Using `math.h`'s
`expf` on the device maps directly to the GPU's `ex2` / `expf` intrinsic.

#### 3. Relationship to SiLU

Sigmoid and SiLU share the same exponential core:

```
sigmoid(x) =          1 / (1 + e^{-x})
SiLU(x)    = x  ·  [ 1 / (1 + e^{-x}) ]
           = x  ·  sigmoid(x)
```

SiLU is simply sigmoid multiplied by its own input. The kernel for SiLU adds
one extra multiplication: `input[id] * (1 / (1 + expf(-input[id])))`.

#### 4. Numerical behaviour at extremes

| Input | `expf(-x)` | denominator | sigmoid(x) |
|-------|-----------|-------------|-----------|
| x = 88 | `expf(-88)` ≈ 6×10⁻³⁹ → underflows to 0 | ≈ 1 | ≈ 1.0 |
| x = 0 | `expf(0)` = 1.0 | = 2.0 | = 0.5 exactly |
| x = -88 | `expf(88)` ≈ 1.6×10³⁸ → near FLT_MAX | huge | ≈ 0.0 |
| x = -104 | `expf(104)` overflows to +Inf | +Inf | 1/Inf = 0.0 |

Float32 handles all these gracefully without any special-casing — IEEE 754
arithmetic produces the correct limiting values automatically.

#### 5. Grid configuration

```cpp
int threadsPerBlock = 256;
int blocksPerGrid   = (N + 255) / 256;
```

For the benchmark size N = 50,000,000: ~195,313 blocks × 256 threads.
sm_50 (GTX 750 Ti) supports up to 2,147,483,647 blocks per grid dimension,
so even N = 100,000,000 (~390,625 blocks) is well within limits — no
grid-stride loop is required.

#### 6. Memory access

Fully **coalesced** reads from `X` and writes to `Y`. Consecutive threads
access consecutive addresses, maximising L2/DRAM bandwidth. For N = 50M,
the kernel reads 200 MB and writes 200 MB — bandwidth is the bottleneck,
not the `expf` arithmetic.

#### 7. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Blocks the host until all GPU threads have finished.

---

## Sigmoid vs SiLU

| Property | Sigmoid | SiLU |
|----------|---------|------|
| Formula | `1 / (1 + e^{-x})` | `x / (1 + e^{-x})` |
| Output range | (0, 1) | (≈−0.28, ∞) |
| Symmetric around | 0.5 (output) | 0 (output) |
| Used as | Output gate, binary classifier | Hidden-layer activation |
| Vanishing gradient? | Yes for large \|x\| | Mitigated |

Sigmoid is most commonly used at the **output** of a binary classification model
(probability output) or inside gating mechanisms (LSTM, attention). SiLU is
preferred as a **hidden-layer activation** because its output is centred near
zero for typical input distributions.

---

## Project Structure

```
sigmoid/
├── sigmoid.cu                 # ← your CUDA solution (unchanged)
├── main.cpp                   # test harness (fixed examples + spot checks + random)
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
cd sigmoid

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 [0,1,-1,2]                      N=        4  ... PASS
Example 2 [0.5,-0.5,3,-3]                 N=        4  ... PASS
sigmoid(0)   == 0.5 exactly               N=        1  ... PASS
sigmoid(88)  ≈ 1.0 (exp underflows)       N=        1  ... PASS
sigmoid(-88) ≈ 0.0 (exp overflows)        N=        1  ... PASS
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

1. Open the `sigmoid/` folder in VSCode.
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
cd sigmoid
git init
git add .
git commit -m "Initial commit: LeetGPU Sigmoid Activation (Easy)"

git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
