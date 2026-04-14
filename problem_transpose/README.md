# LeetGPU – Matrix Transpose (Easy)

A CUDA solution for the **Matrix Transpose** problem on [LeetGPU](https://leetgpu.com).  
All GPU work is in `matrix_transpose.cu` using plain CUDA — no external libraries.

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

Given a matrix `A` of 32-bit floats with dimensions **rows × cols** stored in row-major format,
compute its transpose `Aᵀ` with dimensions **cols × rows**.

Transposing switches rows and columns: the element at position `[r][c]` in the input
moves to position `[c][r]` in the output.

```
input[r][c]  →  output[c][r]

In row-major indexing:
  input  [ r * cols + c ]
  output [ c * rows + r ]
```

The `solve` function signature must not be changed.

---

## Examples

### Example 1 — 2×3 → 3×2

```
Input (2 rows, 3 cols):          Output (3 rows, 2 cols):
  1  2  3                          1  4
  4  5  6                          2  5
                                   3  6
```

Element trace:
- `input[0][0] = 1` → `output[0][0] = 1`
- `input[0][1] = 2` → `output[1][0] = 2`
- `input[1][0] = 4` → `output[0][1] = 4`

### Example 2 — 3×1 → 1×3

```
Input (3 rows, 1 col):    Output (1 row, 3 cols):
  1                          1  2  3
  2
  3
```

---

## Constraints

| Parameter | Range |
|-----------|-------|
| rows      | 1 – 8192 |
| cols      | 1 – 8192 |

Performance is evaluated at **rows = 7,000, cols = 6,000**.

---

## How the Solution Works

### File: `matrix_transpose.cu`

#### 1. Thread-to-element mapping

Each thread is responsible for exactly **one element** of the input matrix.  
The 2-D thread index maps directly to a `(row, col)` coordinate:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;   // column of input
int y = blockIdx.y * blockDim.y + threadIdx.y;   // row    of input
```

The x-dimension covers columns, and the y-dimension covers rows.

#### 2. Boundary guard

```cpp
if (x < cols && y < rows)
```

The grid is rounded up to the nearest multiple of the block size, so some
threads at the edges would otherwise map to out-of-bounds positions.
This guard ensures only valid elements are processed.

#### 3. The transpose write

```cpp
output[x * rows + y] = input[y * cols + x];
```

| Side   | Index formula     | Meaning |
|--------|------------------|---------|
| Read   | `y * cols + x`   | Row `y`, column `x` of the input (rows × cols) |
| Write  | `x * rows + y`   | Row `x`, column `y` of the output (cols × rows) |

This single line encodes the entire transpose: the element that was in
row `y`, column `x` of the input is placed into row `x`, column `y` of
the output — exactly what a transpose requires.

#### 4. Grid and block dimensions

```cpp
dim3 threadsPerBlock(16, 16);                            // 256 threads/block
dim3 blocksPerGrid(
    (cols + 15) / 16,   // blocks to cover all columns
    (rows + 15) / 16    // blocks to cover all rows
);
```

A 16×16 block is a standard choice: 256 threads per block is a common
sweet spot for occupancy on Maxwell-generation GPUs (GTX 750 Ti), and it
divides evenly into the 32-thread warp size.

#### 5. Memory access pattern

| Access | Pattern | Notes |
|--------|---------|-------|
| **Read** `input[y * cols + x]` | Threads in the same warp have consecutive `x` values → **coalesced** reads along a row | Good bandwidth |
| **Write** `output[x * rows + y]` | Threads in the same warp write to `output[x * rows + 0], output[x * rows + 1], ...` — these are separated by `rows` elements → **strided** (non-coalesced) writes | Potential bottleneck for large matrices |

For the performance benchmark size (7000×6000) the strided writes are the
limiting factor. A shared-memory tiling approach could improve this, but
the simple version is correct and passes all tests.

#### 6. Synchronisation

```cpp
cudaDeviceSynchronize();
```

Ensures all GPU work finishes before `solve()` returns to the caller.

---

## Project Structure

```
matrix_transpose/
├── matrix_transpose.cu    # ← your CUDA solution (unchanged)
├── main.cpp               # local test harness (CPU reference + correctness checks)
├── Makefile               # build rules for nvcc + g++-10, sm_50
├── .vscode/
│   ├── c_cpp_properties.json   # IntelliSense config
│   └── tasks.json              # Build / Build & Run / Clean tasks
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
# CUDA Toolkit: follow https://developer.nvidia.com/cuda-downloads
```

---

## Building & Running

```bash
cd matrix_transpose

# Build
make

# Run all tests
make run

# Clean
make clean
```

Expected output:

```
Example 1 (2x3 -> 3x2)           rows=    2 cols=    3  ... PASS
Example 2 (3x1 -> 1x3)           rows=    3 cols=    1  ... PASS
1x1                               rows=    1 cols=    1  ... PASS
1x16 (single row)                 rows=    1 cols=   16  ... PASS
16x1 (single col)                 rows=   16 cols=    1  ... PASS
16x16 square                      rows=   16 cols=   16  ... PASS
32x64                             rows=   32 cols=   64  ... PASS
128x256                           rows=  128 cols=  256  ... PASS
512x512 square                    rows=  512 cols=  512  ... PASS
1000x2000                         rows= 1000 cols= 2000  ... PASS

10 / 10 tests passed.
```

---

## VSCode Integration

1. Open the `matrix_transpose/` folder in VSCode.
2. Recommended extensions:
   - **C/C++** (`ms-vscode.cpptools`)
   - **Makefile Tools** (`ms-vscode.makefile-tools`)
   - **CUDA C++** (optional, adds `.cu` syntax highlighting)
3. Keyboard shortcuts:
   - `Ctrl+Shift+B` → **Build**
   - `Ctrl+Shift+P` → **Tasks: Run Task** → **Build & Run**

---

## Uploading to GitHub

```bash
cd matrix_transpose
git init
git add .
git commit -m "Initial commit: LeetGPU Matrix Transpose (Easy)"

# Create a repo on GitHub, then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```
