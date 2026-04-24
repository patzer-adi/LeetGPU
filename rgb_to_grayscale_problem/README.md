# RGB to Grayscale — CUDA Solution

**Problem:** LeetGPU · RGB to Grayscale (Easy)

---

## What does this program do?

It converts a colour image (stored as RGB floats) into a single-channel
grayscale image entirely on the GPU. Each pixel's luminance is computed using
the standard ITU-R BT.601 luma coefficients:

```
gray = 0.299 × R  +  0.587 × G  +  0.114 × B
```

These weights reflect human perception: the eye is most sensitive to green,
less so to red, and least to blue.

---

## Memory Layout

### Input — interleaved RGB, row-major

```
index:  0   1   2   3   4   5   6   7   8  ...
value: [R0, G0, B0, R1, G1, B1, R2, G2, B2, ...]
         ↑── pixel 0 ──↑  ↑── pixel 1 ──↑
```

Total elements: `width × height × 3`

### Output — flat grayscale, row-major

```
index:  0      1      2   ...
value: [gray0, gray1, gray2, ...]
```

Total elements: `width × height`

---

## Files

| File | Purpose |
|---|---|
| `solution.cu` | CUDA kernel + `solve()` entry point + optional test harness |
| `Makefile` | Builds the shared library and/or the test binary |
| `README.md` | This file |

---

## Implementation Details

### Kernel: `rgb_to_grayscale_kernel`

```cuda
__global__ void rgb_to_grayscale_kernel(const float* __restrict__ input,
                                         float*       __restrict__ output,
                                         int width, int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < width * height) {
        int base  = i * 3;
        float r   = input[base    ];
        float g   = input[base + 1];
        float b   = input[base + 2];
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
```

**Key design choices:**

| Choice | Reason |
|---|---|
| 1-D flat thread index | One thread = one pixel; no 2-D indexing needed |
| `__restrict__` on both pointers | Tells the compiler input and output never alias → better memory scheduling |
| `float` coefficients (`0.299f`) | Avoids implicit double-precision promotion; keeps everything in FP32 |
| `--use_fast_math` (Makefile) | Enables fast FMA and other SASS-level optimizations with no accuracy loss for this formula |

### Grid / Block dimensions

```
threadsPerBlock = 256
blocksPerGrid   = ceil(width * height / 256)
```

For the judge's target (2048 × 2048 = 4,194,304 pixels):
- Threads launched: 4,194,304
- Blocks: 16,384
- Each thread does exactly **3 loads + 1 store** — fully memory-bound, which is ideal for the GPU's memory subsystem.

---

## What was wrong in the original draft?

The original code was functionally correct but had two minor issues:

1. **Double-precision coefficients** — writing `0.299` instead of `0.299f`
   causes the compiler to promote the entire expression to `double`, which is
   slower on GPU and unnecessary.

2. **Inline index arithmetic** — `input[i*3]`, `input[(i*3)+1]`,
   `input[(i*3)+2]` was readable but redundant. Storing `base = i * 3` once
   removes three multiplications per thread (the compiler likely optimises this
   anyway, but explicit is cleaner).

3. **No `__restrict__`** — without it the compiler must conservatively assume
   aliasing between `input` and `output`.

---

## Building

> Requires: NVIDIA CUDA Toolkit ≥ 11.x, `nvcc` on PATH.

### Adjust GPU architecture if needed

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
# produces: grayscale.so
```

### Build and run the test binary

```bash
make test
```

Expected output:

```
Test Example 1 (2x2)        →  output: [76.245, 149.685, 29.070, 128.000]  PASS ✓
Test Example 2 (1x1)        →  output: [140.750]  PASS ✓
Test Stress (2048x2048)     →  output: [...]
```

### Clean

```bash
make clean
```

---

## Example Walkthrough

### Example 1 — 2×2 image

Input: `[255, 0, 0,  0, 255, 0,  0, 0, 255,  128, 128, 128]`

| Pixel | R | G | B | Calculation | Gray |
|---|---|---|---|---|---|
| 0 (pure red) | 255 | 0 | 0 | 0.299×255 | **76.245** |
| 1 (pure green) | 0 | 255 | 0 | 0.587×255 | **149.685** |
| 2 (pure blue) | 0 | 0 | 255 | 0.114×255 | **29.070** |
| 3 (mid-gray) | 128 | 128 | 128 | (0.299+0.587+0.114)×128 = 1.0×128 | **128.000** |

### Example 2 — 1×1 image

```
gray = 0.299×100 + 0.587×150 + 0.114×200
     = 29.9 + 88.05 + 22.8
     = 140.75 ✓
```

---

## Complexity

| | Value |
|---|---|
| Time complexity | O(W × H) — fully parallel, one thread per pixel |
| Memory read | 3 × W × H floats |
| Memory write | 1 × W × H floats |
| Shared memory | None required |
| Synchronisation | None inside kernel |
