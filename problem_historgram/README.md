# Histogram — CUDA Solution

**Problem:** LeetGPU · Histogram (Easy)

---

## What does this program do?

Counts how many times each integer value appears in the input array.
Each value maps directly to a bin index.

```
input  = [0, 1, 2, 1, 3, 2, 2, 4, 0]   (N=9, num_bins=5)
output = [2, 2, 3, 1, 1]
          ↑  ↑  ↑  ↑  ↑
         bin 0  1  2  3  4
```

---

## Files

| File | Purpose |
|---|---|
| `solution.cu` | CUDA kernel + `solve()` entry + optional test harness |
| `Makefile` | Builds the shared library and/or test binary |
| `README.md` | This file |

---

## Implementation Details

### Kernel design

```cuda
int val = -1;
if (idx < N)
    val = input[idx];

if (val >= 0 && val < num_bins)
    atomicAdd(histogram + val, 1);
```

**Two-step sentinel pattern:**
1. Initialise `val = -1` (invalid sentinel)
2. Only read `input[idx]` if in bounds
3. Only call `atomicAdd` if `val` is a valid bin index

This cleanly handles both out-of-bounds threads (idx ≥ N) and defensive input validation.

### Why atomicAdd is essential

Without it, a **race condition** destroys counts:

```
❌ Without atomics (broken):
   Thread A reads histogram[3] = 5
   Thread B reads histogram[3] = 5   ← doesn't see A's update yet
   Thread A writes histogram[3] = 6
   Thread B writes histogram[3] = 6  ← overwrites A's work!
   Final: 6 instead of 7. One count silently lost.

✓ With atomicAdd:
   Thread A: lock → read 5 → write 6 → unlock
   Thread B: lock → read 6 → write 7 → unlock
   Final: 7. Correct.
```

`atomicAdd(ptr, 1)` is a single **read-modify-write** hardware instruction.
The memory controller serialises all requests to the same address — no two
threads execute their RMW simultaneously.

### Performance note

atomicAdd is fast when threads hit **different** bins. It serialises (becomes
slow) when many threads contend on the **same** bin. Worst case: all input
values are identical → one bin gets N sequential atomic operations.

Mitigation (beyond this Easy problem):
- **Shared memory privatisation** — each block builds a local histogram in
  fast shared memory, then merges into global memory with one atomic per bin
  per block.
- For this problem, global atomics are sufficient under the given constraints.

---

## Bugs Fixed From Original Code

| # | Bug | Original | Fixed |
|---|---|---|---|
| 1 | Missing sync | No `cudaDeviceSynchronize()` | Added after kernel launch |
| 2 | Missing bounds check on val | `if (val >= 0)` only | `if (val >= 0 && val < num_bins)` |

Bug 1 is the critical one — without the sync, the judge may read an empty
histogram because the CPU returns before the GPU finishes writing.

---

## Building

```bash
make lib      # → histogram.so  (for the judge)
make test     # → builds + runs test binary
make clean
```

Change `CUDA_ARCH` in the Makefile to match your GPU:

| GPU | CUDA_ARCH |
|---|---|
| RTX 2070/2080 | sm_75 |
| RTX 3080/3090 | sm_86 |
| RTX 4090 | sm_89 |
| A100 | sm_80 |

---

## atomicAdd — Full Explanation

### Signature
```cuda
int atomicAdd(int* address, int val);
```
Atomically adds `val` to `*address` and returns the **old** value.
"Atomically" = the entire read-modify-write is one indivisible hardware op.

### Hardware mechanism
1. Thread issues atomic request to the L2 cache / memory controller.
2. The memory controller has a **reservation station** that queues requests
   to the same address.
3. Each request completes fully (read → add → write) before the next begins.
4. Other threads wanting the same address wait in hardware — not busy-spinning.
5. The old value is returned (useful for indexing; unused here).

### Available CUDA atomic operations

| Function | Operation |
|---|---|
| `atomicAdd(ptr, val)` | `*ptr += val` |
| `atomicSub(ptr, val)` | `*ptr -= val` |
| `atomicMax(ptr, val)` | `*ptr = max(*ptr, val)` |
| `atomicMin(ptr, val)` | `*ptr = min(*ptr, val)` |
| `atomicExch(ptr, val)` | `*ptr = val` (returns old) |
| `atomicCAS(ptr, cmp, val)` | if `*ptr==cmp`: `*ptr = val` (the primitive all others are built on) |

All support `int`, `unsigned int`, `unsigned long long`.
`atomicAdd` also supports `float` on SM 6.0+ and `double` on SM 6.0+.

---

## Complexity

| | Value |
|---|---|
| Time complexity | O(N) — one pass, fully parallel |
| Threads launched | N (padded to multiple of 256) |
| Memory reads | N (one per thread) |
| Atomic operations | N (one atomicAdd per thread) |
| Bottleneck | Atomic contention when many threads hit the same bin |
