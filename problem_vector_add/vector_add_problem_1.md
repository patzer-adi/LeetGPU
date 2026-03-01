# GPU Vector Addition (CUDA)

## Overview

This project implements **element-wise vector addition** using **CUDA**.
Given two vectors **A** and **B** containing 32-bit floating-point numbers, the program computes their sum and stores the result in vector **C**.

This implementation is designed for **large-scale data** and leverages GPU parallelism for high performance.

---

## Problem Statement

Given:

* Vector **A** of size `N`
* Vector **B** of size `N`

Compute:

[
C[i] = A[i] + B[i], \quad \text{for } i = 0 \text{ to } N-1
]

### Example

**Input**

```
A = [1.0, 2.0, 3.0, 4.0]
B = [5.0, 6.0, 7.0, 8.0]
```

**Output**

```
C = [6.0, 8.0, 10.0, 12.0]
```

---

## Why Use GPU?

A CPU processes elements **sequentially**:

```
for i in range(N):
    C[i] = A[i] + B[i]
```

A GPU processes elements **in parallel**:

```
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
...
```

This allows the program to efficiently handle very large vectors (tens of millions of elements).

---

## CUDA Implementation Details

### Kernel Function

Each GPU thread computes one element of the result.

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### Key Concepts

| Term          | Meaning                            |
| ------------- | ---------------------------------- |
| `threadIdx.x` | Thread index within a block        |
| `blockIdx.x`  | Block index within the grid        |
| `blockDim.x`  | Number of threads per block        |
| `idx`         | Global index handled by the thread |

Global index calculation:

```
idx = blockIdx.x * blockDim.x + threadIdx.x
```

This uniquely maps each thread to an element in the vector.

---

## Kernel Launch Configuration

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
cudaDeviceSynchronize();
```

### Why 256 threads per block?

* Multiple of warp size (32)
* Good default for most GPUs
* Provides high occupancy and performance

### Why the bounds check?

The grid may launch slightly more threads than needed.
The condition:

```
if (idx < N)
```

prevents out-of-bounds memory access.

---

## Memory Model

* `A`, `B`, and `C` are **device pointers** (memory already on GPU)
* No host-device memory transfer is required inside `solve()`
* Memory access pattern is **coalesced** (sequential), improving performance

---

## Performance Characteristics

* Time Complexity: **O(N)**
* Parallel execution across thousands of GPU cores
* Designed to handle:

  * Up to **100 million elements**
  * Benchmark size: **25 million elements**

---

## Constraints

* `1 ≤ N ≤ 100,000,000`
* Input vectors have identical length
* External libraries are **not used**
* Result must be stored in vector **C**

---

## Execution Flow

1. Host calls `solve()`
2. Grid and block sizes are calculated
3. GPU kernel is launched
4. Each thread:

   * Computes its index
   * Adds corresponding elements
   * Writes result to `C`
5. Host waits for completion

---

## Core Parallel Pattern

This pattern is widely used in CUDA element-wise operations:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    // process element idx
}
```

Applicable to:

* Vector addition
* Element-wise multiplication
* ReLU activation
* Scaling operations

---

## Hardware Used

* NVIDIA GPU (e.g., Tesla T4)
* CUDA Runtime API

---

## Summary

This project demonstrates:

* Basic CUDA kernel design
* Thread indexing and mapping
* Parallel execution model
* Efficient handling of large datasets

Vector addition is a foundational CUDA problem and serves as a starting point for more advanced GPU algorithms.

---

## Future Improvements

* Grid-stride loop for extremely large inputs
* Memory bandwidth optimization
* Extension to multi-dimensional data
* Integration with shared memory for complex operations
# GPU Vector Addition (CUDA)

## Overview

This project implements **element-wise vector addition** using **CUDA**.
Given two vectors **A** and **B** containing 32-bit floating-point numbers, the program computes their sum and stores the result in vector **C**.

This implementation is designed for **large-scale data** and leverages GPU parallelism for high performance.

---

## Problem Statement

Given:

* Vector **A** of size `N`
* Vector **B** of size `N`

Compute:

[
C[i] = A[i] + B[i], \quad \text{for } i = 0 \text{ to } N-1
]

### Example

**Input**

```
A = [1.0, 2.0, 3.0, 4.0]
B = [5.0, 6.0, 7.0, 8.0]
```

**Output**

```
C = [6.0, 8.0, 10.0, 12.0]
```

---

## Why Use GPU?

A CPU processes elements **sequentially**:

```
for i in range(N):
    C[i] = A[i] + B[i]
```

A GPU processes elements **in parallel**:

```
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
...
```

This allows the program to efficiently handle very large vectors (tens of millions of elements).

---

## CUDA Implementation Details

### Kernel Function

Each GPU thread computes one element of the result.

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### Key Concepts

| Term          | Meaning                            |
| ------------- | ---------------------------------- |
| `threadIdx.x` | Thread index within a block        |
| `blockIdx.x`  | Block index within the grid        |
| `blockDim.x`  | Number of threads per block        |
| `idx`         | Global index handled by the thread |

Global index calculation:

```
idx = blockIdx.x * blockDim.x + threadIdx.x
```

This uniquely maps each thread to an element in the vector.

---

## Kernel Launch Configuration

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
cudaDeviceSynchronize();
```

### Why 256 threads per block?

* Multiple of warp size (32)
* Good default for most GPUs
* Provides high occupancy and performance

### Why the bounds check?

The grid may launch slightly more threads than needed.
The condition:

```
if (idx < N)
```

prevents out-of-bounds memory access.

---

## Memory Model

* `A`, `B`, and `C` are **device pointers** (memory already on GPU)
* No host-device memory transfer is required inside `solve()`
* Memory access pattern is **coalesced** (sequential), improving performance

---

## Performance Characteristics

* Time Complexity: **O(N)**
* Parallel execution across thousands of GPU cores
* Designed to handle:

  * Up to **100 million elements**
  * Benchmark size: **25 million elements**

---

## Constraints

* `1 ≤ N ≤ 100,000,000`
* Input vectors have identical length
* External libraries are **not used**
* Result must be stored in vector **C**

---

## Execution Flow

1. Host calls `solve()`
2. Grid and block sizes are calculated
3. GPU kernel is launched
4. Each thread:

   * Computes its index
   * Adds corresponding elements
   * Writes result to `C`
5. Host waits for completion

---

## Core Parallel Pattern

This pattern is widely used in CUDA element-wise operations:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    // process element idx
}
```

Applicable to:

* Vector addition
* Element-wise multiplication
* ReLU activation
* Scaling operations

---

## Hardware Used

* NVIDIA GPU (e.g., Tesla T4)
* CUDA Runtime API

---

## Summary

This project demonstrates:

* Basic CUDA kernel design
* Thread indexing and mapping
* Parallel execution model
* Efficient handling of large datasets

Vector addition is a foundational CUDA problem and serves as a starting point for more advanced GPU algorithms.

---

## Future Improvements

* Grid-stride loop for extremely large inputs
* Memory bandwidth optimization
* Extension to multi-dimensional data
* Integration with shared memory for complex operations
