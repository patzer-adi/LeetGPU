#include <cuda_runtime.h>
#include <stdio.h>

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Copy Kernel
//
// The N×N matrix is stored in row-major flat layout:
//   element (row, col)  →  index  row*N + col
//
// Each thread copies exactly one element: B[tid] = A[tid]
// No arithmetic, no shared memory — pure memory bandwidth exercise.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void copy_matrix_kernel(const float* __restrict__ A,
                                    float*       __restrict__ B,
                                    int N)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (unsigned int)(N * N))
        B[tid] = A[tid];
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point  (signature must not change)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void solve(const float* A, float* B, int N)
{
    int total           = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (total + threadsPerBlock - 1) / threadsPerBlock;

    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Optional test harness  (compiled only when TEST_MAIN is defined)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef TEST_MAIN
#include <stdlib.h>
#include <math.h>

static void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void run_test(const char* label, const float* h_A, int N)
{
    int    total = N * N;
    size_t sz    = total * sizeof(float);

    float *d_A = NULL, *d_B = NULL;
    check(cudaMalloc(&d_A, sz), "malloc A");
    check(cudaMalloc(&d_B, sz), "malloc B");
    check(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice), "H→D");

    solve(d_A, d_B, N);

    float* h_B = (float*)malloc(sz);
    check(cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost), "D→H");

    int pass = 1;
    for (int i = 0; i < total; i++)
        if (fabsf(h_B[i] - h_A[i]) > 1e-6f) { pass = 0; break; }

    printf("Test %-22s  N=%-5d  %s\n", label, N, pass ? "PASS ✓" : "FAIL ✗");

    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
}

int main(void)
{
    // Example 1: 2×2
    const float A1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    run_test("Example 1 (2x2)", A1, 2);

    // Example 2: 3×3
    const float A2[] = {5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.1f, 11.2f, 12.3f, 13.4f};
    run_test("Example 2 (3x3)", A2, 3);

    // Stress: 4096×4096 (judge's perf target)
    int N = 4096;
    float* h_big = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N * N; i++) h_big[i] = (float)i * 0.001f;
    run_test("Stress (4096x4096)", h_big, N);
    free(h_big);

    return 0;
}
#endif // TEST_MAIN
