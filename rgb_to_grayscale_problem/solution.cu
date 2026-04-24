#include <cuda_runtime.h>
#include <stdio.h>

// ─────────────────────────────────────────────────────────────────────────────
// RGB → Grayscale Kernel
//
// Each thread handles one pixel at flat index `i` (0 … width*height - 1).
//
// Input layout:  [R0, G0, B0,  R1, G1, B1,  …]   (interleaved, row-major)
// Output layout: [gray0, gray1, …]
//
// Formula: gray = 0.299*R + 0.587*G + 0.114*B
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point  (signature must not change)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void solve(const float* input, float* output, int width, int height)
{
    int total_pixels    = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,
                                                                width, height);
    cudaDeviceSynchronize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Optional standalone test harness  (compiled only when TEST_MAIN is defined)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef TEST_MAIN
#include <math.h>
#include <stdlib.h>

static void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void run_test(const char* label,
                     const float* h_input, int width, int height,
                     const float* expected)
{
    int    pixels = width * height;
    size_t in_sz  = pixels * 3 * sizeof(float);
    size_t out_sz = pixels     * sizeof(float);

    float *d_input = NULL, *d_output = NULL;
    check(cudaMalloc(&d_input,  in_sz),  "malloc input");
    check(cudaMalloc(&d_output, out_sz), "malloc output");
    check(cudaMemcpy(d_input, h_input, in_sz, cudaMemcpyHostToDevice), "H→D");

    solve(d_input, d_output, width, height);

    float* h_output = (float*)malloc(out_sz);
    check(cudaMemcpy(h_output, d_output, out_sz, cudaMemcpyDeviceToHost), "D→H");

    printf("Test %-22s  →  output: [", label);
    int pass = 1;
    for (int i = 0; i < pixels; i++) {
        printf("%.3f", h_output[i]);
        if (i < pixels - 1) printf(", ");
        if (expected && fabsf(h_output[i] - expected[i]) > 1e-2f) pass = 0;
    }
    printf("]  %s\n", expected ? (pass ? "PASS ✓" : "FAIL ✗") : "");

    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(void)
{
    // Example 1: 2×2 image
    const float in1[] = {
        255.0f,   0.0f,   0.0f,   // pure red
          0.0f, 255.0f,   0.0f,   // pure green
          0.0f,   0.0f, 255.0f,   // pure blue
        128.0f, 128.0f, 128.0f    // mid-gray
    };
    const float exp1[] = {76.245f, 149.685f, 29.07f, 128.0f};
    run_test("Example 1 (2x2)", in1, 2, 2, exp1);

    // Example 2: 1×1 image
    const float in2[]  = {100.0f, 150.0f, 200.0f};
    const float exp2[] = {140.75f};
    run_test("Example 2 (1x1)", in2, 1, 1, exp2);

    // Stress: 2048×2048 (judge's perf target)
    int W = 2048, H = 2048, pixels = W * H;
    float* h_big = (float*)malloc(pixels * 3 * sizeof(float));
    for (int i = 0; i < pixels * 3; i++) h_big[i] = (float)(i % 256);
    run_test("Stress (2048x2048)", h_big, W, H, NULL);
    free(h_big);

    return 0;
}
#endif // TEST_MAIN
