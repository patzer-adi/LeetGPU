// main.cpp  –  local test harness for matrix_transpose.cu
// Compile with the provided Makefile, then run: ./matrix_transpose_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward-declare the CUDA solve function
extern "C" void solve(const float* input, float* output, int rows, int cols);

// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_random(float* h, int size) {
    for (int i = 0; i < size; ++i)
        h[i] = static_cast<float>(rand()) / RAND_MAX;
}

// CPU reference: output[col][row] = input[row][col]
static void cpu_transpose(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            output[c * rows + r] = input[r * cols + c];
}

static bool check(const float* ref, const float* got, int size, float tol = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - got[i]);
        if (diff > tol) {
            printf("MISMATCH at index %d: ref=%.6f  got=%.6f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// ─── single randomised test ──────────────────────────────────────────────────

static bool run_test(int rows, int cols, const char* label) {
    printf("%-32s  rows=%5d cols=%5d  ... ", label, rows, cols);
    fflush(stdout);

    int szIn  = rows * cols;
    int szOut = cols * rows;   // same count, different shape

    float* hIn  = (float*)malloc(szIn  * sizeof(float));
    float* hOut = (float*)malloc(szOut * sizeof(float));
    float* ref  = (float*)malloc(szOut * sizeof(float));

    fill_random(hIn, szIn);
    cpu_transpose(hIn, ref, rows, cols);

    float *dIn, *dOut;
    cudaMalloc(&dIn,  szIn  * sizeof(float));
    cudaMalloc(&dOut, szOut * sizeof(float));

    cudaMemcpy(dIn, hIn, szIn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, szOut * sizeof(float));

    solve(dIn, dOut, rows, cols);

    cudaMemcpy(hOut, dOut, szOut * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, szOut);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut); free(ref);
    return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    int pass = 0, total = 0;

    // ── Example 1: 2×3  →  3×2 ───────────────────────────────────────────────
    {
        // input:
        //  1 2 3
        //  4 5 6
        // expected output (3×2):
        //  1 4
        //  2 5
        //  3 6
        float hIn[]  = {1,2,3, 4,5,6};
        float ref[]  = {1,4, 2,5, 3,6};
        float hOut[6];

        float *dIn, *dOut;
        cudaMalloc(&dIn,  6*sizeof(float));
        cudaMalloc(&dOut, 6*sizeof(float));
        cudaMemcpy(dIn, hIn, 6*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dOut, 0, 6*sizeof(float));

        solve(dIn, dOut, 2, 3);
        cudaMemcpy(hOut, dOut, 6*sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = check(ref, hOut, 6);
        printf("%-32s  rows=    2 cols=    3  ... %s\n", "Example 1 (2x3 -> 3x2)", ok ? "PASS" : "FAIL");
        if (ok) ++pass; ++total;

        cudaFree(dIn); cudaFree(dOut);
    }

    // ── Example 2: 3×1  →  1×3 ───────────────────────────────────────────────
    {
        float hIn[]  = {1, 2, 3};
        float ref[]  = {1, 2, 3};   // same values, different shape
        float hOut[3];

        float *dIn, *dOut;
        cudaMalloc(&dIn,  3*sizeof(float));
        cudaMalloc(&dOut, 3*sizeof(float));
        cudaMemcpy(dIn, hIn, 3*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dOut, 0, 3*sizeof(float));

        solve(dIn, dOut, 3, 1);
        cudaMemcpy(hOut, dOut, 3*sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = check(ref, hOut, 3);
        printf("%-32s  rows=    3 cols=    1  ... %s\n", "Example 2 (3x1 -> 1x3)", ok ? "PASS" : "FAIL");
        if (ok) ++pass; ++total;

        cudaFree(dIn); cudaFree(dOut);
    }

    // ── Randomised tests ──────────────────────────────────────────────────────
    struct { int rows, cols; const char* label; } cases[] = {
        {   1,    1, "1x1"},
        {   1,   16, "1x16 (single row)"},
        {  16,    1, "16x1 (single col)"},
        {  16,   16, "16x16 square"},
        {  32,   64, "32x64"},
        { 128,  256, "128x256"},
        { 512,  512, "512x512 square"},
        {1000, 2000, "1000x2000"},
    };

    for (auto& c : cases) {
        bool ok = run_test(c.rows, c.cols, c.label);
        if (ok) ++pass;
        ++total;
    }

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
