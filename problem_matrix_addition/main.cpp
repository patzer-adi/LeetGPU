// main.cpp  –  local test harness for matrix_addition.cu
// Compile with the provided Makefile, then run: ./matrix_addition_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" void solve(const float* A, const float* B, float* C, int N);

// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_random(float* h, int size) {
    for (int i = 0; i < size; ++i)
        h[i] = static_cast<float>(rand()) / RAND_MAX * 10.f;
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

// ─── fixed example tests ─────────────────────────────────────────────────────

static bool run_example1() {
    // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  C = [[6,8],[10,12]]
    float hA[] = {1.f, 2.f, 3.f, 4.f};
    float hB[] = {5.f, 6.f, 7.f, 8.f};
    float ref[] = {6.f, 8.f, 10.f, 12.f};
    float hC[4];

    float *dA, *dB, *dC;
    cudaMalloc(&dA, 4*sizeof(float));
    cudaMalloc(&dB, 4*sizeof(float));
    cudaMalloc(&dC, 4*sizeof(float));
    cudaMemcpy(dA, hA, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, 4*sizeof(float));

    solve(dA, dB, dC, 2);
    cudaMemcpy(hC, dC, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hC, 4);
    printf("%-34s  N=   2  ... %s\n", "Example 1 (2x2)", ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

static bool run_example2() {
    // A = [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]
    // B = all 0.5
    // C = [[2,3,4],[5,6,7],[8,9,10]]
    float hA[] = {1.5f,2.5f,3.5f, 4.5f,5.5f,6.5f, 7.5f,8.5f,9.5f};
    float hB[] = {0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f, 0.5f,0.5f,0.5f};
    float ref[] = {2.f, 3.f, 4.f,  5.f, 6.f, 7.f,  8.f, 9.f,10.f};
    float hC[9];

    float *dA, *dB, *dC;
    cudaMalloc(&dA, 9*sizeof(float));
    cudaMalloc(&dB, 9*sizeof(float));
    cudaMalloc(&dC, 9*sizeof(float));
    cudaMemcpy(dA, hA, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, 9*sizeof(float));

    solve(dA, dB, dC, 3);
    cudaMemcpy(hC, dC, 9*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hC, 9);
    printf("%-34s  N=   3  ... %s\n", "Example 2 (3x3)", ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, const char* label) {
    printf("%-34s  N=%4d  ... ", label, N);
    fflush(stdout);

    int sz = N * N;
    float* hA  = (float*)malloc(sz * sizeof(float));
    float* hB  = (float*)malloc(sz * sizeof(float));
    float* ref = (float*)malloc(sz * sizeof(float));
    float* hC  = (float*)malloc(sz * sizeof(float));

    fill_random(hA, sz);
    fill_random(hB, sz);
    for (int i = 0; i < sz; ++i) ref[i] = hA[i] + hB[i];

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz * sizeof(float));
    cudaMalloc(&dB, sz * sizeof(float));
    cudaMalloc(&dC, sz * sizeof(float));
    cudaMemcpy(dA, hA, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sz * sizeof(float));

    solve(dA, dB, dC, N);
    cudaMemcpy(hC, dC, sz * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hC, sz);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(ref); free(hC);
    return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    int pass = 0, total = 0;

    auto tally = [&](bool ok) { if (ok) ++pass; ++total; };

    tally(run_example1());
    tally(run_example2());

    struct { int N; const char* label; } cases[] = {
        {   1, "1x1"},
        {  15, "15x15 (non-power-of-2)"},
        {  16, "16x16"},
        {  32, "32x32"},
        { 128, "128x128"},
        { 256, "256x256"},
        {1024, "1024x1024"},
        {4096, "4096x4096 (benchmark size)"},
    };

    for (auto& c : cases)
        tally(run_test(c.N, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
