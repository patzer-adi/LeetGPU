// main.cpp  –  local test harness for relu.cu
// Compile with the provided Makefile, then run: ./relu_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" void solve(const float* input, float* output, int N);

// ─── helpers ────────────────────────────────────────────────────────────────

// Fill with values in [-range/2, +range/2]
static void fill_mixed(float* h, int N, float range = 10.f) {
    for (int i = 0; i < N; ++i)
        h[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * range;
}

static float cpu_relu(float x) { return x < 0.f ? 0.f : x; }

static bool check(const float* ref, const float* got, int N, float tol = 1e-6f) {
    for (int i = 0; i < N; ++i) {
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
    float hIn[]  = {-2.f, -1.f, 0.f, 1.f, 2.f};
    float ref[]  = { 0.f,  0.f, 0.f, 1.f, 2.f};
    float hOut[5];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  5*sizeof(float));
    cudaMalloc(&dOut, 5*sizeof(float));
    cudaMemcpy(dIn, hIn, 5*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 5*sizeof(float));

    solve(dIn, dOut, 5);
    cudaMemcpy(hOut, dOut, 5*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 5);
    printf("%-40s  N=        5  ... %s\n", "Example 1 [-2,-1,0,1,2]", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

static bool run_example2() {
    float hIn[]  = {-3.5f, 0.f, 4.2f};
    float ref[]  = { 0.f,  0.f, 4.2f};
    float hOut[3];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  3*sizeof(float));
    cudaMalloc(&dOut, 3*sizeof(float));
    cudaMemcpy(dIn, hIn, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 3*sizeof(float));

    solve(dIn, dOut, 3);
    cudaMemcpy(hOut, dOut, 3*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 3);
    printf("%-40s  N=        3  ... %s\n", "Example 2 [-3.5,0,4.2]", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, const char* label) {
    printf("%-40s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));

    fill_mixed(hIn, N);
    for (int i = 0; i < N; ++i) ref[i] = cpu_relu(hIn[i]);

    float *dIn, *dOut;
    cudaMalloc(&dIn,  N * sizeof(float));
    cudaMalloc(&dOut, N * sizeof(float));
    cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, N * sizeof(float));

    solve(dIn, dOut, N);
    cudaMemcpy(hOut, dOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, N);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut); free(ref);
    return ok;
}

// ─── all-negative / all-positive edge cases ───────────────────────────────────

static bool run_uniform(int N, float val, const char* label) {
    printf("%-40s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) { hIn[i] = val; ref[i] = cpu_relu(val); }

    float *dIn, *dOut;
    cudaMalloc(&dIn,  N * sizeof(float));
    cudaMalloc(&dOut, N * sizeof(float));
    cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, N * sizeof(float));

    solve(dIn, dOut, N);
    cudaMemcpy(hOut, dOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, N);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut); free(ref);
    return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    int pass = 0, total = 0;
    auto tally = [&](bool ok) { if (ok) ++pass; ++total; };

    // Fixed examples from LeetGPU
    tally(run_example1());
    tally(run_example2());

    // Edge cases
    tally(run_uniform(1,      0.f,   "N=1, value=0 (boundary)"));
    tally(run_uniform(100, -999.f,   "all negative"));
    tally(run_uniform(100,  999.f,   "all positive"));

    // Randomised sizes
    struct { int N; const char* label; } cases[] = {
        {       1, "N=1"},
        {     255, "N=255 (non-power-of-2)"},
        {     256, "N=256 (one block)"},
        {     257, "N=257 (just over one block)"},
        {   10000, "N=10,000"},
        { 1000000, "N=1,000,000"},
        {25000000, "N=25,000,000 (benchmark size)"},
    };

    for (auto& c : cases)
        tally(run_test(c.N, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
