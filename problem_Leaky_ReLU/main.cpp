// main.cpp  –  local test harness for leaky_relu.cu
// Compile with the provided Makefile, then run: ./leaky_relu_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" void solve(const float* input, float* output, int N);

static const float ALPHA = 0.01f;

// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_mixed(float* h, int N) {
    // values in [-1000, 1000] as per constraints
    for (int i = 0; i < N; ++i)
        h[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2000.f;
}

static float cpu_leaky_relu(float x) { return x > 0.f ? x : ALPHA * x; }

static bool check(const float* ref, const float* got, int N) {
    for (int i = 0; i < N; ++i) {
        // relative tolerance for larger values, absolute for values near zero
        float tol = fmaxf(1e-5f, fabsf(ref[i]) * 1e-5f);
        float diff = fabsf(ref[i] - got[i]);
        if (diff > tol) {
            printf("MISMATCH at index %d: input leads to ref=%.8f  got=%.8f\n",
                   i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// ─── fixed example tests ─────────────────────────────────────────────────────

static bool run_example1() {
    // x = [1.0, -2.0, 3.0, -4.0]
    // y = [1.0, -0.02, 3.0, -0.04]
    float hIn[]  = { 1.f, -2.f,   3.f,  -4.f};
    float ref[]  = { 1.f, -0.02f, 3.f,  -0.04f};
    float hOut[4];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  4*sizeof(float));
    cudaMalloc(&dOut, 4*sizeof(float));
    cudaMemcpy(dIn, hIn, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 4*sizeof(float));

    solve(dIn, dOut, 4);
    cudaMemcpy(hOut, dOut, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 4);
    printf("%-42s  N=        4  ... %s\n", "Example 1 [1,-2,3,-4]", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

static bool run_example2() {
    // x = [-1.5, 0.0, 2.5, -3.0]
    // y = [-0.015, 0.0, 2.5, -0.03]
    float hIn[]  = {-1.5f,  0.f,  2.5f, -3.f};
    float ref[]  = {-0.015f, 0.f, 2.5f, -0.03f};
    float hOut[4];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  4*sizeof(float));
    cudaMalloc(&dOut, 4*sizeof(float));
    cudaMemcpy(dIn, hIn, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 4*sizeof(float));

    solve(dIn, dOut, 4);
    cudaMemcpy(hOut, dOut, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 4);
    printf("%-42s  N=        4  ... %s\n", "Example 2 [-1.5,0,2.5,-3]", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, const char* label) {
    printf("%-42s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));

    fill_mixed(hIn, N);
    for (int i = 0; i < N; ++i) ref[i] = cpu_leaky_relu(hIn[i]);

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

// ─── uniform edge cases ───────────────────────────────────────────────────────

static bool run_uniform(int N, float val, const char* label) {
    printf("%-42s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) { hIn[i] = val; ref[i] = cpu_leaky_relu(val); }

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

    tally(run_example1());
    tally(run_example2());

    // Edge cases
    tally(run_uniform(1,     0.f,      "N=1, value=0.0 (exact boundary)"));
    tally(run_uniform(100, -1000.f,    "all at min (-1000)"));
    tally(run_uniform(100,  1000.f,    "all at max (+1000)"));

    // Block-boundary sizes
    struct { int N; const char* label; } cases[] = {
        {        1, "N=1"},
        {      255, "N=255 (non-power-of-2)"},
        {      256, "N=256 (one full block)"},
        {      257, "N=257 (one block + 1)"},
        {    10000, "N=10,000"},
        {  1000000, "N=1,000,000"},
        { 50000000, "N=50,000,000 (benchmark size)"},
    };

    for (auto& c : cases)
        tally(run_test(c.N, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
