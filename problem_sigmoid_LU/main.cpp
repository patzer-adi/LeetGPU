// main.cpp  –  local test harness for silu.cu
// Compile with the provided Makefile, then run: ./silu_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" void solve(const float* input, float* output, int N);

// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_range(float* h, int N, float lo = -100.f, float hi = 100.f) {
    for (int i = 0; i < N; ++i)
        h[i] = lo + (static_cast<float>(rand()) / RAND_MAX) * (hi - lo);
}

// CPU reference: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static double cpu_silu(double x) { return x / (1.0 + exp(-x)); }

static bool check(const float* ref, const float* got, int N) {
    for (int i = 0; i < N; ++i) {
        // generous relative tolerance – expf has ~1 ULP error on GPU
        float tol = fmaxf(1e-4f, fabsf(ref[i]) * 1e-4f);
        float diff = fabsf(ref[i] - got[i]);
        if (diff > tol) {
            printf("MISMATCH at index %d: ref=%.8f  got=%.8f  diff=%.2e\n",
                   i, ref[i], got[i], diff);
            return false;
        }
    }
    return true;
}

// ─── fixed example tests ─────────────────────────────────────────────────────

static bool run_example1() {
    float hIn[]  = { 0.5f,  1.0f, -0.5f};
    float ref[]  = { 0.3112295f, 0.7310586f, -0.1887705f};
    float hOut[3];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  3*sizeof(float));
    cudaMalloc(&dOut, 3*sizeof(float));
    cudaMemcpy(dIn, hIn, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0,  3*sizeof(float));

    solve(dIn, dOut, 3);
    cudaMemcpy(hOut, dOut, 3*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 3);
    printf("%-44s  N=        3  ... %s\n", "Example 1 [0.5,1,-0.5]", ok ? "PASS" : "FAIL");
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

static bool run_example2() {
    float hIn[]  = {-1.f, -2.f, -3.f, -4.f, -5.f};
    float ref[]  = {-0.26894143f, -0.23840584f, -0.14227763f,
                    -0.07194484f, -0.03346425f};
    float hOut[5];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  5*sizeof(float));
    cudaMalloc(&dOut, 5*sizeof(float));
    cudaMemcpy(dIn, hIn, 5*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0,  5*sizeof(float));

    solve(dIn, dOut, 5);
    cudaMemcpy(hOut, dOut, 5*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 5);
    printf("%-44s  N=        5  ... %s\n", "Example 2 [-1,-2,-3,-4,-5]", ok ? "PASS" : "FAIL");
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, float lo, float hi, const char* label) {
    printf("%-44s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));

    fill_range(hIn, N, lo, hi);
    for (int i = 0; i < N; ++i) ref[i] = (float)cpu_silu((double)hIn[i]);

    float *dIn, *dOut;
    cudaMalloc(&dIn,  N * sizeof(float));
    cudaMalloc(&dOut, N * sizeof(float));
    cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0,  N * sizeof(float));

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

    // Special values
    {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        float hIn[]  = {0.f};
        float hOut[1];
        float *dIn, *dOut;
        cudaMalloc(&dIn,  sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        cudaMemcpy(dIn, hIn, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dOut, 0, sizeof(float));
        solve(dIn, dOut, 1);
        cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        float ref = 0.f;
        bool ok = (fabsf(hOut[0] - ref) < 1e-6f);
        printf("%-44s  N=        1  ... %s\n", "SiLU(0) == 0", ok ? "PASS" : "FAIL");
        if (ok) ++pass; ++total;
        cudaFree(dIn); cudaFree(dOut);
    }
    {
        // SiLU(100) ≈ 100  (sigmoid(100) ≈ 1)
        float hIn[]  = {100.f};
        float hOut[1];
        float *dIn, *dOut;
        cudaMalloc(&dIn,  sizeof(float));
        cudaMalloc(&dOut, sizeof(float));
        cudaMemcpy(dIn, hIn, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(dOut, 0, sizeof(float));
        solve(dIn, dOut, 1);
        cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = (fabsf(hOut[0] - 100.f) < 0.01f);
        printf("%-44s  N=        1  ... %s\n", "SiLU(100) ≈ 100", ok ? "PASS" : "FAIL");
        if (ok) ++pass; ++total;
        cudaFree(dIn); cudaFree(dOut);
    }

    struct { int N; float lo, hi; const char* label; } cases[] = {
        {      1,   -1.f,   1.f, "N=1"},
        {    255,  -10.f,  10.f, "N=255 (non-power-of-2)"},
        {    256,  -10.f,  10.f, "N=256 (one full block)"},
        {    257,  -10.f,  10.f, "N=257 (one block + 1)"},
        {   1000, -100.f, 100.f, "N=1000, full range"},
        {  10000, -100.f, 100.f, "N=10,000 (constraint max)"},
        {  50000, -100.f, 100.f, "N=50,000 (benchmark size)"},
    };

    for (auto& c : cases)
        tally(run_test(c.N, c.lo, c.hi, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
