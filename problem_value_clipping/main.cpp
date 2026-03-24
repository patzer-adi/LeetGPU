// main.cpp  –  local test harness for value_clipping.cu
// Compile with the provided Makefile, then run: ./value_clipping_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" void solve(const float* input, float* output, float lo, float hi, int N);

// ─── helpers ────────────────────────────────────────────────────────────────

static float cpu_clip(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static bool check(const float* ref, const float* got, int N) {
    for (int i = 0; i < N; ++i) {
        if (fabsf(ref[i] - got[i]) > 1e-6f) {
            printf("MISMATCH at index %d: ref=%.6f  got=%.6f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// ─── fixed example tests ─────────────────────────────────────────────────────

static bool run_example1() {
    // [1.5,-2.0,3.0,4.5]  lo=0.0  hi=3.5  →  [1.5,0.0,3.0,3.5]
    float hIn[]  = {1.5f, -2.0f, 3.0f, 4.5f};
    float ref[]  = {1.5f,  0.0f, 3.0f, 3.5f};
    float hOut[4];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  4*sizeof(float));
    cudaMalloc(&dOut, 4*sizeof(float));
    cudaMemcpy(dIn, hIn, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 4*sizeof(float));

    solve(dIn, dOut, 0.0f, 3.5f, 4);
    cudaMemcpy(hOut, dOut, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 4);
    printf("%-46s  N=      4  lo= 0.0  hi=3.5  ... %s\n",
           "Example 1 [1.5,-2,3,4.5]", ok ? "PASS" : "FAIL");
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

static bool run_example2() {
    // [-1.0,2.0,5.0]  lo=-0.5  hi=2.5  →  [-0.5,2.0,2.5]
    float hIn[]  = {-1.0f, 2.0f, 5.0f};
    float ref[]  = {-0.5f, 2.0f, 2.5f};
    float hOut[3];

    float *dIn, *dOut;
    cudaMalloc(&dIn,  3*sizeof(float));
    cudaMalloc(&dOut, 3*sizeof(float));
    cudaMemcpy(dIn, hIn, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, 3*sizeof(float));

    solve(dIn, dOut, -0.5f, 2.5f, 3);
    cudaMemcpy(hOut, dOut, 3*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, 3);
    printf("%-46s  N=      3  lo=-0.5  hi=2.5  ... %s\n",
           "Example 2 [-1,2,5]", ok ? "PASS" : "FAIL");
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, float lo, float hi, const char* label) {
    printf("%-46s  N=%6d  lo=%5.1f  hi=%5.1f  ... ", label, N, lo, hi);
    fflush(stdout);

    float* hIn  = (float*)malloc(N * sizeof(float));
    float* hOut = (float*)malloc(N * sizeof(float));
    float* ref  = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i)
        hIn[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2e6f;
    for (int i = 0; i < N; ++i) ref[i] = cpu_clip(hIn[i], lo, hi);

    float *dIn, *dOut;
    cudaMalloc(&dIn,  N * sizeof(float));
    cudaMalloc(&dOut, N * sizeof(float));
    cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0,  N * sizeof(float));

    solve(dIn, dOut, lo, hi, N);
    cudaMemcpy(hOut, dOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hOut, N);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut); free(ref);
    return ok;
}

// ─── special boundary tests ───────────────────────────────────────────────────

static bool run_boundary(const char* label, float val, float lo, float hi, float expected) {
    float hIn[]  = {val};
    float hOut[1];
    float *dIn, *dOut;
    cudaMalloc(&dIn,  sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dIn, hIn, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dOut, 0, sizeof(float));
    solve(dIn, dOut, lo, hi, 1);
    cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = (fabsf(hOut[0] - expected) < 1e-6f);
    printf("%-46s  N=      1  lo=%5.1f  hi=%5.1f  ... %s\n", label, lo, hi, ok ? "PASS" : "FAIL");
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    int pass = 0, total = 0;
    auto tally = [&](bool ok) { if (ok) ++pass; ++total; };

    tally(run_example1());
    tally(run_example2());

    // Exact boundary values (lo and hi themselves must pass through unchanged)
    tally(run_boundary("x == lo exactly",        0.0f,  0.0f, 1.0f, 0.0f));
    tally(run_boundary("x == hi exactly",        1.0f,  0.0f, 1.0f, 1.0f));
    // lo == hi (degenerate range — all values collapse to lo==hi)
    tally(run_boundary("lo == hi (degenerate)",  5.0f,  2.0f, 2.0f, 2.0f));
    tally(run_boundary("lo == hi, val < lo",     1.0f,  2.0f, 2.0f, 2.0f));
    tally(run_boundary("lo == hi, val > hi",    99.0f,  2.0f, 2.0f, 2.0f));
    // Negative range
    tally(run_boundary("negative range, clamp up",  -5.0f, -3.0f, -1.0f, -3.0f));
    tally(run_boundary("negative range, clamp dn",   0.0f, -3.0f, -1.0f, -1.0f));
    tally(run_boundary("negative range, pass-thru", -2.0f, -3.0f, -1.0f, -2.0f));

    // Randomised sizes with various ranges
    struct { int N; float lo, hi; const char* label; } cases[] = {
        {      1,   -1.f,    1.f, "N=1"},
        {    255,    0.f,   10.f, "N=255 (non-power-of-2)"},
        {    256,    0.f,   10.f, "N=256 (one full block)"},
        {    257,    0.f,   10.f, "N=257 (one block + 1)"},
        {  10000, -100.f,  100.f, "N=10,000"},
        { 100000, -1e6f,   1e6f,  "N=100,000 (benchmark, full range)"},
        { 100000,    0.f,    0.f, "N=100,000 (lo==hi==0, all collapse)"},
    };

    for (auto& c : cases) tally(run_test(c.N, c.lo, c.hi, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
