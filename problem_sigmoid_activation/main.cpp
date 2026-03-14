// main.cpp  –  local test harness for sigmoid.cu
// Compile with the provided Makefile, then run: ./sigmoid_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

extern "C" void solve(const float* X, float* Y, int N);

// ─── helpers ────────────────────────────────────────────────────────────────

static void fill_range(float* h, int N, float lo = -10.f, float hi = 10.f) {
    for (int i = 0; i < N; ++i)
        h[i] = lo + (static_cast<float>(rand()) / RAND_MAX) * (hi - lo);
}

// CPU reference using double for extra accuracy
static float cpu_sigmoid(float x) {
    return (float)(1.0 / (1.0 + exp(-(double)x)));
}

static bool check(const float* ref, const float* got, int N) {
    for (int i = 0; i < N; ++i) {
        float tol = fmaxf(1e-5f, fabsf(ref[i]) * 1e-5f);
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
    // X = [0, 1, -1, 2]
    // Y = [0.5, 0.7311, 0.2689, 0.8808]
    float hX[]  = {0.f, 1.f, -1.f, 2.f};
    float ref[] = {0.5f, 0.7310586f, 0.2689414f, 0.8807970f};
    float hY[4];

    float *dX, *dY;
    cudaMalloc(&dX, 4*sizeof(float));
    cudaMalloc(&dY, 4*sizeof(float));
    cudaMemcpy(dX, hX, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0,  4*sizeof(float));

    solve(dX, dY, 4);
    cudaMemcpy(hY, dY, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hY, 4);
    printf("%-44s  N=        4  ... %s\n", "Example 1 [0,1,-1,2]", ok ? "PASS" : "FAIL");
    cudaFree(dX); cudaFree(dY);
    return ok;
}

static bool run_example2() {
    // X = [0.5, -0.5, 3.0, -3.0]
    // Y = [0.6225, 0.3775, 0.9526, 0.0474]
    float hX[]  = {0.5f, -0.5f, 3.f, -3.f};
    float ref[] = {0.6224593f, 0.3775407f, 0.9525741f, 0.0474259f};
    float hY[4];

    float *dX, *dY;
    cudaMalloc(&dX, 4*sizeof(float));
    cudaMalloc(&dY, 4*sizeof(float));
    cudaMemcpy(dX, hX, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0,  4*sizeof(float));

    solve(dX, dY, 4);
    cudaMemcpy(hY, dY, 4*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hY, 4);
    printf("%-44s  N=        4  ... %s\n", "Example 2 [0.5,-0.5,3,-3]", ok ? "PASS" : "FAIL");
    cudaFree(dX); cudaFree(dY);
    return ok;
}

// ─── known-value spot checks ─────────────────────────────────────────────────

static bool run_spot(float x, float expected, const char* label) {
    float hX[] = {x};
    float hY[1];
    float *dX, *dY;
    cudaMalloc(&dX, sizeof(float));
    cudaMalloc(&dY, sizeof(float));
    cudaMemcpy(dX, hX, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0, sizeof(float));

    solve(dX, dY, 1);
    cudaMemcpy(hY, dY, sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = (fabsf(hY[0] - expected) <= 1e-5f);
    printf("%-44s  N=        1  ... %s\n", label, ok ? "PASS" : "FAIL");
    cudaFree(dX); cudaFree(dY);
    return ok;
}

// ─── randomised test ─────────────────────────────────────────────────────────

static bool run_test(int N, float lo, float hi, const char* label) {
    printf("%-44s  N=%9d  ... ", label, N);
    fflush(stdout);

    float* hX  = (float*)malloc(N * sizeof(float));
    float* hY  = (float*)malloc(N * sizeof(float));
    float* ref = (float*)malloc(N * sizeof(float));

    fill_range(hX, N, lo, hi);
    for (int i = 0; i < N; ++i) ref[i] = cpu_sigmoid(hX[i]);

    float *dX, *dY;
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, N * sizeof(float));
    cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0,  N * sizeof(float));

    solve(dX, dY, N);
    cudaMemcpy(hY, dY, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = check(ref, hY, N);
    printf("%s\n", ok ? "PASS" : "FAIL");

    cudaFree(dX); cudaFree(dY);
    free(hX); free(hY); free(ref);
    return ok;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    srand(42);

    int pass = 0, total = 0;
    auto tally = [&](bool ok) { if (ok) ++pass; ++total; };

    // LeetGPU examples
    tally(run_example1());
    tally(run_example2());

    // Analytical spot checks
    tally(run_spot( 0.f,  0.5f,          "sigmoid(0)   == 0.5 exactly"));
    tally(run_spot( 88.f, 1.f,           "sigmoid(88)  ≈ 1.0 (exp underflows)"));
    tally(run_spot(-88.f, 0.f,           "sigmoid(-88) ≈ 0.0 (exp overflows)"));

    // Randomised sizes
    struct { int N; float lo, hi; const char* label; } cases[] = {
        {        1,  -1.f,   1.f, "N=1"},
        {      255, -10.f,  10.f, "N=255 (non-power-of-2)"},
        {      256, -10.f,  10.f, "N=256 (one full block)"},
        {      257, -10.f,  10.f, "N=257 (one block + 1)"},
        {    10000, -10.f,  10.f, "N=10,000"},
        {  1000000, -10.f,  10.f, "N=1,000,000"},
        { 50000000, -10.f,  10.f, "N=50,000,000 (benchmark size)"},
    };

    for (auto& c : cases)
        tally(run_test(c.N, c.lo, c.hi, c.label));

    printf("\n%d / %d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
