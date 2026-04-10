#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx;

    while (i < N)
    {
        if (input[i] < 0)
        {
            output[i] = 0.0f;
        }
        else if (input[i] >= 0)
        {
            output[i] = input[i];
        }

        i += blockDim.x * gridDim.x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
