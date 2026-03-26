#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = id;

    while(i < N){
        if(input[i] > 0){
            output[i] = input[i];
        }
        else if(input[i] <= 0){
            output[i] = 0.01f * input[i];
        }

        i += blockDim.x * gridDim.x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
