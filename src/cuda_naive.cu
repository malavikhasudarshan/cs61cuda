#include <cuda_runtime.h>
#include <iostream>

//naive CUDA kernel
__global__ void cuda_naive_matmul(float *A, float *B, float *C, int m, int n, int k) {
    //TODO: fill in the blanks to implement naive matrix multiplication
    int row = ________________
    int col = ________________
    if _________ {
        float sum = 0.0f;
        for (______________)
            ______________________
        C[______________] = ___;
    }
}

//kernel launcher
void launch_naive_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    cuda_naive_matmul<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
}