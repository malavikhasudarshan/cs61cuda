#include <cuda_runtime.h>
#include <iostream>

//naive CUDA kernel
__global__ void cuda_naive_matmul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int e = 0; e < k; ++e)
            sum += A[row * k + e] * B[e * n + col];
        C[row * n + col] = sum;
    }
}

//kernel launcher
void launch_naive_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    cuda_naive_matmul<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
}