//TASK 3: NAIVE MATRIX MULTIPLICATION
#include "check.cuh"

__global__ void mm_naive_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int M, int N, int K){
// TODO: fill this out
}

void mm_naive(const float* dA, const float* dB, float* dC,
int M, int N, int K, dim3 block){
dim3 grid((N + block.x - 1)/block.x,
(M + block.y - 1)/block.y);
mm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
checkCuda(cudaGetLastError());
checkCuda(cudaDeviceSynchronize());
}