//TASK 1: 2D COPY KERNEL
#include "check.cuh"

__global__ void copy2D_kernel(const float* __restrict__ in,
float* __restrict__ out,
int rows, int cols) {
// TODO: fill this out
}

void copy2D(const float* d_in, float* d_out, int rows, int cols, dim3 block){
dim3 grid((cols + block.x - 1)/block.x,
(rows + block.y - 1)/block.y);
copy2D_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
checkCuda(cudaGetLastError());
checkCuda(cudaDeviceSynchronize());
}