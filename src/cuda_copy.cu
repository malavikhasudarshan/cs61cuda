//TASK 1: 2D COPY KERNEL
#include "check.cuh"

__global__ void copy2D_kernel(const float* __restrict__ in,
float* __restrict__ out,
int rows, int cols) {
// TODO: compute r, c using blockIdx/threadIdx and blockDim
// TODO: bounds check: if (r >= rows || c >= cols) return;
// TODO: copy one element
}

void copy2D(const float* d_in, float* d_out, int rows, int cols, dim3 block){
// Suggest block = (16,16,1)
dim3 grid((cols + block.x - 1)/block.x,
(rows + block.y - 1)/block.y);
copy2D_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
checkCuda(cudaGetLastError());
checkCuda(cudaDeviceSynchronize());
}