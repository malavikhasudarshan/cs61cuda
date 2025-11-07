//TASK 4: SIMD MATRIX MULTIPLICATION
#include "check.cuh"

template<int V>
__global__ void mm_simd_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int M, int N, int K){
// TODO: compute i (row) and j0 (first column of this thread's vector)
// TODO: maintain acc[V] and handle aligned vs tail paths
}

void mm_simd(const float* dA, const float* dB, float* dC,
int M, int N, int K, int vec, dim3 block){
// TODO: call specialized kernel for vec==4, else fall back
}