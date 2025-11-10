//TASK 4: SIMD MATRIX MULTIPLICATION
#include "check.cuh"

template<int V>
__global__ void mm_simd_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int M, int N, int K){
// TODO: fill this out
}

void mm_simd(const float* dA, const float* dB, float* dC,
int M, int N, int K, int vec, dim3 block){
// TODO: fill this out
}