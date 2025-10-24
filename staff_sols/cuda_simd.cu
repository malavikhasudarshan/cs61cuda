#include <cuda_runtime.h>
__global__ void cuda_simd_matmul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int p = 0; p < k; p += 4) {
        float4 Avals = *((float4*)&A[row * k + p]);
        float4 Bvals = *((float4*)&B[p * n + col]);
        sum += Avals.x * Bvals.x + Avals.y * Bvals.y + Avals.z * Bvals.z + Avals.w * Bvals.w;
    }
    C[row * n + col] = sum;
}