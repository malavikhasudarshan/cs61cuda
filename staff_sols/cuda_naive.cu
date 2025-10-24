__global__ void cuda_naive_matmul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int p = 0; p < k; ++p)
            sum += A[row * k + p] * B[p * n + col];
        C[row * n + col] = sum;
    }
}

void launch_naive_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    cuda_naive_matmul<<<blocks, threads>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
}
