#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

//CPU matmul declaration
void cpu_matmul(const float *A, const float *B, float *C, int m, int n, int k);

int main() {
    srand(42);  //seed for reproducibility
    int m = 512, n = 512, k = 512;
    float *A, *B, *C_cpu, *C_gpu;

    //remember to malloc and free these later!
    malloc_matrix(&A, m, k);
    //...

    //CPU computation baseline
    std::cout << "Running CPU baseline..." << std::endl;
    clock_t startTime = clock();
    cpu_matmul(A, B, C_cpu, m, n, k);
    clock_t endTime = clock();
    std::cout << "CPU time: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s\n";

    //GPU allocation
    //remember to malloc and free these later!
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * k);
    //...

    cudaMemcpy(d_A, A, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*k*n, cudaMemcpyHostToDevice);

    //naive GPU computation
    std::cout << "Running CUDA kernel..." << std::endl;
    startTime = clock();
    launch_naive_kernel(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C_gpu, d_C, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    endTime = clock();
    std::cout << "GPU time (naive): " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s\n";

    //for validating correctness
    bool match = validate(C_gpu, C_cpu, m, n, 1e-3f);
    std::cout << (match ? "✅ Validation passed!\n" : "❌ Validation failed.\n");

    //remember to free memory!
    free_matrix(A);
    //...

    cudaDeviceReset();
    return 0;
}