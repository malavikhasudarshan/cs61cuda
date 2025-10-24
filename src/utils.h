#pragma once
#include <cuda_runtime.h>

void malloc_matrix(float **mat, int rows, int cols);
void free_matrix(float *mat);
void init_matrix(float *mat, int rows, int cols);
bool validate(const float *gpuRes, const float *cpuRes, int rows, int cols, float epsilon);

//kernel launch helper
void launch_naive_kernel(float *d_A, float *d_B, float *d_C, int m, int n, int k);
