#include "utils.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

void malloc_matrix(float **mat, int rows, int cols) {
    *mat = (float*)malloc(sizeof(float) * rows * cols);
}
void free_matrix(float *mat) { free(mat); }

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool validate(const float *gpuRes, const float *cpuRes, int rows, int cols, float epsilon) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(gpuRes[i] - cpuRes[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i
                      << ": GPU=" << gpuRes[i] << " CPU=" << cpuRes[i] << std::endl;
            return false;
        }
    }
    return true;
}
