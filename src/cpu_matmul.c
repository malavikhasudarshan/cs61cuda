#include <iostream>

void cpu_matmul(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int p = 0; p < k; p++)
                acc += A[i * k + p] * B[p * n + j];
            C[i * n + j] = acc;
        }
    }
}