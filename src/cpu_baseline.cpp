//TASK 2: BASELINE MATRIX MULTIPLICATION
#include <cstddef>
void mm_cpu(const float* A, const float* B, float* C,
int M, int N, int K){
// TODO: triple nested loops over i (rows of A), j (cols of B), k (shared dim)
// Use row-major: A[i*K + k], B[k*N + j], C[i*N + j]
}