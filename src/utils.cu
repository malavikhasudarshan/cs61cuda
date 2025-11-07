void mm_naive(const float* dA, const float* dB, float* dC, int M, int N, int K,
dim3 block) {
dim3 grid( (N + block.x - 1)/block.x, (M + block.y - 1)/block.y );
mm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

void mm_simd(const float* dA, const float* dB, float* dC, int M, int N, int K,
int vec, dim3 block) {
if (vec == 4) {
dim3 grid( ((N + 3)/4 + block.x - 1)/block.x,
( M + block.y - 1)/block.y );
mm_simd_kernel<4><<<grid, block>>>(dA, dB, dC, M, N, K);
} else {
dim3 grid( (N + block.x - 1)/block.x, (M + block.y - 1)/block.y );
mm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}
}