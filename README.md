# CS61Cuda: Matrix Multiplication on GPU

This repository contains the starter code for your CUDA-based matrix multiplication assignment.

## Structure
- **CPU Implementation:** `cpu_matmul.cu`
- **Naive CUDA Implementation:** `cuda_naive.cu`
- **SIMD / Optimization Placeholder:** `cuda_simd.cu`
- **Build and Run:** defined in `Makefile`

## Overview  
You'll speed up matrix multiplication \(C = A \times B\) using three progressively faster implementations:  
1. CPU single-thread baseline  
2. CUDA naive kernel — one thread per output element  
3. CUDA SIMD kernel — vectorized loads/stores per thread

You will use Hive machines and CUDA tools for development.

## Tasks

| Task | Description | Points | Learning Goals |
|-------|-------------|--------|---------------|
| 1 | Welcome to 61Cuda: implement a simple copy kernel | 10 | Thread indexing, kernel launch basics |
| 2 | CPU baseline matmul (nested loops) | 15 | Matrix indexing, correctness |
| 3 | CUDA naive matmul kernel | 35 | Thread-level parallelism; grid/block mapping |
| 4 | CUDA SIMD matmul kernel | 40 | Data-level parallelism; vectorization |
| 5 | Performance engineering (optional) | 15 | Creative optimization, leaderboard |

Total: 100 required points, 15 extra credit optional

## Environment Setup  
- Use Hive machines for CUDA development.  
- Use provided starter repo and Makefile.  
- Validate output by comparing to CPU baseline.

## Submission  
Upload your code to Gradescope under the CS61Cuda assignment.

## Notes  
- Document your code well.  
- Test correctness thoroughly.  
- Extra credit is optional but encouraged!

---
