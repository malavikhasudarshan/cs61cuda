# **CS61Cuda Project: Matmul & CUDA Fundamentals**

Welcome to **CS61Cuda\!\!** This is a mini‑project that introduces you to GPU programming with CUDA by building up to a fast matrix multiply. You’ll start with a CPU reference, write your first CUDA kernels, learn to reason about grids/blocks/threads, and finally add simple vectorization (SIMD) on the GPU. An optional performance sandbox lets you explore optimizations for bragging rights.

## **Learning Goals**

* Map data‑parallel work to CUDA’s **grid/block/thread** hierarchy.  
* Practice **indexing** and **bounds checks** in 1D/2D.  
* Understand **memory access patterns** (coalescing) and why they matter.  
* See the benefits of **TLP** (thread‑level parallelism) and **DLP** (data‑level/SIMD) on the GPU.  
* Read simple performance counters and reason about **memory‑ vs compute‑bound** kernels.

## **Repo Layout & What You Edit**

```c
cs61cuda/
├─ CMakeLists.txt # or Makefile (both provided)
├─ include/
│ └─ matmul.h # shared function prototypes
├─ src/
│ ├─ main.cpp # driver: parses flags, allocs, calls your code
│ ├─ cpu_baseline.cpp # ✏️ Task 2: implement CPU matmul
│ ├─ cuda_copy.cu # ✏️ Task 1: 2D copy kernel
│ ├─ cuda_naive.cu # ✏️ Task 3: naive CUDA matmul (1 output/thread)
│ ├─ cuda_simd.cu # ✏️ Task 4: vectorized CUDA matmul (no shared mem)
│ ├─ utils.cu # timers, error checks, random init, compare
│ └─ check.cuh # CUDA error macros (provided)
├─ tests/
│ ├─ correctness_tests.py # local correctness checks
│ └─ perf_runner.py # runs sizes & prints throughput
├─ data/
│ └─ generate.py # makes small sample matrices (optional)
├─ scripts/
│ ├─ build.sh # nvcc or cmake build helper
│ └─ run_all.sh # runs all tasks & tests
└─ README.md # quickstart

```

**You will edit:** `src/cpu_baseline.cpp`, `src/cuda_copy.cu`, `src/cuda_naive.cu`, `src/cuda_simd.cu`.

![][image1]

**Autograder policy.** We check correctness on hidden sizes, run times on a standard GPU, and basic style (clear bounds checks, no UB). We do not require a specific speedup, but we verify your kernels scale sensibly with size.

**Collaboration.** Discuss ideas high‑level with peers; code must be your own. Cite any online sources you consulted.

## **CUDA Primer (read first)**

* A **kernel** is a C/C++ function annotated `__global__` and launched with `<<<grid, block>>>`.  
* Each launch creates a 2‑D/3‑D **grid** of **blocks**; each block contains many **threads**. Every thread runs the same kernel on different data.  
* Built‑in variables inside kernels: `blockIdx.{x,y,z}`, `threadIdx.{x,y,z}`, `blockDim.{x,y,z}`, `gridDim.{x,y,z}`.  
* Memory spaces:  
  * **Global** (device DRAM): large, high‑latency; visible to all threads.  
  * **Shared** (on‑chip, per‑block): small, low‑latency; visible to threads in the same block.  
  * **Registers** (per‑thread): fastest.  
* **Memory coalescing:** threads in a warp should access consecutive addresses to combine loads/stores into few transactions.  
* **Synchronization:** `__syncthreads()` is a barrier for all threads in a block (not across blocks).

You’ll always (1) map data to threads, (2) ensure bounds checks, (3) choose grid/block sizes, and (4) verify results.

## **Task 1 — Welcome to 61Cuda: 2D Copy Kernel (10 pts)**

### **Conceptual Overview**

Warm up with indexing and coalesced global memory access. You’ll copy a dense matrix from device input to device output using a 2D grid of 2D blocks, one element per thread.

### **Data Flow**

* **Input:** `in ∈ R^{rows×cols}` (row‑major `in[r*cols + c]`), allocated in **device** memory.  
* **Processing:** For each `(r,c)` covered by a thread, read `in[r,c]` and write it to `out[r,c]`. Use 2‑D indexing and guard against out‑of‑bounds.  
* **Output:** `out ∈ R^{rows×cols}` identical to `in` (device memory).

### **Your Task**

Implement `copy2D_kernel` and the host wrapper `copy2D(...)`

```c
#include "check.cuh"

__global__ void copy2D_kernel(const float* __restrict__ in,
float* __restrict__ out,
int rows, int cols) {
// TODO: compute r, c using blockIdx/threadIdx and blockDim
// TODO: bounds check: if (r >= rows || c >= cols) return;
// TODO: copy one element
}

void copy2D(const float* d_in, float* d_out, int rows, int cols, dim3 block){
// Suggest block = (16,16,1)
dim3 grid((cols + block.x - 1)/block.x,
(rows + block.y - 1)/block.y);
copy2D_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
checkCuda(cudaGetLastError());
checkCuda(cudaDeviceSynchronize());
}

```


**Testing & Tips**

* Prefer `threadIdx.x` to index **columns** to encourage coalesced row‑major accesses.  
* Run: `./build/cs61cuda --task=copy --M=64 --N=128 --verify`.

## **Task 2 — CPU Baseline Matmul (15 pts)**

### **Conceptual Overview**

Implement a correct triple‑loop matrix multiply in row‑major order. This is the correctness oracle for later tasks.

### **Data Flow**

* **Input:** `A ∈ R^{M×K}`, `B ∈ R^{K×N}` (host memory, row‑major).  
* **Processing:** For every `(i,j)`, compute `C[i,j] = Σ_{k=0..K-1} A[i,k] * B[k,j]`.  
* **Output:** `C ∈ R^{M×N}` (host memory).

### **Your Task**

Fill `mm_cpu(...)` in `src/cpu_baseline.cpp`.

**Starter (skeleton) — `src/cpu_baseline.cpp`**

```c
#include <cstddef>
void mm_cpu(const float* A, const float* B, float* C,
int M, int N, int K){
// TODO: triple nested loops over i (rows of A), j (cols of B), k (shared dim)
// Use row-major: A[i*K + k], B[k*N + j], C[i*N + j]
}

```


**Testing & Tips**

* Use small sizes first (e.g., 8×8×8). GPU results in later tasks are compared against this function with tolerance 1e‑4.

## **Task 3 — CUDA Naive Matmul (30 pts)**

### **Conceptual Overview**

Parallelize Task 2 on the GPU: map each output element `C[i,j]` to a single thread. This exposes **thread‑level parallelism (TLP)**; performance is often limited by memory bandwidth.

### **Data Flow**

* **Input:** `A ∈ R^{M×K}`, `B ∈ R^{K×N}` (device memory), populated from host.  
* **Processing:** Each thread computes one output `C[i,j]` by streaming the `k` dimension.  
* **Output:** `C ∈ R^{M×N}` (device memory), copied back to host by the driver when verifying.

### **Your Task**

Implement `mm_naive_kernel` and the host launcher `mm_naive(...)`.

**Starter (skeleton) — `src/cuda_naive.cu`**

```c
#include "check.cuh"

__global__ void mm_naive_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int M, int N, int K){
// TODO: compute i (row) and j (col) from 2D grid/block
// TODO: bounds guard
// TODO: accumulate over k
}

void mm_naive(const float* dA, const float* dB, float* dC,
int M, int N, int K, dim3 block){
dim3 grid((N + block.x - 1)/block.x,
(M + block.y - 1)/block.y);
mm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
checkCuda(cudaGetLastError());
checkCuda(cudaDeviceSynchronize());
}

```


**Notes on memory access**

* Threads in a warp vary `j` at fixed `k`, so `B[k*N + j]` is coalesced. Accesses to `A[i*K + k]` replicate a single element per thread (cache helps).

**Run** `./build/cs61cuda --task=naive --M=512 --N=512 --K=512 --verify`

## **Task 4 — CUDA SIMD Matmul (Vectorized, No Shared Memory) (30 pts)**

### **Conceptual Overview**

Augment TLP with simple **data‑level parallelism (DLP)**: each thread computes a short contiguous vector of outputs in a row using vector loads/stores (e.g., `float4`). No shared memory yet.

### **Data Flow**

* **Input:** `A ∈ R^{M×K}`, `B ∈ R^{K×N}` (device).  
* **Processing:** A thread at `(i, j0_group)` produces `V` outputs `C[i, j0..j0+V-1]`. Inner loop streams over `k`, reading one scalar `A[i,k]` and a vector of `V` neighbors from row `B[k, :]`.  
* **Output:** `C ∈ R^{M×N}` (device).

### **Your Task**

Implement `mm_simd_kernel<V>()` and its launcher. Default `V=4`; handle tails where `N % V ≠ 0`.

**Starter (skeleton) — `src/cuda_simd.cu`**

```c
#include "check.cuh"

template<int V>
__global__ void mm_simd_kernel(const float* __restrict__ A,
const float* __restrict__ B,
float* __restrict__ C,
int M, int N, int K){
// TODO: compute i (row) and j0 (first column of this thread's vector)
// TODO: maintain acc[V] and handle aligned vs tail paths
}

void mm_simd(const float* dA, const float* dB, float* dC,
int M, int N, int K, int vec, dim3 block){
// TODO: call specialized kernel for vec==4, else fall back
}

```


**Run** `./build/cs61cuda --task=simd --M=1024 --N=1024 --K=1024 --vec=4 --verify`

**Discussion prompts**

* Why does vectorizing along columns improve coalescing for loads from `B` and stores to `C`?  
* What changes would shared‑memory tiling introduce (Task 5 EC)?

## **Task 5 — (Optional) Performance Engineering (15 EC)**

Make it **faster**. Ideas:

* **Shared‑memory tiling** (classic 16×16 or 32×32 tiles).  
* **Register tiling**: each thread computes a small `r×c` tile.  
* **Loop unrolling & `#pragma unroll`** for `k`.  
* **Occupancy tuning**: vary `blockDim` to trade registers vs parallelism.  
* **Mixed precision**: keep FP32 accumulate but try `__half` inputs (only if you also keep a FP32 correctness path for grading).  
* **Software prefetch**: read the next `k` slice early.

We’ll publish a lightweight leaderboard (GFLOP/s). Please write a short **README-perf.md** describing what you tried and why it helped (or didn’t)

**Command‑Line Interface (Driver)**

```c
./cs61cuda --task={copy|cpu|naive|simd}
--M=1024 --N=1024 --K=1024
--block=16 --vec=4 --repeat=10 --verify

```

* `--task=copy` ignores `K`.  
* `--vec` controls SIMD width in Task 4; grader uses 4\.  
* `--verify` runs `mm_cpu` then compares results (L2 relative error ≤ 1e-4).

## **Correctness & Floating‑Point Notes**

* We compare with a small absolute+relative tolerance (`1e-4`).  
* Random inputs in `[-1, 1]` with fixed seed.  
* Your GPU kernels must **not** read/write out of bounds; failing `cuda-memcheck` is an automatic zero for that test.

---

## **Debugging Checklist**

* After each kernel launch:

```c
cudaDeviceSynchronize();
checkCuda(cudaGetLastError());

```

* Use `printf` inside kernels **sparingly** on tiny sizes.  
* Try `cuda-memcheck` for invalid addresses / race conditions.  
* Start small (e.g., `M=N=K=8`) then scale.

## **Style & Submission**

* Clear variable names (`i,j,k`, `M,N,K`).  
* Comments: what the mapping is, what each thread computes, any assumptions.  
* No undefined behavior (no out‑of‑bounds pointer math, no aliasing shenanigans).  
* Submit your edited `.cpp/.cu` files and **README.md** answering the reflection prompts below.

### **Reflection (graded in Task 5 rubric even if you skip perf EC)**

1. Where is your naive kernel memory‑bound? Which array dominates traffic and why?  
2. Why does vectorizing along columns improve coalescing? What’s the trade‑off?  
3. What would shared‑memory tiling change about the access pattern?

---

## **Reference: Shapes & Indexing**

Row‑major:

```c
A: MxK → A[i*K + k]
B: KxN → B[k*N + j]
C: MxN → C[i*N + j]

```

Grid/block formulas used throughout:

```c
int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

```

## **Rubric Details**

**Task 1 (10)**

* (4) correct 2D indexing \+ bounds  
* (3) coalesced mapping (x→cols)  
* (3) passes tests, tidy style

**Task 2 (15)**

* (10) correct triple‑loop for arbitrary M,N,K  
* (5) clear comments & no UB

**Task 3 (30)**

* (10) correct one‑element/thread mapping  
* (10) correct launch geometry & bounds  
* (5) reasonable performance (scaled timing)  
* (5) comments explaining memory pattern

**Task 4 (30)**

* (12) correct SIMD (V outputs/thread), handles tails  
* (8) proper vector loads/stores when aligned  
* (5) coalesced writes & justified mapping  
* (5) performance better than naive on large sizes

**Task 5 EC (15)**

* (10) measurable speedup over Task 4  
* (5) README‑perf.md analysis

---

## **Appendix A: Minimal Host Launchers (provided in skeleton)**

```c
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

```

**Appendix B: Starter main.cpp (driver skeleton)**

```c
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include "check.cuh"

void mm_cpu(const float*, const float*, float*, int,int,int);
void copy2D(const float*, float*, int,int, dim3);
void mm_naive(const float*, const float*, float*, int,int,int, dim3);
void mm_simd(const float*, const float*, float*, int,int,int, int, dim3);

int main(int argc, char** argv){
int M=256,N=256,K=256, repeat=5, vec=4; dim3 block(16,16);
enum {COPY, CPU, NAIVE, SIMD} task = NAIVE;
for (int i=1;i<argc;i++){
if (!strncmp(argv[i],"--M=",4)) M=atoi(argv[i]+4);
else if (!strncmp(argv[i],"--N=",4)) N=atoi(argv[i]+4);
else if (!strncmp(argv[i],"--K=",4)) K=atoi(argv[i]+4);
else if (!strncmp(argv[i],"--block=",8)) { int b=atoi(argv[i]+8); block=dim3(b,b); }
else if (!strncmp(argv[i],"--vec=",6)) vec=atoi(argv[i]+6);
else if (!strcmp(argv[i],"--task=copy")) task=COPY;
else if (!strcmp(argv[i],"--task=cpu")) task=CPU;
else if (!strcmp(argv[i],"--task=naive")) task=NAIVE;
else if (!strcmp(argv[i],"--task=simd")) task=SIMD;
}

std::mt19937 gen(42); std::uniform_real_distribution<float> d(-1,1);
std::vector<float> hA(M*K), hB(K*N), hC(M*N), hRef(M*N);
for (auto& x: hA) x=d(gen); for (auto& x: hB) x=d(gen);

float *dA=nullptr,*dB=nullptr,*dC=nullptr;
checkCuda(cudaMalloc(&dA, sizeof(float)*M*K));
checkCuda(cudaMalloc(&dB, sizeof(float)*K*N));
checkCuda(cudaMalloc(&dC, sizeof(float)*M*N));
checkCuda(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
checkCuda(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

if (task==COPY){
std::vector<float> tmp(M*N);
float *dIn=nullptr,*dOut=nullptr;
checkCuda(cudaMalloc(&dIn, sizeof(float)*M*N));
checkCuda(cudaMalloc(&dOut, sizeof(float)*M*N));
checkCuda(cudaMemcpy(dIn, hA.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));
copy2D(dIn, dOut, M, N, block);
checkCuda(cudaMemcpy(tmp.data(), dOut, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
// verify copy
} else if (task==CPU){
mm_cpu(hA.data(), hB.data(), hRef.data(), M,N,K);
} else if (task==NAIVE){
mm_naive(dA,dB,dC,M,N,K,block);
checkCuda(cudaMemcpy(hC.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
mm_cpu(hA.data(), hB.data(), hRef.data(), M,N,K);
} else if (task==SIMD){
mm_simd(dA,dB,dC,M,N,K,vec,block);
checkCuda(cudaMemcpy(hC.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
mm_cpu(hA.data(), hB.data(), hRef.data(), M,N,K);
}
// compare hC vs hRef if computed
return 0;
}

```

**Appendix C: check.cuh (error helpers)**

```c
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#define checkCuda(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ \
fprintf(stderr,"CUDA error %s at %s:%d
", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(1);} }while(0)

```

[image1]: <data:image/png;base64,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>