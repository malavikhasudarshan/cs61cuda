# CS61Cuda

Tasks: lead up to performing the equivalent of CS61C Project 4, but broken down into manageable chunks.

"Your goal is to speed up a simple, common numeric kernel: matrix multiplication. You’ve seen this before in Project 2!”

We’ll be benchmarking three versions in this assignment.
CPU baseline - single thread 
CUDA naive - one thread per output 
CUDA vectorized - one thread per output, with vectorized operations 

// setup … hive machines … git … //

Task 
Goal
Learning Goals
1
[DIR]
Task 1 - Welcome to 61Cuda!

Learn four basic tenets of CUDA syntax: 
Thread indexing: mapping elements to CUDA threads
Tiling on shared memory 
Synchronization (maybe not in scope…)
Coalesce memory access

Through “copy matrix” kernel
How to Cuda!

Teach the basic mechanics of CUDA with a lot of direction, scaffolding and references for syntax. 


2
[DIR]
Task 2 - Baseline CPU Matmul 

Compute C = A × B (row-major; A: m×k, B: k×n, C: m×n) in float
Stable triple loop order: for i (rows of A) for j (cols of B) acc = Σ_k A[i,k]*B[k,j]
No SIMD/threads; correctness first.



Basic kernel to start

Directed scaffolding portion - help ensure students have a good understanding of indices, indexing and matmul process before jumping into CUDA. 

Considering adding a “draw a 3 x 3 example of matmul” as a suggestion, to ensure students have visual intuition. 
3
[DIR]
Task 3 - CUDA Naive Matmul 

Now that students can index threads and copy matrices, it’s time to parallelize a real computation.
In this task, write your first CUDA version of matrix multiplication; still naive, but now massively parallel! Every CUDA thread will compute exactly one output element of C.


TLP - Thread Level Parallelism
Intuition in mapping 2D output space to 2D CUDA grid and blocks
See speedup from parallelization 
Inquire about potential for memory-focused optimization (memory bound, not compute bound here)
4
[DIR]
Task 4 - CUDA SIMD Matmul

Introduce Data-Level Parallelism on the GPU by speeding up the naive CUDA matmul without shared memory. Add to naive TLP-level kernel to add SIMD. (I think we should do vectorized loads, matches lab the best). 

Goal: show that each thread can also process multiple elements per instruction.
DLP - Data Level Parallelism
Teach intuition for vectorization; build off exposure in lab to repeat similar syntax + logichere
Show importance of considering memory access pattern to implement more effective vectorization + tiling
5
[OPT]
Task 5 - Performance Engineering!

Go wild. :) 
Leaderboard. Allow students free reign to attempt any more complex optimizations they deem fit for bragging rights. Not required for full credit at all. (or maybe a very small boost beyond full.)



