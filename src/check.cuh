//ERROR HELPERS
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#define checkCuda(call) do{ cudaError_t err=(call); if(err!=cudaSuccess){ \
fprintf(stderr,"CUDA error %s at %s:%d
", cudaGetErrorString(err), __FILE__, __LINE__); \
exit(1);} }while(0)