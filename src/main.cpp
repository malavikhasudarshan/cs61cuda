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
