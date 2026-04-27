#include <cuda_runtime.h>
#include "cuda_backend.hpp"

/*Computation*/
__global__ void kernel(float* d_A, float* d_B,float* d_C, int N){
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;
if( i < N && j < N){
    float sum= 0.0f;
    for(int k=0; k < N; k++){
       sum+= d_A[i*N+k]*d_B[k*N+j];
}
d_C[i*N+j] = sum
}
}
void GEMM_CUDA(float* A, float* B, float* C, int N){
/*Device*/
cudaSetDevice(0);
/*Memory allocation*/
float* d_A;
float* d_B;
float* d_C;
cudaMalloc(&d_A, N*N*sizeof(float));
cudaMalloc(&d_B, N*N*sizeof(float));
cudaMalloc(&d_C, N*N*sizeof(float));
/*Copying from CPU to GPU*/
cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
/*Launching Kernel*/
dim3 threadsPerBlock(16, 16);
dim3 numBlocks((N+15)/16, (N+15)/16);
kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();
/*Send the results to the CPU*/
cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
/*Free the memory*/
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
}
