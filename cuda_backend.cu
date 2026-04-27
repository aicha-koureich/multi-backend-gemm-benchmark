#include <cuda_runtime.h>
#include "cuda_backend.hpp"

/*Computation*/
/*Update: added shared memory tiling*/
__global__ void kernel(float* d_A, float* d_B,float* d_C, int N){
    __shared__ float s_A[16][16];       
    __shared__ float s_B[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;
    if( i < N && j < N){
        float sum= 0.0f;
        for(int t=0; t < N/16; t++){
            float val_A = d_A[i*N +t*16+ tx];
            float val_B = d_B[(ty + 16*t)*N + j];
            s_A[ty][tx] = val_A;
            s_B[ty][tx] = val_B;
            /*The syncthreads can stay inside the for loop because all N are multiples of 16*/
            __syncthreads(); 
            for(int k=0; k<16; k++){
                sum+= s_A[ty][k]*s_B[k][tx];
            }
           __syncthreads();
        }
        d_C[i*N+j] = sum;
    }
}
void GEMM_CUDA(float* A, float* B, float* C, int N){
    cudaSetDevice(0);
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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
/*__global__ void kernel(float* d_A, float* d_B,float* d_C, int N){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if( i < N && j < N){
        float sum= 0.0f;
        for(int k=0; k < N; k++){
           sum+= d_A[i*N+k]*d_B[k*N+j];
        }
    d_C[i*N+j] = sum;
    }
}*/