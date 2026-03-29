/*GEMM Kernel for the OpenCL backend.

Problem encountered: double precision is not supported 
on my laptop's intel uhd graphics so we use float for fair comparison.*/

__kernel void gemmKernel(__global float* A, __global float* B, __global float* C, int N){
int j = get_global_id(0);
int i = get_global_id(1);
int k=0;
C[i*N+j]= 0.0;
while(k<N){
    C[i*N+j]+= A[i*N+k]*B[k*N+j];
    k+=1;
}
}