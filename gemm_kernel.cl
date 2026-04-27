/*GEMM Kernel for the OpenCL backend.

Problem encountered: double precision is not supported 
on my laptop's intel uhd graphics so we use float for fair comparison.*/

__kernel void gemmKernel(__global float* A, __global float* B, __global float* C, int N){
    int j = get_global_id(0);
    int i = get_global_id(1);
    float sum = 0.0f;
    for(int k = 0; k < N; k++){
        sum+= A[i*N+k]*B[k*N+j];
    }   
    C[i*N+j] = sum

}
