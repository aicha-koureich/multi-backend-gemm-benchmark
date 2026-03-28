#include <iostream>

void GEMM_CPU(float* A, float* B, float *C, int N){
     for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            C[i*N+j]= 0.0;
            int k=0;
            while(k<N){
            C[i*N + j] += A[i*N + k]*B[k*N +j];
            k+=1;
            }
        }        
    }
}