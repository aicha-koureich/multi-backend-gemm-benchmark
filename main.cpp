#include <iostream>
#include "cpu_backend.hpp"
#include "cl_backend.hpp"
#include "cuda_backend.hpp"
#include <chrono>
#include <omp.h>

using std::cout;
using std::cin;
using namespace std::chrono;
int main(int argc, char** argv){
    if (argc<3){ //We need N and the mode, either OpencCL or CUDA for
        cout<<"N and/or mode required, <mode: 0=cuda, 1=OpenCL>\n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    int mode = std::stoi(argv[2]);

    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];
    for(int i=0; i<N; i++){
        for(int j=0; j< N; j++){
            A[i*N+j] = rand();
            B[i*N+j] = rand();
        }
    
    } 
    time_point<steady_clock> start, end;
    duration<double> duration_cpu, duration_cl, duration_cuda;
    if (mode == 0){
    /*CUDA*/   
        start= steady_clock::now();
        GEMM_CUDA(A,B,C,N);
        end= steady_clock::now();
        duration_cuda = end-start;
        double GFLOPS0 = (2.0*N*N*N)/(duration_cuda.count()*1e9);
        cout<< "GPU w/cuda Compute time: " << duration_cuda.count()<<" s\n";
        cout<<"GPU w/cuda GFLOPS: "<<GFLOPS0<<" gflops\n"; 
        }
    else if (mode == 1){
        start = steady_clock::now();
        GEMM_OPENCL(A,B,C,N);
        end = steady_clock::now();
        duration_cl = end - start;
        double GFLOPS1 = (2.0*N*N*N)/(duration_cl.count()*1e9);
        cout<< "\n";
        cout<< "GPU w/opencl Compute time: " << duration_cl.count()<<" s\n";
        cout<<"GPU w/opencl GFLOPS: "<<GFLOPS1<<" gflops\n";  
    }
    /*CPU*/
    start= steady_clock::now();
    GEMM_CPU(A,B,C,N);
    end= steady_clock::now();
    duration_cpu = end-start;
    double GFLOPS2 = (2.0*N*N*N)/(duration_cpu.count()*1e9);
    cout<< "\n";
    cout<< "CPU Compute time: " << duration_cpu.count()<<" s\n";
    cout<<"CPU GFLOPS: "<<GFLOPS2<<" gflops\n";
    cout<< '\n';
    
    delete[] A; 
    delete[] B; 
    delete[] C;
    return 0;
    
}