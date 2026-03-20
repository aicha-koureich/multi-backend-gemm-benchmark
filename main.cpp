#include <iostream>
#include "gemm_cpu.hpp"
#include "cl_backend.hpp"
#include <chrono>

using std::cout;
using std::cin;
using namespace std::chrono;

int main(){
    int N;
    cout<< "Enter the size of your matrices: ";
    cin >> N;
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
    duration<double> duration_cpu, duration_cl;
    start = steady_clock::now();
    GEMM_CPU(A,B,C,N);
    end = steady_clock::now();
    duration_cpu = end - start;
    start= steady_clock::now();
    GEMM_OPENCL(A,B,C,N);
    end= steady_clock::now();
    duration_cl = end-start;

    double GFLOPS0 = (2.0*N*N*N)/(duration_cpu.count()*1e9);
    double GFLOPS1 = (2.0*N*N*N)/(duration_cl.count()*1e9);
    cout<< "-----------RESULTS----------\n";
    cout<< "\n";
    cout<< "CPU Compute time: " << duration_cpu.count()<<" s\n";
    cout<<"CPU GFLOPS: "<<GFLOPS0<<" gflops/s\n";
    cout<< "\n";
    cout<< "GPU w/opencl Compute time: " << duration_cl.count()<<" s\n";
    cout<<"GPU w/opencl GFLOPS: "<<GFLOPS1<<" gflops/s\n";
    delete[] A; 
    delete[] B; 
    delete[] C;
    return 0;
    
}