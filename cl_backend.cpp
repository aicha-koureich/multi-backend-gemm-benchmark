#include <iostream>
#include <CL/opencl.hpp>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <fstream>
using std::cout;

void GEMM_OPENCL(float* A, float* B, float* C, int N){
/*Getting my computer platform*/
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);
cl::Platform myplatform = platforms[0];
cout<<"Machine's Platform: "<< myplatform.getInfo<CL_PLATFORM_NAME>()<<'\n';
/*Getting my device*/
std::vector<cl::Device> devices;
myplatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
cl::Device mydevice = devices[0];
cout<<"Device used: "<<mydevice.getInfo<CL_DEVICE_NAME>()<<'\n';
/*Setting up the context*/
cl::Context context(mydevice);
cl::CommandQueue commandqueue(context,mydevice);
/*Allocating buffers to the matrices*/
cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, N*N*sizeof(float));
cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, N*N*sizeof(float));
cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, N*N*sizeof(float));
/*Copying the matrices from the CPU to the GPU*/
commandqueue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, N*N*sizeof(float), A);
commandqueue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, N*N*sizeof(float), B);
/*loading the kernel file as a text*/
std::ifstream file("kernel_gemm.cl");
std::string kernel_string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
/*Compiling*/
cl::Program program(context, kernel_string);
program.build({mydevice});
/*Calling the kernel*/
cl::Kernel kernel(program,"gemmKernel");
kernel.setArg(0, buffer_A);
kernel.setArg(1, buffer_B);
kernel.setArg(2, buffer_C);
kernel.setArg(3, N);
commandqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N,N), cl::NullRange);
/* Send the results to the GPU*/
commandqueue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, N*N*sizeof(float), C);
}