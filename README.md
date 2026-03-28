# multi-backend-gemm-benchmark
This is a personal project for me to understand the GPU software stack from the ground up, from CPU baselines to multi-backend GPU compute. And to get hands-on with low-level GPU programming across different APIs.

It implements GEMM (General Matrix Multiplication) across multiple backends and profiles performance across matrix sizes. I've developed it on my GTX 1660 Super, with additional runs on Google Colab (T4) and an Intel UHD Graphics iGPU.

## Architecture
```
. ├──main.cpp 
├── cpu_backend.cpp       # Naive triple-loop GEMM
├── cl_backend.cpp        # OpenCL host + kernel loading
│   └── kernel_gemm.cl    # OpenCL C kernel
└── cuda_backend.cu       # CUDA kernel + host code
```
## Backends 
| Backend | Hardware | Status |
|--------|----------|--------|
| CPU | x86 | Done |
| OpenCL | Intel UHD Graphics, GTX 1660s| Done |
| CUDA | GTX 1660s, NVIDIA T4 | Done |
| ROCm | — | soon |
| SYCL | — | soon |
 
 OpenCL on NVIDIA hardware may require manual driver setup. Tested on Intel UHD Graphics. Full cross-backend comparison on a single machine in progress.
## Command
```
# NVIDIA GPU (CUDA + CPU)
nvcc main.cpp cpu_backend.cpp cuda_backend.cu -o benchmarkCuda
```
```
# No NVIDIA GPU
g++ main.cpp cpu_backend.cpp cl_backend.cpp -o benchmarkOpencl -lOpenCL
```
## Results 
Soon

## Next steps
- CUDA shared memory tiling
- CNN convolution kernel
- SYCL backend (via Intel oneAPI)
- ROCm backend (AMD)
