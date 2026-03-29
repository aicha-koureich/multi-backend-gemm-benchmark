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
| CPU | Ryzen 5 3600, x86 | Done |
| OpenCL | Intel UHD Graphics| Done |
| CUDA | GTX 1660s, NVIDIA T4 | Done |
| ROCm | — | soon |
| SYCL | — | soon |
 
 OpenCL on NVIDIA hardware does not work for me currently. Tested on Intel UHD Graphics. Full cross-backend comparison on a single machine under investigation.
## Command
```
# NVIDIA GPU (CUDA + CPU)
nvcc main.cpp cpu_backend.cpp cuda_backend.cu -o benchmarkCuda
```
```
# No NVIDIA GPU (OpenCL + CPU)
g++ main.cpp cpu_backend.cpp cl_backend.cpp -o benchmarkOpencl -lOpenCL
```
## Results 
### CUDA vs CPU — GTX 1660 Super / Ryzen 5 3600 (above N = 2048 CPU GFLOPS are too insignificant)
Graph
![GFLOPS vs N](results.png)
> Peak naive kernel efficiency (155 GFLOPS) ~2.8% of GTX 1660 Super theoretical GFLOPS (5.5 TFLOPS)
Speedup
| N | CUDA/CPU |
|--------|-----|
| 64 |~0.003x  |
| 128 |~0.027x | 
| 256 | ~0.22x | 
| 512 | ~2x | 
| 1024 | ~17x | 
| 2048 | ~477x | 

 
## Next steps
- CUDA shared memory tiling
- SYCL backend (via Intel oneAPI)
- ROCm backend (AMD)
