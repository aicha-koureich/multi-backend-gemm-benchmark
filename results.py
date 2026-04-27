import matplotlib.pyplot as plt
import subprocess
import sys 

N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

def benchmark(x):
    res_cpu = [] 
    res_cuda = [] 
    res_opencl = [] 
    for n in N:
        print(f"Running N= {n}")
        cmd = subprocess.check_output([f"./benchmark", str(n), str(x)])
        raw_output = cmd.decode("utf-8")
        print(raw_output)
        for line in raw_output.split('\n'):
            if "CPU GFLOPS:" in line :
                value = line.split(":")[1].split()[0]
                res_cpu.append(float(value))
            if x==0:
                if "GPU w/cuda GFLOPS:" in line:
                    value = line.split(":")[1].split()[0]
                    res_cuda.append(float(value))
            if x==1:
                if "GPU w/opencl GFLOPS:" in line:
                    value = line.split(":")[1].split()[0]
                    res_opencl.append(float(value))
    return res_cuda, res_opencl, res_cpu

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 results.py <mode: 0 for CUDA, 1 for OpenCL>")
        sys.exit(1)
    selected_mode = int(sys.argv[1])
    
    # Run the benchmark with that mode
    if selected_mode == 0:
        print("Running CUDA vs CPU Benchmark")
        res_cuda, _, res_cpu = benchmark(0)
    elif selected_mode == 1:
        print("Running OpenCL vs CPU Benchmark")
        _, res_opencl, res_cpu = benchmark(1)
    elif selected_mode == 2:
        print("Running CUDA vs OpenCL Benchmark")
        res_cuda, _, res_cpu = benchmark(0)
        _, res_opencl, _ = benchmark(1)
    else:
        print("Error: Invalid mode. Use 0 for CUDA, 1 for OpenCL, 2 for both.")
        sys.exit(1)

    fig, ax1 = plt.subplots(1, figsize=(10, 6))
    if res_cuda:
        ax1.plot(N, res_cuda, marker='s', label='GPU CUDA', color='green')
    if res_opencl:
        ax1.plot(N, res_opencl, marker='^', label='GPU OpenCL', color='orange')
    if res_cpu:
        ax1.plot(N, res_cpu, marker='o', label='CPU', color='blue')
    title = "Cuda vs OpenCL (Tesla T4)" if selected_mode == 2 else f" { 'CUDA vs CPU (Tesla T4)' if selected_mode == 0 else 'OpenCL vs CPU (Tesla T4)' }"
    fig.suptitle(f'GEMM Benchmark: {title}', fontsize=12)

    ax1.set_xlabel('Matrix size N')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('Performance vs Matrix Size')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Saved to results.png")
