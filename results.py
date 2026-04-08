import matplotlib.pyplot as plt
import subprocess
import sys 

N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

def benchmark(x):
    res_cpu = [] 
    res_gpu = [] 
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
                    res_gpu.append(float(value))
            if x==1:
                if "GPU w/opencl GFLOPS:" in line:
                    value = line.split(":")[1].split()[0]
                    res_gpu.append(float(value))
    return res_gpu, res_cpu

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 results.py <mode: 0 for CUDA, 1 for OpenCL>")
        sys.exit(1)
    selected_mode = int(sys.argv[1])
    
    # Run the benchmark with that mode
    # We ignore the other list returns by using "_"
    if selected_mode == 0:
        print("Running CUDA Benchmark")
        res_gpu, res_cpu = benchmark(0)
        label = "CUDA"
    elif selected_mode == 1:
        print("Running OpenCL Benchmark")
        res_gpu, res_cpu = benchmark(1)
        label = "OpenCL"
    else:
        print("Error: Invalid mode. Use 0 for CUDA or 1 for OpenCL.")
        sys.exit(1)

    fig, ax1 = plt.subplots(1, figsize=(10, 6))
    if label == "CUDA":
        fig.suptitle('GEMM Benchmark: Cuda (GTX 1660 Super) vs CPU (Ryzen 5 3600)', fontsize=10)
        ax1.plot(N, res_gpu,  marker='o', label='CUDA' )
    elif label == "OpenCL":
        fig.suptitle('GEMM Benchmark: OpenCL (Intel UHD Graphics) vs CPU (Intel)', fontsize=10)
        ax1.plot(N, res_gpu,  marker='o', label='OpenCL' )

    ax1.plot(N, res_cpu, marker='o', label='CPU')

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