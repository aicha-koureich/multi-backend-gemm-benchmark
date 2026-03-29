import matplotlib.pyplot as plt

N = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

cpu_gflops   = [0.542256, 0.550067, 0.525797, 0.463896, 0.412675, 0.0819666, float('nan'), float('nan'), float('nan')]  
cuda_gflops  = [0.00176468, 0.0149282, 0.116127, 0.916598, 6.97508, 39.0911, 111.698, 147.143, 155.942]  
opencl_gflops = []

fig, ax1 = plt.subplots(1, figsize=(7, 5))
fig.suptitle('GEMM Benchmark: GTX 1660 Super vs Ryzen 5 3600', fontsize=13)

# GFLOPS plot
ax1.plot(N, cpu_gflops,  marker='o', label='CPU (Ryzen 5 3600)')
ax1.plot(N, cuda_gflops, marker='o', label='CUDA (GTX 1660 Super)')
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