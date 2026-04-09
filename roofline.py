import matplotlib.pyplot as plt
import numpy as np

N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
AI_measured = [n/6 for n in N] 
GFLOPS_measured = [0.00197045, 0.0180278, 0.152447, 1.2061, 8.64523, 44.6688, 128.483, 162.929]

P_peak = 5027 # GFLOPS
B_peak = 336  # GB/s
ridge_point = P_peak / B_peak

AI_line = np.logspace(0, 3.5, 500) 
roofline_curve = np.minimum(P_peak, B_peak * AI_line)
fig, ax0 = plt.subplots(figsize=(10, 6))
ax0.plot(AI_line, roofline_curve, color='red', label="Theoretical Limit (Roofline)", linewidth=2)
ax0.scatter(AI_measured, GFLOPS_measured, color='blue', label="Naive CUDA Implementation", zorder=5)
ax0.axvline(ridge_point, color='gray', linestyle='--', alpha=0.5, label=f"Ridge Point ({ridge_point:.2f})")
ax0.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax0.set_ylabel('Performance (GFLOPS)')
ax0.set_title('GTX 1660s Roofline Model: Naive GEMM')
ax0.legend()
ax0.grid(True, which="both", ls="-", alpha=0.2)
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_ylim(1e-3, 1e4) 

plt.tight_layout()
plt.savefig('roofline1660scuda.png', dpi=150)
print("Saved to roofline1660scuda.png")