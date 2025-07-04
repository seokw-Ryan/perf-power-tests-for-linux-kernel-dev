import sys
import torch
import time

# Configuration
N = 2048
iterations = 5
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
# print(torch.__version__)
# print(f"xpu torch is available: {torch.xpu.is_available()}")
# sys.exit()
print(device)

# Total FLOPs for one matmul: 2·N³
flops_per_matmul = 2 * N**3
total_flops = iterations * flops_per_matmul

# Create random tensors on GPU (float16 for better GPU throughput)
a = torch.rand(N, N, dtype=torch.float16, device=device)
b = torch.rand(N, N, dtype=torch.float16, device=device)

# Warm‐up (one matmul) to initialize kernels
_ = a @ b
torch.xpu.synchronize()

# Measure time (GPU)
start = time.time()
for _ in range(iterations):
    c = a @ b
torch.xpu.synchronize()
end = time.time()

elapsed = end - start
gflops_per_sec = (total_flops / elapsed) / 1e9

print(f"GFLOPS/sec (measured): {gflops_per_sec:.2f}")
print(f"Total GFLOPs: {total_flops/1e9:.2f}")
print(f"Elapsed time: {elapsed:.3f} sec")

