import torch
import time

N = 2048 
iterations = 5

# Total FLOPs
flops_per_matmul = 2 * N**3
total_flops = iterations * flops_per_matmul

#original tensors
torch_input_tensor_a = torch.rand(N, N, dtype=torch.float16)
torch_input_tensor_b = torch.rand(N, N, dtype=torch.float16)

# measure time
start = time.time()
for i in range(iterations):
    matmul_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
end = time.time()

time_elapsed = end - start
gflops_per_sec = (total_flops / time_elapsed) / 1e9

# print(f"Total theoretical FLOPs: {total_flops: } ({total_flops/1e9:.2f} GFLOPS)")
print(f"GFLOPS per sec: {gflops_per_sec}")
print(f"Total GFLOPs: {total_flops/1e9:.2f}")
print(f"Elapsed time: {time_elapsed:.3f}")
print(f"Percent utilized: {gflops_per_sec/614*100:.2f}%")

# Theoretical Max FLOPS = Cores * Clock Speed (Hz) * FLOPs per cycler per core
# 
# GBP 5,  Intel(R) Core(TM) Ultra 7 258V
# Cores: 8
# Max Clock Speed (Hz): 4800 MHz = 4800000000 Hz
# 
# FLOPs per cycle per core: 
# Vector width: 256 bits
# float32 = 32 bits -> 8 float32 values per vector
# With FMA, each vector op = 2 FLOPs per float
# 8*2 = 16 FLOPs per cycle per core
#
# GBP Theoretical FLOPs = 8 * 4800000000 * 16 = around 614 GFLOPs 