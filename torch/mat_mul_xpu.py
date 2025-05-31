# intel_gpu_flops_test.py

import time
import torch


def main():
    # --- 1) XPU check ---
    if not torch.xpu.is_available():
        raise RuntimeError("Intel GPU (xpu) is not available. Check your driver installation.")
    device = torch.device("xpu")
    print(f"Using device: {device}")

    # --- 2) FLOPS test parameters ---
    N = 2048
    iterations = 5

    # For a dense N×N matmul: ≃2*N^3 FLOPs each
    flops_per_matmul = 2 * N**3
    total_flops = iterations * flops_per_matmul

    # --- 3) Prepare half-precision tensors on XPU ---
    a = torch.rand(N, N, dtype=torch.float16, device=device)
    b = torch.rand(N, N, dtype=torch.float16, device=device)

    # Warm-up
    with torch.no_grad():
        _ = a @ b
        torch.xpu.synchronize()

    # --- 4) Timed matmuls ---
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = a @ b
            torch.xpu.synchronize()
    end = time.time()

    elapsed = end - start
    achieved_gflops = (total_flops / elapsed) / 1e9

    # --- 5) Print results ---
    print(f"GFLOPS achieved:    {achieved_gflops:.2f} GFLOPS/s")
    print(f"Total GFLOPs run:   {total_flops/1e9:.2f} GFLOPs")
    print(f"Elapsed time:       {elapsed:.3f} s")

    # Theoretical max for your CPU = 8 cores × 4.8 GHz × 16 FLOPs/cycle = 614.4 GFLOPS
    theoretical = 8 * 4.8e9 * 16 / 1e9
    util = achieved_gflops / theoretical * 100
    print(f"Theoretical max:    {theoretical:.1f} GFLOPS")
    print(f"Percent utilized:   {util:.2f}%")

if __name__ == "__main__":
    main()
