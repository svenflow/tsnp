#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Benchmark 100x100 f32 matrix multiplication with NumPy.
Runs 100 iterations, reports median timing.
"""

import time
import numpy as np

def benchmark_matmul(size: int = 100, iterations: int = 100) -> None:
    # Create random f32 matrices
    np.random.seed(42)  # For reproducibility
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)

    # Warmup run
    _ = np.matmul(a, b)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = np.matmul(a, b)
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    times_ms = [t * 1000 for t in times]
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    mean_ms = sum(times_ms) / len(times_ms)

    # Checksum for verification
    c = np.matmul(a, b)
    checksum = float(c.sum())

    # Calculate theoretical FLOPs
    # matmul: 2 * n^3 FLOPs (n^3 multiplies + n^3 adds for n x n matrices)
    flops = 2 * size * size * size
    gflops = (flops / (median_ms / 1000)) / 1e9

    print(f"NumPy {size}x{size} f32 matmul benchmark")
    print(f"=" * 40)
    print(f"Iterations: {iterations}")
    print(f"Median:     {median_ms:.4f} ms")
    print(f"Min:        {min_ms:.4f} ms")
    print(f"Max:        {max_ms:.4f} ms")
    print(f"Mean:       {mean_ms:.4f} ms")
    print(f"Checksum:   {checksum:.6f}")
    print(f"")
    print(f"Performance:")
    print(f"  FLOPs:    {flops:,} ({flops/1e6:.1f}M)")
    print(f"  GFLOPs/s: {gflops:.1f} (at median timing)")
    print(f"")
    print(f"NumPy config:")
    print(f"  BLAS:     {np.__config__.show() if hasattr(np.__config__, 'show') else 'unknown'}")

if __name__ == "__main__":
    benchmark_matmul()
