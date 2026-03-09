#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///

import numpy as np
import time
import json

def benchmark_matmul(n, iters=5):
    """Benchmark matrix multiplication for size n x n"""
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)

    # Warmup
    _ = a @ b

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = a @ b
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return min(times)

if __name__ == "__main__":
    sizes = [512, 1024, 2048, 4096, 8192]
    results = {}

    print("NumPy GEMM Benchmark (float32)")
    print(f"NumPy version: {np.__version__}")
    print(f"BLAS info: {np.show_config()}")
    print()
    print("Size | Time (ms)")
    print("-----|----------")

    for n in sizes:
        t = benchmark_matmul(n)
        results[n] = t
        print(f"{n:4d} | {t:8.1f}")

    # Output JSON for easy parsing
    print()
    print("JSON:", json.dumps(results))
