# WASM Numeric Libraries Benchmark

**Date:** February 25, 2026
**Environment:** Headless Chromium (Playwright), WASM backend
**Config:** Array 10,000 | Matrix 100×100 | 100 iterations

## Summary

| Operation | Fastest | Speedup |
|-----------|---------|---------|
| add | native JS | — |
| mul | **rumpy-ts** | tie |
| matmul | **tensorflow.js** | 6.7× over rumpy |
| sum | **rumpy-ts** | 4× over tfjs |
| mean | **rumpy-ts** | 16× over tfjs |
| chained | **tensorflow.js** | 5.3× over rumpy |

## Element-wise Operations (10k elements)

| Operation | rumpy-ts | tensorflow.js | native JS |
|-----------|----------|---------------|-----------|
| **add** | 0.018ms | 0.020ms | **0.016ms** |
| **mul** | **0.015ms** | 0.016ms | — |

### Analysis

- All WASM libraries perform similarly for element-wise ops
- Native JS wins add slightly (no WASM call overhead)
- rumpy-ts and tfjs essentially tied for multiply

## Matrix Operations (100×100)

| Operation | rumpy-ts | tensorflow.js |
|-----------|----------|---------------|
| **matmul** | 0.257ms | **0.038ms** |
| **chained** (matmul+add+mean) | 0.223ms | **0.042ms** |

### Analysis

**TensorFlow.js WASM is 6.7× faster at matmul!**

Likely reasons:
1. **SIMD optimization**: tfjs WASM backend uses explicit SIMD kernels for GEMM
2. **WebAssembly SIMD**: Modern browsers support WASM SIMD, tfjs exploits this
3. **faer (rumpy's backend)**: Optimized for native CPU, not WASM-specific

The chained ops test shows the same pattern - tfjs dominates matrix-heavy workloads in WASM.

## Reductions

| Operation | rumpy-ts | tensorflow.js | native JS |
|-----------|----------|---------------|-----------|
| **sum** | **0.005ms** | 0.020ms | 0.070ms |
| **mean** | **0.001ms** | 0.015ms | — |

### Analysis

- rumpy-ts dominates reductions: **4× faster than tfjs, 14× faster than native**
- WASM overhead hurts native JS here (function call overhead per element)
- rumpy's reduction ops are extremely fast

## Linear Regression Pipeline

| Library | Time |
|---------|------|
| **rumpy-ts** | 0.062ms |
| tensorflow.js | N/A (pinv not in WASM) |

The full closed-form linear regression (1000 samples):
- `w = (X^T X)^-1 X^T y`
- Involves transpose, matmul, matrix inverse, more matmul

rumpy-ts completes in 0.062ms, but we couldn't compare to tfjs because `tf.linalg.pinv` isn't available in the WASM backend.

## jax-js Status

jax-js **initialized but didn't run any operations**.

Likely cause: jax-js requires **WebGPU** for matrix operations, which isn't available in headless Chromium. Need to test in a real browser with WebGPU support.

jax-js is expected to be fastest for:
- Large matrix operations (7000+ GFLOP/s claimed)
- Chained operations (kernel fusion via JIT compilation)
- GPU-accelerated workloads

## Conclusions

### TensorFlow.js WASM Strengths
- **Matrix multiplication**: 6.7× faster than rumpy-ts
- **Optimized GEMM kernels**: SIMD-enabled WASM implementation
- **Mature library**: Years of optimization

### rumpy-ts Strengths
- **Reductions**: 4-16× faster than tfjs
- **Element-wise ops**: Competitive with native JS
- **Full linalg**: Matrix inverse, solve, etc.

### Recommendations

For WASM-based numeric computing:
- **Heavy matmul workloads**: Use TensorFlow.js WASM
- **Reduction-heavy workloads**: Use rumpy-ts
- **Mixed workloads**: Need to profile your specific use case

### Future Improvements for rumpy-ts

1. **SIMD GEMM**: Implement WASM SIMD matmul kernels
2. **Faer WASM**: Check if faer has WASM-specific optimizations
3. **WebGPU backend**: Add WebGPU support for GPU acceleration

## Raw Results

```json
{
  "add": { "rumpy-ts": 0.018, "tensorflow.js": 0.020, "native": 0.016 },
  "mul": { "rumpy-ts": 0.015, "tensorflow.js": 0.016 },
  "matmul": { "rumpy-ts": 0.257, "tensorflow.js": 0.038 },
  "sum": { "rumpy-ts": 0.005, "tensorflow.js": 0.020, "native": 0.070 },
  "mean": { "rumpy-ts": 0.001, "tensorflow.js": 0.015 },
  "chained": { "rumpy-ts": 0.223, "tensorflow.js": 0.042 },
  "linear-regression": { "rumpy-ts": 0.062 }
}
```
