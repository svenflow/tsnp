# numpyjs

**NumPy for JavaScript** — High-performance NumPy-like library for TypeScript, with pure JS and WebGPU backends.

Part of the [pydatajs](https://github.com/svenflow/pydatajs) ecosystem.

[![CI](https://github.com/svenflow/pydatajs/actions/workflows/ci.yml/badge.svg)](https://github.com/svenflow/pydatajs/actions/workflows/ci.yml)

## Features

- **NumPy-compatible API** - Familiar interface for Python developers
- **Three backends** - Pure JS (CPU), WebGPU (GPU-accelerated), and WASM SIMD (near-native CPU)
- **TypeScript-first** - Full type definitions included
- **Browser & Node.js** - JS backend works everywhere; WebGPU in supported browsers

## Installation

```bash
npm install numpyjs
```

## Quick Start

```typescript
import { createBackend } from 'numpyjs';

// Create a JS (CPU) backend - works everywhere
const np = await createBackend('js');

// Or create a WebGPU (GPU) backend - requires WebGPU support
// const np = await createBackend('webgpu');

// Array creation
const a = np.zeros([3, 4]); // 3x4 array of zeros
const b = np.ones([2, 3]); // 2x3 array of ones
const c = np.arange(0, 10, 1); // [0, 1, 2, ..., 9]
const d = np.linspace(0, 1, 5); // [0, 0.25, 0.5, 0.75, 1]
const I = np.eye(3); // 3x3 identity matrix

// Math operations
const angles = np.linspace(0, Math.PI, 100);
const sines = np.sin(angles);
const cosines = np.cos(angles);

// Linear algebra
const A = np.array([1, 2, 3, 4], [2, 2]);
const B = np.array([5, 6, 7, 8], [2, 2]);
const C = np.matmul(A, B); // Matrix multiplication
```

## Backends

| Backend  | Environment                               | Performance                                       |
| -------- | ----------------------------------------- | ------------------------------------------------- |
| `js`     | Node.js, Bun, all browsers                | CPU-bound, good for small-medium arrays           |
| `webgpu` | Chrome 113+, Edge 113+, Firefox (nightly) | GPU-accelerated, best for large arrays and matmul |
| `wasm`   | All modern browsers, Node.js, Bun         | SIMD-optimized, near-native CPU performance       |

## Benchmarks

Matrix multiplication (matmul) performance in GFLOPS — higher is better.

### Apple M4 (Mac)

| Size | numpyjs WebGPU | numpyjs WASM | tfjs WebGPU | tfjs WASM |
| ---- | -------------- | ------------ | ----------- | --------- |
| 256  | 66             | 68           | 61          | 247       |
| 512  | 446            | 199          | 334         | 352       |
| 1024 | 1,580          | 419          | 1,136       | 382       |
| 2048 | 2,818          | 501          | 2,264       | 408       |
| 4096 | **3,272**      | 625          | 2,264       | 407       |

**Peak: 3,272 GFLOPS** — 1.45× faster than tfjs WebGPU at 4096.

### iPhone 17 Pro Max (A18 Pro)

| Size | numpyjs WebGPU | tfjs WebGPU |
| ---- | -------------- | ----------- |
| 512  | **312**        | 198         |
| 1024 | **610**        | 480         |
| 2048 | **781**        | 650         |

**Peak: 781 GFLOPS** — beats tfjs at every size tested.

### How it works

- **WebGPU**: Custom BCACHE WGSL shader (4×4 output per thread) optimized for Apple GPUs. GPU-resident computation with no CPU readback.
- **WASM**: Rust-compiled SIMD f32 with futex thread pool for parallel matmul. Runtime feature detection with relaxed-SIMD fast path and simd128 compat fallback.

> **Run it yourself:** Open [bench.html](../../bench.html) in a WebGPU-capable browser to benchmark on your own device.

## License

MIT
