# numpyjs

**NumPy for JavaScript** — High-performance NumPy-like library for TypeScript, powered by Rust/WASM and WebGPU.

Part of the [pydatajs](https://github.com/svenflow/pydatajs) ecosystem.

[![CI](https://github.com/svenflow/pydatajs/actions/workflows/ci.yml/badge.svg)](https://github.com/svenflow/pydatajs/actions/workflows/ci.yml)

## Features

- **NumPy-compatible API** - Familiar interface for Python developers
- **Rust performance** - Compiled to native code or WebAssembly
- **Pluggable backends** - CPU (ndarray/faer), WASM, WebGPU
- **TypeScript-first** - Full type definitions included
- **Zero dependencies** - Self-contained WASM bundle

## Installation

```bash
npm install numpyjs
```

## Quick Start

```typescript
import { initNumpy, zeros, ones, arange, linspace, eye } from 'numpyjs';
import { sin, cos, exp, log, sqrt } from 'numpyjs';
import { linalg, random } from 'numpyjs';

// Initialize WASM module
await initNumpy();

// Array creation
const a = zeros([3, 4]);           // 3x4 array of zeros
const b = ones([2, 3]);            // 2x3 array of ones
const c = arange(0, 10, 1);        // [0, 1, 2, ..., 9]
const d = linspace(0, 1, 5);       // [0, 0.25, 0.5, 0.75, 1]
const I = eye(3);                  // 3x3 identity matrix

// Math operations
const angles = linspace(0, Math.PI, 100);
const sines = sin(angles);
const cosines = cos(angles);

// Linear algebra
const A = NDArray.fromArray([[1, 2], [3, 4]]);
const B = NDArray.fromArray([[5, 6], [7, 8]]);
const C = linalg.matmul(A, B);     // Matrix multiplication
const Ainv = linalg.inv(A);        // Matrix inverse
```

## License

MIT
