/**
 * torchjs - PyTorch-style neural network operations for JavaScript
 *
 * Provides torch.nn.functional equivalent ops with dual backends:
 * - WASM: Rust-compiled WebAssembly (works everywhere)
 * - WebGPU: GPU compute shaders (browsers with WebGPU support)
 *
 * Usage:
 *   import { nn, createWebGPUBackend } from 'torchjs';
 *
 *   // Use convenience functions (auto-detect backend)
 *   const output = nn.relu(input);
 *   const conv = nn.conv2d(input, weight, bias, [1, 1], [1, 1]);
 *
 *   // Or create specific backend
 *   const backend = await createWebGPUBackend();
 *   const output = backend.relu(input);
 */

export * from './types';
export { WebGPUBackend, WebGPUTensor, createWebGPUBackend, hasF16Support } from './webgpu-backend';
export { WasmBackend, createWasmBackend } from './wasm-backend';

// NN operations and backend management
export { nn, setBackend, getBackend, beginBatch, endBatch, withBatch } from './nn';

import type { Tensor } from './types';

// ============ Tensor Creation Helpers ============

/**
 * Create a tensor from array data
 */
export function tensor(data: number[] | Float32Array, shape: number[]): Tensor {
  const f32 = data instanceof Float32Array ? data : new Float32Array(data);
  return { data: f32, shape: [...shape] };
}

/**
 * Create a zeros tensor
 */
export function zeros(shape: number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return { data: new Float32Array(size), shape: [...shape] };
}

/**
 * Create a ones tensor
 */
export function ones(shape: number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  data.fill(1);
  return { data, shape: [...shape] };
}

/**
 * Create a random tensor (uniform [0, 1))
 */
export function rand(shape: number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.random();
  }
  return { data, shape: [...shape] };
}

/**
 * Create a random normal tensor
 */
export function randn(shape: number[]): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    data[i] = r * Math.cos(theta);
    if (i + 1 < size) {
      data[i + 1] = r * Math.sin(theta);
    }
  }
  return { data, shape: [...shape] };
}
