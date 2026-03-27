/**
 * WASM Backend for torchjs
 *
 * Wraps the Rust/WASM crate (torchjs-wasm) with TypeScript interface.
 * Falls back to pure JS for missing ops.
 */

import type { Tensor, TorchBackend } from './types';

// WASM module interface (matches torchjs-wasm exports)
interface TorchjsWasm {
  // Activations
  relu_f32(input: Float32Array): Float32Array;
  relu6_f32(input: Float32Array): Float32Array;
  leaky_relu_f32(input: Float32Array, alpha: number): Float32Array;
  gelu_f32(input: Float32Array): Float32Array;
  sigmoid_f32(input: Float32Array): Float32Array;
  softmax_f32(input: Float32Array, axis_size: number): Float32Array;

  // Convolutions
  conv2d_f32(
    input: Float32Array,
    kernel: Float32Array,
    n: number, c_in: number, h_in: number, w_in: number,
    c_out: number, k_h: number, k_w: number,
    stride_h: number, stride_w: number,
    pad_h: number, pad_w: number,
  ): Float32Array;

  depthwise_conv2d_f32(
    input: Float32Array,
    kernel: Float32Array,
    n: number, c_in: number, h_in: number, w_in: number,
    depth_mult: number, k_h: number, k_w: number,
    stride_h: number, stride_w: number,
    pad_h: number, pad_w: number,
  ): Float32Array;

  // Pooling
  max_pool2d_f32(
    input: Float32Array,
    n: number, c: number, h_in: number, w_in: number,
    kernel_h: number, kernel_w: number,
    stride_h: number, stride_w: number,
    pad_h: number, pad_w: number,
  ): Float32Array;

  avg_pool2d_f32(
    input: Float32Array,
    n: number, c: number, h_in: number, w_in: number,
    kernel_h: number, kernel_w: number,
    stride_h: number, stride_w: number,
    pad_h: number, pad_w: number,
  ): Float32Array;

  // Batch norm
  batch_norm_f32(
    input: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    running_mean: Float32Array,
    running_var: Float32Array,
    n: number, c: number, h: number, w: number,
    eps: number,
  ): Float32Array;
}

// Simple tensor wrapper for WASM backend
class WasmTensor implements Tensor {
  readonly data: Float32Array;
  readonly shape: number[];

  constructor(data: Float32Array, shape: number[]) {
    this.data = data;
    this.shape = [...shape];
  }
}

export class WasmBackend implements TorchBackend {
  readonly name = 'wasm';
  private wasm: TorchjsWasm | null = null;

  async init(wasmModule: TorchjsWasm): Promise<void> {
    this.wasm = wasmModule;
  }

  private getWasm(): TorchjsWasm {
    if (!this.wasm) {
      throw new Error('WASM backend not initialized. Call init() first.');
    }
    return this.wasm;
  }

  private asF32(tensor: Tensor): Float32Array {
    return tensor.data instanceof Float32Array
      ? tensor.data
      : new Float32Array(tensor.data);
  }

  // ============ Activations ============

  relu(input: Tensor): Tensor {
    const result = this.getWasm().relu_f32(this.asF32(input));
    return new WasmTensor(result, input.shape);
  }

  relu6(input: Tensor): Tensor {
    const result = this.getWasm().relu6_f32(this.asF32(input));
    return new WasmTensor(result, input.shape);
  }

  leakyRelu(input: Tensor, negativeSlope: number): Tensor {
    const result = this.getWasm().leaky_relu_f32(this.asF32(input), negativeSlope);
    return new WasmTensor(result, input.shape);
  }

  gelu(input: Tensor): Tensor {
    const result = this.getWasm().gelu_f32(this.asF32(input));
    return new WasmTensor(result, input.shape);
  }

  sigmoid(input: Tensor): Tensor {
    const result = this.getWasm().sigmoid_f32(this.asF32(input));
    return new WasmTensor(result, input.shape);
  }

  softmax(input: Tensor, dim: number): Tensor {
    const shape = input.shape;
    const axisSize = shape[dim < 0 ? shape.length + dim : dim];
    const result = this.getWasm().softmax_f32(this.asF32(input), axisSize);
    return new WasmTensor(result, shape);
  }

  prelu(input: Tensor, weight: Tensor): Tensor {
    // PReLU not in WASM yet - fall back to JS
    const data = this.asF32(input);
    const w = this.asF32(weight);
    const result = new Float32Array(data.length);

    // Simple per-element PReLU
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const alpha = w[i % w.length];
      result[i] = x > 0 ? x : alpha * x;
    }
    return new WasmTensor(result, input.shape);
  }

  // ============ Convolutions ============

  conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const [n, cIn, hIn, wIn] = input.shape;
    const [cOut, , kH, kW] = weight.shape;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;

    const result = this.getWasm().conv2d_f32(
      this.asF32(input),
      this.asF32(weight),
      n, cIn, hIn, wIn,
      cOut, kH, kW,
      stride[0], stride[1],
      padding[0], padding[1],
    );

    // Add bias if provided
    if (bias) {
      const biasData = this.asF32(bias);
      const outHW = hOut * wOut;
      for (let b = 0; b < n; b++) {
        for (let c = 0; c < cOut; c++) {
          const base = b * cOut * outHW + c * outHW;
          for (let i = 0; i < outHW; i++) {
            result[base + i] += biasData[c];
          }
        }
      }
    }

    return new WasmTensor(result, [n, cOut, hOut, wOut]);
  }

  depthwiseConv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const [n, channels, hIn, wIn] = input.shape;
    const [, , kH, kW] = weight.shape;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;

    const result = this.getWasm().depthwise_conv2d_f32(
      this.asF32(input),
      this.asF32(weight),
      n, channels, hIn, wIn,
      1, kH, kW,
      stride[0], stride[1],
      padding[0], padding[1],
    );

    // Add bias if provided
    if (bias) {
      const biasData = this.asF32(bias);
      const outHW = hOut * wOut;
      for (let b = 0; b < n; b++) {
        for (let c = 0; c < channels; c++) {
          const base = b * channels * outHW + c * outHW;
          for (let i = 0; i < outHW; i++) {
            result[base + i] += biasData[c];
          }
        }
      }
    }

    return new WasmTensor(result, [n, channels, hOut, wOut]);
  }

  // ============ Pooling ============

  maxPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const [n, c, hIn, wIn] = input.shape;
    const hOut = Math.floor((hIn + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1;

    const result = this.getWasm().max_pool2d_f32(
      this.asF32(input),
      n, c, hIn, wIn,
      kernelSize[0], kernelSize[1],
      stride[0], stride[1],
      padding[0], padding[1],
    );

    return new WasmTensor(result, [n, c, hOut, wOut]);
  }

  avgPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const [n, c, hIn, wIn] = input.shape;
    const hOut = Math.floor((hIn + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1;

    const result = this.getWasm().avg_pool2d_f32(
      this.asF32(input),
      n, c, hIn, wIn,
      kernelSize[0], kernelSize[1],
      stride[0], stride[1],
      padding[0], padding[1],
    );

    return new WasmTensor(result, [n, c, hOut, wOut]);
  }

  globalAvgPool2d(input: Tensor): Tensor {
    // Global avg pool = avg pool with kernel = input spatial dims
    const [, , h, w] = input.shape;
    return this.avgPool2d(input, [h, w], [1, 1], [0, 0]);
  }

  // ============ Normalization ============

  batchNorm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    runningMean: Tensor,
    runningVar: Tensor,
    eps: number,
  ): Tensor {
    const [n, c, h, w] = input.shape;

    const result = this.getWasm().batch_norm_f32(
      this.asF32(input),
      this.asF32(gamma),
      this.asF32(beta),
      this.asF32(runningMean),
      this.asF32(runningVar),
      n, c, h, w,
      eps,
    );

    return new WasmTensor(result, input.shape);
  }

  // ============ Image Processing ============

  resizeBilinear(input: Tensor, outputSize: [number, number]): Tensor {
    // Not in WASM yet - fall back to JS implementation
    const [batch, channels, inH, inW] = input.shape;
    const [outH, outW] = outputSize;
    const data = this.asF32(input);
    const result = new Float32Array(batch * channels * outH * outW);

    const scaleY = inH / outH;
    const scaleX = inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        const inBase = b * channels * inH * inW + c * inH * inW;
        const outBase = b * channels * outH * outW + c * outH * outW;

        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            const srcY = (oy + 0.5) * scaleY - 0.5;
            const srcX = (ox + 0.5) * scaleX - 0.5;

            const y0 = Math.max(0, Math.floor(srcY));
            const x0 = Math.max(0, Math.floor(srcX));
            const y1 = Math.min(y0 + 1, inH - 1);
            const x1 = Math.min(x0 + 1, inW - 1);

            const ly = srcY - y0;
            const lx = srcX - x0;
            const hy = 1 - ly;
            const hx = 1 - lx;

            const v00 = data[inBase + y0 * inW + x0];
            const v01 = data[inBase + y0 * inW + x1];
            const v10 = data[inBase + y1 * inW + x0];
            const v11 = data[inBase + y1 * inW + x1];

            const val = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
            result[outBase + oy * outW + ox] = val;
          }
        }
      }
    }

    return new WasmTensor(result, [batch, channels, outH, outW]);
  }

  // ============ Tensor Manipulation (JS fallbacks) ============

  pad(
    input: Tensor,
    padding: number[],
    mode: 'constant' | 'reflect' | 'replicate' = 'constant',
    value = 0,
  ): Tensor {
    const data = this.asF32(input);
    const shape = input.shape;

    if (padding.length === 4 && shape.length >= 2) {
      const [padL, padR, padT, padB] = padding;
      const [n, c, h, w] = shape.length === 4 ? shape : [1, 1, ...shape.slice(-2)];
      const newH = h + padT + padB;
      const newW = w + padL + padR;
      const result = new Float32Array(n * c * newH * newW);
      result.fill(value);

      for (let b = 0; b < n; b++) {
        for (let ch = 0; ch < c; ch++) {
          for (let ih = 0; ih < h; ih++) {
            for (let iw = 0; iw < w; iw++) {
              const oh = ih + padT;
              const ow = iw + padL;
              const srcIdx = b * c * h * w + ch * h * w + ih * w + iw;
              const dstIdx = b * c * newH * newW + ch * newH * newW + oh * newW + ow;
              result[dstIdx] = data[srcIdx];
            }
          }
        }
      }

      const newShape = shape.length === 4 ? [n, c, newH, newW] : [newH, newW];
      return new WasmTensor(result, newShape);
    }

    throw new Error(`Unsupported padding length: ${padding.length}`);
  }

  permute(input: Tensor, dims: number[]): Tensor {
    const data = this.asF32(input);
    const oldShape = input.shape;
    const newShape = dims.map(d => oldShape[d]);
    const result = new Float32Array(data.length);

    const oldStrides = new Array(oldShape.length);
    oldStrides[oldShape.length - 1] = 1;
    for (let i = oldShape.length - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1] * oldShape[i + 1];
    }

    const newStrides = new Array(newShape.length);
    newStrides[newShape.length - 1] = 1;
    for (let i = newShape.length - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    for (let i = 0; i < data.length; i++) {
      let remaining = i;
      const newCoords = new Array(newShape.length);
      for (let d = 0; d < newShape.length; d++) {
        newCoords[d] = Math.floor(remaining / newStrides[d]);
        remaining %= newStrides[d];
      }

      let oldIdx = 0;
      for (let d = 0; d < dims.length; d++) {
        oldIdx += newCoords[d] * oldStrides[dims[d]];
      }

      result[i] = data[oldIdx];
    }

    return new WasmTensor(result, newShape);
  }

  reshape(input: Tensor, newShape: number[]): Tensor {
    const data = this.asF32(input);
    const inferIdx = newShape.indexOf(-1);
    if (inferIdx >= 0) {
      const knownSize = newShape.filter(d => d !== -1).reduce((a, b) => a * b, 1);
      const inferredSize = data.length / knownSize;
      newShape = [...newShape];
      newShape[inferIdx] = inferredSize;
    }
    return new WasmTensor(data, newShape);
  }

  squeeze(input: Tensor, dim?: number): Tensor {
    const data = this.asF32(input);
    let newShape: number[];
    if (dim !== undefined) {
      if (input.shape[dim] === 1) {
        newShape = [...input.shape.slice(0, dim), ...input.shape.slice(dim + 1)];
      } else {
        newShape = [...input.shape];
      }
    } else {
      newShape = input.shape.filter(d => d !== 1);
    }
    return new WasmTensor(data, newShape);
  }

  add(a: Tensor, b: Tensor): Tensor {
    const dataA = this.asF32(a);
    const dataB = this.asF32(b);
    const result = new Float32Array(dataA.length);
    for (let i = 0; i < dataA.length; i++) {
      result[i] = dataA[i] + dataB[i];
    }
    return new WasmTensor(result, [...a.shape]);
  }

  upsampleBilinear(input: Tensor, scaleFactor: number): Tensor {
    const [n, c, h, w] = input.shape;
    const outH = Math.floor(h * scaleFactor);
    const outW = Math.floor(w * scaleFactor);
    return this.resizeBilinear(input, [outH, outW]);
  }

  mulScalar(input: Tensor, scalar: number): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = data[i] * scalar;
    }
    return new WasmTensor(result, [...input.shape]);
  }
}

// ============ Factory ============

export async function createWasmBackend(wasmPath?: string): Promise<WasmBackend> {
  const backend = new WasmBackend();
  // Load WASM module - caller should provide initialized module
  // This is a placeholder - actual loading depends on build setup
  return backend;
}
