/**
 * Neural network operations and backend management
 *
 * This module contains the backend system and nn.* convenience API.
 * Separated from index.ts to avoid circular dependencies with model files.
 */

import type { Tensor, TorchBackend } from './types';

// ============ Pure JS Implementation (fallback) ============

class JsBackend implements TorchBackend {
  readonly name = 'js';

  private asF32(tensor: Tensor): Float32Array {
    return tensor.data instanceof Float32Array
      ? tensor.data
      : new Float32Array(tensor.data);
  }

  relu(input: Tensor): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = Math.max(0, data[i]);
    }
    return { data: result, shape: input.shape };
  }

  relu6(input: Tensor): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = Math.min(6, Math.max(0, data[i]));
    }
    return { data: result, shape: input.shape };
  }

  leakyRelu(input: Tensor, negativeSlope: number): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      result[i] = x > 0 ? x : negativeSlope * x;
    }
    return { data: result, shape: input.shape };
  }

  gelu(input: Tensor): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    const sqrt2pi = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const inner = sqrt2pi * (x + 0.044715 * x * x * x);
      result[i] = x * 0.5 * (1 + Math.tanh(inner));
    }
    return { data: result, shape: input.shape };
  }

  sigmoid(input: Tensor): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = 1 / (1 + Math.exp(-data[i]));
    }
    return { data: result, shape: input.shape };
  }

  softmax(input: Tensor, dim: number): Tensor {
    const data = this.asF32(input);
    const shape = input.shape;
    const axisSize = shape[dim < 0 ? shape.length + dim : dim];
    const numBatches = data.length / axisSize;
    const result = new Float32Array(data.length);

    for (let b = 0; b < numBatches; b++) {
      const start = b * axisSize;
      let max = -Infinity;
      for (let i = 0; i < axisSize; i++) {
        max = Math.max(max, data[start + i]);
      }
      let sum = 0;
      for (let i = 0; i < axisSize; i++) {
        const exp = Math.exp(data[start + i] - max);
        result[start + i] = exp;
        sum += exp;
      }
      for (let i = 0; i < axisSize; i++) {
        result[start + i] /= sum;
      }
    }
    return { data: result, shape };
  }

  prelu(input: Tensor, weight: Tensor): Tensor {
    const data = this.asF32(input);
    const w = this.asF32(weight);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      result[i] = x > 0 ? x : w[i % w.length] * x;
    }
    return { data: result, shape: input.shape };
  }

  conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const inputData = this.asF32(input);
    const weightData = this.asF32(weight);
    const [n, cIn, hIn, wIn] = input.shape;
    const [cOut, , kH, kW] = weight.shape;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;
    const result = new Float32Array(n * cOut * hOut * wOut);

    for (let batch = 0; batch < n; batch++) {
      for (let co = 0; co < cOut; co++) {
        for (let oh = 0; oh < hOut; oh++) {
          for (let ow = 0; ow < wOut; ow++) {
            let sum = 0;
            for (let ci = 0; ci < cIn; ci++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh * stride[0] + kh - padding[0];
                  const iw = ow * stride[1] + kw - padding[1];
                  if (ih >= 0 && ih < hIn && iw >= 0 && iw < wIn) {
                    const inIdx = batch * cIn * hIn * wIn + ci * hIn * wIn + ih * wIn + iw;
                    const wIdx = co * cIn * kH * kW + ci * kH * kW + kh * kW + kw;
                    sum += inputData[inIdx] * weightData[wIdx];
                  }
                }
              }
            }
            if (bias) {
              sum += this.asF32(bias)[co];
            }
            const outIdx = batch * cOut * hOut * wOut + co * hOut * wOut + oh * wOut + ow;
            result[outIdx] = sum;
          }
        }
      }
    }

    return { data: result, shape: [n, cOut, hOut, wOut] };
  }

  depthwiseConv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const inputData = this.asF32(input);
    const weightData = this.asF32(weight);
    const [n, c, hIn, wIn] = input.shape;
    const [, , kH, kW] = weight.shape;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;
    const result = new Float32Array(n * c * hOut * wOut);

    for (let batch = 0; batch < n; batch++) {
      for (let ch = 0; ch < c; ch++) {
        for (let oh = 0; oh < hOut; oh++) {
          for (let ow = 0; ow < wOut; ow++) {
            let sum = 0;
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * stride[0] + kh - padding[0];
                const iw = ow * stride[1] + kw - padding[1];
                if (ih >= 0 && ih < hIn && iw >= 0 && iw < wIn) {
                  const inIdx = batch * c * hIn * wIn + ch * hIn * wIn + ih * wIn + iw;
                  const wIdx = ch * kH * kW + kh * kW + kw;
                  sum += inputData[inIdx] * weightData[wIdx];
                }
              }
            }
            if (bias) {
              sum += this.asF32(bias)[ch];
            }
            const outIdx = batch * c * hOut * wOut + ch * hOut * wOut + oh * wOut + ow;
            result[outIdx] = sum;
          }
        }
      }
    }

    return { data: result, shape: [n, c, hOut, wOut] };
  }

  maxPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const data = this.asF32(input);
    const [n, c, hIn, wIn] = input.shape;
    const [kH, kW] = kernelSize;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;
    const result = new Float32Array(n * c * hOut * wOut);

    for (let batch = 0; batch < n; batch++) {
      for (let ch = 0; ch < c; ch++) {
        for (let oh = 0; oh < hOut; oh++) {
          for (let ow = 0; ow < wOut; ow++) {
            let max = -Infinity;
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * stride[0] + kh - padding[0];
                const iw = ow * stride[1] + kw - padding[1];
                if (ih >= 0 && ih < hIn && iw >= 0 && iw < wIn) {
                  const idx = batch * c * hIn * wIn + ch * hIn * wIn + ih * wIn + iw;
                  max = Math.max(max, data[idx]);
                }
              }
            }
            const outIdx = batch * c * hOut * wOut + ch * hOut * wOut + oh * wOut + ow;
            result[outIdx] = max === -Infinity ? 0 : max;
          }
        }
      }
    }

    return { data: result, shape: [n, c, hOut, wOut] };
  }

  avgPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const data = this.asF32(input);
    const [n, c, hIn, wIn] = input.shape;
    const [kH, kW] = kernelSize;

    const hOut = Math.floor((hIn + 2 * padding[0] - kH) / stride[0]) + 1;
    const wOut = Math.floor((wIn + 2 * padding[1] - kW) / stride[1]) + 1;
    const result = new Float32Array(n * c * hOut * wOut);

    for (let batch = 0; batch < n; batch++) {
      for (let ch = 0; ch < c; ch++) {
        for (let oh = 0; oh < hOut; oh++) {
          for (let ow = 0; ow < wOut; ow++) {
            let sum = 0;
            let count = 0;
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * stride[0] + kh - padding[0];
                const iw = ow * stride[1] + kw - padding[1];
                if (ih >= 0 && ih < hIn && iw >= 0 && iw < wIn) {
                  const idx = batch * c * hIn * wIn + ch * hIn * wIn + ih * wIn + iw;
                  sum += data[idx];
                  count++;
                }
              }
            }
            const outIdx = batch * c * hOut * wOut + ch * hOut * wOut + oh * wOut + ow;
            result[outIdx] = count > 0 ? sum / count : 0;
          }
        }
      }
    }

    return { data: result, shape: [n, c, hOut, wOut] };
  }

  globalAvgPool2d(input: Tensor): Tensor {
    const [, , h, w] = input.shape;
    return this.avgPool2d(input, [h, w], [1, 1], [0, 0]);
  }

  batchNorm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    runningMean: Tensor,
    runningVar: Tensor,
    eps: number,
  ): Tensor {
    const data = this.asF32(input);
    const gammaData = this.asF32(gamma);
    const betaData = this.asF32(beta);
    const meanData = this.asF32(runningMean);
    const varData = this.asF32(runningVar);
    const [n, c, h, w] = input.shape;
    const result = new Float32Array(data.length);

    const hw = h * w;
    for (let batch = 0; batch < n; batch++) {
      for (let ch = 0; ch < c; ch++) {
        const invStd = 1 / Math.sqrt(varData[ch] + eps);
        const scale = gammaData[ch] * invStd;
        const shift = betaData[ch] - meanData[ch] * scale;
        for (let i = 0; i < hw; i++) {
          const idx = batch * c * hw + ch * hw + i;
          result[idx] = data[idx] * scale + shift;
        }
      }
    }

    return { data: result, shape: input.shape };
  }

  resizeBilinear(input: Tensor, outputSize: [number, number]): Tensor {
    const data = this.asF32(input);
    const [batch, channels, inH, inW] = input.shape;
    const [outH, outW] = outputSize;
    const result = new Float32Array(batch * channels * outH * outW);

    const scaleY = inH / outH;
    const scaleX = inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        const inBase = b * channels * inH * inW + c * inH * inW;
        const outBase = b * channels * outH * outW + c * outH * outW;

        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            // Compute source coordinates (align_corners=False)
            const srcY = (oy + 0.5) * scaleY - 0.5;
            const srcX = (ox + 0.5) * scaleX - 0.5;

            // Clamp source coordinates to valid range
            const clampedY = Math.max(0, Math.min(srcY, inH - 1));
            const clampedX = Math.max(0, Math.min(srcX, inW - 1));

            const y0 = Math.floor(clampedY);
            const x0 = Math.floor(clampedX);
            const y1 = Math.min(y0 + 1, inH - 1);
            const x1 = Math.min(x0 + 1, inW - 1);

            // Compute interpolation weights from clamped coordinates
            const ly = clampedY - y0;
            const lx = clampedX - x0;
            const hy = 1 - ly;
            const hx = 1 - lx;

            const v00 = data[inBase + y0 * inW + x0];
            const v01 = data[inBase + y0 * inW + x1];
            const v10 = data[inBase + y1 * inW + x0];
            const v11 = data[inBase + y1 * inW + x1];

            result[outBase + oy * outW + ox] = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
          }
        }
      }
    }

    return { data: result, shape: [batch, channels, outH, outW] };
  }

  /**
   * F.pad equivalent - supports asymmetric padding
   * @param padding [left, right, top, bottom, ...] (PyTorch order)
   */
  pad(input: Tensor, padding: number[], mode: 'constant' | 'reflect' | 'replicate' = 'constant', value = 0): Tensor {
    const data = this.asF32(input);
    const shape = input.shape;

    // Handle 2D padding (last 2 dimensions): [left, right, top, bottom]
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
      return { data: result, shape: newShape };
    }

    // Handle channel padding [left, right, top, bottom, front, back]
    if (padding.length === 6 && shape.length >= 3) {
      const [padL, padR, padT, padB, padF, padBack] = padding;
      const [n, c, h, w] = shape.length === 4 ? shape : [1, ...shape.slice(-3)];
      const newC = c + padF + padBack;
      const newH = h + padT + padB;
      const newW = w + padL + padR;
      const result = new Float32Array(n * newC * newH * newW);
      result.fill(value);

      for (let b = 0; b < n; b++) {
        for (let ch = 0; ch < c; ch++) {
          for (let ih = 0; ih < h; ih++) {
            for (let iw = 0; iw < w; iw++) {
              const oc = ch + padF;
              const oh = ih + padT;
              const ow = iw + padL;
              const srcIdx = b * c * h * w + ch * h * w + ih * w + iw;
              const dstIdx = b * newC * newH * newW + oc * newH * newW + oh * newW + ow;
              result[dstIdx] = data[srcIdx];
            }
          }
        }
      }

      return { data: result, shape: [n, newC, newH, newW] };
    }

    throw new Error(`Unsupported padding length: ${padding.length} for shape: ${shape}`);
  }

  /**
   * Permute tensor dimensions
   */
  permute(input: Tensor, dims: number[]): Tensor {
    const data = this.asF32(input);
    const oldShape = input.shape;
    const newShape = dims.map(d => oldShape[d]);
    const result = new Float32Array(data.length);

    // Compute strides for old shape
    const oldStrides = new Array(oldShape.length);
    oldStrides[oldShape.length - 1] = 1;
    for (let i = oldShape.length - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1] * oldShape[i + 1];
    }

    // Compute strides for new shape
    const newStrides = new Array(newShape.length);
    newStrides[newShape.length - 1] = 1;
    for (let i = newShape.length - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    const totalSize = data.length;
    for (let i = 0; i < totalSize; i++) {
      // Convert flat index to multi-dim index in new shape
      let remaining = i;
      const newCoords = new Array(newShape.length);
      for (let d = 0; d < newShape.length; d++) {
        newCoords[d] = Math.floor(remaining / newStrides[d]);
        remaining %= newStrides[d];
      }

      // Map back to old coordinates
      let oldIdx = 0;
      for (let d = 0; d < dims.length; d++) {
        oldIdx += newCoords[d] * oldStrides[dims[d]];
      }

      result[i] = data[oldIdx];
    }

    return { data: result, shape: newShape };
  }

  /**
   * Reshape tensor
   */
  reshape(input: Tensor, newShape: number[]): Tensor {
    const data = this.asF32(input);
    // Handle -1 dimension
    const inferIdx = newShape.indexOf(-1);
    if (inferIdx >= 0) {
      const knownSize = newShape.filter(d => d !== -1).reduce((a, b) => a * b, 1);
      const inferredSize = data.length / knownSize;
      newShape = [...newShape];
      newShape[inferIdx] = inferredSize;
    }
    return { data, shape: newShape };
  }

  /**
   * Squeeze - remove dimensions of size 1
   */
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
    return { data, shape: newShape };
  }

  /**
   * Element-wise addition
   */
  add(a: Tensor, b: Tensor): Tensor {
    const dataA = this.asF32(a);
    const dataB = this.asF32(b);
    const result = new Float32Array(dataA.length);
    for (let i = 0; i < dataA.length; i++) {
      result[i] = dataA[i] + dataB[i];
    }
    return { data: result, shape: [...a.shape] };
  }

  /**
   * Upsample with scale factor (bilinear interpolation)
   * Equivalent to nn.Upsample(scale_factor=N, mode='bilinear', align_corners=False)
   */
  upsampleBilinear(input: Tensor, scaleFactor: number): Tensor {
    const [n, c, h, w] = input.shape;
    const outH = Math.floor(h * scaleFactor);
    const outW = Math.floor(w * scaleFactor);
    return this.resizeBilinear(input, [outH, outW]);
  }

  /**
   * Multiply tensor by scalar
   */
  mulScalar(input: Tensor, scalar: number): Tensor {
    const data = this.asF32(input);
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      result[i] = data[i] * scalar;
    }
    return { data: result, shape: [...input.shape] };
  }
}

// ============ Global Backend & Convenience API ============

const jsBackend = new JsBackend();
let currentBackend: TorchBackend = jsBackend;

/**
 * Set the global backend for all nn.* operations
 * @param backend - 'js' for pure JS, or a TorchBackend instance (e.g. WebGPUBackend)
 */
export function setBackend(backend: 'js' | TorchBackend): void {
  if (backend === 'js') {
    currentBackend = jsBackend;
  } else {
    currentBackend = backend;
  }
}

/**
 * Get the current global backend
 */
export function getBackend(): TorchBackend {
  return currentBackend;
}

/**
 * Begin batching GPU commands (if backend supports it)
 * All subsequent ops will be collected into a single command buffer
 * Call endBatch() when done
 */
export function beginBatch(): void {
  if ('beginBatch' in currentBackend && typeof currentBackend.beginBatch === 'function') {
    currentBackend.beginBatch();
  }
}

/**
 * End batching and submit all collected GPU commands at once
 */
export function endBatch(): void {
  if ('flush' in currentBackend && typeof currentBackend.flush === 'function') {
    currentBackend.flush();
  }
}

/**
 * Execute a function with command batching
 * Automatically calls beginBatch() before and endBatch() after
 *
 * @param fn Function to execute with batching
 * @returns Result of fn()
 *
 * @example
 * const output = withBatch(() => {
 *   return handLandmarks(input, weights);
 * });
 */
export function withBatch<T>(fn: () => T): T {
  beginBatch();
  try {
    return fn();
  } finally {
    endBatch();
  }
}

/**
 * torch.nn.functional equivalent API
 *
 * Uses global backend (default: pure JS).
 * To switch to WebGPU:
 *   const backend = await createWebGPUBackend();
 *   setBackend(backend);
 */
export const nn = {
  relu: (input: Tensor) => currentBackend.relu(input),
  relu6: (input: Tensor) => currentBackend.relu6(input),
  leakyRelu: (input: Tensor, negativeSlope = 0.01) => currentBackend.leakyRelu(input, negativeSlope),
  gelu: (input: Tensor) => currentBackend.gelu(input),
  sigmoid: (input: Tensor) => currentBackend.sigmoid(input),
  softmax: (input: Tensor, dim = -1) => currentBackend.softmax(input, dim),
  prelu: (input: Tensor, weight: Tensor) => currentBackend.prelu(input, weight),

  conv2d: (
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null = null,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
  ) => currentBackend.conv2d(input, weight, bias, stride, padding),

  depthwiseConv2d: (
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null = null,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
  ) => currentBackend.depthwiseConv2d(input, weight, bias, stride, padding),

  maxPool2d: (
    input: Tensor,
    kernelSize: [number, number],
    stride?: [number, number],
    padding: [number, number] = [0, 0],
  ) => currentBackend.maxPool2d(input, kernelSize, stride ?? kernelSize, padding),

  avgPool2d: (
    input: Tensor,
    kernelSize: [number, number],
    stride?: [number, number],
    padding: [number, number] = [0, 0],
  ) => currentBackend.avgPool2d(input, kernelSize, stride ?? kernelSize, padding),

  globalAvgPool2d: (input: Tensor) => currentBackend.globalAvgPool2d(input),

  batchNorm: (
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    runningMean: Tensor,
    runningVar: Tensor,
    eps = 1e-5,
  ) => currentBackend.batchNorm(input, gamma, beta, runningMean, runningVar, eps),

  resizeBilinear: (input: Tensor, outputSize: [number, number]) =>
    currentBackend.resizeBilinear(input, outputSize),

  pad: (
    input: Tensor,
    padding: number[],
    mode: 'constant' | 'reflect' | 'replicate' = 'constant',
    value = 0,
  ) => currentBackend.pad(input, padding, mode, value),

  permute: (input: Tensor, dims: number[]) => currentBackend.permute(input, dims),

  reshape: (input: Tensor, newShape: number[]) => currentBackend.reshape(input, newShape),

  squeeze: (input: Tensor, dim?: number) => currentBackend.squeeze(input, dim),

  add: (a: Tensor, b: Tensor) => currentBackend.add(a, b),

  // Fused add + relu - uses optimized shader on WebGPU, falls back to add+relu on JS
  addRelu: (a: Tensor, b: Tensor) => {
    if (currentBackend.addRelu) {
      return currentBackend.addRelu(a, b);
    }
    // Fallback: separate add + relu
    const sum = currentBackend.add(a, b);
    return currentBackend.relu(sum);
  },

  upsampleBilinear: (input: Tensor, scaleFactor: number) =>
    currentBackend.upsampleBilinear(input, scaleFactor),

  mulScalar: (input: Tensor, scalar: number) => currentBackend.mulScalar(input, scalar),

  // ============ Fused Operations for Performance ============
  // These combine multiple ops into single GPU dispatches

  /**
   * Fused Conv2d + Bias + Activation
   * Falls back to separate ops on backends that don't support it
   */
  conv2dFused: (
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null = null,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
    activation: 'none' | 'relu' | 'relu6' = 'none',
  ) => {
    if ('conv2dFused' in currentBackend && typeof currentBackend.conv2dFused === 'function') {
      return (currentBackend as ExtendedBackend).conv2dFused(input, weight, bias, stride, padding, activation);
    }
    // Fallback: separate ops
    let out = currentBackend.conv2d(input, weight, bias, stride, padding);
    if (activation === 'relu') out = currentBackend.relu(out);
    else if (activation === 'relu6') out = currentBackend.relu6(out);
    return out;
  },

  /**
   * Fused Depthwise Conv2d + Bias + Activation
   */
  depthwiseConv2dFused: (
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null = null,
    stride: [number, number] = [1, 1],
    padding: [number, number] = [0, 0],
    activation: 'none' | 'relu' | 'relu6' = 'none',
  ) => {
    if ('depthwiseConv2dFused' in currentBackend && typeof currentBackend.depthwiseConv2dFused === 'function') {
      return (currentBackend as ExtendedBackend).depthwiseConv2dFused(input, weight, bias, stride, padding, activation);
    }
    // Fallback
    let out = currentBackend.depthwiseConv2d(input, weight, bias, stride, padding);
    if (activation === 'relu') out = currentBackend.relu(out);
    else if (activation === 'relu6') out = currentBackend.relu6(out);
    return out;
  },

  /**
   * Fused Depthwise Conv + Pointwise Conv + Skip + ReLU
   * This is the full ResModule pattern - single kernel dispatch instead of 4
   */
  depthwisePointwiseSkipRelu: (
    input: Tensor,
    dwWeight: Tensor,
    dwBias: Tensor | null,
    pwWeight: Tensor,
    pwBias: Tensor | null,
    skip: Tensor | null,
    dwPadding: [number, number],
  ) => {
    if ('depthwisePointwiseSkipRelu' in currentBackend && typeof currentBackend.depthwisePointwiseSkipRelu === 'function') {
      return (currentBackend as ExtendedBackend).depthwisePointwiseSkipRelu(input, dwWeight, dwBias, pwWeight, pwBias, skip, dwPadding);
    }
    // Fallback: separate ops
    const dw = currentBackend.depthwiseConv2d(input, dwWeight, dwBias, [1, 1], dwPadding);
    const pw = currentBackend.conv2d(dw, pwWeight, pwBias, [1, 1], [0, 0]);
    if (skip) {
      const sum = currentBackend.add(pw, skip);
      return currentBackend.relu(sum);
    }
    return currentBackend.relu(pw);
  },
};

// Extended backend interface for fused operations
interface ExtendedBackend extends TorchBackend {
  conv2dFused: (input: Tensor, weight: Tensor, bias: Tensor | null, stride: [number, number], padding: [number, number], activation: 'none' | 'relu' | 'relu6') => Tensor;
  depthwiseConv2dFused: (input: Tensor, weight: Tensor, bias: Tensor | null, stride: [number, number], padding: [number, number], activation: 'none' | 'relu' | 'relu6') => Tensor;
  depthwisePointwiseSkipRelu: (input: Tensor, dwWeight: Tensor, dwBias: Tensor | null, pwWeight: Tensor, pwBias: Tensor | null, skip: Tensor | null, dwPadding: [number, number]) => Tensor;
}
