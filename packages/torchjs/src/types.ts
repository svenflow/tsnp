/**
 * Type definitions for torchjs
 */

/**
 * Tensor interface - compatible with numpyjs NDArray
 */
export interface Tensor {
  readonly data: Float32Array | Float64Array;
  readonly shape: number[];
}

/**
 * Backend interface for torchjs ops
 */
export interface TorchBackend {
  readonly name: string;

  // Activation functions
  relu(input: Tensor): Tensor;
  relu6(input: Tensor): Tensor;
  leakyRelu(input: Tensor, negativeSlope: number): Tensor;
  gelu(input: Tensor): Tensor;
  sigmoid(input: Tensor): Tensor;
  softmax(input: Tensor, dim: number): Tensor;
  prelu(input: Tensor, weight: Tensor): Tensor;

  // Convolution ops
  conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor;

  depthwiseConv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor;

  // Pooling ops
  maxPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor;

  avgPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor;

  globalAvgPool2d(input: Tensor): Tensor;

  // Normalization
  batchNorm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    runningMean: Tensor,
    runningVar: Tensor,
    eps: number,
  ): Tensor;

  // Image processing
  resizeBilinear(input: Tensor, outputSize: [number, number]): Tensor;

  // Tensor manipulation
  pad(
    input: Tensor,
    padding: number[],
    mode?: 'constant' | 'reflect' | 'replicate',
    value?: number,
  ): Tensor;
  permute(input: Tensor, dims: number[]): Tensor;
  reshape(input: Tensor, newShape: number[]): Tensor;
  squeeze(input: Tensor, dim?: number): Tensor;
  add(a: Tensor, b: Tensor): Tensor;
  addRelu?(a: Tensor, b: Tensor): Tensor;  // Fused add+relu for perf
  upsampleBilinear(input: Tensor, scaleFactor: number): Tensor;
  mulScalar(input: Tensor, scalar: number): Tensor;
}

/**
 * Conv2d parameters following PyTorch conventions
 */
export interface Conv2dParams {
  stride?: [number, number];
  padding?: [number, number];
  dilation?: [number, number];
  groups?: number;
}

/**
 * Pool2d parameters
 */
export interface Pool2dParams {
  kernelSize: [number, number];
  stride?: [number, number];
  padding?: [number, number];
}

/**
 * BatchNorm parameters
 */
export interface BatchNormParams {
  eps?: number;
  momentum?: number;
}
