/**
 * WebGPU Backend for torchjs
 *
 * Provides GPU-accelerated neural network operations using WGSL compute shaders.
 * All data lives in GPUBuffer (f32 format) for efficient GPU execution.
 *
 * Architecture follows numpyjs-webgpu pattern:
 * - Data stays on GPU between ops
 * - Explicit getData() for CPU readback
 * - Shader caching for performance
 */

import type { Tensor, TorchBackend } from './types';

// ============ GPU Device & Shader Cache ============

let gpuDevice: GPUDevice | null = null;
let hasShaderF16 = false; // Track if shader-f16 extension is available
const shaderCache = new Map<string, GPUComputePipeline>();

// ============ Staging Buffer Pool ============
// Reuse MAP_READ buffers to avoid allocation overhead
const stagingBufferPool = new Map<number, GPUBuffer[]>();

function acquireStagingBuffer(device: GPUDevice, size: number): GPUBuffer {
  // Round up to power of 2 for better reuse
  const roundedSize = Math.pow(2, Math.ceil(Math.log2(Math.max(size, 256))));

  const pool = stagingBufferPool.get(roundedSize);
  if (pool && pool.length > 0) {
    return pool.pop()!;
  }

  return device.createBuffer({
    size: roundedSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

function releaseStagingBuffer(size: number, buffer: GPUBuffer): void {
  const roundedSize = Math.pow(2, Math.ceil(Math.log2(Math.max(size, 256))));

  if (!stagingBufferPool.has(roundedSize)) {
    stagingBufferPool.set(roundedSize, []);
  }
  stagingBufferPool.get(roundedSize)!.push(buffer);
}

async function getDevice(): Promise<GPUDevice> {
  if (gpuDevice) return gpuDevice;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU not supported in this environment');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('Failed to get GPU adapter');
  }

  // Check for shader-f16 support
  // NOTE: FP16 with f32 storage was tested - adds conversion overhead that exceeds compute benefit.
  // For FP16 to help, need native f16 storage throughout the pipeline.
  // Keeping disabled until that's implemented.
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) {
    console.log('[WebGPU] shader-f16 available (not enabled - need native f16 storage)');
  } else {
    console.log('[WebGPU] shader-f16 not available');
  }

  gpuDevice = await adapter.requestDevice({
    requiredFeatures,
  });
  return gpuDevice;
}

/**
 * Check if FP16 is available (for external benchmarks)
 */
export function hasF16Support(): boolean {
  return hasShaderF16;
}

// ============ GPU Tensor Implementation ============

/**
 * GPU-resident tensor - data lives in GPUBuffer (f32)
 */
export class WebGPUTensor implements Tensor {
  readonly buffer: GPUBuffer;
  readonly shape: number[];
  readonly device: GPUDevice;
  private _cachedData: Float32Array | null = null;

  constructor(buffer: GPUBuffer, shape: number[], device: GPUDevice) {
    this.buffer = buffer;
    this.shape = [...shape];
    this.device = device;
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get data(): Float32Array {
    if (!this._cachedData) {
      throw new Error('Data not cached. Call await getData() first.');
    }
    return this._cachedData;
  }

  async getData(): Promise<Float32Array> {
    if (this._cachedData) return this._cachedData;

    const n = this.size;
    const byteSize = n * 4;

    // Acquire reusable staging buffer from pool
    const readBuffer = acquireStagingBuffer(this.device, byteSize);

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, byteSize);
    this.device.queue.submit([commandEncoder.finish()]);

    // Try mapSync if available (Chrome with --enable-features=WebGPUMapSyncOnWorkers)
    // This is much faster when available in workers
    const bufferAny = readBuffer as unknown as { mapSync?: (mode: number) => void };
    if (typeof bufferAny.mapSync === 'function') {
      bufferAny.mapSync(GPUMapMode.READ);
    } else {
      await readBuffer.mapAsync(GPUMapMode.READ);
    }

    const f32Data = new Float32Array(readBuffer.getMappedRange().slice(0, byteSize));
    readBuffer.unmap();

    // Return buffer to pool for reuse
    releaseStagingBuffer(byteSize, readBuffer);

    this._cachedData = f32Data;
    return f32Data;
  }

  /**
   * Batch read multiple tensors with a single mapAsync call
   * This is faster than calling getData() on each tensor separately
   * because it uses one staging buffer and one mapAsync
   */
  static async batchGetData(tensors: WebGPUTensor[]): Promise<Float32Array[]> {
    if (tensors.length === 0) return [];
    if (tensors.length === 1) return [await tensors[0].getData()];

    const device = tensors[0].device;

    // Calculate total size and offsets
    const sizes = tensors.map(t => t.size);
    const totalSize = sizes.reduce((a, b) => a + b, 0);
    const totalBytes = totalSize * 4;
    const offsets: number[] = [];
    let offset = 0;
    for (const size of sizes) {
      offsets.push(offset);
      offset += size;
    }

    // Acquire reusable staging buffer from pool
    const readBuffer = acquireStagingBuffer(device, totalBytes);

    // Copy all tensors to the staging buffer in one command buffer
    const commandEncoder = device.createCommandEncoder();
    for (let i = 0; i < tensors.length; i++) {
      commandEncoder.copyBufferToBuffer(
        tensors[i].buffer, 0,
        readBuffer, offsets[i] * 4,
        sizes[i] * 4
      );
    }
    device.queue.submit([commandEncoder.finish()]);

    // Try mapSync if available (Chrome with --enable-features=WebGPUMapSyncOnWorkers)
    const bufferAny = readBuffer as unknown as { mapSync?: (mode: number) => void };
    if (typeof bufferAny.mapSync === 'function') {
      bufferAny.mapSync(GPUMapMode.READ);
    } else {
      await readBuffer.mapAsync(GPUMapMode.READ);
    }

    const fullData = new Float32Array(readBuffer.getMappedRange().slice(0, totalBytes));
    readBuffer.unmap();

    // Return buffer to pool for reuse
    releaseStagingBuffer(totalBytes, readBuffer);

    // Split into individual arrays and cache
    const results: Float32Array[] = [];
    for (let i = 0; i < tensors.length; i++) {
      const data = fullData.subarray(offsets[i], offsets[i] + sizes[i]);
      const copy = new Float32Array(data);
      tensors[i]._cachedData = copy;
      results.push(copy);
    }

    return results;
  }

  static fromArray(data: Float32Array | number[], shape: number[], device: GPUDevice): WebGPUTensor {
    const f32Data = data instanceof Float32Array ? data : new Float32Array(data);
    const n = f32Data.length;

    const buffer = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, f32Data.buffer, f32Data.byteOffset, f32Data.byteLength);

    const tensor = new WebGPUTensor(buffer, shape, device);
    tensor._cachedData = f32Data;
    return tensor;
  }

  static empty(shape: number[], device: GPUDevice): WebGPUTensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const buffer = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return new WebGPUTensor(buffer, shape, device);
  }

  destroy(): void {
    this.buffer.destroy();
  }
}

// ============ Shader Templates ============

// Unary element-wise activation shader
function makeUnaryShader(op: string): string {
  return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }
      let x = input[idx];
      output[idx] = ${op};
    }
  `;
}

// ============ WGSL Shader Definitions ============

const SHADERS = {
  // Activation functions
  relu: makeUnaryShader('max(0.0, x)'),
  relu6: makeUnaryShader('clamp(x, 0.0, 6.0)'),
  sigmoid: makeUnaryShader('1.0 / (1.0 + exp(-x))'),

  leaky_relu: `
    struct Params {
      size: u32,
      negative_slope: f32,
    }
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= params.size) { return; }
      let x = input[idx];
      output[idx] = select(params.negative_slope * x, x, x > 0.0);
    }
  `,

  gelu: makeUnaryShader(`
    x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
  `),

  prelu: `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }
      let x = input[idx];
      // PReLU weight is per-channel, but for simplicity we use element-wise
      // Real implementation would need channel indexing
      let w = weight[idx % arrayLength(&weight)];
      output[idx] = select(w * x, x, x > 0.0);
    }
  `,

  softmax: `
    struct Params {
      total_size: u32,
      axis_size: u32,
    }
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    // Shared memory for reduction
    var<workgroup> shared_max: array<f32, 256>;
    var<workgroup> shared_sum: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch_idx = wid.x;
      let tid = lid.x;
      let axis_size = params.axis_size;

      // Each workgroup handles one softmax batch
      let base = batch_idx * axis_size;

      // Find max for numerical stability
      var local_max: f32 = -1e30;
      for (var i = tid; i < axis_size; i = i + 256u) {
        local_max = max(local_max, input[base + i]);
      }
      shared_max[tid] = local_max;
      workgroupBarrier();

      // Reduce to find global max
      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
      }
      let max_val = shared_max[0];
      workgroupBarrier();

      // Compute exp(x - max) and sum
      var local_sum: f32 = 0.0;
      for (var i = tid; i < axis_size; i = i + 256u) {
        let exp_val = exp(input[base + i] - max_val);
        output[base + i] = exp_val;
        local_sum = local_sum + exp_val;
      }
      shared_sum[tid] = local_sum;
      workgroupBarrier();

      // Reduce to find global sum
      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
      }
      let sum_val = shared_sum[0];
      workgroupBarrier();

      // Normalize
      for (var i = tid; i < axis_size; i = i + 256u) {
        output[base + i] = output[base + i] / sum_val;
      }
    }
  `,

  // 2D Convolution - NCHW format
  conv2d: `
    struct Conv2dParams {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Conv2dParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let out_c_batch = gid.z;

      let out_c = out_c_batch % params.out_channels;
      let batch = out_c_batch / params.out_channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var sum: f32 = 0.0;

      // Convolve
      for (var ic = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
          for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
              let in_idx = batch * params.in_channels * params.in_height * params.in_width
                         + ic * params.in_height * params.in_width
                         + u32(in_y) * params.in_width
                         + u32(in_x);

              let w_idx = out_c * params.in_channels * params.kernel_h * params.kernel_w
                        + ic * params.kernel_h * params.kernel_w
                        + kh * params.kernel_w
                        + kw;

              sum = sum + input[in_idx] * weight[w_idx];
            }
          }
        }
      }

      // Add bias
      if (params.has_bias == 1u) {
        sum = sum + bias[out_c];
      }

      let out_idx = batch * params.out_channels * params.out_height * params.out_width
                  + out_c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum;
    }
  `,

  // Depthwise Convolution - NCHW format
  depthwise_conv2d: `
    struct DepthwiseConv2dParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv2dParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var sum: f32 = 0.0;

      for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
          let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
          let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

          if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
            let in_idx = batch * params.channels * params.in_height * params.in_width
                       + c * params.in_height * params.in_width
                       + u32(in_y) * params.in_width
                       + u32(in_x);

            let w_idx = c * params.kernel_h * params.kernel_w
                      + kh * params.kernel_w
                      + kw;

            sum = sum + input[in_idx] * weight[w_idx];
          }
        }
      }

      if (params.has_bias == 1u) {
        sum = sum + bias[c];
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum;
    }
  `,

  // Max Pooling 2D
  max_pool2d: `
    struct Pool2dParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Pool2dParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var max_val: f32 = -1e30;

      for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
          let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
          let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

          if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
            let in_idx = batch * params.channels * params.in_height * params.in_width
                       + c * params.in_height * params.in_width
                       + u32(in_y) * params.in_width
                       + u32(in_x);
            max_val = max(max_val, input[in_idx]);
          }
        }
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = max_val;
    }
  `,

  // Average Pooling 2D
  avg_pool2d: `
    struct Pool2dParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Pool2dParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var sum: f32 = 0.0;
      var count: u32 = 0u;

      for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
          let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
          let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

          if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
            let in_idx = batch * params.channels * params.in_height * params.in_width
                       + c * params.in_height * params.in_width
                       + u32(in_y) * params.in_width
                       + u32(in_x);
            sum = sum + input[in_idx];
            count = count + 1u;
          }
        }
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum / f32(max(count, 1u));
    }
  `,

  // Global Average Pooling 2D
  global_avg_pool2d: `
    struct GlobalPool2dParams {
      batch: u32,
      channels: u32,
      height: u32,
      width: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: GlobalPool2dParams;

    var<workgroup> shared_sum: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let c_batch = wid.x;
      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;
      let tid = lid.x;

      let hw = params.height * params.width;
      let base = batch * params.channels * hw + c * hw;

      // Each thread sums multiple elements
      var local_sum: f32 = 0.0;
      for (var i = tid; i < hw; i = i + 256u) {
        local_sum = local_sum + input[base + i];
      }
      shared_sum[tid] = local_sum;
      workgroupBarrier();

      // Reduce
      for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
      }

      if (tid == 0u) {
        output[c_batch] = shared_sum[0] / f32(hw);
      }
    }
  `,

  // Batch Normalization (inference mode)
  batch_norm: `
    struct BatchNormParams {
      batch: u32,
      channels: u32,
      height: u32,
      width: u32,
      eps: f32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> gamma: array<f32>;
    @group(0) @binding(2) var<storage, read> beta: array<f32>;
    @group(0) @binding(3) var<storage, read> running_mean: array<f32>;
    @group(0) @binding(4) var<storage, read> running_var: array<f32>;
    @group(0) @binding(5) var<storage, read_write> output: array<f32>;
    @group(0) @binding(6) var<uniform> params: BatchNormParams;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let total = params.batch * params.channels * params.height * params.width;
      if (idx >= total) { return; }

      let hw = params.height * params.width;
      let c = (idx / hw) % params.channels;

      let inv_std = 1.0 / sqrt(running_var[c] + params.eps);
      let scale = gamma[c] * inv_std;
      let shift = beta[c] - running_mean[c] * scale;

      output[idx] = input[idx] * scale + shift;
    }
  `,

  // Permute (transpose) - handles 4D tensors
  permute4d: `
    struct Permute4dParams {
      old_shape_0: u32,
      old_shape_1: u32,
      old_shape_2: u32,
      old_shape_3: u32,
      new_shape_0: u32,
      new_shape_1: u32,
      new_shape_2: u32,
      new_shape_3: u32,
      dims_0: u32,
      dims_1: u32,
      dims_2: u32,
      dims_3: u32,
      total_size: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Permute4dParams;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_idx = gid.x;
      if (out_idx >= params.total_size) { return; }

      // Compute new coordinates from flat output index
      let new_strides = array<u32, 4>(
        params.new_shape_1 * params.new_shape_2 * params.new_shape_3,
        params.new_shape_2 * params.new_shape_3,
        params.new_shape_3,
        1u
      );

      var remaining = out_idx;
      let new_coords = array<u32, 4>(
        remaining / new_strides[0],
        (remaining % new_strides[0]) / new_strides[1],
        (remaining % new_strides[1]) / new_strides[2],
        remaining % new_strides[2]
      );

      // Map to old coordinates using dims permutation
      let dims = array<u32, 4>(params.dims_0, params.dims_1, params.dims_2, params.dims_3);
      let old_strides = array<u32, 4>(
        params.old_shape_1 * params.old_shape_2 * params.old_shape_3,
        params.old_shape_2 * params.old_shape_3,
        params.old_shape_3,
        1u
      );

      var in_idx: u32 = 0u;
      in_idx = in_idx + new_coords[0] * old_strides[dims[0]];
      in_idx = in_idx + new_coords[1] * old_strides[dims[1]];
      in_idx = in_idx + new_coords[2] * old_strides[dims[2]];
      in_idx = in_idx + new_coords[3] * old_strides[dims[3]];

      output[out_idx] = input[in_idx];
    }
  `,

  // Constant padding (2D spatial padding)
  pad2d: `
    struct Pad2dParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      out_height: u32,
      out_width: u32,
      pad_left: u32,
      pad_top: u32,
      pad_value: f32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Pad2dParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;

      // Check if within input bounds
      let in_x = i32(out_x) - i32(params.pad_left);
      let in_y = i32(out_y) - i32(params.pad_top);

      if (in_x >= 0 && in_x < i32(params.in_width) && in_y >= 0 && in_y < i32(params.in_height)) {
        let in_idx = batch * params.channels * params.in_height * params.in_width
                   + c * params.in_height * params.in_width
                   + u32(in_y) * params.in_width
                   + u32(in_x);
        output[out_idx] = input[in_idx];
      } else {
        output[out_idx] = params.pad_value;
      }
    }
  `,

  // Channel padding (6-value padding: left, right, top, bottom, front, back)
  pad_channel: `
    struct PadChannelParams {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      height: u32,
      width: u32,
      pad_front: u32,
      pad_value: f32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: PadChannelParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let x = gid.x;
      let y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.out_channels;
      let batch = c_batch / params.out_channels;

      if (x >= params.width || y >= params.height || batch >= params.batch) {
        return;
      }

      let out_idx = batch * params.out_channels * params.height * params.width
                  + c * params.height * params.width
                  + y * params.width
                  + x;

      // Check if channel is in input range
      let in_c = i32(c) - i32(params.pad_front);

      if (in_c >= 0 && in_c < i32(params.in_channels)) {
        let in_idx = batch * params.in_channels * params.height * params.width
                   + u32(in_c) * params.height * params.width
                   + y * params.width
                   + x;
        output[out_idx] = input[in_idx];
      } else {
        output[out_idx] = params.pad_value;
      }
    }
  `,

  // Element-wise add
  add: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }
      output[idx] = a[idx] + b[idx];
    }
  `,

  // Fused add + relu (used in ResModule skip connection)
  add_relu: `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }
      output[idx] = max(0.0, a[idx] + b[idx]);
    }
  `,

  // Multiply by scalar
  mul_scalar: `
    struct Params {
      size: u32,
      scalar: f32,
    }
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= params.size) { return; }
      output[idx] = input[idx] * params.scalar;
    }
  `,

  // ============ OPTIMIZED CONVOLUTIONS ============

  // Optimized 5x5 Depthwise Convolution with padding=2, stride=1
  // Fully unrolled kernel loop for common hand landmark model pattern
  depthwise_conv2d_5x5: `
    struct DepthwiseConv5x5Params {
      batch: u32,
      channels: u32,
      height: u32,
      width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;  // [channels, 1, 5, 5]
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv5x5Params;

    // Branchless bounds check using select() - returns 0 if out of bounds
    fn safe_read(base: u32, dy: i32, dx: i32, h: u32, w: u32, width: u32) -> f32 {
      let y = i32(h) + dy;
      let x = i32(w) + dx;
      let in_bounds = y >= 0 && y < i32(params.height) && x >= 0 && x < i32(params.width);
      let clamped_y = clamp(y, 0, i32(params.height) - 1);
      let clamped_x = clamp(x, 0, i32(params.width) - 1);
      let val = input[base + u32(clamped_y) * params.width + u32(clamped_x)];
      return select(0.0, val, in_bounds);
    }

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let x = gid.x;
      let y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (x >= params.width || y >= params.height || batch >= params.batch) {
        return;
      }

      let in_base = batch * params.channels * params.height * params.width
                  + c * params.height * params.width;
      let w_base = c * 25u;  // 5x5 = 25 weights per channel

      // Unrolled 5x5 convolution with padding=2
      // dy, dx range from -2 to +2
      var sum: f32 = 0.0;

      // Row -2
      sum = sum + safe_read(in_base, -2, -2, y, x, params.width) * weight[w_base + 0u];
      sum = sum + safe_read(in_base, -2, -1, y, x, params.width) * weight[w_base + 1u];
      sum = sum + safe_read(in_base, -2,  0, y, x, params.width) * weight[w_base + 2u];
      sum = sum + safe_read(in_base, -2,  1, y, x, params.width) * weight[w_base + 3u];
      sum = sum + safe_read(in_base, -2,  2, y, x, params.width) * weight[w_base + 4u];

      // Row -1
      sum = sum + safe_read(in_base, -1, -2, y, x, params.width) * weight[w_base + 5u];
      sum = sum + safe_read(in_base, -1, -1, y, x, params.width) * weight[w_base + 6u];
      sum = sum + safe_read(in_base, -1,  0, y, x, params.width) * weight[w_base + 7u];
      sum = sum + safe_read(in_base, -1,  1, y, x, params.width) * weight[w_base + 8u];
      sum = sum + safe_read(in_base, -1,  2, y, x, params.width) * weight[w_base + 9u];

      // Row 0
      sum = sum + safe_read(in_base,  0, -2, y, x, params.width) * weight[w_base + 10u];
      sum = sum + safe_read(in_base,  0, -1, y, x, params.width) * weight[w_base + 11u];
      sum = sum + safe_read(in_base,  0,  0, y, x, params.width) * weight[w_base + 12u];
      sum = sum + safe_read(in_base,  0,  1, y, x, params.width) * weight[w_base + 13u];
      sum = sum + safe_read(in_base,  0,  2, y, x, params.width) * weight[w_base + 14u];

      // Row +1
      sum = sum + safe_read(in_base,  1, -2, y, x, params.width) * weight[w_base + 15u];
      sum = sum + safe_read(in_base,  1, -1, y, x, params.width) * weight[w_base + 16u];
      sum = sum + safe_read(in_base,  1,  0, y, x, params.width) * weight[w_base + 17u];
      sum = sum + safe_read(in_base,  1,  1, y, x, params.width) * weight[w_base + 18u];
      sum = sum + safe_read(in_base,  1,  2, y, x, params.width) * weight[w_base + 19u];

      // Row +2
      sum = sum + safe_read(in_base,  2, -2, y, x, params.width) * weight[w_base + 20u];
      sum = sum + safe_read(in_base,  2, -1, y, x, params.width) * weight[w_base + 21u];
      sum = sum + safe_read(in_base,  2,  0, y, x, params.width) * weight[w_base + 22u];
      sum = sum + safe_read(in_base,  2,  1, y, x, params.width) * weight[w_base + 23u];
      sum = sum + safe_read(in_base,  2,  2, y, x, params.width) * weight[w_base + 24u];

      // Add bias
      if (params.has_bias == 1u) {
        sum = sum + bias[c];
      }

      let out_idx = batch * params.channels * params.height * params.width
                  + c * params.height * params.width
                  + y * params.width
                  + x;
      output[out_idx] = sum;
    }
  `,

  // Optimized 5x5 Depthwise Convolution - 2x2 outputs per thread
  // Reduces memory accesses by computing 4 adjacent outputs with shared data loading
  // For 5x5 kernel computing 2x2 outputs: need 6x6 input region
  // Memory access: ~36 reads for 4 outputs vs ~100 reads for 4 separate threads
  depthwise_conv2d_5x5_2x2: `
    struct DepthwiseConv5x5_2x2Params {
      batch: u32,
      channels: u32,
      height: u32,
      width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv5x5_2x2Params;

    // Each thread computes 2x2 outputs
    // Workgroup size 8x8 = 64 threads, each computing 4 outputs = 256 outputs per workgroup
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      // Each thread handles a 2x2 output block
      let out_x = gid.x * 2u;
      let out_y = gid.y * 2u;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
        return;
      }

      let H = params.height;
      let W = params.width;
      let in_base = batch * params.channels * H * W + c * H * W;
      let w_base = c * 25u;

      // Load 6x6 input region (covers 5x5 kernel for 2x2 outputs)
      // Use local array to avoid redundant global memory reads
      var tile: array<f32, 36>;  // 6x6

      for (var dy: i32 = -2; dy <= 3; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 3; dx = dx + 1) {
          let in_y = i32(out_y) + dy;
          let in_x = i32(out_x) + dx;
          let in_bounds = in_y >= 0 && in_y < i32(H) && in_x >= 0 && in_x < i32(W);
          let clamped_y = clamp(in_y, 0, i32(H) - 1);
          let clamped_x = clamp(in_x, 0, i32(W) - 1);
          let val = input[in_base + u32(clamped_y) * W + u32(clamped_x)];
          let tile_idx = (dy + 2) * 6 + (dx + 2);
          tile[tile_idx] = select(0.0, val, in_bounds);
        }
      }

      // Load weights once
      var w: array<f32, 25>;
      for (var i = 0u; i < 25u; i = i + 1u) {
        w[i] = weight[w_base + i];
      }

      // Compute 4 outputs: (out_y, out_x), (out_y, out_x+1), (out_y+1, out_x), (out_y+1, out_x+1)
      var sum00: f32 = 0.0;
      var sum01: f32 = 0.0;
      var sum10: f32 = 0.0;
      var sum11: f32 = 0.0;

      // Manually unroll 5x5 kernel
      // Output (0,0) uses tile[0:5, 0:5], output (0,1) uses tile[0:5, 1:6], etc.
      for (var ky = 0u; ky < 5u; ky = ky + 1u) {
        for (var kx = 0u; kx < 5u; kx = kx + 1u) {
          let wgt = w[ky * 5u + kx];
          sum00 = sum00 + tile[(ky + 0u) * 6u + (kx + 0u)] * wgt;
          sum01 = sum01 + tile[(ky + 0u) * 6u + (kx + 1u)] * wgt;
          sum10 = sum10 + tile[(ky + 1u) * 6u + (kx + 0u)] * wgt;
          sum11 = sum11 + tile[(ky + 1u) * 6u + (kx + 1u)] * wgt;
        }
      }

      // Add bias
      if (params.has_bias == 1u) {
        let b = bias[c];
        sum00 = sum00 + b;
        sum01 = sum01 + b;
        sum10 = sum10 + b;
        sum11 = sum11 + b;
      }

      // Write outputs (check bounds for edge cases)
      let out_base = batch * params.channels * H * W + c * H * W;

      output[out_base + out_y * W + out_x] = sum00;

      if (out_x + 1u < W) {
        output[out_base + out_y * W + (out_x + 1u)] = sum01;
      }

      if (out_y + 1u < H) {
        output[out_base + (out_y + 1u) * W + out_x] = sum10;

        if (out_x + 1u < W) {
          output[out_base + (out_y + 1u) * W + (out_x + 1u)] = sum11;
        }
      }
    }
  `,

  // Optimized 5x5 Depthwise Conv2d with stride=2, no padding (input pre-padded)
  // For handlandmarks ResModule stride=2 case: input is padded (1,2,1,2) before calling
  depthwise_conv2d_5x5_stride2: `
    struct DepthwiseConv5x5S2Params {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      out_height: u32,
      out_width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;  // [channels, 1, 5, 5]
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv5x5S2Params;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      // With stride=2, input position is (out_y*2, out_x*2)
      // Input is pre-padded, so we just read directly
      let in_y = out_y * 2u;
      let in_x = out_x * 2u;

      let in_base = batch * params.channels * params.in_height * params.in_width
                  + c * params.in_height * params.in_width;
      let w_base = c * 25u;

      // Unrolled 5x5 convolution - no bounds checking (input is pre-padded)
      var sum: f32 = 0.0;

      // Row 0
      let row0 = in_base + in_y * params.in_width + in_x;
      sum = sum + input[row0 + 0u] * weight[w_base + 0u];
      sum = sum + input[row0 + 1u] * weight[w_base + 1u];
      sum = sum + input[row0 + 2u] * weight[w_base + 2u];
      sum = sum + input[row0 + 3u] * weight[w_base + 3u];
      sum = sum + input[row0 + 4u] * weight[w_base + 4u];

      // Row 1
      let row1 = in_base + (in_y + 1u) * params.in_width + in_x;
      sum = sum + input[row1 + 0u] * weight[w_base + 5u];
      sum = sum + input[row1 + 1u] * weight[w_base + 6u];
      sum = sum + input[row1 + 2u] * weight[w_base + 7u];
      sum = sum + input[row1 + 3u] * weight[w_base + 8u];
      sum = sum + input[row1 + 4u] * weight[w_base + 9u];

      // Row 2
      let row2 = in_base + (in_y + 2u) * params.in_width + in_x;
      sum = sum + input[row2 + 0u] * weight[w_base + 10u];
      sum = sum + input[row2 + 1u] * weight[w_base + 11u];
      sum = sum + input[row2 + 2u] * weight[w_base + 12u];
      sum = sum + input[row2 + 3u] * weight[w_base + 13u];
      sum = sum + input[row2 + 4u] * weight[w_base + 14u];

      // Row 3
      let row3 = in_base + (in_y + 3u) * params.in_width + in_x;
      sum = sum + input[row3 + 0u] * weight[w_base + 15u];
      sum = sum + input[row3 + 1u] * weight[w_base + 16u];
      sum = sum + input[row3 + 2u] * weight[w_base + 17u];
      sum = sum + input[row3 + 3u] * weight[w_base + 18u];
      sum = sum + input[row3 + 4u] * weight[w_base + 19u];

      // Row 4
      let row4 = in_base + (in_y + 4u) * params.in_width + in_x;
      sum = sum + input[row4 + 0u] * weight[w_base + 20u];
      sum = sum + input[row4 + 1u] * weight[w_base + 21u];
      sum = sum + input[row4 + 2u] * weight[w_base + 22u];
      sum = sum + input[row4 + 3u] * weight[w_base + 23u];
      sum = sum + input[row4 + 4u] * weight[w_base + 24u];

      // Add bias
      if (params.has_bias == 1u) {
        sum = sum + bias[c];
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum;
    }
  `,

  // 1x1 Convolution (Pointwise) with tiled matmul using shared memory
  // Treats conv as: output[spatial_idx, out_c] = sum_ic(input[spatial_idx, ic] * weight[out_c, ic])
  // This is equivalent to: C = A @ B^T where A=[spatial, in_c], B=[out_c, in_c]
  //
  // Uses 8x8 workgroups (64 threads), each computing 1 output.
  // Manual loop unrolling and swapped x/y for better cache.
  conv2d_1x1_tiled: `
    struct Conv1x1Params {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      spatial_size: u32,  // H * W
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Conv1x1Params;

    const TILE_M: u32 = 8u;   // spatial tile = workgroup cols (swapped for cache)
    const TILE_N: u32 = 8u;   // out_channels tile = workgroup rows
    const TILE_K: u32 = 8u;   // in_channels tile for shared memory

    var<workgroup> tile_A: array<f32, 64>;   // TILE_M x TILE_K = 8x8
    var<workgroup> tile_B: array<f32, 64>;   // TILE_N x TILE_K = 8x8

    @compute @workgroup_size(8, 8)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch = wid.z;
      // Swapped x/y for better cache - x increments first
      let out_c_tile = wid.x;
      let spatial_tile = wid.y;

      let local_col = lid.x;  // out_channel within tile (0-7)
      let local_row = lid.y;  // spatial position within tile (0-7)

      let global_out_c = out_c_tile * TILE_N + local_col;
      let global_spatial = spatial_tile * TILE_M + local_row;

      var sum: f32 = 0.0;

      let num_k_tiles = (params.in_channels + TILE_K - 1u) / TILE_K;

      for (var kt: u32 = 0u; kt < num_k_tiles; kt = kt + 1u) {
        let k_start = kt * TILE_K;

        // Load input tile: A[spatial, in_c]
        let load_k_a = k_start + local_col;
        if (global_spatial < params.spatial_size && load_k_a < params.in_channels) {
          let a_idx = batch * params.in_channels * params.spatial_size
                    + load_k_a * params.spatial_size
                    + global_spatial;
          tile_A[local_row * TILE_K + local_col] = input[a_idx];
        } else {
          tile_A[local_row * TILE_K + local_col] = 0.0;
        }

        // Load weight tile: B[out_c, in_c]
        let load_k_b = k_start + local_row;
        if (global_out_c < params.out_channels && load_k_b < params.in_channels) {
          tile_B[local_col * TILE_K + local_row] = weight[global_out_c * params.in_channels + load_k_b];
        } else {
          tile_B[local_col * TILE_K + local_row] = 0.0;
        }

        workgroupBarrier();

        // Manually unrolled inner loop for better ILP
        sum = sum + tile_A[local_row * TILE_K + 0u] * tile_B[local_col * TILE_K + 0u];
        sum = sum + tile_A[local_row * TILE_K + 1u] * tile_B[local_col * TILE_K + 1u];
        sum = sum + tile_A[local_row * TILE_K + 2u] * tile_B[local_col * TILE_K + 2u];
        sum = sum + tile_A[local_row * TILE_K + 3u] * tile_B[local_col * TILE_K + 3u];
        sum = sum + tile_A[local_row * TILE_K + 4u] * tile_B[local_col * TILE_K + 4u];
        sum = sum + tile_A[local_row * TILE_K + 5u] * tile_B[local_col * TILE_K + 5u];
        sum = sum + tile_A[local_row * TILE_K + 6u] * tile_B[local_col * TILE_K + 6u];
        sum = sum + tile_A[local_row * TILE_K + 7u] * tile_B[local_col * TILE_K + 7u];

        workgroupBarrier();
      }

      // Add bias and write output
      if (global_spatial < params.spatial_size && global_out_c < params.out_channels) {
        if (params.has_bias == 1u) {
          sum = sum + bias[global_out_c];
        }

        let out_idx = batch * params.out_channels * params.spatial_size
                    + global_out_c * params.spatial_size
                    + global_spatial;
        output[out_idx] = sum;
      }
    }
  `,

  // Vec4 1x1 Convolution - uses dot(vec4, vec4) for hardware DP4 instruction
  // Process 4 input channels at once, accumulate via dot product
  conv2d_1x1_vec4: `
    struct Conv1x1Params {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      spatial_size: u32,  // H * W
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Conv1x1Params;

    const TILE_SIZE: u32 = 8u;

    // Shared memory for input tile (spatial x in_channels/4 vec4s)
    var<workgroup> tile_A: array<vec4<f32>, 128>;  // 8x16 = 128 vec4s
    // Shared memory for weight tile (out_c x in_channels/4 vec4s)
    var<workgroup> tile_B: array<vec4<f32>, 128>;  // 8x16 = 128 vec4s

    @compute @workgroup_size(8, 8)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch = wid.z;
      let out_c_tile = wid.x;
      let spatial_tile = wid.y;

      let local_col = lid.x;  // out_channel within tile (0-7)
      let local_row = lid.y;  // spatial position within tile (0-7)

      let global_out_c = out_c_tile * TILE_SIZE + local_col;
      let global_spatial = spatial_tile * TILE_SIZE + local_row;

      var sum: f32 = 0.0;

      // Process input channels in groups of 4 (vec4)
      let in_channels_vec4 = (params.in_channels + 3u) / 4u;

      for (var kv: u32 = 0u; kv < in_channels_vec4; kv = kv + 1u) {
        let k_start = kv * 4u;

        // Load 4 consecutive input channels as vec4
        // Threads load different spatial positions, same channel range
        if (global_spatial < params.spatial_size) {
          var a_vec: vec4<f32> = vec4<f32>(0.0);
          for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let ic = k_start + i;
            if (ic < params.in_channels) {
              let a_idx = batch * params.in_channels * params.spatial_size
                        + ic * params.spatial_size
                        + global_spatial;
              a_vec[i] = input[a_idx];
            }
          }
          tile_A[local_row * 16u + local_col] = a_vec;
        } else {
          tile_A[local_row * 16u + local_col] = vec4<f32>(0.0);
        }

        // Load 4 consecutive weight channels as vec4
        if (global_out_c < params.out_channels) {
          var b_vec: vec4<f32> = vec4<f32>(0.0);
          for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let ic = k_start + i;
            if (ic < params.in_channels) {
              b_vec[i] = weight[global_out_c * params.in_channels + ic];
            }
          }
          tile_B[local_col * 16u + local_row] = b_vec;
        } else {
          tile_B[local_col * 16u + local_row] = vec4<f32>(0.0);
        }

        workgroupBarrier();

        // Dot products - each processes 4 channels at once
        // This uses hardware DP4 instruction
        sum = sum + dot(tile_A[local_row * 16u + 0u], tile_B[local_col * 16u + 0u]);
        sum = sum + dot(tile_A[local_row * 16u + 1u], tile_B[local_col * 16u + 1u]);
        sum = sum + dot(tile_A[local_row * 16u + 2u], tile_B[local_col * 16u + 2u]);
        sum = sum + dot(tile_A[local_row * 16u + 3u], tile_B[local_col * 16u + 3u]);
        sum = sum + dot(tile_A[local_row * 16u + 4u], tile_B[local_col * 16u + 4u]);
        sum = sum + dot(tile_A[local_row * 16u + 5u], tile_B[local_col * 16u + 5u]);
        sum = sum + dot(tile_A[local_row * 16u + 6u], tile_B[local_col * 16u + 6u]);
        sum = sum + dot(tile_A[local_row * 16u + 7u], tile_B[local_col * 16u + 7u]);

        workgroupBarrier();
      }

      // Add bias and write output
      if (global_spatial < params.spatial_size && global_out_c < params.out_channels) {
        if (params.has_bias == 1u) {
          sum = sum + bias[global_out_c];
        }

        let out_idx = batch * params.out_channels * params.spatial_size
                    + global_out_c * params.spatial_size
                    + global_spatial;
        output[out_idx] = sum;
      }
    }
  `,

  // ============ FP16 SHADERS (shader-f16 extension) ============
  // These use half-precision for 2x compute throughput on M1/M2 (2x FP16 FLOPS)
  //
  // TWO VERSIONS:
  // 1. conv2d_1x1_tiled_f16 - reads f32 storage, converts to f16 (slower due to conversion)
  // 2. conv2d_1x1_native_f16 - reads f16 storage directly (faster, requires f16 weight buffers)

  // Version 1: FP16 compute with f32 storage (has conversion overhead)
  conv2d_1x1_tiled_f16: `
    enable f16;

    struct Conv1x1Params {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      spatial_size: u32,  // H * W
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Conv1x1Params;

    const TILE_M: u32 = 8u;   // spatial tile = workgroup cols (swapped for cache)
    const TILE_N: u32 = 8u;   // out_channels tile = workgroup rows
    const TILE_K: u32 = 8u;   // in_channels tile for shared memory

    var<workgroup> tile_A: array<f16, 64>;   // TILE_M x TILE_K = 8x8 (half memory!)
    var<workgroup> tile_B: array<f16, 64>;   // TILE_N x TILE_K = 8x8

    @compute @workgroup_size(8, 8)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch = wid.z;
      // Swapped x/y for better cache - x increments first
      let out_c_tile = wid.x;
      let spatial_tile = wid.y;

      let local_col = lid.x;  // out_channel within tile (0-7)
      let local_row = lid.y;  // spatial position within tile (0-7)

      let global_out_c = out_c_tile * TILE_N + local_col;
      let global_spatial = spatial_tile * TILE_M + local_row;

      var sum: f16 = f16(0.0);

      let num_k_tiles = (params.in_channels + TILE_K - 1u) / TILE_K;

      for (var kt: u32 = 0u; kt < num_k_tiles; kt = kt + 1u) {
        let k_start = kt * TILE_K;

        // Load input tile: A[spatial, in_c] - convert f32 to f16
        let load_k_a = k_start + local_col;
        if (global_spatial < params.spatial_size && load_k_a < params.in_channels) {
          let a_idx = batch * params.in_channels * params.spatial_size
                    + load_k_a * params.spatial_size
                    + global_spatial;
          tile_A[local_row * TILE_K + local_col] = f16(input[a_idx]);
        } else {
          tile_A[local_row * TILE_K + local_col] = f16(0.0);
        }

        // Load weight tile: B[out_c, in_c] - convert f32 to f16
        let load_k_b = k_start + local_row;
        if (global_out_c < params.out_channels && load_k_b < params.in_channels) {
          tile_B[local_col * TILE_K + local_row] = f16(weight[global_out_c * params.in_channels + load_k_b]);
        } else {
          tile_B[local_col * TILE_K + local_row] = f16(0.0);
        }

        workgroupBarrier();

        // Manually unrolled inner loop - all FP16 math
        sum = sum + tile_A[local_row * TILE_K + 0u] * tile_B[local_col * TILE_K + 0u];
        sum = sum + tile_A[local_row * TILE_K + 1u] * tile_B[local_col * TILE_K + 1u];
        sum = sum + tile_A[local_row * TILE_K + 2u] * tile_B[local_col * TILE_K + 2u];
        sum = sum + tile_A[local_row * TILE_K + 3u] * tile_B[local_col * TILE_K + 3u];
        sum = sum + tile_A[local_row * TILE_K + 4u] * tile_B[local_col * TILE_K + 4u];
        sum = sum + tile_A[local_row * TILE_K + 5u] * tile_B[local_col * TILE_K + 5u];
        sum = sum + tile_A[local_row * TILE_K + 6u] * tile_B[local_col * TILE_K + 6u];
        sum = sum + tile_A[local_row * TILE_K + 7u] * tile_B[local_col * TILE_K + 7u];

        workgroupBarrier();
      }

      // Add bias and write output (convert back to f32)
      if (global_spatial < params.spatial_size && global_out_c < params.out_channels) {
        if (params.has_bias == 1u) {
          sum = sum + f16(bias[global_out_c]);
        }

        let out_idx = batch * params.out_channels * params.spatial_size
                    + global_out_c * params.spatial_size
                    + global_spatial;
        output[out_idx] = f32(sum);
      }
    }
  `,

  // FP16 5x5 Depthwise Conv with 2x2 output-per-thread
  depthwise_conv2d_5x5_2x2_f16: `
    enable f16;

    struct DepthwiseConv5x5_2x2Params {
      batch: u32,
      channels: u32,
      height: u32,
      width: u32,
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv5x5_2x2Params;

    // Each thread computes 2x2 outputs using f16
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      // Each thread handles a 2x2 output block
      let out_x = gid.x * 2u;
      let out_y = gid.y * 2u;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
        return;
      }

      let H = params.height;
      let W = params.width;
      let in_base = batch * params.channels * H * W + c * H * W;
      let w_base = c * 25u;

      // Load 6x6 input region (covers 5x5 kernel for 2x2 outputs) - FP16
      var tile: array<f16, 36>;  // 6x6

      for (var dy: i32 = -2; dy <= 3; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 3; dx = dx + 1) {
          let in_y = i32(out_y) + dy;
          let in_x = i32(out_x) + dx;
          let in_bounds = in_y >= 0 && in_y < i32(H) && in_x >= 0 && in_x < i32(W);
          let clamped_y = clamp(in_y, 0, i32(H) - 1);
          let clamped_x = clamp(in_x, 0, i32(W) - 1);
          let val = f16(input[in_base + u32(clamped_y) * W + u32(clamped_x)]);
          let tile_idx = (dy + 2) * 6 + (dx + 2);
          tile[tile_idx] = select(f16(0.0), val, in_bounds);
        }
      }

      // Load weights once - FP16
      var w: array<f16, 25>;
      for (var i = 0u; i < 25u; i = i + 1u) {
        w[i] = f16(weight[w_base + i]);
      }

      // Compute 4 outputs - all FP16 math
      var sum00: f16 = f16(0.0);
      var sum01: f16 = f16(0.0);
      var sum10: f16 = f16(0.0);
      var sum11: f16 = f16(0.0);

      // Manually unroll 5x5 kernel
      for (var ky = 0u; ky < 5u; ky = ky + 1u) {
        for (var kx = 0u; kx < 5u; kx = kx + 1u) {
          let wgt = w[ky * 5u + kx];
          sum00 = sum00 + tile[(ky + 0u) * 6u + (kx + 0u)] * wgt;
          sum01 = sum01 + tile[(ky + 0u) * 6u + (kx + 1u)] * wgt;
          sum10 = sum10 + tile[(ky + 1u) * 6u + (kx + 0u)] * wgt;
          sum11 = sum11 + tile[(ky + 1u) * 6u + (kx + 1u)] * wgt;
        }
      }

      // Add bias
      if (params.has_bias == 1u) {
        let b = f16(bias[c]);
        sum00 = sum00 + b;
        sum01 = sum01 + b;
        sum10 = sum10 + b;
        sum11 = sum11 + b;
      }

      // Write outputs (convert back to f32)
      let out_base = batch * params.channels * H * W + c * H * W;

      output[out_base + out_y * W + out_x] = f32(sum00);

      if (out_x + 1u < W) {
        output[out_base + out_y * W + (out_x + 1u)] = f32(sum01);
      }

      if (out_y + 1u < H) {
        output[out_base + (out_y + 1u) * W + out_x] = f32(sum10);

        if (out_x + 1u < W) {
          output[out_base + (out_y + 1u) * W + (out_x + 1u)] = f32(sum11);
        }
      }
    }
  `,

  // ============ NATIVE F16 SHADERS ============
  // These read f16 storage directly (no conversion overhead)
  // Requires weights to be pre-converted to f16 format at model load time

  // Native F16 1x1 Conv - reads weights as f16, activations as f32 (first layer)
  // For first layer: input f32, weights f16, output f16
  // For middle layers: input f16, weights f16, output f16
  conv2d_1x1_native_f16: `
    enable f16;

    struct Conv1x1Params {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      spatial_size: u32,
      has_bias: u32,
      input_f16: u32,  // 1 if input is f16, 0 if f32
    }

    // Input can be either f32 (first layer) or f16 (subsequent layers)
    @group(0) @binding(0) var<storage, read> input_f32: array<f32>;
    @group(0) @binding(1) var<storage, read> input_f16_buf: array<f16>;
    // Weights and bias are always f16
    @group(0) @binding(2) var<storage, read> weight: array<f16>;
    @group(0) @binding(3) var<storage, read> bias: array<f16>;
    // Output is f16 (will be read by next layer)
    @group(0) @binding(4) var<storage, read_write> output: array<f16>;
    @group(0) @binding(5) var<uniform> params: Conv1x1Params;

    const TILE_M: u32 = 8u;
    const TILE_N: u32 = 8u;
    const TILE_K: u32 = 8u;

    var<workgroup> tile_A: array<f16, 64>;
    var<workgroup> tile_B: array<f16, 64>;

    @compute @workgroup_size(8, 8)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch = wid.z;
      let out_c_tile = wid.x;
      let spatial_tile = wid.y;

      let local_col = lid.x;
      let local_row = lid.y;

      let global_out_c = out_c_tile * TILE_N + local_col;
      let global_spatial = spatial_tile * TILE_M + local_row;

      var sum: f16 = f16(0.0);

      let num_k_tiles = (params.in_channels + TILE_K - 1u) / TILE_K;

      for (var kt: u32 = 0u; kt < num_k_tiles; kt = kt + 1u) {
        let k_start = kt * TILE_K;

        // Load input - check if f16 or f32
        let load_k_a = k_start + local_col;
        if (global_spatial < params.spatial_size && load_k_a < params.in_channels) {
          let a_idx = batch * params.in_channels * params.spatial_size
                    + load_k_a * params.spatial_size
                    + global_spatial;
          if (params.input_f16 == 1u) {
            tile_A[local_row * TILE_K + local_col] = input_f16_buf[a_idx];
          } else {
            tile_A[local_row * TILE_K + local_col] = f16(input_f32[a_idx]);
          }
        } else {
          tile_A[local_row * TILE_K + local_col] = f16(0.0);
        }

        // Load weight - always f16 (no conversion needed!)
        let load_k_b = k_start + local_row;
        if (global_out_c < params.out_channels && load_k_b < params.in_channels) {
          tile_B[local_col * TILE_K + local_row] = weight[global_out_c * params.in_channels + load_k_b];
        } else {
          tile_B[local_col * TILE_K + local_row] = f16(0.0);
        }

        workgroupBarrier();

        // F16 compute
        sum = sum + tile_A[local_row * TILE_K + 0u] * tile_B[local_col * TILE_K + 0u];
        sum = sum + tile_A[local_row * TILE_K + 1u] * tile_B[local_col * TILE_K + 1u];
        sum = sum + tile_A[local_row * TILE_K + 2u] * tile_B[local_col * TILE_K + 2u];
        sum = sum + tile_A[local_row * TILE_K + 3u] * tile_B[local_col * TILE_K + 3u];
        sum = sum + tile_A[local_row * TILE_K + 4u] * tile_B[local_col * TILE_K + 4u];
        sum = sum + tile_A[local_row * TILE_K + 5u] * tile_B[local_col * TILE_K + 5u];
        sum = sum + tile_A[local_row * TILE_K + 6u] * tile_B[local_col * TILE_K + 6u];
        sum = sum + tile_A[local_row * TILE_K + 7u] * tile_B[local_col * TILE_K + 7u];

        workgroupBarrier();
      }

      // Add bias and write f16 output
      if (global_spatial < params.spatial_size && global_out_c < params.out_channels) {
        if (params.has_bias == 1u) {
          sum = sum + bias[global_out_c];
        }

        let out_idx = batch * params.out_channels * params.spatial_size
                    + global_out_c * params.spatial_size
                    + global_spatial;
        output[out_idx] = sum;
      }
    }
  `,

  // ============ FUSED KERNELS ============
  // These combine multiple operations into single GPU dispatches for efficiency

  // Fused 1x1 Conv + Skip Add + ReLU (for ResModule)
  // This is the critical path: pointwise conv -> add skip -> relu
  // Uses same tiled matmul as conv2d_1x1_tiled but adds skip connection and relu
  conv2d_1x1_skip_relu: `
    struct Conv1x1SkipReluParams {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      spatial_size: u32,  // H * W
      has_bias: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read> skip: array<f32>;
    @group(0) @binding(4) var<storage, read_write> output: array<f32>;
    @group(0) @binding(5) var<uniform> params: Conv1x1SkipReluParams;

    const TILE_M: u32 = 8u;
    const TILE_N: u32 = 8u;
    const TILE_K: u32 = 8u;

    var<workgroup> tile_A: array<f32, 64>;
    var<workgroup> tile_B: array<f32, 64>;

    @compute @workgroup_size(8, 8)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let batch = wid.z;
      let out_c_tile = wid.x;
      let spatial_tile = wid.y;

      let local_col = lid.x;
      let local_row = lid.y;

      let global_out_c = out_c_tile * TILE_N + local_col;
      let global_spatial = spatial_tile * TILE_M + local_row;

      var sum: f32 = 0.0;

      let num_k_tiles = (params.in_channels + TILE_K - 1u) / TILE_K;

      for (var kt: u32 = 0u; kt < num_k_tiles; kt = kt + 1u) {
        let k_start = kt * TILE_K;

        let load_k_a = k_start + local_col;
        if (global_spatial < params.spatial_size && load_k_a < params.in_channels) {
          let a_idx = batch * params.in_channels * params.spatial_size
                    + load_k_a * params.spatial_size
                    + global_spatial;
          tile_A[local_row * TILE_K + local_col] = input[a_idx];
        } else {
          tile_A[local_row * TILE_K + local_col] = 0.0;
        }

        let load_k_b = k_start + local_row;
        if (global_out_c < params.out_channels && load_k_b < params.in_channels) {
          tile_B[local_col * TILE_K + local_row] = weight[global_out_c * params.in_channels + load_k_b];
        } else {
          tile_B[local_col * TILE_K + local_row] = 0.0;
        }

        workgroupBarrier();

        sum = sum + tile_A[local_row * TILE_K + 0u] * tile_B[local_col * TILE_K + 0u];
        sum = sum + tile_A[local_row * TILE_K + 1u] * tile_B[local_col * TILE_K + 1u];
        sum = sum + tile_A[local_row * TILE_K + 2u] * tile_B[local_col * TILE_K + 2u];
        sum = sum + tile_A[local_row * TILE_K + 3u] * tile_B[local_col * TILE_K + 3u];
        sum = sum + tile_A[local_row * TILE_K + 4u] * tile_B[local_col * TILE_K + 4u];
        sum = sum + tile_A[local_row * TILE_K + 5u] * tile_B[local_col * TILE_K + 5u];
        sum = sum + tile_A[local_row * TILE_K + 6u] * tile_B[local_col * TILE_K + 6u];
        sum = sum + tile_A[local_row * TILE_K + 7u] * tile_B[local_col * TILE_K + 7u];

        workgroupBarrier();
      }

      // Add bias, skip, and apply relu
      if (global_spatial < params.spatial_size && global_out_c < params.out_channels) {
        if (params.has_bias == 1u) {
          sum = sum + bias[global_out_c];
        }

        let out_idx = batch * params.out_channels * params.spatial_size
                    + global_out_c * params.spatial_size
                    + global_spatial;

        // Add skip connection and apply ReLU
        sum = sum + skip[out_idx];
        output[out_idx] = max(0.0, sum);
      }
    }
  `,

  // Fused Conv2d + Bias + ReLU (or no activation)
  // activation: 0=none, 1=relu, 2=relu6
  conv2d_fused: `
    struct Conv2dFusedParams {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
      has_bias: u32,
      activation: u32,  // 0=none, 1=relu, 2=relu6
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Conv2dFusedParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let out_c_batch = gid.z;

      let out_c = out_c_batch % params.out_channels;
      let batch = out_c_batch / params.out_channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var sum: f32 = 0.0;

      // Convolve
      for (var ic = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
          for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
            let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

            if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
              let in_idx = batch * params.in_channels * params.in_height * params.in_width
                         + ic * params.in_height * params.in_width
                         + u32(in_y) * params.in_width
                         + u32(in_x);

              let w_idx = out_c * params.in_channels * params.kernel_h * params.kernel_w
                        + ic * params.kernel_h * params.kernel_w
                        + kh * params.kernel_w
                        + kw;

              sum = sum + input[in_idx] * weight[w_idx];
            }
          }
        }
      }

      // Add bias
      if (params.has_bias == 1u) {
        sum = sum + bias[out_c];
      }

      // Apply activation
      if (params.activation == 1u) {
        sum = max(0.0, sum);
      } else if (params.activation == 2u) {
        sum = clamp(sum, 0.0, 6.0);
      }

      let out_idx = batch * params.out_channels * params.out_height * params.out_width
                  + out_c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum;
    }
  `,

  // Fused Depthwise Conv2d + Bias + ReLU
  depthwise_conv2d_fused: `
    struct DepthwiseConv2dFusedParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      kernel_h: u32,
      kernel_w: u32,
      stride_h: u32,
      stride_w: u32,
      pad_h: u32,
      pad_w: u32,
      out_height: u32,
      out_width: u32,
      has_bias: u32,
      activation: u32,  // 0=none, 1=relu, 2=relu6
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: DepthwiseConv2dFusedParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      var sum: f32 = 0.0;

      for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
          let in_y = i32(out_y * params.stride_h + kh) - i32(params.pad_h);
          let in_x = i32(out_x * params.stride_w + kw) - i32(params.pad_w);

          if (in_y >= 0 && in_y < i32(params.in_height) && in_x >= 0 && in_x < i32(params.in_width)) {
            let in_idx = batch * params.channels * params.in_height * params.in_width
                       + c * params.in_height * params.in_width
                       + u32(in_y) * params.in_width
                       + u32(in_x);

            let w_idx = c * params.kernel_h * params.kernel_w
                      + kh * params.kernel_w
                      + kw;

            sum = sum + input[in_idx] * weight[w_idx];
          }
        }
      }

      if (params.has_bias == 1u) {
        sum = sum + bias[c];
      }

      // Apply activation
      if (params.activation == 1u) {
        sum = max(0.0, sum);
      } else if (params.activation == 2u) {
        sum = clamp(sum, 0.0, 6.0);
      }

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = sum;
    }
  `,

  // Fused Depthwise Conv (5x5) + Pointwise Conv (1x1) + Skip Add + ReLU
  // This is the full ResModule pattern - single kernel dispatch instead of 4
  depthwise_pointwise_skip_relu: `
    struct DPSkipReluParams {
      batch: u32,
      in_channels: u32,
      out_channels: u32,
      height: u32,
      width: u32,
      dw_kernel_h: u32,  // typically 5
      dw_kernel_w: u32,  // typically 5
      dw_pad_h: u32,     // typically 2
      dw_pad_w: u32,     // typically 2
      has_dw_bias: u32,
      has_pw_bias: u32,
      has_skip: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> dw_weight: array<f32>;  // [in_channels, 1, kH, kW]
    @group(0) @binding(2) var<storage, read> dw_bias: array<f32>;    // [in_channels]
    @group(0) @binding(3) var<storage, read> pw_weight: array<f32>;  // [out_channels, in_channels, 1, 1]
    @group(0) @binding(4) var<storage, read> pw_bias: array<f32>;    // [out_channels]
    @group(0) @binding(5) var<storage, read> skip: array<f32>;       // [batch, out_channels, H, W]
    @group(0) @binding(6) var<storage, read_write> output: array<f32>;
    @group(0) @binding(7) var<uniform> params: DPSkipReluParams;

    // Shared memory for depthwise output - reused across channels
    var<workgroup> dw_cache: array<f32, 256>;  // Enough for 16x16 tile or 8x8x4 channels

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let out_c_batch = gid.z;

      let out_c = out_c_batch % params.out_channels;
      let batch = out_c_batch / params.out_channels;

      if (out_x >= params.width || out_y >= params.height || batch >= params.batch) {
        return;
      }

      // Pointwise convolution: sum over all input channels
      var pw_sum: f32 = 0.0;

      for (var ic = 0u; ic < params.in_channels; ic = ic + 1u) {
        // First compute depthwise conv for this input channel
        var dw_sum: f32 = 0.0;

        for (var kh = 0u; kh < params.dw_kernel_h; kh = kh + 1u) {
          for (var kw = 0u; kw < params.dw_kernel_w; kw = kw + 1u) {
            let in_y = i32(out_y + kh) - i32(params.dw_pad_h);
            let in_x = i32(out_x + kw) - i32(params.dw_pad_w);

            if (in_y >= 0 && in_y < i32(params.height) && in_x >= 0 && in_x < i32(params.width)) {
              let in_idx = batch * params.in_channels * params.height * params.width
                         + ic * params.height * params.width
                         + u32(in_y) * params.width
                         + u32(in_x);

              let dw_w_idx = ic * params.dw_kernel_h * params.dw_kernel_w
                           + kh * params.dw_kernel_w
                           + kw;

              dw_sum = dw_sum + input[in_idx] * dw_weight[dw_w_idx];
            }
          }
        }

        // Add depthwise bias
        if (params.has_dw_bias == 1u) {
          dw_sum = dw_sum + dw_bias[ic];
        }

        // Multiply by pointwise weight for this output channel
        let pw_w_idx = out_c * params.in_channels + ic;
        pw_sum = pw_sum + dw_sum * pw_weight[pw_w_idx];
      }

      // Add pointwise bias
      if (params.has_pw_bias == 1u) {
        pw_sum = pw_sum + pw_bias[out_c];
      }

      // Add skip connection
      if (params.has_skip == 1u) {
        let skip_idx = batch * params.out_channels * params.height * params.width
                     + out_c * params.height * params.width
                     + out_y * params.width
                     + out_x;
        pw_sum = pw_sum + skip[skip_idx];
      }

      // Apply ReLU
      pw_sum = max(0.0, pw_sum);

      let out_idx = batch * params.out_channels * params.height * params.width
                  + out_c * params.height * params.width
                  + out_y * params.width
                  + out_x;
      output[out_idx] = pw_sum;
    }
  `,

  // Bilinear resize
  resize_bilinear: `
    struct ResizeParams {
      batch: u32,
      channels: u32,
      in_height: u32,
      in_width: u32,
      out_height: u32,
      out_width: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: ResizeParams;

    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let out_x = gid.x;
      let out_y = gid.y;
      let c_batch = gid.z;

      let c = c_batch % params.channels;
      let batch = c_batch / params.channels;

      if (out_x >= params.out_width || out_y >= params.out_height || batch >= params.batch) {
        return;
      }

      // Compute source coordinates (align_corners=False)
      let scale_y = f32(params.in_height) / f32(params.out_height);
      let scale_x = f32(params.in_width) / f32(params.out_width);

      let src_y = (f32(out_y) + 0.5) * scale_y - 0.5;
      let src_x = (f32(out_x) + 0.5) * scale_x - 0.5;

      // Clamp source coordinates to valid range before computing weights
      let clamped_y = max(0.0, min(src_y, f32(params.in_height - 1u)));
      let clamped_x = max(0.0, min(src_x, f32(params.in_width - 1u)));

      let y0 = u32(floor(clamped_y));
      let x0 = u32(floor(clamped_x));
      let y1 = min(y0 + 1u, params.in_height - 1u);
      let x1 = min(x0 + 1u, params.in_width - 1u);

      // Compute interpolation weights from clamped coordinates
      let ly = clamped_y - f32(y0);
      let lx = clamped_x - f32(x0);
      let hy = 1.0 - ly;
      let hx = 1.0 - lx;

      let base = batch * params.channels * params.in_height * params.in_width
               + c * params.in_height * params.in_width;

      let v00 = input[base + y0 * params.in_width + x0];
      let v01 = input[base + y0 * params.in_width + x1];
      let v10 = input[base + y1 * params.in_width + x0];
      let v11 = input[base + y1 * params.in_width + x1];

      let val = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);

      let out_idx = batch * params.channels * params.out_height * params.out_width
                  + c * params.out_height * params.out_width
                  + out_y * params.out_width
                  + out_x;
      output[out_idx] = val;
    }
  `,
};

// ============ Shader Pipeline Helper ============

function getOrCreatePipelineSync(
  device: GPUDevice,
  shaderKey: string,
  shaderCode: string,
): GPUComputePipeline {
  if (shaderCache.has(shaderKey)) {
    return shaderCache.get(shaderKey)!;
  }

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  // Use synchronous pipeline creation for immediate availability
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  shaderCache.set(shaderKey, pipeline);
  return pipeline;
}

// Keep async version for optional pre-compilation
async function getOrCreatePipeline(
  device: GPUDevice,
  shaderKey: string,
  shaderCode: string,
): Promise<GPUComputePipeline> {
  if (shaderCache.has(shaderKey)) {
    return shaderCache.get(shaderKey)!;
  }

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });

  shaderCache.set(shaderKey, pipeline);
  return pipeline;
}

// ============ WebGPU Backend Implementation ============

export class WebGPUBackend implements TorchBackend {
  readonly name = 'webgpu';
  private device: GPUDevice | null = null;

  // Command batching state
  private batchMode = false;
  private commandEncoder: GPUCommandEncoder | null = null;
  private pendingBuffers: GPUBuffer[] = [];  // Buffers to destroy after submit

  async init(): Promise<void> {
    this.device = await getDevice();
    // Pre-compile all shader pipelines for faster first inference
    await this.warmupPipelines();
  }

  /**
   * Pre-compile all shader pipelines asynchronously
   * This avoids blocking on first use of each op
   */
  private async warmupPipelines(): Promise<void> {
    const device = this.device!;

    // All shaders we use in the model
    const shaderEntries: [string, string][] = [
      ['relu', SHADERS.relu],
      ['relu6', SHADERS.relu6],
      ['sigmoid', SHADERS.sigmoid],
      ['leaky_relu', SHADERS.leaky_relu],
      ['softmax', SHADERS.softmax],
      ['prelu', SHADERS.prelu],
      ['conv2d', SHADERS.conv2d],
      ['conv2d_1x1_tiled', SHADERS.conv2d_1x1_tiled],
      ['conv2d_1x1_vec4', SHADERS.conv2d_1x1_vec4],
      ['conv2d_fused', SHADERS.conv2d_fused],
      ['depthwise_conv2d', SHADERS.depthwise_conv2d],
      ['depthwise_conv2d_5x5', SHADERS.depthwise_conv2d_5x5],
      ['depthwise_conv2d_5x5_2x2', SHADERS.depthwise_conv2d_5x5_2x2],
      ['depthwise_conv2d_5x5_stride2', SHADERS.depthwise_conv2d_5x5_stride2],
      ['depthwise_conv2d_fused', SHADERS.depthwise_conv2d_fused],
      ['depthwise_pointwise_skip_relu', SHADERS.depthwise_pointwise_skip_relu],
      ['batch_norm', SHADERS.batch_norm],
      ['resize_bilinear', SHADERS.resize_bilinear],
      ['pad2d', SHADERS.pad2d],
      ['pad_channel', SHADERS.pad_channel],
      ['permute4d', SHADERS.permute4d],
      ['add', SHADERS.add],
      ['add_relu', SHADERS.add_relu],
      ['mul_scalar', SHADERS.mul_scalar],
      ['global_avg_pool2d', SHADERS.global_avg_pool2d],
      ['max_pool2d', SHADERS.max_pool2d],
    ];

    // Compile all in parallel using async pipeline creation
    await Promise.all(
      shaderEntries.map(async ([key, code]) => {
        await getOrCreatePipeline(device, key, code);
      })
    );
  }

  private getDevice(): GPUDevice {
    if (!this.device) {
      throw new Error('WebGPU backend not initialized. Call init() first.');
    }
    return this.device;
  }

  /**
   * Start batching mode - collects all dispatches into one command buffer
   * Call flush() when done to submit all at once
   */
  beginBatch(): void {
    if (this.batchMode) return;
    this.batchMode = true;
    this.commandEncoder = this.getDevice().createCommandEncoder();
    this.pendingBuffers = [];
  }

  /**
   * End batching mode and submit all collected commands at once
   */
  flush(): void {
    if (!this.batchMode || !this.commandEncoder) return;

    const device = this.getDevice();
    device.queue.submit([this.commandEncoder.finish()]);

    // Clean up temporary buffers after submission
    for (const buf of this.pendingBuffers) {
      buf.destroy();
    }

    this.batchMode = false;
    this.commandEncoder = null;
    this.pendingBuffers = [];
  }

  /**
   * Get a command encoder - either the batched one or create a new one
   */
  private getEncoder(): { encoder: GPUCommandEncoder; autoSubmit: boolean } {
    if (this.batchMode && this.commandEncoder) {
      return { encoder: this.commandEncoder, autoSubmit: false };
    }
    return { encoder: this.getDevice().createCommandEncoder(), autoSubmit: true };
  }

  /**
   * Submit commands or defer if batching
   */
  private submitOrDefer(encoder: GPUCommandEncoder, autoSubmit: boolean, tempBuffers: GPUBuffer[]): void {
    if (autoSubmit) {
      this.getDevice().queue.submit([encoder.finish()]);
      for (const buf of tempBuffers) {
        buf.destroy();
      }
    } else {
      // Collect buffers for later cleanup
      this.pendingBuffers.push(...tempBuffers);
    }
  }

  private asGPUTensor(tensor: Tensor): WebGPUTensor {
    if (tensor instanceof WebGPUTensor) {
      return tensor;
    }
    // Upload CPU tensor to GPU
    const f32 = tensor.data instanceof Float32Array
      ? tensor.data
      : new Float32Array(tensor.data);
    return WebGPUTensor.fromArray(f32, tensor.shape, this.getDevice());
  }

  // ============ Activation Functions ============

  relu(input: Tensor): Tensor {
    return this.runUnaryOp(input, 'relu', SHADERS.relu);
  }

  relu6(input: Tensor): Tensor {
    return this.runUnaryOp(input, 'relu6', SHADERS.relu6);
  }

  leakyRelu(input: Tensor, negativeSlope: number): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const output = WebGPUTensor.empty(input.shape, device);

    // Run shader synchronously (for now)
    this.runLeakyReluAsync(gpuInput, output, negativeSlope);
    return output;
  }

  private async runLeakyReluAsync(
    input: WebGPUTensor,
    output: WebGPUTensor,
    negativeSlope: number,
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'leaky_relu', SHADERS.leaky_relu);

    const uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([input.size]));
    device.queue.writeBuffer(uniformBuffer, 4, new Float32Array([negativeSlope]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(input.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  gelu(input: Tensor): Tensor {
    return this.runUnaryOp(input, 'gelu', SHADERS.gelu);
  }

  sigmoid(input: Tensor): Tensor {
    return this.runUnaryOp(input, 'sigmoid', SHADERS.sigmoid);
  }

  softmax(input: Tensor, dim: number): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const output = WebGPUTensor.empty(input.shape, device);

    // Run asynchronously
    this.runSoftmaxAsync(gpuInput, output, dim);
    return output;
  }

  private async runSoftmaxAsync(
    input: WebGPUTensor,
    output: WebGPUTensor,
    dim: number,
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'softmax', SHADERS.softmax);

    // Compute axis size
    const shape = input.shape;
    const axisSize = shape[dim < 0 ? shape.length + dim : dim];
    const numBatches = input.size / axisSize;

    const uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([input.size, axisSize]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(numBatches);
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  prelu(input: Tensor, weight: Tensor): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);
    const output = WebGPUTensor.empty(input.shape, device);

    this.runPreluAsync(gpuInput, gpuWeight, output);
    return output;
  }

  private async runPreluAsync(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    output: WebGPUTensor,
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'prelu', SHADERS.prelu);

    const uniformBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([input.size]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: output.buffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(input.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  // Helper for simple unary ops - uses sync pipeline creation
  private runUnaryOp(input: Tensor, key: string, shader: string): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const output = WebGPUTensor.empty(input.shape, device);

    const pipeline = getOrCreatePipelineSync(device, key, shader);

    const uniformBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([gpuInput.size]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gpuInput.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

    return output;
  }

  // ============ Convolution Ops ============

  conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);

    // Input: [N, C_in, H, W], Weight: [C_out, C_in, kH, kW]
    const [batch, inChannels, inHeight, inWidth] = input.shape;
    const [outChannels, , kernelH, kernelW] = weight.shape;

    const outHeight = Math.floor((inHeight + 2 * padding[0] - kernelH) / stride[0]) + 1;
    const outWidth = Math.floor((inWidth + 2 * padding[1] - kernelW) / stride[1]) + 1;

    const output = WebGPUTensor.empty([batch, outChannels, outHeight, outWidth], device);

    // Use optimized tiled shader for 1x1 convolutions (common in MobileNet-style architectures)
    if (kernelH === 1 && kernelW === 1 && stride[0] === 1 && stride[1] === 1 && padding[0] === 0 && padding[1] === 0) {
      this.runConv2d1x1Tiled(gpuInput, gpuWeight, bias, output, {
        batch,
        inChannels,
        outChannels,
        spatialSize: outHeight * outWidth,
      });
    } else {
      this.runConv2dAsync(gpuInput, gpuWeight, bias, output, {
        batch,
        inChannels,
        outChannels,
        inHeight,
        inWidth,
        kernelH,
        kernelW,
        strideH: stride[0],
        strideW: stride[1],
        padH: padding[0],
        padW: padding[1],
        outHeight,
        outWidth,
      });
    }

    return output;
  }

  /**
   * Tiled 1x1 convolution using shared memory for weight caching
   */
  private runConv2d1x1Tiled(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      inChannels: number;
      outChannels: number;
      spatialSize: number;
    },
  ): void {
    const device = this.getDevice();
    // Use scalar tiled shader (vec4 was slower due to NCHW gather overhead)
    const pipeline = getOrCreatePipelineSync(device, 'conv2d_1x1_tiled', SHADERS.conv2d_1x1_tiled);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.inChannels,
        params.outChannels,
        params.spatialSize,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    // Tile sizes must match the shader (8x8 workgroups, swapped x/y)
    const TILE_M = 8;  // spatial
    const TILE_N = 8;  // out_channels
    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outChannels / TILE_N),   // x: out_channel tiles (swapped for cache)
      Math.ceil(params.spatialSize / TILE_M),   // y: spatial tiles
      params.batch,                              // z: batch
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  /**
   * Fused 1x1 Conv + Skip Add + ReLU (for ResModule)
   * Single kernel dispatch instead of 3 (conv + add + relu)
   */
  conv2d1x1SkipRelu(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    skip: Tensor,
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);
    const gpuSkip = this.asGPUTensor(skip);

    const [batch, inChannels, height, width] = input.shape;
    const [outChannels] = weight.shape;
    const spatialSize = height * width;

    const output = WebGPUTensor.empty([batch, outChannels, height, width], device);

    const pipeline = getOrCreatePipelineSync(
      device,
      'conv2d_1x1_skip_relu',
      SHADERS.conv2d_1x1_skip_relu,
    );

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batch,
        inChannels,
        outChannels,
        spatialSize,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: gpuWeight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: gpuSkip.buffer } },
        { binding: 4, resource: { buffer: output.buffer } },
        { binding: 5, resource: { buffer: uniformBuffer } },
      ],
    });

    const TILE_M = 8;
    const TILE_N = 8;
    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(outChannels / TILE_N),
      Math.ceil(spatialSize / TILE_M),
      batch,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);

    return output;
  }

  private async runConv2dAsync(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      inChannels: number;
      outChannels: number;
      inHeight: number;
      inWidth: number;
      kernelH: number;
      kernelW: number;
      strideH: number;
      strideW: number;
      padH: number;
      padW: number;
      outHeight: number;
      outWidth: number;
    },
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'conv2d', SHADERS.conv2d);

    // Create bias buffer (empty if no bias)
    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 56,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.inChannels,
        params.outChannels,
        params.inHeight,
        params.inWidth,
        params.kernelH,
        params.kernelW,
        params.strideH,
        params.strideW,
        params.padH,
        params.padW,
        params.outHeight,
        params.outWidth,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / 8),
      Math.ceil(params.outHeight / 8),
      params.batch * params.outChannels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  depthwiseConv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);

    const [batch, channels, inHeight, inWidth] = input.shape;
    const [, , kernelH, kernelW] = weight.shape;

    const outHeight = Math.floor((inHeight + 2 * padding[0] - kernelH) / stride[0]) + 1;
    const outWidth = Math.floor((inWidth + 2 * padding[1] - kernelW) / stride[1]) + 1;

    const output = WebGPUTensor.empty([batch, channels, outHeight, outWidth], device);

    // Use optimized 5x5 shaders for common cases
    if (kernelH === 5 && kernelW === 5 && stride[0] === 1 && stride[1] === 1 && padding[0] === 2 && padding[1] === 2) {
      // stride=1, padding=2: same output size
      // Use 2x2 output per thread optimization for larger feature maps
      if (inHeight >= 8 && inWidth >= 8) {
        this.runDepthwiseConv2d5x5_2x2(gpuInput, gpuWeight, bias, output, {
          batch,
          channels,
          height: inHeight,
          width: inWidth,
        });
      } else {
        this.runDepthwiseConv2d5x5(gpuInput, gpuWeight, bias, output, {
          batch,
          channels,
          height: inHeight,
          width: inWidth,
        });
      }
    } else if (kernelH === 5 && kernelW === 5 && stride[0] === 2 && stride[1] === 2 && padding[0] === 0 && padding[1] === 0) {
      // stride=2, no padding: input is pre-padded by caller (asymmetric 1,2,1,2)
      this.runDepthwiseConv2d5x5Stride2(gpuInput, gpuWeight, bias, output, {
        batch,
        channels,
        inHeight,
        inWidth,
        outHeight,
        outWidth,
      });
    } else {
      this.runDepthwiseConv2dAsync(gpuInput, gpuWeight, bias, output, {
        batch,
        channels,
        inHeight,
        inWidth,
        kernelH,
        kernelW,
        strideH: stride[0],
        strideW: stride[1],
        padH: padding[0],
        padW: padding[1],
        outHeight,
        outWidth,
      });
    }

    return output;
  }

  /**
   * Optimized 5x5 depthwise convolution with unrolled loops
   */
  private runDepthwiseConv2d5x5(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      height: number;
      width: number;
    },
  ): void {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'depthwise_conv2d_5x5', SHADERS.depthwise_conv2d_5x5);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.height,
        params.width,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.width / 8),
      Math.ceil(params.height / 8),
      params.batch * params.channels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  /**
   * Optimized 5x5 depthwise convolution with 2x2 outputs per thread
   * Each thread computes 4 adjacent outputs, reducing memory accesses ~2.5x
   */
  private runDepthwiseConv2d5x5_2x2(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      height: number;
      width: number;
    },
  ): void {
    const device = this.getDevice();
    // Use scalar shader (FP16 was slower due to f32<->f16 conversion overhead)
    const pipeline = getOrCreatePipelineSync(device, 'depthwise_conv2d_5x5_2x2', SHADERS.depthwise_conv2d_5x5_2x2);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.height,
        params.width,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // Each thread handles 2x2 outputs, so dispatch half the workgroups in each dimension
    pass.dispatchWorkgroups(
      Math.ceil(params.width / 16),  // 8 threads * 2 outputs per thread = 16
      Math.ceil(params.height / 16),
      params.batch * params.channels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  /**
   * Optimized 5x5 depthwise convolution with stride=2, no padding
   * Input is pre-padded asymmetrically (1,2,1,2) by caller
   */
  private runDepthwiseConv2d5x5Stride2(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      inHeight: number;
      inWidth: number;
      outHeight: number;
      outWidth: number;
    },
  ): void {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'depthwise_conv2d_5x5_stride2', SHADERS.depthwise_conv2d_5x5_stride2);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 28,  // 7 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.inHeight,
        params.inWidth,
        params.outHeight,
        params.outWidth,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / 8),
      Math.ceil(params.outHeight / 8),
      params.batch * params.channels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  private async runDepthwiseConv2dAsync(
    input: WebGPUTensor,
    weight: WebGPUTensor,
    bias: Tensor | null,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      inHeight: number;
      inWidth: number;
      kernelH: number;
      kernelW: number;
      strideH: number;
      strideW: number;
      padH: number;
      padW: number;
      outHeight: number;
      outWidth: number;
    },
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'depthwise_conv2d', SHADERS.depthwise_conv2d);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 52,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.inHeight,
        params.inWidth,
        params.kernelH,
        params.kernelW,
        params.strideH,
        params.strideW,
        params.padH,
        params.padW,
        params.outHeight,
        params.outWidth,
        bias ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: weight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / 8),
      Math.ceil(params.outHeight / 8),
      params.batch * params.channels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);
  }

  // ============ Pooling Ops ============

  maxPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    return this.runPool2d(input, kernelSize, stride, padding, 'max_pool2d', SHADERS.max_pool2d);
  }

  avgPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
  ): Tensor {
    return this.runPool2d(input, kernelSize, stride, padding, 'avg_pool2d', SHADERS.avg_pool2d);
  }

  private runPool2d(
    input: Tensor,
    kernelSize: [number, number],
    stride: [number, number],
    padding: [number, number],
    key: string,
    shader: string,
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);

    const [batch, channels, inHeight, inWidth] = input.shape;
    const outHeight = Math.floor((inHeight + 2 * padding[0] - kernelSize[0]) / stride[0]) + 1;
    const outWidth = Math.floor((inWidth + 2 * padding[1] - kernelSize[1]) / stride[1]) + 1;

    const output = WebGPUTensor.empty([batch, channels, outHeight, outWidth], device);

    this.runPool2dAsync(gpuInput, output, {
      batch,
      channels,
      inHeight,
      inWidth,
      kernelH: kernelSize[0],
      kernelW: kernelSize[1],
      strideH: stride[0],
      strideW: stride[1],
      padH: padding[0],
      padW: padding[1],
      outHeight,
      outWidth,
    }, key, shader);

    return output;
  }

  private async runPool2dAsync(
    input: WebGPUTensor,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      inHeight: number;
      inWidth: number;
      kernelH: number;
      kernelW: number;
      strideH: number;
      strideW: number;
      padH: number;
      padW: number;
      outHeight: number;
      outWidth: number;
    },
    key: string,
    shader: string,
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, key, shader);

    const uniformBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.inHeight,
        params.inWidth,
        params.kernelH,
        params.kernelW,
        params.strideH,
        params.strideW,
        params.padH,
        params.padW,
        params.outHeight,
        params.outWidth,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / 8),
      Math.ceil(params.outHeight / 8),
      params.batch * params.channels,
    );
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  globalAvgPool2d(input: Tensor): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);

    const [batch, channels, height, width] = input.shape;
    const output = WebGPUTensor.empty([batch, channels, 1, 1], device);

    this.runGlobalAvgPool2dAsync(gpuInput, output, { batch, channels, height, width });
    return output;
  }

  private async runGlobalAvgPool2dAsync(
    input: WebGPUTensor,
    output: WebGPUTensor,
    params: { batch: number; channels: number; height: number; width: number },
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'global_avg_pool2d', SHADERS.global_avg_pool2d);

    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([params.batch, params.channels, params.height, params.width]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(params.batch * params.channels);
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
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
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuGamma = this.asGPUTensor(gamma);
    const gpuBeta = this.asGPUTensor(beta);
    const gpuMean = this.asGPUTensor(runningMean);
    const gpuVar = this.asGPUTensor(runningVar);

    const output = WebGPUTensor.empty(input.shape, device);

    const [batch, channels, height, width] = input.shape;
    this.runBatchNormAsync(gpuInput, gpuGamma, gpuBeta, gpuMean, gpuVar, output, {
      batch,
      channels,
      height,
      width,
      eps,
    });

    return output;
  }

  private async runBatchNormAsync(
    input: WebGPUTensor,
    gamma: WebGPUTensor,
    beta: WebGPUTensor,
    runningMean: WebGPUTensor,
    runningVar: WebGPUTensor,
    output: WebGPUTensor,
    params: { batch: number; channels: number; height: number; width: number; eps: number },
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'batch_norm', SHADERS.batch_norm);

    const uniformBuffer = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uniformData = new ArrayBuffer(20);
    new Uint32Array(uniformData, 0, 4).set([params.batch, params.channels, params.height, params.width]);
    new Float32Array(uniformData, 16, 1).set([params.eps]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: gamma.buffer } },
        { binding: 2, resource: { buffer: beta.buffer } },
        { binding: 3, resource: { buffer: runningMean.buffer } },
        { binding: 4, resource: { buffer: runningVar.buffer } },
        { binding: 5, resource: { buffer: output.buffer } },
        { binding: 6, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(input.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  // ============ Image Processing ============

  resizeBilinear(input: Tensor, outputSize: [number, number]): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);

    const [batch, channels, , ] = input.shape;
    const [outHeight, outWidth] = outputSize;

    const output = WebGPUTensor.empty([batch, channels, outHeight, outWidth], device);

    this.runResizeBilinearAsync(gpuInput, output, {
      batch,
      channels,
      inHeight: input.shape[2],
      inWidth: input.shape[3],
      outHeight,
      outWidth,
    });

    return output;
  }

  private async runResizeBilinearAsync(
    input: WebGPUTensor,
    output: WebGPUTensor,
    params: {
      batch: number;
      channels: number;
      inHeight: number;
      inWidth: number;
      outHeight: number;
      outWidth: number;
    },
  ): Promise<void> {
    const device = this.getDevice();
    const pipeline = getOrCreatePipelineSync(device, 'resize_bilinear', SHADERS.resize_bilinear);

    const uniformBuffer = device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        params.batch,
        params.channels,
        params.inHeight,
        params.inWidth,
        params.outHeight,
        params.outWidth,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / 8),
      Math.ceil(params.outHeight / 8),
      params.batch * params.channels,
    );
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);
  }

  // ============ Tensor Manipulation ============

  pad(
    input: Tensor,
    padding: number[],
    mode: 'constant' | 'reflect' | 'replicate' = 'constant',
    value = 0,
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const shape = gpuInput.shape;

    // Handle 2D padding (last 2 dimensions): [left, right, top, bottom]
    if (padding.length === 4 && shape.length >= 2) {
      const [padL, padR, padT, padB] = padding;
      const [n, c, h, w] = shape.length === 4 ? shape : [1, 1, ...shape.slice(-2)];
      const newH = h + padT + padB;
      const newW = w + padL + padR;
      const newShape = shape.length === 4 ? [n, c, newH, newW] : [newH, newW];
      const output = WebGPUTensor.empty(newShape, device);

      const pipeline = getOrCreatePipelineSync(device, 'pad2d', SHADERS.pad2d);

      const uniformBuffer = device.createBuffer({
        size: 36,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const uniformData = new ArrayBuffer(36);
      new Uint32Array(uniformData, 0, 8).set([n, c, h, w, newH, newW, padL, padT]);
      new Float32Array(uniformData, 32, 1).set([value]);
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gpuInput.buffer } },
          { binding: 1, resource: { buffer: output.buffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const { encoder, autoSubmit } = this.getEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(newW / 8),
        Math.ceil(newH / 8),
        n * c,
      );
      pass.end();
      this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

      return output;
    }

    // Handle channel padding [left, right, top, bottom, front, back]
    if (padding.length === 6 && shape.length >= 3) {
      const [padL, padR, padT, padB, padF, padBack] = padding;
      const [n, c, h, w] = shape.length === 4 ? shape : [1, ...shape.slice(-3)];
      const newC = c + padF + padBack;
      const newH = h + padT + padB;
      const newW = w + padL + padR;

      // For simplicity, handle the common case: only channel padding (spatial unchanged)
      if (padL === 0 && padR === 0 && padT === 0 && padB === 0) {
        const output = WebGPUTensor.empty([n, newC, h, w], device);

        const pipeline = getOrCreatePipelineSync(device, 'pad_channel', SHADERS.pad_channel);

        const uniformBuffer = device.createBuffer({
          size: 28,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const uniformData = new ArrayBuffer(28);
        new Uint32Array(uniformData, 0, 6).set([n, c, newC, h, w, padF]);
        new Float32Array(uniformData, 24, 1).set([value]);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: gpuInput.buffer } },
            { binding: 1, resource: { buffer: output.buffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
          ],
        });

        const { encoder, autoSubmit } = this.getEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil(w / 8),
          Math.ceil(h / 8),
          n * newC,
        );
        pass.end();
        this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

        return output;
      }
    }

    throw new Error(`Unsupported padding length: ${padding.length} for shape: ${shape}`);
  }

  permute(input: Tensor, dims: number[]): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const oldShape = gpuInput.shape;
    const newShape = dims.map(d => oldShape[d]);

    // Handle 4D permute on GPU
    if (oldShape.length === 4) {
      const output = WebGPUTensor.empty(newShape, device);

      const pipeline = getOrCreatePipelineSync(device, 'permute4d', SHADERS.permute4d);

      const uniformBuffer = device.createBuffer({
        size: 52,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        uniformBuffer,
        0,
        new Uint32Array([
          oldShape[0], oldShape[1], oldShape[2], oldShape[3],
          newShape[0], newShape[1], newShape[2], newShape[3],
          dims[0], dims[1], dims[2], dims[3],
          gpuInput.size,
        ]),
      );

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gpuInput.buffer } },
          { binding: 1, resource: { buffer: output.buffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });

      const { encoder, autoSubmit } = this.getEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(gpuInput.size / 256));
      pass.end();
      this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

      return output;
    }

    // For non-4D tensors, use CPU fallback (rare case, would need data)
    throw new Error(`WebGPU permute only supports 4D tensors, got ${oldShape.length}D`);
  }

  reshape(input: Tensor, newShape: number[]): Tensor {
    const gpuTensor = this.asGPUTensor(input);
    const inferIdx = newShape.indexOf(-1);
    if (inferIdx >= 0) {
      const knownSize = newShape.filter(d => d !== -1).reduce((a, b) => a * b, 1);
      const inferredSize = gpuTensor.size / knownSize;
      newShape = [...newShape];
      newShape[inferIdx] = inferredSize;
    }
    // Zero-copy reshape - create new tensor wrapper with same buffer
    return new WebGPUTensor(gpuTensor.buffer, newShape, this.device!);
  }

  squeeze(input: Tensor, dim?: number): Tensor {
    const gpuTensor = this.asGPUTensor(input);
    let newShape: number[];
    if (dim !== undefined) {
      if (gpuTensor.shape[dim] === 1) {
        newShape = [...gpuTensor.shape.slice(0, dim), ...gpuTensor.shape.slice(dim + 1)];
      } else {
        newShape = [...gpuTensor.shape];
      }
    } else {
      newShape = gpuTensor.shape.filter((d: number) => d !== 1);
    }
    // Zero-copy squeeze - create new tensor wrapper with same buffer
    return new WebGPUTensor(gpuTensor.buffer, newShape, this.device!);
  }

  add(a: Tensor, b: Tensor): Tensor {
    const device = this.getDevice();
    const tensorA = this.asGPUTensor(a);
    const tensorB = this.asGPUTensor(b);
    const output = WebGPUTensor.empty(tensorA.shape, device);

    const pipeline = getOrCreatePipelineSync(device, 'add', SHADERS.add);

    const uniformBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([tensorA.size]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorA.buffer } },
        { binding: 1, resource: { buffer: tensorB.buffer } },
        { binding: 2, resource: { buffer: output.buffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tensorA.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

    return output;
  }

  /**
   * Fused add + relu (used in ResModule skip connections)
   * Saves one GPU dispatch vs separate add() + relu()
   */
  addRelu(a: Tensor, b: Tensor): Tensor {
    const device = this.getDevice();
    const tensorA = this.asGPUTensor(a);
    const tensorB = this.asGPUTensor(b);
    const output = WebGPUTensor.empty(tensorA.shape, device);

    const pipeline = getOrCreatePipelineSync(device, 'add_relu', SHADERS.add_relu);

    const uniformBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([tensorA.size]));

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorA.buffer } },
        { binding: 1, resource: { buffer: tensorB.buffer } },
        { binding: 2, resource: { buffer: output.buffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tensorA.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

    return output;
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

  // ============ Fused Operations for Performance ============

  /**
   * Fused Conv2d + Bias + Activation
   * activation: 'none' | 'relu' | 'relu6'
   */
  conv2dFused(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
    activation: 'none' | 'relu' | 'relu6' = 'none',
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);

    const [batch, inChannels, inHeight, inWidth] = input.shape;
    const [outChannels, , kernelH, kernelW] = weight.shape;

    const outHeight = Math.floor((inHeight + 2 * padding[0] - kernelH) / stride[0]) + 1;
    const outWidth = Math.floor((inWidth + 2 * padding[1] - kernelW) / stride[1]) + 1;

    const output = WebGPUTensor.empty([batch, outChannels, outHeight, outWidth], device);

    const pipeline = getOrCreatePipelineSync(device, 'conv2d_fused', SHADERS.conv2d_fused);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const activationCode = activation === 'relu' ? 1 : activation === 'relu6' ? 2 : 0;

    const uniformBuffer = device.createBuffer({
      size: 60,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batch,
        inChannels,
        outChannels,
        inHeight,
        inWidth,
        kernelH,
        kernelW,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        outHeight,
        outWidth,
        bias ? 1 : 0,
        activationCode,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: gpuWeight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(outWidth / 8),
      Math.ceil(outHeight / 8),
      batch * outChannels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);

    return output;
  }

  /**
   * Fused Depthwise Conv2d + Bias + Activation
   */
  depthwiseConv2dFused(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | null,
    stride: [number, number],
    padding: [number, number],
    activation: 'none' | 'relu' | 'relu6' = 'none',
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuWeight = this.asGPUTensor(weight);

    const [batch, channels, inHeight, inWidth] = input.shape;
    const [, , kernelH, kernelW] = weight.shape;

    const outHeight = Math.floor((inHeight + 2 * padding[0] - kernelH) / stride[0]) + 1;
    const outWidth = Math.floor((inWidth + 2 * padding[1] - kernelW) / stride[1]) + 1;

    const output = WebGPUTensor.empty([batch, channels, outHeight, outWidth], device);

    const pipeline = getOrCreatePipelineSync(device, 'depthwise_conv2d_fused', SHADERS.depthwise_conv2d_fused);

    const gpuBias = bias
      ? this.asGPUTensor(bias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const activationCode = activation === 'relu' ? 1 : activation === 'relu6' ? 2 : 0;

    const uniformBuffer = device.createBuffer({
      size: 56,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batch,
        channels,
        inHeight,
        inWidth,
        kernelH,
        kernelW,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        outHeight,
        outWidth,
        bias ? 1 : 0,
        activationCode,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: gpuWeight.buffer } },
        { binding: 2, resource: { buffer: gpuBias.buffer } },
        { binding: 3, resource: { buffer: output.buffer } },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(outWidth / 8),
      Math.ceil(outHeight / 8),
      batch * channels,
    );
    pass.end();

    const tempBuffers = bias ? [uniformBuffer] : [uniformBuffer, gpuBias.buffer];
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);

    return output;
  }

  /**
   * Fused Depthwise Conv (5x5) + Pointwise Conv (1x1) + Skip Connection + ReLU
   * This is the full ResModule pattern - single kernel dispatch instead of 4
   *
   * @param input Input tensor [N, C_in, H, W]
   * @param dwWeight Depthwise weight [C_in, 1, kH, kW]
   * @param dwBias Depthwise bias [C_in] or null
   * @param pwWeight Pointwise weight [C_out, C_in, 1, 1]
   * @param pwBias Pointwise bias [C_out] or null
   * @param skip Skip connection tensor [N, C_out, H, W] or null
   * @param dwPadding Depthwise padding [padH, padW]
   */
  depthwisePointwiseSkipRelu(
    input: Tensor,
    dwWeight: Tensor,
    dwBias: Tensor | null,
    pwWeight: Tensor,
    pwBias: Tensor | null,
    skip: Tensor | null,
    dwPadding: [number, number],
  ): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const gpuDwWeight = this.asGPUTensor(dwWeight);
    const gpuPwWeight = this.asGPUTensor(pwWeight);

    const [batch, inChannels, height, width] = input.shape;
    const [outChannels] = pwWeight.shape;
    const [, , kernelH, kernelW] = dwWeight.shape;

    const output = WebGPUTensor.empty([batch, outChannels, height, width], device);

    const pipeline = getOrCreatePipelineSync(device, 'depthwise_pointwise_skip_relu', SHADERS.depthwise_pointwise_skip_relu);

    // Create placeholder buffers for optional inputs
    const gpuDwBias = dwBias
      ? this.asGPUTensor(dwBias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);
    const gpuPwBias = pwBias
      ? this.asGPUTensor(pwBias)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);
    const gpuSkip = skip
      ? this.asGPUTensor(skip)
      : WebGPUTensor.fromArray(new Float32Array(1), [1], device);

    const uniformBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([
        batch,
        inChannels,
        outChannels,
        height,
        width,
        kernelH,
        kernelW,
        dwPadding[0],
        dwPadding[1],
        dwBias ? 1 : 0,
        pwBias ? 1 : 0,
        skip ? 1 : 0,
      ]),
    );

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: gpuDwWeight.buffer } },
        { binding: 2, resource: { buffer: gpuDwBias.buffer } },
        { binding: 3, resource: { buffer: gpuPwWeight.buffer } },
        { binding: 4, resource: { buffer: gpuPwBias.buffer } },
        { binding: 5, resource: { buffer: gpuSkip.buffer } },
        { binding: 6, resource: { buffer: output.buffer } },
        { binding: 7, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / 8),
      Math.ceil(height / 8),
      batch * outChannels,
    );
    pass.end();

    // Collect temp buffers
    const tempBuffers: GPUBuffer[] = [uniformBuffer];
    if (!dwBias) tempBuffers.push(gpuDwBias.buffer);
    if (!pwBias) tempBuffers.push(gpuPwBias.buffer);
    if (!skip) tempBuffers.push(gpuSkip.buffer);
    this.submitOrDefer(encoder, autoSubmit, tempBuffers);

    return output;
  }

  /**
   * Multiply tensor by scalar
   */
  mulScalar(input: Tensor, scalar: number): Tensor {
    const device = this.getDevice();
    const gpuInput = this.asGPUTensor(input);
    const output = WebGPUTensor.empty(gpuInput.shape, device);

    const pipeline = getOrCreatePipelineSync(device, 'mul_scalar', SHADERS.mul_scalar);

    const uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uniformData = new ArrayBuffer(8);
    new Uint32Array(uniformData, 0, 1).set([gpuInput.size]);
    new Float32Array(uniformData, 4, 1).set([scalar]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuInput.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const { encoder, autoSubmit } = this.getEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(gpuInput.size / 256));
    pass.end();
    this.submitOrDefer(encoder, autoSubmit, [uniformBuffer]);

    return output;
  }
}

// ============ Factory Function ============

let backendInstance: WebGPUBackend | null = null;

export async function createWebGPUBackend(): Promise<WebGPUBackend> {
  if (!backendInstance) {
    backendInstance = new WebGPUBackend();
    await backendInstance.init();
  }
  return backendInstance;
}
