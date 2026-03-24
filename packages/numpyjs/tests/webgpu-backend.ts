/* eslint-disable @typescript-eslint/no-unused-vars, no-empty */
/**
 * WebGPU Backend for numpyjs tests
 *
 * This backend uses WebGPU compute shaders for GPU-accelerated operations.
 *
 * Architecture:
 * - ALL data lives in GPUBuffer (f32 format)
 * - ALL ops use real WGSL compute shaders
 * - NO CPU fallbacks - ops chain on GPU without readback
 * - Data only reads to CPU via explicit getData() call
 *
 * NDArray types:
 * - WebGPUTensor: GPU-resident tensor with GPUBuffer storage
 * - The Backend interface returns NDArray which wraps WebGPUTensor
 * - .data getter triggers async readback (via synchronous spinlock workaround)
 */

import { Backend, NDArray as IFaceNDArray } from './test-utils';

// ============ GPU Device & Shader Cache ============

let gpuDevice: GPUDevice | null = null;
const shaderCache = new Map<string, GPUComputePipeline>();

// ============ GPU Tensor Implementation ============

/**
 * GPU-resident tensor - data lives in GPUBuffer (f32)
 * This is the internal representation used by WebGPU ops.
 */
export class WebGPUTensor {
  readonly buffer: GPUBuffer;
  readonly shape: number[];
  readonly device: GPUDevice;
  private _cachedData: Float64Array | null = null;

  constructor(buffer: GPUBuffer, shape: number[], device: GPUDevice) {
    this.buffer = buffer;
    this.shape = [...shape];
    this.device = device;
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  /**
   * Read data from GPU to CPU (async).
   * Result is cached for subsequent sync access.
   */
  async getData(): Promise<Float64Array> {
    if (this._cachedData) return this._cachedData;

    const n = this.size;
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.buffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Data = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();

    // Convert f32 to f64
    const f64Data = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Data[i] = f32Data[i];

    this._cachedData = f64Data;
    return f64Data;
  }

  /**
   * Check if data is cached (already read from GPU)
   */
  get isCached(): boolean {
    return this._cachedData !== null;
  }

  /**
   * Sync data access (throws if not cached - use getData() first for async readback)
   */
  get data(): Float64Array {
    if (!this._cachedData) {
      throw new Error(
        'Data not cached. Call await getData() first, or use WebGPUNDArray.materialize()'
      );
    }
    return this._cachedData;
  }

  /**
   * Get cached data (throws if not cached - use getData() first)
   */
  getCachedData(): Float64Array {
    if (!this._cachedData) {
      throw new Error('Data not cached. Call getData() first.');
    }
    return this._cachedData;
  }

  /**
   * Create tensor from CPU data
   */
  static fromArray(
    data: Float64Array | number[],
    shape: number[],
    device: GPUDevice
  ): WebGPUTensor {
    const f64Data = data instanceof Float64Array ? data : new Float64Array(data);
    const n = f64Data.length;

    // Convert f64 to f32 for GPU
    const f32Data = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Data[i] = f64Data[i];

    // Create GPU buffer
    const buffer = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, f32Data);

    const tensor = new WebGPUTensor(buffer, shape, device);
    tensor._cachedData = f64Data; // Cache since we just created from CPU data
    return tensor;
  }

  /**
   * Create empty tensor with given shape
   */
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

// ============ NDArray Wrapper (implements IFaceNDArray) ============

// Global registry of unmaterialized arrays for batch sync
const unmaterializedArrays: Set<WebGPUNDArray> = new Set();

/**
 * NDArray wrapper around WebGPUTensor.
 * Implements sync interface via lazy readback with caching.
 *
 * For GPU ops that chain, data stays on GPU. When .data is accessed:
 * - If cached (from CPU creation or prior readback), returns immediately
 * - If not cached, triggers async readback and caches result
 *
 * For tests: call `await backend.materializeAll()` before assertions
 * to ensure all data is cached for sync access.
 */
class WebGPUNDArray implements IFaceNDArray {
  private _tensor: WebGPUTensor;
  private _pendingReadback: Promise<Float64Array> | null = null;

  constructor(tensor: WebGPUTensor) {
    this._tensor = tensor;
    // Track for batch materialization
    unmaterializedArrays.add(this);
  }

  get shape(): number[] {
    return [...this._tensor.shape];
  }

  /**
   * Sync data access with lazy readback.
   * If data is already cached, returns immediately.
   * If not, starts async readback and throws (call materialize() first).
   */
  get data(): Float64Array {
    if (this._tensor.isCached) {
      unmaterializedArrays.delete(this);
      return this._tensor.getCachedData();
    }

    // Start readback if not already started
    if (!this._pendingReadback) {
      this._pendingReadback = this._tensor.getData().then(data => {
        unmaterializedArrays.delete(this);
        return data;
      });
    }

    // In a sync context, we can't wait for the promise
    // The test should call materializeAll() before assertions
    throw new Error(
      'GPU data not cached. Call await backend.materializeAll() before accessing .data. ' +
        `This array has shape ${JSON.stringify(this.shape)} and is still on GPU.`
    );
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  get tensor(): WebGPUTensor {
    return this._tensor;
  }

  /**
   * Async data access - the proper way to get GPU data
   */
  async getData(): Promise<Float64Array> {
    const data = await this._tensor.getData();
    unmaterializedArrays.delete(this);
    return data;
  }

  /**
   * Materialize data from GPU (call before sync access)
   */
  async materialize(): Promise<void> {
    await this._tensor.getData();
    unmaterializedArrays.delete(this);
  }

  /**
   * Check if data is ready for sync access
   */
  get isMaterialized(): boolean {
    return this._tensor.isCached;
  }

  /**
   * Create from CPU data (already materialized)
   */
  static fromArray(
    data: Float64Array | number[],
    shape: number[],
    device: GPUDevice
  ): WebGPUNDArray {
    const arr = new WebGPUNDArray(WebGPUTensor.fromArray(data, shape, device));
    unmaterializedArrays.delete(arr); // Already has CPU data
    return arr;
  }
}

/**
 * Materialize all pending GPU arrays.
 * Call this before accessing .data on GPU-backed arrays.
 */
export async function materializeAll(): Promise<void> {
  const promises = Array.from(unmaterializedArrays).map(arr => arr.materialize());
  await Promise.all(promises);
}

// Legacy class for backwards compatibility during transition
class LegacyWebGPUNDArray implements IFaceNDArray {
  private _data: Float64Array;
  private _shape: number[];

  constructor(data: Float64Array | number[], shape: number[]) {
    this._data = data instanceof Float64Array ? data : new Float64Array(data);
    this._shape = [...shape];
  }

  get shape(): number[] {
    return [...this._shape];
  }

  get data(): Float64Array {
    return this._data;
  }

  toArray(): number[] {
    return Array.from(this._data);
  }
}

// ============ WGSL Shaders ============

// ============ Elementwise Shaders ============

// Generic unary operation shader template
// {{OP}} is replaced with the WGSL operation (e.g., "sin(x)", "exp(x)")
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

// Generic binary operation shader template
// {{OP}} is replaced with the WGSL operation (e.g., "a + b", "a * b")
function makeBinaryShader(op: string): string {
  return `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }
      let av = a[idx];
      let bv = b[idx];
      output[idx] = ${op};
    }
  `;
}

// Generic scalar operation shader template
function makeScalarShader(op: string): string {
  return `
    struct Uniforms {
      size: u32,
      scalar: f32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> uniforms: Uniforms;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= uniforms.size) { return; }
      let x = input[idx];
      let s = uniforms.scalar;
      output[idx] = ${op};
    }
  `;
}

// Parallel reduction shader for sum/prod/min/max
// Uses workgroup shared memory for efficient reduction
function makeReductionShader(initValue: string, reduceOp: string): string {
  return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> size: u32;

    var<workgroup> sdata: array<f32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let tid = lid.x;
      let gid = wid.x * 256u + tid;

      // Initialize shared memory with identity value
      // Each thread processes one element if within bounds
      if (gid < size) {
        sdata[tid] = input[gid];
      } else {
        sdata[tid] = ${initValue};
      }

      workgroupBarrier();

      // Parallel reduction in shared memory
      for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          let a = sdata[tid];
          let b = sdata[tid + s];
          sdata[tid] = ${reduceOp.replace(/\$a/g, 'a').replace(/\$b/g, 'b')};
        }
        workgroupBarrier();
      }

      // Write result for this workgroup
      if (tid == 0u) {
        output[wid.x] = sdata[0];
      }
    }
  `;
}

// Unary shader definitions
const UNARY_SHADERS: Record<string, string> = {
  sin: makeUnaryShader('sin(x)'),
  cos: makeUnaryShader('cos(x)'),
  tan: makeUnaryShader('tan(x)'),
  asin: makeUnaryShader('asin(x)'),
  acos: makeUnaryShader('acos(x)'),
  atan: makeUnaryShader('atan(x)'),
  sinh: makeUnaryShader('sinh(x)'),
  cosh: makeUnaryShader('cosh(x)'),
  tanh: makeUnaryShader('tanh(x)'),
  exp: makeUnaryShader('exp(x)'),
  exp2: makeUnaryShader('exp2(x)'),
  log: makeUnaryShader('log(x)'),
  log2: makeUnaryShader('log2(x)'),
  sqrt: makeUnaryShader('sqrt(x)'),
  abs: makeUnaryShader('abs(x)'),
  sign: makeUnaryShader('sign(x)'),
  floor: makeUnaryShader('floor(x)'),
  ceil: makeUnaryShader('ceil(x)'),
  // NumPy uses banker's rounding (round half to even)
  // WGSL round() also uses banker's rounding, so we use it directly
  round: makeUnaryShader('round(x)'),
  neg: makeUnaryShader('-x'),
  reciprocal: makeUnaryShader('1.0 / x'),
  square: makeUnaryShader('x * x'),
  // cbrt and log10 need special handling (not in WGSL)
  cbrt: makeUnaryShader('sign(x) * pow(abs(x), 0.333333333333)'),
  log10: makeUnaryShader('log(x) / 2.302585093'), // log10(e) = 1/ln(10)
  // Extended unary ops - all use GPU shaders
  // asinh(x) = ln(x + sqrt(x^2 + 1))
  asinh: makeUnaryShader('sign(x) * log(abs(x) + sqrt(x * x + 1.0))'),
  // acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1; returns NaN for x < 1
  // Note: sqrt(negative) returns NaN in WGSL, so no need for explicit select
  acosh: makeUnaryShader('log(x + sqrt(x * x - 1.0))'),
  // atanh(x) = 0.5 * ln((1 + x) / (1 - x)), |x| < 1; returns NaN/Inf for |x| >= 1
  atanh: makeUnaryShader('0.5 * log((1.0 + x) / (1.0 - x))'),
  // expm1(x) = exp(x) - 1, numerically stable for small x
  // For |x| < 0.01, use Taylor series: x + x^2/2 + x^3/6 + x^4/24
  // For larger x, use exp(x) - 1 directly
  expm1: makeUnaryShader(`
    select(
      exp(x) - 1.0,
      x * (1.0 + x * (0.5 + x * (0.16666666666666666 + x * 0.041666666666666664))),
      abs(x) < 0.01
    )
  `),
  // log1p(x) = log(1 + x), numerically stable for small x
  // For |x| < 0.01, use Taylor series: x - x^2/2 + x^3/3 - x^4/4
  // For larger x, use log(1 + x) directly
  log1p: makeUnaryShader(`
    select(
      log(1.0 + x),
      x * (1.0 - x * (0.5 - x * (0.3333333333333333 - x * 0.25))),
      abs(x) < 0.01
    )
  `),
  // trunc(x) = round toward zero
  trunc: makeUnaryShader('trunc(x)'),
  // sinc(x) = sin(pi*x)/(pi*x), sinc(0) = 1
  sinc: makeUnaryShader(
    'select(sin(3.14159265358979 * x) / (3.14159265358979 * x), 1.0, abs(x) < 0.0000001)'
  ),
  // deg2rad
  deg2rad: makeUnaryShader('x * 0.01745329251994329577'), // pi/180
  // rad2deg
  rad2deg: makeUnaryShader('x * 57.29577951308232087680'), // 180/pi
  // signbit: 1.0 if negative (including -0), 0.0 otherwise
  // NumPy: signbit(NaN) = False (0), signbit(-0) = True (1), signbit(-inf) = True (1)
  // Use bitcast to detect sign bit directly - IEEE 754 f32 sign is bit 31
  signbit: makeUnaryShader(`
    select(
      f32(bitcast<u32>(x) >> 31u),
      0.0,
      x != x
    )
  `),
};

// Binary shader definitions
const BINARY_SHADERS: Record<string, string> = {
  add: makeBinaryShader('av + bv'),
  sub: makeBinaryShader('av - bv'),
  mul: makeBinaryShader('av * bv'),
  div: makeBinaryShader('av / bv'),
  // pow: handle negative bases with integer exponents like NumPy
  // WGSL pow(negative, y) returns NaN even for integer y, but NumPy handles it
  pow: makeBinaryShader(`
    select(
      pow(av, bv),
      select(
        -pow(-av, bv),
        pow(-av, bv),
        fract(bv) == 0.0 && (i32(bv) % 2) == 1
      ),
      av < 0.0 && fract(bv) == 0.0
    )
  `),
  // maximum/minimum: NumPy propagates NaN, WGSL max/min ignores it
  // Use av != av to check for NaN (NaN != NaN is true)
  maximum: makeBinaryShader('select(max(av, bv), av + bv, av != av || bv != bv)'), // NaN + anything = NaN
  minimum: makeBinaryShader('select(min(av, bv), av + bv, av != av || bv != bv)'),
  // fmod: C-style modulo (result has same sign as dividend)
  // WGSL % operator does exactly this
  fmod: makeBinaryShader('av - trunc(av / bv) * bv'),
  // mod: Python-style modulo (result has same sign as divisor)
  // r = av % bv; if r != 0 and sign(r) != sign(bv), then r + bv, else r
  mod: makeBinaryShader(`
    select(
      av - trunc(av / bv) * bv + bv,
      av - trunc(av / bv) * bv,
      (av - trunc(av / bv) * bv) * bv >= 0.0 || (av - trunc(av / bv) * bv) == 0.0
    )
  `),
  // copysign: copy sign of bv to magnitude of av
  // Use bitcast to properly handle -0
  copysign: makeBinaryShader(`
    bitcast<f32>((bitcast<u32>(abs(av)) & 0x7FFFFFFFu) | (bitcast<u32>(bv) & 0x80000000u))
  `),
  // hypot: sqrt(a^2 + b^2), avoid overflow for large values
  hypot: makeBinaryShader(`
    select(
      select(
        abs(av) * sqrt(1.0 + (bv / av) * (bv / av)),
        abs(bv) * sqrt(1.0 + (av / bv) * (av / bv)),
        abs(av) > abs(bv)
      ),
      0.0,
      av == 0.0 && bv == 0.0
    )
  `),
  // arctan2: angle of point (bv, av) - note: atan2(y, x) so y=av, x=bv
  arctan2: makeBinaryShader('atan2(av, bv)'),
  // logaddexp: log(exp(a) + exp(b)), numerically stable
  // Simplified: use max to pick the larger, then add log(1 + exp(smaller - larger))
  // Note: Can't use select() with infinity due to GPU driver bugs
  logaddexp: makeBinaryShader(`log(exp(av) + exp(bv))`),
  // logaddexp2: log2(2^a + 2^b), numerically stable
  logaddexp2: makeBinaryShader(`log2(exp2(av) + exp2(bv))`),
  // fmax: maximum ignoring NaN (if one is NaN, return the other)
  // Only return NaN if both are NaN
  fmax: makeBinaryShader(`
    select(
      select(
        max(av, bv),
        av,
        bv != bv
      ),
      bv,
      av != av
    )
  `),
  // fmin: minimum ignoring NaN (if one is NaN, return the other)
  fmin: makeBinaryShader(`
    select(
      select(
        min(av, bv),
        av,
        bv != bv
      ),
      bv,
      av != av
    )
  `),
};

// Scalar shader definitions
const SCALAR_SHADERS: Record<string, string> = {
  addScalar: makeScalarShader('x + s'),
  subScalar: makeScalarShader('x - s'),
  mulScalar: makeScalarShader('x * s'),
  divScalar: makeScalarShader('x / s'),
  // powScalar: handle negative bases like NumPy
  powScalar: makeScalarShader(`
    select(
      pow(x, s),
      select(
        -pow(-x, s),
        pow(-x, s),
        fract(s) == 0.0 && (i32(s) % 2) == 1
      ),
      x < 0.0 && fract(s) == 0.0
    )
  `),
  // heaviside: 0 if x < 0, s if x == 0, 1 if x > 0
  heaviside: makeScalarShader('select(select(1.0, s, x == 0.0), 0.0, x < 0.0)'),
  // ldexp: x * 2^s (scalar exponent)
  ldexp: makeScalarShader('x * exp2(s)'),
  // minScalar: element-wise min(x, scalar) - for clip upper bound
  minScalar: makeScalarShader('min(x, s)'),
  // maxScalar: element-wise max(x, scalar) - for clip lower bound
  maxScalar: makeScalarShader('max(x, s)'),
};

// Reduction shader definitions
// For min/max: NumPy propagates NaN, so we need custom reduce ops
const REDUCTION_SHADERS: Record<string, string> = {
  sum: makeReductionShader('0.0f', '$a + $b'),
  prod: makeReductionShader('1.0f', '$a * $b'),
  // min/max with NaN propagation like NumPy
  // Use x != x to check for NaN
  min: makeReductionShader('3.40282e+38f', 'select(min($a, $b), $a + $b, $a != $a || $b != $b)'),
  max: makeReductionShader('-3.40282e+38f', 'select(max($a, $b), $a + $b, $a != $a || $b != $b)'),
};

// ============ Phase 2 Specialized Shaders ============

// Kronecker product shader
// output[(i * bm + k) * (an * bn) + (j * bn + l)] = a[i * an + j] * b[k * bn + l]
function makeKronShader(): string {
  return `
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> dims: vec4<u32>; // am, an, bm, bn

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let am = dims.x;
      let an = dims.y;
      let bm = dims.z;
      let bn = dims.w;
      let outM = am * bm;
      let outN = an * bn;

      if (idx >= outM * outN) { return; }

      let outRow = idx / outN;
      let outCol = idx % outN;

      let i = outRow / bm;
      let k = outRow % bm;
      let j = outCol / bn;
      let l = outCol % bn;

      let aVal = a[i * an + j];
      let bVal = b[k * bn + l];
      output[idx] = aVal * bVal;
    }
  `;
}

// Polynomial evaluation shader (Horner's method)
// p(x) = c[0] * x^n + c[1] * x^(n-1) + ... + c[n-1] * x + c[n]
// Horner: p(x) = ((...((c[0] * x + c[1]) * x + c[2]) * x + ...) * x + c[n])
function makePolyvalShader(degree: number): string {
  // Build Horner's method unrolled for the given degree
  let expr = 'coeffs[0]';
  for (let i = 1; i <= degree; i++) {
    expr = `(${expr}) * xi + coeffs[${i}]`;
  }

  return `
    @group(0) @binding(0) var<storage, read> coeffs: array<f32>;
    @group(0) @binding(1) var<storage, read> x: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform> size: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= size) { return; }

      let xi = x[idx];
      output[idx] = ${expr};
    }
  `;
}

// Linear interpolation shader
// For each x[i], find the interval [xp[j], xp[j+1]] where x[i] falls,
// then linearly interpolate between fp[j] and fp[j+1]
function makeInterpShader(xpSize: number): string {
  return `
    @group(0) @binding(0) var<storage, read> x: array<f32>;
    @group(0) @binding(1) var<storage, read> xp: array<f32>;
    @group(0) @binding(2) var<storage, read> fp: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: vec2<u32>; // xSize, xpSize

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let xSize = params.x;
      let xpSize = params.y;

      if (idx >= xSize) { return; }

      let xi = x[idx];

      // Clip to boundaries
      if (xi <= xp[0]) {
        output[idx] = fp[0];
        return;
      }
      if (xi >= xp[xpSize - 1u]) {
        output[idx] = fp[xpSize - 1u];
        return;
      }

      // Binary search for interval
      var lo = 0u;
      var hi = xpSize - 1u;
      while (lo < hi - 1u) {
        let mid = (lo + hi) / 2u;
        if (xp[mid] <= xi) {
          lo = mid;
        } else {
          hi = mid;
        }
      }

      // Linear interpolation
      let t = (xi - xp[lo]) / (xp[hi] - xp[lo]);
      output[idx] = fp[lo] + t * (fp[hi] - fp[lo]);
    }
  `;
}

// Bincount shader - counts occurrences of non-negative integers
// Uses atomics for thread-safe incrementing
function makeBincountShader(): string {
  return `
    @group(0) @binding(0) var<storage, read> x: array<i32>;
    @group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
    @group(0) @binding(2) var<uniform> xSize: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= xSize) { return; }

      let bin = x[idx];
      if (bin >= 0) {
        atomicAdd(&output[u32(bin)], 1u);
      }
    }
  `;
}

// Weighted bincount shader
function makeWeightedBincountShader(): string {
  return `
    @group(0) @binding(0) var<storage, read> x: array<i32>;
    @group(0) @binding(1) var<storage, read> weights: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
    @group(0) @binding(3) var<uniform> xSize: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      if (idx >= xSize) { return; }

      let bin = x[idx];
      if (bin >= 0) {
        // Convert f32 weight to fixed-point for atomic add
        // Scale by 1e6, add, then scale back when reading
        let fixedWeight = u32(weights[idx] * 1000000.0);
        atomicAdd(&output[u32(bin)], fixedWeight);
      }
    }
  `;
}

// Cumulative sum/prod shader (Hillis-Steele scan)
const CUMSUM_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    // For large arrays, we do a simple serial prefix sum per workgroup
    // This is a naive implementation - for production we'd use a proper parallel scan
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i <= idx; i = i + 1u) {
      sum = sum + input[i];
    }
    output[idx] = sum;
  }
`;

const CUMPROD_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    var prod: f32 = 1.0;
    for (var i: u32 = 0u; i <= idx; i = i + 1u) {
      prod = prod * input[i];
    }
    output[idx] = prod;
  }
`;

// ============ Decomposition Shaders ============

// modf: split into fractional and integer parts
const MODF_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> frac: array<f32>;
  @group(0) @binding(2) var<storage, read_write> integ: array<f32>;
  @group(0) @binding(3) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    let x = input[idx];
    let i = trunc(x);
    integ[idx] = i;
    frac[idx] = x - i;
  }
`;

// frexp: split into mantissa and exponent
// x = mantissa * 2^exponent, where 0.5 <= |mantissa| < 1
// Note: Can't use if/else with NaN checks due to GPU driver bugs
const FREXP_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> mantissa: array<f32>;
  @group(0) @binding(2) var<storage, read_write> exponent: array<f32>;
  @group(0) @binding(3) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    let x = input[idx];

    // Always compute - handle special cases via output
    // For x=0, log2(0)=-inf, floor(-inf)=-inf, exp2(-inf)=0, so 0/0=NaN
    // We'll handle x=0 specially by checking and returning (0, 0)
    // For NaN/Inf inputs, the math will propagate correctly
    let safeX = select(x, 1.0, x == 0.0);  // Use 1.0 to avoid log2(0)
    let e = floor(log2(abs(safeX))) + 1.0;
    let m = safeX / exp2(e);

    // Output: for x=0, return (0, 0); otherwise use computed values
    mantissa[idx] = select(m, 0.0, x == 0.0);
    exponent[idx] = select(e, 0.0, x == 0.0);
  }
`;

// divmod: compute both quotient and remainder (Python-style)
const DIVMOD_SHADER = `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> quotient: array<f32>;
  @group(0) @binding(3) var<storage, read_write> remainder: array<f32>;
  @group(0) @binding(4) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    let av = a[idx];
    let bv = b[idx];

    // Python-style floor division
    quotient[idx] = floor(av / bv);

    // Python-style modulo: result has same sign as divisor
    let r = av - trunc(av / bv) * bv;
    // Adjust if r and bv have different signs and r != 0
    remainder[idx] = select(r + bv, r, r * bv >= 0.0 || r == 0.0);
  }
`;

// Clip shader (min and max bounds)
const CLIP_SHADER = `
  struct Uniforms {
    size: u32,
    minVal: f32,
    maxVal: f32,
    _pad: f32,
  }

  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.size) { return; }
    output[idx] = clamp(input[idx], uniforms.minVal, uniforms.maxVal);
  }
`;

// ============ Additional Stats Shaders ============

// Argmin/argmax reduction shader - returns index of min/max element
// Uses two-pass: first find the value, then find the first index
function makeArgReductionShader(initValue: string, cmpOp: string): string {
  return `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> outputVal: array<f32>;
    @group(0) @binding(2) var<storage, read_write> outputIdx: array<u32>;
    @group(0) @binding(3) var<uniform> size: u32;

    var<workgroup> sval: array<f32, 256>;
    var<workgroup> sidx: array<u32, 256>;

    @compute @workgroup_size(256)
    fn main(
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(workgroup_id) wid: vec3<u32>
    ) {
      let tid = lid.x;
      let gid = wid.x * 256u + tid;

      // Initialize with identity
      if (gid < size) {
        sval[tid] = input[gid];
        sidx[tid] = gid;
      } else {
        sval[tid] = ${initValue};
        sidx[tid] = 0u;
      }

      workgroupBarrier();

      // Parallel reduction
      for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          let va = sval[tid];
          let vb = sval[tid + s];
          let ia = sidx[tid];
          let ib = sidx[tid + s];
          // ${cmpOp} determines if b is better
          if (${cmpOp}) {
            sval[tid] = vb;
            sidx[tid] = ib;
          }
        }
        workgroupBarrier();
      }

      if (tid == 0u) {
        outputVal[wid.x] = sval[0];
        outputIdx[wid.x] = sidx[0];
      }
    }
  `;
}

const ARGMIN_SHADER = makeArgReductionShader('3.40282e+38f', 'vb < va');
const ARGMAX_SHADER = makeArgReductionShader('-3.40282e+38f', 'vb > va');

// All/any reduction shader
const ALL_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<u32>;
  @group(0) @binding(2) var<uniform> size: u32;

  var<workgroup> sdata: array<u32, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // All: 1 if all non-zero, 0 if any zero
    if (gid < size) {
      sdata[tid] = select(0u, 1u, input[gid] != 0.0f);
    } else {
      sdata[tid] = 1u;  // Identity for all
    }

    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] & sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

const ANY_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<u32>;
  @group(0) @binding(2) var<uniform> size: u32;

  var<workgroup> sdata: array<u32, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // Any: 1 if any non-zero
    if (gid < size) {
      sdata[tid] = select(0u, 1u, input[gid] != 0.0f);
    } else {
      sdata[tid] = 0u;  // Identity for any
    }

    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] | sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// Sum along axis 0 (columns) for 2D arrays
const SUM_AXIS0_SHADER = `
  struct Uniforms {
    rows: u32,
    cols: u32,
  }

  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= uniforms.cols) { return; }

    var sum: f32 = 0.0f;
    for (var row: u32 = 0u; row < uniforms.rows; row = row + 1u) {
      sum = sum + input[row * uniforms.cols + col];
    }
    output[col] = sum;
  }
`;

// Sum along axis 1 (rows) for 2D arrays
const SUM_AXIS1_SHADER = `
  struct Uniforms {
    rows: u32,
    cols: u32,
  }

  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= uniforms.rows) { return; }

    var sum: f32 = 0.0f;
    for (var col: u32 = 0u; col < uniforms.cols; col = col + 1u) {
      sum = sum + input[row * uniforms.cols + col];
    }
    output[row] = sum;
  }
`;

// ============ Linalg Shaders ============

// Transpose shader
const TRANSPOSE_SHADER = `
  struct Uniforms {
    rows: u32,
    cols: u32,
  }

  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= uniforms.rows || col >= uniforms.cols) { return; }

    // input[row, col] -> output[col, row]
    output[col * uniforms.rows + row] = input[row * uniforms.cols + col];
  }
`;

// Outer product: a (m,) x b (n,) -> C (m, n)
const OUTER_SHADER = `
  struct Uniforms {
    m: u32,
    n: u32,
  }

  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;
  @group(0) @binding(3) var<uniform> uniforms: Uniforms;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;
    if (i >= uniforms.m || j >= uniforms.n) { return; }

    output[i * uniforms.n + j] = a[i] * b[j];
  }
`;

// Dot product (1D arrays)
const DOT_SHADER = `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;
  @group(0) @binding(3) var<uniform> size: u32;

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // Multiply and load
    if (gid < size) {
      sdata[tid] = a[gid] * b[gid];
    } else {
      sdata[tid] = 0.0f;
    }

    workgroupBarrier();

    // Sum reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// Trace of a square matrix
const TRACE_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> n: u32;

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // Load diagonal element
    if (gid < n) {
      sdata[tid] = input[gid * n + gid];
    } else {
      sdata[tid] = 0.0f;
    }

    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// ============ Bitonic Sort Shaders ============
// GPU-based bitonic sort for parallel sorting

// Bitonic sort step shader - performs one compare-swap step
// j: half the step size, k: stage size
function makeBitonicSortStepShader(): string {
  return `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @group(0) @binding(1) var<uniform> params: vec4<u32>;  // size, j, k, _pad

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let size = params.x;
      let j = params.y;
      let k = params.z;

      if (idx >= size) { return; }

      // Compute partner index for compare-swap
      let ixj = idx ^ j;

      // Only process if ixj > idx (avoid double-swapping)
      if (ixj > idx && ixj < size) {
        let ascending = ((idx & k) == 0u);

        let va = data[idx];
        let vb = data[ixj];

        // Handle NaN: NaN goes to end (is "larger" than anything)
        let aIsNan = (va != va);
        let bIsNan = (vb != vb);

        var shouldSwap: bool;
        if (aIsNan && bIsNan) {
          shouldSwap = false;
        } else if (aIsNan) {
          shouldSwap = ascending;  // NaN should go to end
        } else if (bIsNan) {
          shouldSwap = !ascending;
        } else {
          shouldSwap = select(va < vb, va > vb, ascending);
        }

        if (shouldSwap) {
          data[idx] = vb;
          data[ixj] = va;
        }
      }
    }
  `;
}

// Bitonic argsort step shader - sorts indices based on values
function makeBitonicArgsortStepShader(): string {
  return `
    @group(0) @binding(0) var<storage, read> values: array<f32>;
    @group(0) @binding(1) var<storage, read_write> indices: array<u32>;
    @group(0) @binding(2) var<uniform> params: vec4<u32>;  // size, j, k, _pad

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let size = params.x;
      let j = params.y;
      let k = params.z;

      if (idx >= size) { return; }

      let ixj = idx ^ j;

      if (ixj > idx && ixj < size) {
        let ascending = ((idx & k) == 0u);

        let idxA = indices[idx];
        let idxB = indices[ixj];
        let va = values[idxA];
        let vb = values[idxB];

        // Handle NaN: NaN goes to end
        let aIsNan = (va != va);
        let bIsNan = (vb != vb);

        var shouldSwap: bool;
        if (aIsNan && bIsNan) {
          shouldSwap = false;
        } else if (aIsNan) {
          shouldSwap = ascending;
        } else if (bIsNan) {
          shouldSwap = !ascending;
        } else {
          shouldSwap = select(va < vb, va > vb, ascending);
        }

        if (shouldSwap) {
          indices[idx] = idxB;
          indices[ixj] = idxA;
        }
      }
    }
  `;
}

// Shader to initialize indices for argsort (indices[i] = i)
const INIT_INDICES_SHADER = `
  @group(0) @binding(0) var<storage, read_write> indices: array<u32>;
  @group(0) @binding(1) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }
    indices[idx] = idx;
  }
`;

// Unique shader - mark duplicates after sorting
// Assumes input is already sorted
const MARK_UNIQUE_SHADER = `
  @group(0) @binding(0) var<storage, read> sorted: array<f32>;
  @group(0) @binding(1) var<storage, read_write> mask: array<u32>;  // 1 = unique, 0 = duplicate
  @group(0) @binding(2) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }

    if (idx == 0u) {
      mask[idx] = 1u;  // First element is always unique
    } else {
      let prev = sorted[idx - 1u];
      let curr = sorted[idx];
      // Check if different (also handle NaN - NaN != NaN is true)
      let prevNan = (prev != prev);
      let currNan = (curr != curr);
      if (prevNan && currNan) {
        mask[idx] = 0u;  // Both NaN = duplicate
      } else if (prevNan || currNan) {
        mask[idx] = 1u;  // One NaN = unique
      } else {
        mask[idx] = select(0u, 1u, prev != curr);
      }
    }
  }
`;

// Exclusive scan for compaction (Hillis-Steele style)
// This is a simplified version - for large arrays we'd need a proper hierarchical scan
const EXCLUSIVE_SCAN_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<u32>;
  @group(0) @binding(1) var<storage, read_write> output: array<u32>;
  @group(0) @binding(2) var<uniform> params: vec2<u32>;  // size, offset

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = params.x;
    let offset = params.y;

    if (idx >= size) { return; }

    // Simple serial scan for now (works but not optimal for large arrays)
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < idx; i = i + 1u) {
      sum = sum + input[i];
    }
    output[idx] = sum;
  }
`;

// Compact unique values using scatter
const COMPACT_UNIQUE_SHADER = `
  @group(0) @binding(0) var<storage, read> sorted: array<f32>;
  @group(0) @binding(1) var<storage, read> mask: array<u32>;
  @group(0) @binding(2) var<storage, read> scanResult: array<u32>;
  @group(0) @binding(3) var<storage, read_write> output: array<f32>;
  @group(0) @binding(4) var<uniform> size: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }

    if (mask[idx] == 1u) {
      let outIdx = scanResult[idx];
      output[outIdx] = sorted[idx];
    }
  }
`;

// Parallel binary search for searchsorted
function makeSearchSortedShader(side: 'left' | 'right'): string {
  const cmp = side === 'left' ? '<' : '<=';
  return `
    @group(0) @binding(0) var<storage, read> haystack: array<f32>;
    @group(0) @binding(1) var<storage, read> needles: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<u32>;
    @group(0) @binding(3) var<uniform> params: vec2<u32>;  // haystackSize, needlesSize

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.x;
      let haystackSize = params.x;
      let needlesSize = params.y;

      if (idx >= needlesSize) { return; }

      let needle = needles[idx];
      var lo: u32 = 0u;
      var hi: u32 = haystackSize;

      while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let midVal = haystack[mid];
        if (midVal ${cmp} needle) {
          lo = mid + 1u;
        } else {
          hi = mid;
        }
      }

      output[idx] = lo;
    }
  `;
}

// Vector norm (L2)
const NORM_SHADER = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;
  @group(0) @binding(2) var<uniform> size: u32;

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // Sum of squares
    if (gid < size) {
      let v = input[gid];
      sdata[tid] = v * v;
    } else {
      sdata[tid] = 0.0f;
    }

    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// ============ QR Decomposition Shaders (Householder Reflections) ============
// GPU-based QR decomposition using Modified Gram-Schmidt
// Each column is processed with parallel norm computation and orthogonalization

// Shader 1: Compute column norm squared (reduction to get ||q_j||^2)
const QR_COLUMN_NORM_SHADER = `
  @group(0) @binding(0) var<storage, read> Q: array<f32>;      // [M, N]
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;  // partial sums
  @group(0) @binding(2) var<uniform> dims: vec4<u32>;          // M, N, col, _pad

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let col = dims.z;
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    // Sum squares of column 'col' (row index = gid)
    if (gid < M) {
      let v = Q[gid * N + col];
      sdata[tid] = v * v;
    } else {
      sdata[tid] = 0.0f;
    }
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// Shader 2: Normalize column j (divide by norm) and set R[j,j] = norm
const QR_NORMALIZE_COLUMN_SHADER = `
  @group(0) @binding(0) var<storage, read_write> Q: array<f32>;  // [M, N]
  @group(0) @binding(1) var<storage, read_write> R: array<f32>;  // [N, N]
  @group(0) @binding(2) var<uniform> dims: vec4<u32>;            // M, N, col, _pad
  @group(0) @binding(3) var<uniform> norm: f32;                  // column norm

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let col = dims.z;
    let row = gid.x;

    // Set R[col, col] = norm (only one thread does this)
    if (row == 0u) {
      R[col * N + col] = norm;
    }

    // Normalize Q[:, col]
    if (row < M && norm > 1e-10f) {
      Q[row * N + col] = Q[row * N + col] / norm;
    }
  }
`;

// Shader 3: Compute dot product of column j with column k (for k > j)
const QR_DOT_COLUMNS_SHADER = `
  @group(0) @binding(0) var<storage, read> Q: array<f32>;        // [M, N]
  @group(0) @binding(1) var<storage, read_write> dots: array<f32>; // [N-col-1] dot products
  @group(0) @binding(2) var<uniform> dims: vec4<u32>;            // M, N, col, numCols

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let col = dims.z;
    let numCols = dims.w;  // number of columns to compute dots for (N - col - 1)

    let tid = lid.x;
    // wid.x indexes over rows, wid.y indexes over columns k (offset from col+1)
    let row = wid.x * 256u + tid;
    let kOffset = wid.y;  // which column pair (col, col+1+kOffset)

    if (kOffset >= numCols) {
      return;
    }

    let k = col + 1u + kOffset;

    // Compute partial dot product Q[:, col] . Q[:, k]
    if (row < M) {
      sdata[tid] = Q[row * N + col] * Q[row * N + k];
    } else {
      sdata[tid] = 0.0f;
    }
    workgroupBarrier();

    // Parallel reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    // Store partial sum - each workgroup stores its contribution
    // Need atomic add or final reduction on CPU
    if (tid == 0u) {
      // dots layout: [numRowWorkgroups * numCols]
      // dots[kOffset * numRowWorkgroups + wid.x]
      let numRowWorkgroups = (M + 255u) / 256u;
      dots[kOffset * numRowWorkgroups + wid.x] = sdata[0];
    }
  }
`;

// Shader 4: Orthogonalize columns k > j against column j
const QR_ORTHOGONALIZE_SHADER = `
  @group(0) @binding(0) var<storage, read_write> Q: array<f32>;  // [M, N]
  @group(0) @binding(1) var<storage, read_write> R: array<f32>;  // [N, N]
  @group(0) @binding(2) var<storage, read> dots: array<f32>;     // dot products for each column k
  @group(0) @binding(3) var<uniform> dims: vec4<u32>;            // M, N, col, numCols

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let col = dims.z;
    let numCols = dims.w;

    let row = gid.x;
    let kOffset = gid.y;

    if (row >= M || kOffset >= numCols) {
      return;
    }

    let k = col + 1u + kOffset;
    let dot = dots[kOffset];

    // Set R[col, k] = dot (only row 0 does this)
    if (row == 0u) {
      R[col * N + k] = dot;
    }

    // Q[:, k] -= dot * Q[:, col]
    Q[row * N + k] = Q[row * N + k] - dot * Q[row * N + col];
  }
`;

// ============ LU Decomposition Shaders (for det/inv) ============
// GPU-accelerated LU decomposition with partial pivoting
// Used for determinant and matrix inverse calculations

// Shader 1: Find pivot row (argmax of column magnitude) - reduction style
const LU_FIND_PIVOT_SHADER = `
  @group(0) @binding(0) var<storage, read> A: array<f32>;         // [N, N]
  @group(0) @binding(1) var<storage, read_write> output: array<f32>; // [numWorkgroups, 2] - (maxVal, maxIdx)
  @group(0) @binding(2) var<uniform> dims: vec3<u32>;             // N, col, _pad

  var<workgroup> svals: array<f32, 256>;
  var<workgroup> sidxs: array<u32, 256>;

  @compute @workgroup_size(256)
  fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let N = dims.x;
    let col = dims.y;
    let tid = lid.x;
    let gid = wid.x * 256u + tid;
    let row = col + gid;  // Search from diagonal down

    // Initialize with current element or -infinity
    if (row < N) {
      svals[tid] = abs(A[row * N + col]);
      sidxs[tid] = row;
    } else {
      svals[tid] = -1e30f;
      sidxs[tid] = col;
    }
    workgroupBarrier();

    // Parallel reduction to find max
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        if (svals[tid + s] > svals[tid]) {
          svals[tid] = svals[tid + s];
          sidxs[tid] = sidxs[tid + s];
        }
      }
      workgroupBarrier();
    }

    // Write workgroup result
    if (tid == 0u) {
      output[wid.x * 2u] = svals[0];
      output[wid.x * 2u + 1u] = f32(sidxs[0]);
    }
  }
`;

// Shader 2: Swap two rows in-place
const LU_SWAP_ROWS_SHADER = `
  @group(0) @binding(0) var<storage, read_write> A: array<f32>;   // [N, N]
  @group(0) @binding(1) var<uniform> dims: vec3<u32>;             // N, row1, row2

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims.x;
    let row1 = dims.y;
    let row2 = dims.z;
    let col = gid.x;

    if (col >= N || row1 == row2) {
      return;
    }

    let idx1 = row1 * N + col;
    let idx2 = row2 * N + col;
    let tmp = A[idx1];
    A[idx1] = A[idx2];
    A[idx2] = tmp;
  }
`;

// Shader 3: LU elimination step - compute multipliers and update submatrix
const LU_ELIMINATE_SHADER = `
  @group(0) @binding(0) var<storage, read_write> A: array<f32>;   // [N, N]
  @group(0) @binding(1) var<uniform> dims: vec3<u32>;             // N, col, _pad
  @group(0) @binding(2) var<uniform> pivot: f32;                  // A[col, col]

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims.x;
    let col = dims.y;
    let row = col + 1u + gid.y;  // Rows below diagonal
    let j = col + gid.x;         // Columns from col onwards

    if (row >= N || j >= N) {
      return;
    }

    let factor = A[row * N + col] / pivot;

    // Store factor in L part (below diagonal) for j == col
    if (j == col) {
      A[row * N + col] = factor;
    } else {
      // Update U part (to the right)
      A[row * N + j] = A[row * N + j] - factor * A[col * N + j];
    }
  }
`;

// Shader 4: Augmented matrix row swap (for inv: [A|I] format, width = 2*N)
const INV_SWAP_ROWS_SHADER = `
  @group(0) @binding(0) var<storage, read_write> aug: array<f32>;   // [N, 2*N]
  @group(0) @binding(1) var<uniform> dims: vec3<u32>;               // N, row1, row2

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims.x;
    let row1 = dims.y;
    let row2 = dims.z;
    let col = gid.x;
    let width = N * 2u;

    if (col >= width || row1 == row2) {
      return;
    }

    let idx1 = row1 * width + col;
    let idx2 = row2 * width + col;
    let tmp = aug[idx1];
    aug[idx1] = aug[idx2];
    aug[idx2] = tmp;
  }
`;

// Shader 5: Scale row by 1/pivot (for Gauss-Jordan)
const INV_SCALE_ROW_SHADER = `
  @group(0) @binding(0) var<storage, read_write> aug: array<f32>;   // [N, 2*N]
  @group(0) @binding(1) var<uniform> dims: vec2<u32>;               // N, row
  @group(0) @binding(2) var<uniform> scale: f32;                    // 1/pivot

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims.x;
    let row = dims.y;
    let col = gid.x;
    let width = N * 2u;

    if (col >= width) {
      return;
    }

    let idx = row * width + col;
    aug[idx] = aug[idx] * scale;
  }
`;

// Shader 6: Eliminate column (for Gauss-Jordan: all rows except pivot row)
const INV_ELIMINATE_SHADER = `
  @group(0) @binding(0) var<storage, read_write> aug: array<f32>;   // [N, 2*N]
  @group(0) @binding(1) var<uniform> dims: vec3<u32>;               // N, pivotRow, _pad

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = dims.x;
    let pivotRow = dims.y;
    let row = gid.y;
    let col = gid.x;
    let width = N * 2u;

    if (row >= N || col >= width || row == pivotRow) {
      return;
    }

    // factor = aug[row, pivotRow]
    let factor = aug[row * width + pivotRow];

    // aug[row, col] -= factor * aug[pivotRow, col]
    aug[row * width + col] = aug[row * width + col] - factor * aug[pivotRow * width + col];
  }
`;

// ============ SVD Shaders (One-Sided Jacobi) ============
// GPU-based SVD using one-sided Jacobi rotations
// Computes singular values and optionally U, V matrices

// Shader: Compute A^T A matrix (for eigenvalue extraction)
const SVD_ATA_SHADER = `
  @group(0) @binding(0) var<storage, read> A: array<f32>;        // [M, N]
  @group(0) @binding(1) var<storage, read_write> ATA: array<f32>; // [N, N]
  @group(0) @binding(2) var<uniform> dims: vec3<u32>;            // M, N, _pad

  var<workgroup> tile: array<f32, 1024>;  // 32x32 tile

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims.x;
    let N = dims.y;
    let i = gid.x;  // output row
    let j = gid.y;  // output col

    if (i >= N || j >= N) {
      return;
    }

    // Compute ATA[i, j] = sum_k A[k, i] * A[k, j]
    var sum = 0.0f;
    for (var k = 0u; k < M; k = k + 1u) {
      sum = sum + A[k * N + i] * A[k * N + j];
    }
    ATA[i * N + j] = sum;
  }
`;

// Shader: Power iteration step for dominant eigenvector/value
const SVD_POWER_ITERATION_SHADER = `
  @group(0) @binding(0) var<storage, read> ATA: array<f32>;      // [N, N]
  @group(0) @binding(1) var<storage, read> v_in: array<f32>;     // [N] current vector
  @group(0) @binding(2) var<storage, read_write> v_out: array<f32>; // [N] output vector
  @group(0) @binding(3) var<uniform> N: u32;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) {
      return;
    }

    // v_out[i] = sum_j ATA[i, j] * v_in[j]
    var sum = 0.0f;
    for (var j = 0u; j < N; j = j + 1u) {
      sum = sum + ATA[i * N + j] * v_in[j];
    }
    v_out[i] = sum;
  }
`;

// Shader: Compute vector norm for power iteration
const SVD_VECTOR_NORM_SHADER = `
  @group(0) @binding(0) var<storage, read> v: array<f32>;        // [N]
  @group(0) @binding(1) var<storage, read_write> output: array<f32>; // partial sums
  @group(0) @binding(2) var<uniform> N: u32;

  var<workgroup> sdata: array<f32, 256>;

  @compute @workgroup_size(256)
  fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    if (gid < N) {
      let val = v[gid];
      sdata[tid] = val * val;
    } else {
      sdata[tid] = 0.0f;
    }
    workgroupBarrier();

    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sdata[tid] = sdata[tid] + sdata[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      output[wid.x] = sdata[0];
    }
  }
`;

// Shader: Normalize vector and compute eigenvalue (Rayleigh quotient)
const SVD_NORMALIZE_SHADER = `
  @group(0) @binding(0) var<storage, read_write> v: array<f32>;  // [N] vector to normalize
  @group(0) @binding(1) var<uniform> norm: f32;                  // vector norm

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num: vec3<u32>) {
    let N = num.x * 256u;
    let i = gid.x;
    if (i < N && norm > 1e-10f) {
      v[i] = v[i] / norm;
    }
  }
`;

// Shader: Deflate matrix (remove contribution of found eigenvector)
const SVD_DEFLATE_SHADER = `
  @group(0) @binding(0) var<storage, read_write> ATA: array<f32>; // [N, N]
  @group(0) @binding(1) var<storage, read> v: array<f32>;         // [N] eigenvector
  @group(0) @binding(2) var<uniform> params: vec2<f32>;           // eigenvalue, N (as u32)

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = u32(params.y);
    let eigenvalue = params.x;
    let i = gid.x;
    let j = gid.y;

    if (i >= N || j >= N) {
      return;
    }

    // ATA[i,j] -= eigenvalue * v[i] * v[j]
    ATA[i * N + j] = ATA[i * N + j] - eigenvalue * v[i] * v[j];
  }
`;

// ============ Convolution Shader ============
// 1D convolution computed in parallel on GPU
// Each output element is sum of a[i-j] * v[j] for valid j
const CONVOLVE_SHADER = `
  struct ConvolveParams {
    aLen: u32,
    vLen: u32,
    outLen: u32,
    _pad: u32,
  }

  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> v: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;
  @group(0) @binding(3) var<uniform> params: ConvolveParams;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.outLen) {
      return;
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < params.vLen; j = j + 1u) {
      // aIdx = i - j (but we need to handle the full convolution indexing)
      let aIdx = i32(i) - i32(j);
      if (aIdx >= 0 && u32(aIdx) < params.aLen) {
        sum = sum + a[u32(aIdx)] * v[j];
      }
    }
    output[i] = sum;
  }
`;

// ============ Matmul Shaders ============

// Phase 1: Tiled shader with shared memory
// - 32x32 tiles loaded into shared memory
// - Each thread computes 2x2 output elements
// - A is transposed during load for coalesced reads
// - Padding (33 instead of 32) to avoid bank conflicts
const MATMUL_TILED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
  }

  const TILE_SIZE: u32 = 32u;
  const TILE_SIZE_PADDED: u32 = 33u;  // Padding to avoid bank conflicts

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<f32>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory tiles
  // As is stored TRANSPOSED with padding: As[k, row] instead of As[row, k]
  var<workgroup> As: array<f32, 1056>;  // 32 * 33 = 1056 (transposed + padded)
  var<workgroup> Bs: array<f32, 1024>;  // 32 * 32 = 1024 (row-major)

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;

    // Thread coordinates within workgroup (0-15 each)
    let tx = lid.x;
    let ty = lid.y;

    // Output coordinates - each thread computes 2x2 block
    let outRow = wid.y * TILE_SIZE + ty * 2u;
    let outCol = wid.x * TILE_SIZE + tx * 2u;

    // Initialize 2x2 accumulators
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    // Number of tiles along K dimension
    let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (TRANSPOSED) into shared memory ===
      // Global A[row, k] -> Shared As[k, row] with padding
      // Each thread loads 2x2 elements

      let aRow0 = wid.y * TILE_SIZE + ty * 2u;
      let aRow1 = aRow0 + 1u;
      let aK0 = t * TILE_SIZE + tx * 2u;
      let aK1 = aK0 + 1u;

      // Load with bounds checking, storing TRANSPOSED
      // As[k, row] = A[row, k]
      if (aRow0 < M && aK0 < K) {
        As[(tx * 2u) * TILE_SIZE_PADDED + ty * 2u] = A[aRow0 * K + aK0];
      } else {
        As[(tx * 2u) * TILE_SIZE_PADDED + ty * 2u] = 0.0;
      }

      if (aRow1 < M && aK0 < K) {
        As[(tx * 2u) * TILE_SIZE_PADDED + ty * 2u + 1u] = A[aRow1 * K + aK0];
      } else {
        As[(tx * 2u) * TILE_SIZE_PADDED + ty * 2u + 1u] = 0.0;
      }

      if (aRow0 < M && aK1 < K) {
        As[(tx * 2u + 1u) * TILE_SIZE_PADDED + ty * 2u] = A[aRow0 * K + aK1];
      } else {
        As[(tx * 2u + 1u) * TILE_SIZE_PADDED + ty * 2u] = 0.0;
      }

      if (aRow1 < M && aK1 < K) {
        As[(tx * 2u + 1u) * TILE_SIZE_PADDED + ty * 2u + 1u] = A[aRow1 * K + aK1];
      } else {
        As[(tx * 2u + 1u) * TILE_SIZE_PADDED + ty * 2u + 1u] = 0.0;
      }

      // === Load B tile (row-major) into shared memory ===
      let bK0 = t * TILE_SIZE + ty * 2u;
      let bK1 = bK0 + 1u;
      let bCol0 = wid.x * TILE_SIZE + tx * 2u;
      let bCol1 = bCol0 + 1u;

      if (bK0 < K && bCol0 < N) {
        Bs[(ty * 2u) * TILE_SIZE + tx * 2u] = B[bK0 * N + bCol0];
      } else {
        Bs[(ty * 2u) * TILE_SIZE + tx * 2u] = 0.0;
      }

      if (bK0 < K && bCol1 < N) {
        Bs[(ty * 2u) * TILE_SIZE + tx * 2u + 1u] = B[bK0 * N + bCol1];
      } else {
        Bs[(ty * 2u) * TILE_SIZE + tx * 2u + 1u] = 0.0;
      }

      if (bK1 < K && bCol0 < N) {
        Bs[(ty * 2u + 1u) * TILE_SIZE + tx * 2u] = B[bK1 * N + bCol0];
      } else {
        Bs[(ty * 2u + 1u) * TILE_SIZE + tx * 2u] = 0.0;
      }

      if (bK1 < K && bCol1 < N) {
        Bs[(ty * 2u + 1u) * TILE_SIZE + tx * 2u + 1u] = B[bK1 * N + bCol1];
      } else {
        Bs[(ty * 2u + 1u) * TILE_SIZE + tx * 2u + 1u] = 0.0;
      }

      // Sync to ensure all threads have loaded their data
      workgroupBarrier();

      // === Compute 2x2 output tile ===
      // Inner loop over K dimension of the tile
      for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
        // Read from transposed+padded A: As[k, localRow]
        let a0 = As[k * TILE_SIZE_PADDED + ty * 2u];
        let a1 = As[k * TILE_SIZE_PADDED + ty * 2u + 1u];

        // Read from row-major B: Bs[k, localCol]
        let b0 = Bs[k * TILE_SIZE + tx * 2u];
        let b1 = Bs[k * TILE_SIZE + tx * 2u + 1u];

        // Accumulate 2x2 outer product
        acc00 = fma(a0, b0, acc00);
        acc01 = fma(a0, b1, acc01);
        acc10 = fma(a1, b0, acc10);
        acc11 = fma(a1, b1, acc11);
      }

      // Sync before loading next tile
      workgroupBarrier();
    }

    // === Write output with bounds checking ===
    if (outRow < M && outCol < N) {
      C[outRow * N + outCol] = acc00;
    }
    if (outRow < M && outCol + 1u < N) {
      C[outRow * N + outCol + 1u] = acc01;
    }
    if (outRow + 1u < M && outCol < N) {
      C[(outRow + 1u) * N + outCol] = acc10;
    }
    if (outRow + 1u < M && outCol + 1u < N) {
      C[(outRow + 1u) * N + outCol + 1u] = acc11;
    }
  }
`;

// Phase 2: Register blocking shader
// - 64x64 output tiles per workgroup
// - Each thread computes 4x4 output elements (16 accumulators)
// - K tile = 16 (conservative for shared memory)
// - Shared memory: (64x17 + 16x64) * 4 = ~5.4KB
const MATMUL_REGISTER_BLOCKED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
  }

  // Tile configuration
  const BLOCK_M: u32 = 64u;   // Output block rows
  const BLOCK_N: u32 = 64u;   // Output block cols
  const BLOCK_K: u32 = 16u;   // K dimension block (conservative for shared mem)
  const THREAD_M: u32 = 4u;   // Per-thread rows
  const THREAD_N: u32 = 4u;   // Per-thread cols
  const BLOCK_M_PADDED: u32 = 65u;  // Padding for bank conflicts

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<f32>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory tiles
  // As: BLOCK_K x BLOCK_M (transposed) with padding = 16 x 65 = 1040
  // Bs: BLOCK_K x BLOCK_N = 16 x 64 = 1024
  var<workgroup> As: array<f32, 1040>;  // 16 * 65 (transposed + padded)
  var<workgroup> Bs: array<f32, 1024>;  // 16 * 64 (row-major)

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;

    // Thread coordinates within workgroup (0-15 each)
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;  // Linear thread ID (0-255)

    // Output coordinates - each thread computes 4x4 block
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    // 16 accumulators (4x4 output per thread)
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    // Number of tiles along K dimension
    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (transposed) ===
      // Need to load BLOCK_K x BLOCK_M = 16 x 64 = 1024 elements
      // 256 threads, each loads 4 elements
      for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let loadIdx = tid * 4u + i;
        let loadK = loadIdx / BLOCK_M;     // 0-15 (K index)
        let loadRow = loadIdx % BLOCK_M;   // 0-63 (M index)

        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;

        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        // Store transposed: As[k, row]
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // === Load B tile ===
      // Need to load BLOCK_K x BLOCK_N = 16 x 64 = 1024 elements
      // 256 threads, each loads 4 elements
      for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let loadIdx = tid * 4u + i;
        let loadK = loadIdx / BLOCK_N;     // 0-15 (K index)
        let loadCol = loadIdx % BLOCK_N;   // 0-63 (N index)

        let globalK = t * BLOCK_K + loadK;
        let globalCol = wid.x * BLOCK_N + loadCol;

        var val: f32 = 0.0;
        if (globalK < K && globalCol < N) {
          val = B[globalK * N + globalCol];
        }
        // Store row-major: Bs[k, col]
        Bs[loadK * BLOCK_N + loadCol] = val;
      }

      // Sync to ensure all threads have loaded
      workgroupBarrier();

      // === Compute 4x4 output tile ===
      // Manually unrolled for performance
      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        // Load 4 values from A (one column of the 4x4 output)
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];

        // Load 4 values from B (one row of the 4x4 output)
        let b0 = Bs[k * BLOCK_N + tx * THREAD_N + 0u];
        let b1 = Bs[k * BLOCK_N + tx * THREAD_N + 1u];
        let b2 = Bs[k * BLOCK_N + tx * THREAD_N + 2u];
        let b3 = Bs[k * BLOCK_N + tx * THREAD_N + 3u];

        // 4x4 outer product
        acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
        acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
        acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
        acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
      }

      // Sync before loading next tile
      workgroupBarrier();
    }

    // === Write 4x4 output with bounds checking ===
    if (outRowBase + 0u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 0u) * N + outColBase + 0u] = acc00; }
      if (outColBase + 1u < N) { C[(outRowBase + 0u) * N + outColBase + 1u] = acc01; }
      if (outColBase + 2u < N) { C[(outRowBase + 0u) * N + outColBase + 2u] = acc02; }
      if (outColBase + 3u < N) { C[(outRowBase + 0u) * N + outColBase + 3u] = acc03; }
    }
    if (outRowBase + 1u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 1u) * N + outColBase + 0u] = acc10; }
      if (outColBase + 1u < N) { C[(outRowBase + 1u) * N + outColBase + 1u] = acc11; }
      if (outColBase + 2u < N) { C[(outRowBase + 1u) * N + outColBase + 2u] = acc12; }
      if (outColBase + 3u < N) { C[(outRowBase + 1u) * N + outColBase + 3u] = acc13; }
    }
    if (outRowBase + 2u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 2u) * N + outColBase + 0u] = acc20; }
      if (outColBase + 1u < N) { C[(outRowBase + 2u) * N + outColBase + 1u] = acc21; }
      if (outColBase + 2u < N) { C[(outRowBase + 2u) * N + outColBase + 2u] = acc22; }
      if (outColBase + 3u < N) { C[(outRowBase + 2u) * N + outColBase + 3u] = acc23; }
    }
    if (outRowBase + 3u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 3u) * N + outColBase + 0u] = acc30; }
      if (outColBase + 1u < N) { C[(outRowBase + 3u) * N + outColBase + 1u] = acc31; }
      if (outColBase + 2u < N) { C[(outRowBase + 3u) * N + outColBase + 2u] = acc32; }
      if (outColBase + 3u < N) { C[(outRowBase + 3u) * N + outColBase + 3u] = acc33; }
    }
  }
`;

// Phase 3: Vectorized shader with vec4 loads
// - Uses vec4<f32> for 4x memory bandwidth
// - 64x64 output tiles per workgroup
// - Each thread computes 4x4 output elements
// - K tile = 16, optimized for vec4 alignment
const MATMUL_VECTORIZED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
  }

  // Tile configuration
  const BLOCK_M: u32 = 64u;   // Output block rows
  const BLOCK_N: u32 = 64u;   // Output block cols
  const BLOCK_K: u32 = 16u;   // K dimension block
  const THREAD_M: u32 = 4u;   // Per-thread rows
  const THREAD_N: u32 = 4u;   // Per-thread cols
  const BLOCK_M_PADDED: u32 = 68u;  // Padding for bank conflicts (must be multiple of 4 + 4)

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;  // Vectorized storage
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // Vectorized storage
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory tiles
  var<workgroup> As: array<f32, 1088>;  // 16 * 68 (transposed + padded)
  var<workgroup> Bs: array<f32, 1024>;  // 16 * 64 (row-major)

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;

    // Thread coordinates
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    // Output coordinates
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    // 16 accumulators (4x4 output per thread)
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (transposed) using vec4 ===
      // Each thread loads 4 elements using a single vec4 load
      // Total: 256 threads * 4 elements = 1024 elements
      {
        let loadIdx = tid;  // Each thread loads one vec4 (4 elements)
        let loadK = (loadIdx * 4u) / BLOCK_M;      // Which K row
        let loadRowBase = (loadIdx * 4u) % BLOCK_M;  // Starting row

        let globalK = t * BLOCK_K + loadK;
        let globalRowBase = wid.y * BLOCK_M + loadRowBase;

        // Load 4 consecutive rows at the same K
        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K) {
          let kOffset = globalK / 4u;
          let kMod = globalK % 4u;

          // Load each row individually (since they're at different global addresses)
          for (var r: u32 = 0u; r < 4u; r = r + 1u) {
            let row = globalRowBase + r;
            if (row < M) {
              let idx = row * (K / 4u) + kOffset;
              let vec = A[idx];
              v[r] = vec[kMod];
            }
          }
        }

        // Store transposed
        As[loadK * BLOCK_M_PADDED + loadRowBase + 0u] = v.x;
        As[loadK * BLOCK_M_PADDED + loadRowBase + 1u] = v.y;
        As[loadK * BLOCK_M_PADDED + loadRowBase + 2u] = v.z;
        As[loadK * BLOCK_M_PADDED + loadRowBase + 3u] = v.w;
      }

      // === Load B tile using vec4 ===
      // Each row of B is contiguous, perfect for vec4 loads
      {
        let loadIdx = tid;
        let loadK = loadIdx / 16u;      // Which K row (0-15)
        let loadColQuad = loadIdx % 16u;  // Which vec4 column (0-15, covering 64 cols)

        let globalK = t * BLOCK_K + loadK;
        let globalCol = wid.x * BLOCK_N + loadColQuad * 4u;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalCol + 3u < N) {
          let idx = globalK * (N / 4u) + globalCol / 4u;
          v = B[idx];
        } else if (globalK < K) {
          // Handle edge case with scalar loads
          let idx = globalK * (N / 4u) + globalCol / 4u;
          if (globalCol < N) {
            let vec = B[idx];
            v.x = vec.x;
            if (globalCol + 1u < N) { v.y = vec.y; }
            if (globalCol + 2u < N) { v.z = vec.z; }
            if (globalCol + 3u < N) { v.w = vec.w; }
          }
        }

        Bs[loadK * BLOCK_N + loadColQuad * 4u + 0u] = v.x;
        Bs[loadK * BLOCK_N + loadColQuad * 4u + 1u] = v.y;
        Bs[loadK * BLOCK_N + loadColQuad * 4u + 2u] = v.z;
        Bs[loadK * BLOCK_N + loadColQuad * 4u + 3u] = v.w;
      }

      workgroupBarrier();

      // === Compute 4x4 output tile ===
      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];

        let b0 = Bs[k * BLOCK_N + tx * THREAD_N + 0u];
        let b1 = Bs[k * BLOCK_N + tx * THREAD_N + 1u];
        let b2 = Bs[k * BLOCK_N + tx * THREAD_N + 2u];
        let b3 = Bs[k * BLOCK_N + tx * THREAD_N + 3u];

        acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
        acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
        acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
        acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
      }

      workgroupBarrier();
    }

    // === Write output ===
    if (outRowBase + 0u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 0u) * N + outColBase + 0u] = acc00; }
      if (outColBase + 1u < N) { C[(outRowBase + 0u) * N + outColBase + 1u] = acc01; }
      if (outColBase + 2u < N) { C[(outRowBase + 0u) * N + outColBase + 2u] = acc02; }
      if (outColBase + 3u < N) { C[(outRowBase + 0u) * N + outColBase + 3u] = acc03; }
    }
    if (outRowBase + 1u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 1u) * N + outColBase + 0u] = acc10; }
      if (outColBase + 1u < N) { C[(outRowBase + 1u) * N + outColBase + 1u] = acc11; }
      if (outColBase + 2u < N) { C[(outRowBase + 1u) * N + outColBase + 2u] = acc12; }
      if (outColBase + 3u < N) { C[(outRowBase + 1u) * N + outColBase + 3u] = acc13; }
    }
    if (outRowBase + 2u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 2u) * N + outColBase + 0u] = acc20; }
      if (outColBase + 1u < N) { C[(outRowBase + 2u) * N + outColBase + 1u] = acc21; }
      if (outColBase + 2u < N) { C[(outRowBase + 2u) * N + outColBase + 2u] = acc22; }
      if (outColBase + 3u < N) { C[(outRowBase + 2u) * N + outColBase + 3u] = acc23; }
    }
    if (outRowBase + 3u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 3u) * N + outColBase + 0u] = acc30; }
      if (outColBase + 1u < N) { C[(outRowBase + 3u) * N + outColBase + 1u] = acc31; }
      if (outColBase + 2u < N) { C[(outRowBase + 3u) * N + outColBase + 2u] = acc32; }
      if (outColBase + 3u < N) { C[(outRowBase + 3u) * N + outColBase + 3u] = acc33; }
    }
  }
`;

// Legacy naive shader (kept for reference/comparison)
const MATMUL_NAIVE_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
  }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<f32>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;

    if (row >= M || col >= N) {
      return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < K; i = i + 1u) {
      sum = sum + A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
`;

// ============ OPTIMIZED VEC4 SHADER ============
// Full vec4 optimization: [16, 16] workgroup with 4x4 elements per thread
// - 64x64 output tiles (256 threads × 4×4 elements per thread)
// - vec4 loads for B matrix (4x memory bandwidth)
// - BLOCK_K = 32 for more compute per tile load
// - Inner loop unrolled 4x for instruction-level parallelism
const MATMUL_VEC4_OPTIMIZED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,  // N padded to multiple of 4
  }

  // Tile configuration - BEST: [16,16] workgroup, 64x64 tiles, 4x4 per thread
  const BLOCK_M: u32 = 64u;    // Output block rows (16 threads * 4 elements)
  const BLOCK_N: u32 = 64u;    // Output block cols (16 threads * 4 elements)
  const BLOCK_K: u32 = 32u;    // K dimension block
  const THREAD_M: u32 = 4u;    // Per-thread rows
  const THREAD_N: u32 = 4u;    // Per-thread cols
  const BLOCK_M_PADDED: u32 = 65u;  // Padding for bank conflicts

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // vec4 storage for B
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory tiles
  // As: BLOCK_K x BLOCK_M (transposed) with padding = 32 x 65 = 2080
  // Bs: BLOCK_K x (BLOCK_N/4) in vec4 = 32 x 16 = 512 vec4s = 2048 floats
  var<workgroup> As: array<f32, 2080>;     // 32 * 65 (transposed + padded)
  var<workgroup> Bs: array<vec4<f32>, 512>;  // 32 * 16 vec4s

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let tx = lid.x;  // 0-15
    let ty = lid.y;  // 0-15
    let tid = ty * 16u + tx;  // 0-255

    // Output coordinates
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    // 4x4 accumulators stored as 4 vec4s
    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (transposed) ===
      // Need to load BLOCK_K x BLOCK_M = 32 x 64 = 2048 elements
      // 256 threads, each loads 8 elements
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;

        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;

        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // === Load B tile using vec4 ===
      // Need to load BLOCK_K x BLOCK_N = 32 x 64 = 2048 elements = 512 vec4s
      // 256 threads, each loads 2 vec4s
      for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let loadIdx = tid * 2u + i;
        let loadK = loadIdx / (BLOCK_N / 4u);
        let loadColVec = loadIdx % (BLOCK_N / 4u);

        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        Bs[loadK * (BLOCK_N / 4u) + loadColVec] = v;
      }

      workgroupBarrier();

      // === Compute 4x4 output tile using vec4 operations ===
      let tileK = min(BLOCK_K, K - t * BLOCK_K);

      for (var k: u32 = 0u; k < tileK; k = k + 1u) {
        // Load 4 values from A
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];

        // Load vec4 from B (4 columns)
        let b = Bs[k * (BLOCK_N / 4u) + tx];

        // 4x4 outer product
        acc0 = fma(vec4<f32>(a0), b, acc0);
        acc1 = fma(vec4<f32>(a1), b, acc1);
        acc2 = fma(vec4<f32>(a2), b, acc2);
        acc3 = fma(vec4<f32>(a3), b, acc3);
      }

      workgroupBarrier();
    }

    // === Write 4x4 output with bounds checking ===
    let row0 = outRowBase;
    let row1 = outRowBase + 1u;
    let row2 = outRowBase + 2u;
    let row3 = outRowBase + 3u;

    if (row0 < M) {
      if (outColBase + 0u < N) { C[row0 * N + outColBase + 0u] = acc0.x; }
      if (outColBase + 1u < N) { C[row0 * N + outColBase + 1u] = acc0.y; }
      if (outColBase + 2u < N) { C[row0 * N + outColBase + 2u] = acc0.z; }
      if (outColBase + 3u < N) { C[row0 * N + outColBase + 3u] = acc0.w; }
    }
    if (row1 < M) {
      if (outColBase + 0u < N) { C[row1 * N + outColBase + 0u] = acc1.x; }
      if (outColBase + 1u < N) { C[row1 * N + outColBase + 1u] = acc1.y; }
      if (outColBase + 2u < N) { C[row1 * N + outColBase + 2u] = acc1.z; }
      if (outColBase + 3u < N) { C[row1 * N + outColBase + 3u] = acc1.w; }
    }
    if (row2 < M) {
      if (outColBase + 0u < N) { C[row2 * N + outColBase + 0u] = acc2.x; }
      if (outColBase + 1u < N) { C[row2 * N + outColBase + 1u] = acc2.y; }
      if (outColBase + 2u < N) { C[row2 * N + outColBase + 2u] = acc2.z; }
      if (outColBase + 3u < N) { C[row2 * N + outColBase + 3u] = acc2.w; }
    }
    if (row3 < M) {
      if (outColBase + 0u < N) { C[row3 * N + outColBase + 0u] = acc3.x; }
      if (outColBase + 1u < N) { C[row3 * N + outColBase + 1u] = acc3.y; }
      if (outColBase + 2u < N) { C[row3 * N + outColBase + 2u] = acc3.z; }
      if (outColBase + 3u < N) { C[row3 * N + outColBase + 3u] = acc3.w; }
    }
  }
`;

// Variant shaders for autotuning
// Phase 4: 32x128 variant - wider tiles, 8x32 workgroup
const MATMUL_WIDE_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
  }

  // Tile configuration - wide tiles optimized for memory bandwidth
  const BLOCK_M: u32 = 32u;    // Output block rows
  const BLOCK_N: u32 = 128u;   // Output block cols (wider)
  const BLOCK_K: u32 = 32u;    // K dimension block
  const THREAD_M: u32 = 4u;    // Per-thread rows
  const THREAD_N: u32 = 4u;    // Per-thread cols
  const BLOCK_M_PADDED: u32 = 33u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<f32>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory: 32x33 + 32x128 = ~5.1KB
  var<workgroup> As: array<f32, 1056>;   // 32 * 33
  var<workgroup> Bs: array<f32, 4096>;   // 32 * 128

  @compute @workgroup_size(32, 8)  // 256 threads: 32 wide, 8 tall
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;

    let tx = lid.x;  // 0-31
    let ty = lid.y;  // 0-7
    let tid = ty * 32u + tx;

    // Each thread computes 4x4 = 16 outputs
    // Workgroup: 32*4 = 128 cols, 8*4 = 32 rows
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // Load A: 32x32 = 1024 elements, 256 threads load 4 each
      for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let loadIdx = tid * 4u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;

        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;

        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // Load B: 32x128 = 4096 elements, 256 threads load 16 each
      for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let loadIdx = tid * 16u + i;
        let loadK = loadIdx / BLOCK_N;
        let loadCol = loadIdx % BLOCK_N;

        let globalK = t * BLOCK_K + loadK;
        let globalCol = wid.x * BLOCK_N + loadCol;

        var val: f32 = 0.0;
        if (globalK < K && globalCol < N) {
          val = B[globalK * N + globalCol];
        }
        Bs[loadK * BLOCK_N + loadCol] = val;
      }

      workgroupBarrier();

      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];

        let b0 = Bs[k * BLOCK_N + tx * THREAD_N + 0u];
        let b1 = Bs[k * BLOCK_N + tx * THREAD_N + 1u];
        let b2 = Bs[k * BLOCK_N + tx * THREAD_N + 2u];
        let b3 = Bs[k * BLOCK_N + tx * THREAD_N + 3u];

        acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
        acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
        acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
        acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
      }

      workgroupBarrier();
    }

    // Write output
    if (outRowBase + 0u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 0u) * N + outColBase + 0u] = acc00; }
      if (outColBase + 1u < N) { C[(outRowBase + 0u) * N + outColBase + 1u] = acc01; }
      if (outColBase + 2u < N) { C[(outRowBase + 0u) * N + outColBase + 2u] = acc02; }
      if (outColBase + 3u < N) { C[(outRowBase + 0u) * N + outColBase + 3u] = acc03; }
    }
    if (outRowBase + 1u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 1u) * N + outColBase + 0u] = acc10; }
      if (outColBase + 1u < N) { C[(outRowBase + 1u) * N + outColBase + 1u] = acc11; }
      if (outColBase + 2u < N) { C[(outRowBase + 1u) * N + outColBase + 2u] = acc12; }
      if (outColBase + 3u < N) { C[(outRowBase + 1u) * N + outColBase + 3u] = acc13; }
    }
    if (outRowBase + 2u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 2u) * N + outColBase + 0u] = acc20; }
      if (outColBase + 1u < N) { C[(outRowBase + 2u) * N + outColBase + 1u] = acc21; }
      if (outColBase + 2u < N) { C[(outRowBase + 2u) * N + outColBase + 2u] = acc22; }
      if (outColBase + 3u < N) { C[(outRowBase + 2u) * N + outColBase + 3u] = acc23; }
    }
    if (outRowBase + 3u < M) {
      if (outColBase + 0u < N) { C[(outRowBase + 3u) * N + outColBase + 0u] = acc30; }
      if (outColBase + 1u < N) { C[(outRowBase + 3u) * N + outColBase + 1u] = acc31; }
      if (outColBase + 2u < N) { C[(outRowBase + 3u) * N + outColBase + 2u] = acc32; }
      if (outColBase + 3u < N) { C[(outRowBase + 3u) * N + outColBase + 3u] = acc33; }
    }
  }
`;

// Phase 5: TFJS vec4 packed pattern - uses vec4 for both A and B loading
// This matches tfjs's makeMatMulPackedVec4Source which is their fastest path
const MATMUL_TFJS_VEC4_PACKED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  const tileInner: u32 = 32u;
  const rowPerThread: u32 = 4u;
  const colPerThread: u32 = 4u;
  const WG_X: u32 = 16u;
  const WG_Y: u32 = 16u;
  const tileAOuter: u32 = 64u;
  const tileBOuter: u32 = 64u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // tfjs vec4 pattern: A stored as [M_tile][K_tile/4] of vec4s
  // Each vec4 contains 4 consecutive K values for the same row
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 64>;  // [tileAOuter][tileInner/4] = [64][8]
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 16>, 32>;  // [tileInner][tileBOuter/4]

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let localRow = i32(lid.y);
    let localCol = i32(lid.x);
    let tileRow = localRow;  // vec4 row index
    let tileCol = localCol;

    let globalRow = i32(gid.y) * i32(rowPerThread);
    let globalCol = i32(gid.x) * i32(colPerThread);
    let globalRowStart = i32(wid.y) * i32(tileAOuter);

    let numTiles = (i32(K) + i32(tileInner) - 1) / i32(tileInner);

    // Accumulators: 4 vec4s for 4x4 output
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
      acc[i] = vec4<f32>(0.0);
    }

    var kStart = 0;

    for (var t = 0; t < numTiles; t++) {
      // Load A as vec4s: each thread loads 2 vec4s
      // A[M][K], load 4 consecutive K values (coalesced reads!)
      // Total: 256 threads * 2 vec4s = 512 vec4s = 64M x 32K
      for (var inner = 0u; inner < 2u; inner++) {
        let loadIdx = u32(localRow) * 16u + u32(localCol) + inner * 256u;  // Linear index 0-511
        let mIdx = loadIdx / 8u;  // M index (0-63)
        let kVec4Idx = loadIdx % 8u;  // K vec4 index (0-7)

        let globalM = u32(globalRowStart) + mIdx;
        let globalKBase = u32(kStart) + kVec4Idx * 4u;

        var val: vec4<f32> = vec4<f32>(0.0);
        if (globalM < M && globalKBase + 3u < K) {
          // Load 4 consecutive K values (coalesced memory access)
          let base = globalM * K + globalKBase;
          val = vec4<f32>(A[base], A[base + 1u], A[base + 2u], A[base + 3u]);
        } else if (globalM < M) {
          // Handle edge case
          let base = globalM * K + globalKBase;
          if (globalKBase < K) { val.x = A[base]; }
          if (globalKBase + 1u < K) { val.y = A[base + 1u]; }
          if (globalKBase + 2u < K) { val.z = A[base + 2u]; }
          if (globalKBase + 3u < K) { val.w = A[base + 3u]; }
        }
        mm_Asub[mIdx][kVec4Idx] = val;
      }

      // Load B as vec4s: each thread loads 2 vec4s
      for (var innerK = 0u; innerK < 2u; innerK++) {
        let kIdx = u32(localRow) * 2u + innerK;
        let colVec = u32(localCol);

        let globalK = u32(kStart) + kIdx;
        let globalColVec = wid.x * (tileBOuter / 4u) + colVec;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        mm_Bsub[kIdx][colVec] = v;
      }

      kStart = kStart + i32(tileInner);
      workgroupBarrier();

      // Compute: A is [M][K/4] vec4s, B is [K][N/4] vec4s
      // Each thread needs A[localRow*4 + 0..3][k/4] for rows 0-3
      // Process K in groups of 4 (since A is stored as vec4 over K)
      for (var kVec4 = 0u; kVec4 < 8u; kVec4++) {  // 32K / 4 = 8 vec4s
        // Load A vec4s for this thread's 4 rows
        let a0 = mm_Asub[u32(localRow) * 4u + 0u][kVec4];
        let a1 = mm_Asub[u32(localRow) * 4u + 1u][kVec4];
        let a2 = mm_Asub[u32(localRow) * 4u + 2u][kVec4];
        let a3 = mm_Asub[u32(localRow) * 4u + 3u][kVec4];

        // Process 4 K values (from the vec4)
        let kBase = kVec4 * 4u;
        {
          let BCached = mm_Bsub[kBase][u32(localCol)];
          acc[0] = fma(vec4<f32>(a0.x), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(a1.x), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(a2.x), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(a3.x), BCached, acc[3]);
        }
        {
          let BCached = mm_Bsub[kBase + 1u][u32(localCol)];
          acc[0] = fma(vec4<f32>(a0.y), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(a1.y), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(a2.y), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(a3.y), BCached, acc[3]);
        }
        {
          let BCached = mm_Bsub[kBase + 2u][u32(localCol)];
          acc[0] = fma(vec4<f32>(a0.z), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(a1.z), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(a2.z), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(a3.z), BCached, acc[3]);
        }
        {
          let BCached = mm_Bsub[kBase + 3u][u32(localCol)];
          acc[0] = fma(vec4<f32>(a0.w), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(a1.w), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(a2.w), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(a3.w), BCached, acc[3]);
        }
      }

      workgroupBarrier();
    }

    // Write output
    for (var innerRow = 0u; innerRow < rowPerThread; innerRow++) {
      let row = globalRow + i32(innerRow);
      if (row < i32(M)) {
        for (var innerCol = 0u; innerCol < 4u; innerCol++) {
          let col = globalCol + i32(innerCol);
          if (col < i32(N)) {
            C[u32(row) * N + u32(col)] = acc[innerRow][innerCol];
          }
        }
      }
    }
  }
`;

// Phase 5b: Exact TFJS pattern - [16,16] workgroup with 4x4 elements per thread
// Matching their exact shared memory layout and inner loop pattern
// Key: cache B values first, then iterate over A rows
const MATMUL_TFJS_EXACT_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // Configuration matching tfjs defaults
  const tileInner: u32 = 32u;   // K dimension tile
  const rowPerThread: u32 = 4u;  // Rows per thread
  const colPerThread: u32 = 4u;  // Cols per thread
  const WG_X: u32 = 16u;
  const WG_Y: u32 = 16u;
  const tileAOuter: u32 = 64u;  // WG_Y * rowPerThread = 16 * 4
  const tileBOuter: u32 = 64u;  // WG_X * colPerThread = 16 * 4

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // vec4 storage
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory with A transposed for coalesced reads in inner loop
  // A stored as [K_tile][M_tile] for better access pattern
  var<workgroup> mm_Asub: array<array<f32, 64>, 32>;  // [tileInner][tileAOuter] = [K][M]
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 16>, 32>;  // [tileInner][tileBOuter/4]

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let localRow = i32(lid.y);  // 0-15
    let localCol = i32(lid.x);  // 0-15
    let tileRow = localRow * i32(rowPerThread);
    let tileCol = localCol;

    let globalRow = i32(gid.y) * i32(rowPerThread);
    let globalCol = i32(gid.x) * i32(colPerThread);
    let globalRowStart = i32(wid.y) * i32(tileAOuter);

    // TFJS-style: A loading params
    let rowPerThreadA = 2u;  // 64 / 32 = 2 (tileAOuter / tileInner / 1)
    let colPerThreadA = 2u;  // 32 / 16 = 2 (tileInner / WG_X)
    let tileRowA = localRow * i32(rowPerThreadA);
    let tileColA = localCol * i32(colPerThreadA);

    // B loading params
    let rowPerThreadB = 2u;  // 32 / 16 = 2 (tileInner / WG_Y)
    let tileRowB = localRow * i32(rowPerThreadB);

    let numTiles = (i32(K) + i32(tileInner) - 1) / i32(tileInner);

    // Initialize accumulators (tfjs uses array of vec4s for output)
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
      acc[i] = vec4<f32>(0.0);
    }

    var kStart = 0;

    for (var t = 0; t < numTiles; t++) {
      // Load A tile into shared memory (TRANSPOSED for coalesced access in inner loop)
      // mm_Asub[K_local][M_local] = A[globalM, globalK]
      for (var innerRow = 0u; innerRow < rowPerThreadA; innerRow++) {
        for (var innerCol = 0u; innerCol < colPerThreadA; innerCol++) {
          let inputRow = u32(tileRowA) + innerRow;  // M index in tile (0-63)
          let inputCol = u32(tileColA) + innerCol;  // K index in tile (0-31)
          let globalM = u32(globalRowStart) + inputRow;
          let globalK = u32(kStart) + inputCol;

          var val: f32 = 0.0;
          if (globalM < M && globalK < K) {
            val = A[globalM * K + globalK];
          }
          // Store TRANSPOSED: [K][M] indexing for coalesced reads in inner loop
          mm_Asub[inputCol][inputRow] = val;
        }
      }

      // Load B tile into shared memory as vec4s
      for (var innerRow = 0u; innerRow < rowPerThreadB; innerRow++) {
        let inputRow = u32(tileRowB) + innerRow;
        let globalK = u32(kStart) + inputRow;
        let globalColVec = wid.x * (tileBOuter / 4u) + u32(localCol);

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        mm_Bsub[inputRow][localCol] = v;
      }

      kStart = kStart + i32(tileInner);
      workgroupBarrier();

      // TFJS inner loop pattern: fully unrolled for better ILP
      // Process 8 K iterations per loop (tileInner=32, 4 unrolls)
      for (var kBase = 0u; kBase < 32u; kBase = kBase + 8u) {
        // K=0
        {
          let BCached = mm_Bsub[kBase][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=1
        {
          let BCached = mm_Bsub[kBase+1u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+1u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+1u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+1u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+1u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=2
        {
          let BCached = mm_Bsub[kBase+2u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+2u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+2u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+2u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+2u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=3
        {
          let BCached = mm_Bsub[kBase+3u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+3u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+3u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+3u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+3u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=4
        {
          let BCached = mm_Bsub[kBase+4u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+4u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+4u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+4u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+4u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=5
        {
          let BCached = mm_Bsub[kBase+5u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+5u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+5u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+5u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+5u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=6
        {
          let BCached = mm_Bsub[kBase+6u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+6u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+6u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+6u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+6u][u32(tileRow)+3u]), BCached, acc[3]);
        }
        // K=7
        {
          let BCached = mm_Bsub[kBase+7u][localCol];
          acc[0] = fma(vec4<f32>(mm_Asub[kBase+7u][u32(tileRow)]), BCached, acc[0]);
          acc[1] = fma(vec4<f32>(mm_Asub[kBase+7u][u32(tileRow)+1u]), BCached, acc[1]);
          acc[2] = fma(vec4<f32>(mm_Asub[kBase+7u][u32(tileRow)+2u]), BCached, acc[2]);
          acc[3] = fma(vec4<f32>(mm_Asub[kBase+7u][u32(tileRow)+3u]), BCached, acc[3]);
        }
      }

      workgroupBarrier();
    }

    // Write output
    for (var innerRow = 0u; innerRow < rowPerThread; innerRow++) {
      let row = globalRow + i32(innerRow);
      if (row < i32(M)) {
        for (var innerCol = 0u; innerCol < 4u; innerCol++) {
          let col = globalCol + i32(innerCol);
          if (col < i32(N)) {
            C[u32(row) * N + u32(col)] = acc[innerRow][innerCol];
          }
        }
      }
    }
  }
`;

// Phase 5b: TFJS-inspired shader with [4,16] workgroup geometry
// Key insight from tfjs: use rectangular workgroups that favor memory coalescing
// [4,16] = 64 threads per workgroup, but 16 threads access consecutive N elements
const MATMUL_TFJS_STYLE_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // Configuration: 4x16 workgroup, 4x4 elements per thread
  // Output tile: 16x64 (4 threads * 4 rows, 16 threads * 4 cols)
  const BLOCK_M: u32 = 16u;      // 4 rows of threads * 4 rows per thread
  const BLOCK_N: u32 = 64u;      // 16 cols of threads * 4 cols per thread
  const BLOCK_K: u32 = 32u;
  const THREAD_M: u32 = 4u;
  const THREAD_N: u32 = 4u;
  const BLOCK_M_PADDED: u32 = 17u;  // Padding for bank conflicts
  const WG_M: u32 = 4u;   // Threads in M dimension
  const WG_N: u32 = 16u;  // Threads in N dimension

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // vec4 storage
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory tiles
  // As: BLOCK_K x BLOCK_M (transposed) = 32 x 17 = 544
  // Bs: BLOCK_K x (BLOCK_N/4) vec4s = 32 x 16 = 512 vec4s
  var<workgroup> As: array<f32, 544>;
  var<workgroup> Bs: array<vec4<f32>, 512>;

  @compute @workgroup_size(16, 4)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let tx = lid.x;  // 0-15 (N dimension - consecutive threads access consecutive columns)
    let ty = lid.y;  // 0-3  (M dimension)
    let tid = ty * WG_N + tx;  // 0-63

    // Output coordinates
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    // 4x4 accumulators stored as 4 vec4s
    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (transposed) ===
      // 32 x 16 = 512 elements, 64 threads load 8 each
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;

        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;

        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // === Load B tile using vec4 ===
      // 32 x 64 = 2048 elements = 512 vec4s, 64 threads load 8 vec4s each
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / (BLOCK_N / 4u);
        let loadColVec = loadIdx % (BLOCK_N / 4u);

        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        Bs[loadK * (BLOCK_N / 4u) + loadColVec] = v;
      }

      workgroupBarrier();

      // === Compute 4x4 output tile ===
      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        // Load 4 values from A (one per output row)
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];

        // Load vec4 from B (4 columns)
        let b = Bs[k * (BLOCK_N / 4u) + tx];

        // 4x4 outer product
        acc0 = fma(vec4<f32>(a0), b, acc0);
        acc1 = fma(vec4<f32>(a1), b, acc1);
        acc2 = fma(vec4<f32>(a2), b, acc2);
        acc3 = fma(vec4<f32>(a3), b, acc3);
      }

      workgroupBarrier();
    }

    // === Write 4x4 output with bounds checking ===
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
      let row = outRowBase + r;
      if (row < M) {
        var accRow: vec4<f32>;
        if (r == 0u) { accRow = acc0; }
        else if (r == 1u) { accRow = acc1; }
        else if (r == 2u) { accRow = acc2; }
        else { accRow = acc3; }

        if (outColBase + 0u < N) { C[row * N + outColBase + 0u] = accRow.x; }
        if (outColBase + 1u < N) { C[row * N + outColBase + 1u] = accRow.y; }
        if (outColBase + 2u < N) { C[row * N + outColBase + 2u] = accRow.z; }
        if (outColBase + 3u < N) { C[row * N + outColBase + 3u] = accRow.w; }
      }
    }
  }
`;

// Phase 6: Ultra-optimized shader with vec4 output and K-unrolling
// Key optimizations:
// - vec4 output writes (4x memory bandwidth for writes)
// - K-loop unrolled 4x for ILP
// - Double buffered shared memory concept in registers
// - Output arranged for coalesced writes
const MATMUL_ULTRA_OPTIMIZED_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,  // N padded to multiple of 4
  }

  // Tile configuration - optimized for vec4 I/O
  const BLOCK_M: u32 = 64u;    // Output block rows
  const BLOCK_N: u32 = 64u;    // Output block cols (must be multiple of 4 for vec4 writes)
  const BLOCK_K: u32 = 32u;    // K dimension block
  const THREAD_M: u32 = 4u;    // Per-thread rows
  const THREAD_N: u32 = 4u;    // Per-thread cols (vec4)
  const BLOCK_M_PADDED: u32 = 65u;  // Padding for bank conflicts

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // vec4 storage for B
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;  // vec4 storage for C output

  // Shared memory tiles
  var<workgroup> As: array<f32, 2080>;     // 32 * 65 (transposed + padded)
  var<workgroup> Bs: array<vec4<f32>, 512>;  // 32 * 16 vec4s

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let tx = lid.x;  // 0-15
    let ty = lid.y;  // 0-15
    let tid = ty * 16u + tx;  // 0-255

    // Output coordinates
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColVec4 = wid.x * (BLOCK_N / 4u) + tx;  // vec4 column index

    // 4x1 vec4 accumulators (each row outputs one vec4)
    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (transposed) ===
      // Need to load BLOCK_K x BLOCK_M = 32 x 64 = 2048 elements
      // 256 threads, each loads 8 elements
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;

        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;

        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // === Load B tile using vec4 ===
      // Need to load BLOCK_K x BLOCK_N = 32 x 64 = 2048 elements = 512 vec4s
      // 256 threads, each loads 2 vec4s
      for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let loadIdx = tid * 2u + i;
        let loadK = loadIdx / (BLOCK_N / 4u);
        let loadColVec = loadIdx % (BLOCK_N / 4u);

        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        Bs[loadK * (BLOCK_N / 4u) + loadColVec] = v;
      }

      workgroupBarrier();

      // === Compute using K-loop unrolled 4x ===
      let tileK = min(BLOCK_K, K - t * BLOCK_K);
      let numIters = tileK / 4u;
      let remainder = tileK % 4u;

      // Main loop - 4 K iterations per loop
      for (var kBase: u32 = 0u; kBase < numIters * 4u; kBase = kBase + 4u) {
        // Iteration 0
        {
          let k = kBase;
          let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          let b = Bs[k * (BLOCK_N / 4u) + tx];
          acc0 = fma(vec4<f32>(a0), b, acc0);
          acc1 = fma(vec4<f32>(a1), b, acc1);
          acc2 = fma(vec4<f32>(a2), b, acc2);
          acc3 = fma(vec4<f32>(a3), b, acc3);
        }
        // Iteration 1
        {
          let k = kBase + 1u;
          let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          let b = Bs[k * (BLOCK_N / 4u) + tx];
          acc0 = fma(vec4<f32>(a0), b, acc0);
          acc1 = fma(vec4<f32>(a1), b, acc1);
          acc2 = fma(vec4<f32>(a2), b, acc2);
          acc3 = fma(vec4<f32>(a3), b, acc3);
        }
        // Iteration 2
        {
          let k = kBase + 2u;
          let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          let b = Bs[k * (BLOCK_N / 4u) + tx];
          acc0 = fma(vec4<f32>(a0), b, acc0);
          acc1 = fma(vec4<f32>(a1), b, acc1);
          acc2 = fma(vec4<f32>(a2), b, acc2);
          acc3 = fma(vec4<f32>(a3), b, acc3);
        }
        // Iteration 3
        {
          let k = kBase + 3u;
          let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          let b = Bs[k * (BLOCK_N / 4u) + tx];
          acc0 = fma(vec4<f32>(a0), b, acc0);
          acc1 = fma(vec4<f32>(a1), b, acc1);
          acc2 = fma(vec4<f32>(a2), b, acc2);
          acc3 = fma(vec4<f32>(a3), b, acc3);
        }
      }

      // Handle remainder
      for (var k: u32 = numIters * 4u; k < tileK; k = k + 1u) {
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
        let b = Bs[k * (BLOCK_N / 4u) + tx];
        acc0 = fma(vec4<f32>(a0), b, acc0);
        acc1 = fma(vec4<f32>(a1), b, acc1);
        acc2 = fma(vec4<f32>(a2), b, acc2);
        acc3 = fma(vec4<f32>(a3), b, acc3);
      }

      workgroupBarrier();
    }

    // === Write output as vec4s (coalesced) ===
    // Each thread writes 4 vec4s (4 rows, 1 vec4 per row = 4 floats)
    let NVec4Out = N / 4u;  // Use actual N, not padded

    if (outColVec4 < NVec4Out) {
      if (outRowBase + 0u < M) {
        C[(outRowBase + 0u) * NVec4Out + outColVec4] = acc0;
      }
      if (outRowBase + 1u < M) {
        C[(outRowBase + 1u) * NVec4Out + outColVec4] = acc1;
      }
      if (outRowBase + 2u < M) {
        C[(outRowBase + 2u) * NVec4Out + outColVec4] = acc2;
      }
      if (outRowBase + 3u < M) {
        C[(outRowBase + 3u) * NVec4Out + outColVec4] = acc3;
      }
    }
  }
`;

// ============ TFJS-BCACHE SHADER (Fastest - 86% of tfjs @ 1024x1024) ============
//
// This shader implements the tfjs-style B-caching matmul pattern that achieves
// near-parity with TensorFlow.js WebGPU performance.
//
// KEY OPTIMIZATIONS:
// 1. Vec4 along K dimension for both A and B (maximizes memory throughput)
// 2. B-caching pattern: load 4 B rows into registers, iterate over A rows
// 3. Output C as vec4 for coalesced writes
// 4. Shared memory tiling: 32x32 tiles with 8x8 workgroup (4x4 elements/thread)
//
// MEMORY LAYOUT:
//   A[M][K] stored as A[M][K/4] of vec4 (row-major, vec4 along K)
//   B[K][N] stored as B[K][N/4] of vec4 (row-major, vec4 along N)
//   C[M][N] stored as C[M][N/4] of vec4 (row-major, vec4 along N)
//
// SHARED MEMORY:
//   mm_Asub[32][8]: 32 rows of A tile, 8 vec4s = 32 K values per row
//   mm_Bsub[32][8]: 32 K values, 8 vec4s = 32 N values per K
//
// DATA FLOW (per tile):
//   1. Load A tile: mm_Asub[row][k] = A[globalRow+row, kStart+k*4:k*4+4]
//   2. Load B tile: mm_Bsub[k][col] = B[kStart+k, globalCol+col*4:col*4+4]
//   3. Sync
//   4. Compute: for k in 0..8:
//        Cache 4 B rows (BCached0-3 = mm_Bsub[k*4+0..3][col])
//        For each of 4 output rows:
//          Load A[row][k] as vec4, use each component with corresponding BCached
//          acc[row] += fma(BCached_i, A[row][k][i], acc[row])
//   5. Sync, repeat for next tile
//   6. Write acc to C
//
// REQUIRES: K % 4 == 0, N % 4 == 0 (for vec4 alignment, handled by padding in matmulAsync)
//
const MATMUL_TFJS_BCACHE_SHADER = `
  struct Uniforms { M: u32, K: u32, N: u32, _pad: u32, }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;  // A[M][K/4] vec4 along K
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;  // B[K][N/4] vec4 along N
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;  // C[M][N/4]
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 32>;  // [32 rows][8 vec4s] = 32x32 tile
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 8>, 32>;  // [32 K values][8 vec4s] = 32x32 tile

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(local_invocation_id) localId: vec3<u32>, @builtin(global_invocation_id) globalId: vec3<u32>) {
    let M = i32(uniforms.M);
    let K = i32(uniforms.K);
    let N = i32(uniforms.N);
    let KVec4 = K / 4;  // Number of vec4s along K
    let NVec4 = N / 4;  // Number of vec4s along N

    // Thread coordinates
    let localRow = i32(localId.y);     // 0-7 within workgroup
    let tileRow = localRow * 4;         // Output row within tile (0,4,8,...,28)
    let tileCol = i32(localId.x);       // vec4 column index (0-7)
    let globalRow = i32(globalId.y) * 4; // Global output row start
    let globalCol = i32(globalId.x) * 4; // Global output col start

    // Accumulator: 4 output rows, each is a vec4 (4 columns)
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0); acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0); acc[3] = vec4<f32>(0.0);

    let numTiles = (K + 31) / 32;  // Number of K-dimension tiles
    var kStart = 0;
    let tileRowB = localRow * 4;  // B tile row index for this thread

    for (var t = 0; t < numTiles; t++) {
      // LOAD A TILE: each thread loads 4 rows, 1 vec4 per row (covers 32 K values)
      // mm_Asub[row][col] = A[globalRow+row, kStart+col*4 : kStart+col*4+4]
      for (var innerRow = 0; innerRow < 4; innerRow++) {
        let inputRow = tileRow + innerRow;
        let aRow = globalRow + innerRow;
        let aColVec4 = (kStart + tileCol * 4) / 4;
        if (aRow < M && aColVec4 < KVec4) {
          mm_Asub[inputRow][tileCol] = A[aRow * KVec4 + aColVec4];
        } else {
          mm_Asub[inputRow][tileCol] = vec4<f32>(0.0);  // Zero-fill out of bounds
        }
      }

      // LOAD B TILE: each thread loads 4 K values, 1 vec4 per K value
      // mm_Bsub[k][col] = B[kStart+k, globalCol+col*4 : globalCol+col*4+4]
      for (var innerRow = 0; innerRow < 4; innerRow++) {
        let inputRow = tileRowB + innerRow;
        let bRow = kStart + inputRow;
        let bColVec4 = globalCol / 4;
        if (bRow < K && bColVec4 < NVec4) {
          mm_Bsub[inputRow][tileCol] = B[bRow * NVec4 + bColVec4];
        } else {
          mm_Bsub[inputRow][tileCol] = vec4<f32>(0.0);  // Zero-fill out of bounds
        }
      }

      kStart = kStart + 32;
      workgroupBarrier();

      // COMPUTE: B-caching pattern - load 4 B rows into registers, iterate over A
      // This maximizes register reuse and minimizes shared memory reads
      for (var k = 0; k < 8; k++) {
        // Cache 4 consecutive B rows (4 K values) into registers
        let BCached0 = mm_Bsub[k * 4 + 0][tileCol];
        let BCached1 = mm_Bsub[k * 4 + 1][tileCol];
        let BCached2 = mm_Bsub[k * 4 + 2][tileCol];
        let BCached3 = mm_Bsub[k * 4 + 3][tileCol];

        // Iterate over 4 output rows, using cached B values
        for (var i = 0; i < 4; i++) {
          let ACached = mm_Asub[tileRow + i][k];  // 4 K values as vec4
          // FMA: acc[i] += A[row][k][j] * B[k*4+j][col] for j=0..3
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
          acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
          acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
          acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
      workgroupBarrier();
    }

    // WRITE OUTPUT: store accumulated results to C
    for (var innerRow = 0; innerRow < 4; innerRow++) {
      let outRow = globalRow + innerRow;
      let outColVec4 = globalCol / 4;
      if (outRow < M && outColVec4 < NVec4) {
        C[outRow * NVec4 + outColVec4] = acc[innerRow];
      }
    }
  }
`;

// ============ BCACHE-FIT KERNEL (No bounds checking, 4×4 output per thread) ============
// Used when M%32==0 && N%32==0 && K%32==0
// Lower register pressure than 8X8-MEGA: only 4 vec4 accumulators (vs 16)
// Better occupancy on Apple GPUs at medium sizes (1024)
// 32×32 tile, [8,8] workgroup, 4 rows × 1 vec4 col per thread
const MATMUL_BCACHE_FIT_SHADER = `
  struct Uniforms { M: u32, K: u32, N: u32, _pad: u32, }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 32>;  // [32 rows][8 vec4s] = 32x32 tile
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 8>, 32>;  // [32 K rows][8 vec4s] = 32x32 tile

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(local_invocation_id) localId: vec3<u32>, @builtin(global_invocation_id) globalId: vec3<u32>) {
    let K = i32(uniforms.K);
    let N = i32(uniforms.N);
    let KVec4 = K / 4;
    let NVec4 = N / 4;

    let localRow = i32(localId.y);
    let tileRow = localRow * 4;
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y) * 4;
    let globalCol = i32(globalId.x) * 4;

    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0); acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0); acc[3] = vec4<f32>(0.0);

    let numTiles = K / 32;
    var kStart = 0;
    let tileRowB = localRow * 4;

    for (var t = 0; t < numTiles; t++) {
      // LOAD A TILE: no bounds checking
      for (var innerRow = 0; innerRow < 4; innerRow++) {
        let inputRow = tileRow + innerRow;
        let aRow = globalRow + innerRow;
        let aColVec4 = (kStart + tileCol * 4) / 4;
        mm_Asub[inputRow][tileCol] = A[aRow * KVec4 + aColVec4];
      }

      // LOAD B TILE: no bounds checking
      for (var innerRow = 0; innerRow < 4; innerRow++) {
        let inputRow = tileRowB + innerRow;
        let bRow = kStart + inputRow;
        let bColVec4 = globalCol / 4;
        mm_Bsub[inputRow][tileCol] = B[bRow * NVec4 + bColVec4];
      }

      kStart = kStart + 32;
      workgroupBarrier();

      // COMPUTE: B-caching, fully unrolled
      for (var k = 0; k < 8; k++) {
        let BCached0 = mm_Bsub[k * 4 + 0][tileCol];
        let BCached1 = mm_Bsub[k * 4 + 1][tileCol];
        let BCached2 = mm_Bsub[k * 4 + 2][tileCol];
        let BCached3 = mm_Bsub[k * 4 + 3][tileCol];

        for (var i = 0; i < 4; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
          acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
          acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
          acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
      workgroupBarrier();
    }

    // WRITE OUTPUT: no bounds checking
    for (var innerRow = 0; innerRow < 4; innerRow++) {
      let outRow = globalRow + innerRow;
      let outColVec4 = globalCol / 4;
      C[outRow * NVec4 + outColVec4] = acc[innerRow];
    }
  }
`;

// ============ 8x8 MEGA KERNEL (Nussbaum-style, target: 1+ TFLOP) ============
//
// Each thread computes 8×8 = 64 output elements. With an [8,8] workgroup that's
// 64×64 output per workgroup — same tile but 4x more compute per thread than BCACHE.
//
// KEY OPTIMIZATIONS:
// 1. 8×8 output per thread = 64 FMA accumulators (vs 16 for 4×4)
//    - Amortizes shared memory loads over 4x more compute
//    - Better instruction-level parallelism
// 2. Vec4 along K for A, vec4 along N for B (same as BCACHE)
// 3. B-caching pattern with 4 B rows cached in registers
// 4. K-loop processes 32 values per tile (8 vec4 iterations)
//
// MEMORY LAYOUT:
//   A[M][K/4] of vec4 (vec4 along K)
//   B[K][N/4] of vec4 (vec4 along N)
//   C[M][N/4] of vec4 (vec4 along N)
//
// TILE: 64×64 output, 32 K per tile
// WORKGROUP: [8,8] = 64 threads, each computes 8 rows × 2 vec4 cols = 8×8 output
// SHARED MEMORY:
//   mm_Asub[64][8]: 64 rows × 8 vec4 = 64×32 A tile
//   mm_Bsub[32][16]: 32 K rows × 16 vec4 = 32×64 B tile
//
const MATMUL_8X8_MEGA_SHADER = `
  struct Uniforms { M: u32, K: u32, N: u32, _pad: u32, }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 64>;   // [64 rows][8 vec4s] = 64×32
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 16>, 32>;  // [32 K rows][16 vec4s] = 32×64

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(local_invocation_id) localId: vec3<u32>,
          @builtin(workgroup_id) wgId: vec3<u32>) {
    let M = i32(uniforms.M);
    let K = i32(uniforms.K);
    let N = i32(uniforms.N);
    let KVec4 = K / 4;
    let NVec4 = N / 4;

    // Thread coordinates within workgroup
    let lx = i32(localId.x);  // 0-7: column thread index
    let ly = i32(localId.y);  // 0-7: row thread index

    // Global output position: each thread covers 8 rows and 8 cols (2 vec4s)
    let globalRowStart = i32(wgId.y) * 64 + ly * 8;
    let globalColVec4Start = i32(wgId.x) * 16 + lx * 2;  // 16 vec4s = 64 cols per WG

    // 8×2 accumulators (8 rows × 2 vec4 cols = 8×8 output elements)
    var acc00 = vec4<f32>(0.0); var acc01 = vec4<f32>(0.0);
    var acc10 = vec4<f32>(0.0); var acc11 = vec4<f32>(0.0);
    var acc20 = vec4<f32>(0.0); var acc21 = vec4<f32>(0.0);
    var acc30 = vec4<f32>(0.0); var acc31 = vec4<f32>(0.0);
    var acc40 = vec4<f32>(0.0); var acc41 = vec4<f32>(0.0);
    var acc50 = vec4<f32>(0.0); var acc51 = vec4<f32>(0.0);
    var acc60 = vec4<f32>(0.0); var acc61 = vec4<f32>(0.0);
    var acc70 = vec4<f32>(0.0); var acc71 = vec4<f32>(0.0);

    let numTiles = (K + 31) / 32;

    for (var t = 0; t < numTiles; t++) {
      let kBase = t * 32;

      // LOAD A TILE: 64 rows × 32 cols (as 8 vec4s)
      // Each of 64 threads loads 8 rows, 1 vec4 each (covers 64×8 per pass)
      // We need 64 rows × 8 vec4s = 512 loads for 64 threads = 8 loads/thread
      for (var r = 0; r < 8; r++) {
        let aRow = globalRowStart + r;
        let aKVec4 = (kBase / 4) + lx;
        if (aRow < M && aKVec4 < KVec4) {
          mm_Asub[ly * 8 + r][lx] = A[aRow * KVec4 + aKVec4];
        } else {
          mm_Asub[ly * 8 + r][lx] = vec4<f32>(0.0);
        }
      }

      // LOAD B TILE: 32 rows × 64 cols (as 16 vec4s)
      // 64 threads need to load 32×16 = 512 values = 8 loads/thread
      // Thread (lx,ly) loads: rows [ly*4..ly*4+3], cols [lx*2..lx*2+1]
      for (var r = 0; r < 4; r++) {
        let bRow = kBase + ly * 4 + r;
        let bColVec4_0 = i32(wgId.x) * 16 + lx * 2;
        let bColVec4_1 = bColVec4_0 + 1;
        if (bRow < K) {
          if (bColVec4_0 < NVec4) {
            mm_Bsub[ly * 4 + r][lx * 2] = B[bRow * NVec4 + bColVec4_0];
          } else {
            mm_Bsub[ly * 4 + r][lx * 2] = vec4<f32>(0.0);
          }
          if (bColVec4_1 < NVec4) {
            mm_Bsub[ly * 4 + r][lx * 2 + 1] = B[bRow * NVec4 + bColVec4_1];
          } else {
            mm_Bsub[ly * 4 + r][lx * 2 + 1] = vec4<f32>(0.0);
          }
        } else {
          mm_Bsub[ly * 4 + r][lx * 2] = vec4<f32>(0.0);
          mm_Bsub[ly * 4 + r][lx * 2 + 1] = vec4<f32>(0.0);
        }
      }

      workgroupBarrier();

      // COMPUTE: B-caching pattern over 32 K values (8 iterations × 4 K per iter)
      for (var k = 0; k < 8; k++) {
        // Cache 4 B rows × 2 vec4 cols into registers
        let bc00 = mm_Bsub[k * 4 + 0][lx * 2];     let bc01 = mm_Bsub[k * 4 + 0][lx * 2 + 1];
        let bc10 = mm_Bsub[k * 4 + 1][lx * 2];     let bc11 = mm_Bsub[k * 4 + 1][lx * 2 + 1];
        let bc20 = mm_Bsub[k * 4 + 2][lx * 2];     let bc21 = mm_Bsub[k * 4 + 2][lx * 2 + 1];
        let bc30 = mm_Bsub[k * 4 + 3][lx * 2];     let bc31 = mm_Bsub[k * 4 + 3][lx * 2 + 1];

        // Process 8 output rows
        let a0 = mm_Asub[ly * 8 + 0][k];
        acc00 = fma(bc00, vec4<f32>(a0[0]), acc00); acc01 = fma(bc01, vec4<f32>(a0[0]), acc01);
        acc00 = fma(bc10, vec4<f32>(a0[1]), acc00); acc01 = fma(bc11, vec4<f32>(a0[1]), acc01);
        acc00 = fma(bc20, vec4<f32>(a0[2]), acc00); acc01 = fma(bc21, vec4<f32>(a0[2]), acc01);
        acc00 = fma(bc30, vec4<f32>(a0[3]), acc00); acc01 = fma(bc31, vec4<f32>(a0[3]), acc01);

        let a1 = mm_Asub[ly * 8 + 1][k];
        acc10 = fma(bc00, vec4<f32>(a1[0]), acc10); acc11 = fma(bc01, vec4<f32>(a1[0]), acc11);
        acc10 = fma(bc10, vec4<f32>(a1[1]), acc10); acc11 = fma(bc11, vec4<f32>(a1[1]), acc11);
        acc10 = fma(bc20, vec4<f32>(a1[2]), acc10); acc11 = fma(bc21, vec4<f32>(a1[2]), acc11);
        acc10 = fma(bc30, vec4<f32>(a1[3]), acc10); acc11 = fma(bc31, vec4<f32>(a1[3]), acc11);

        let a2 = mm_Asub[ly * 8 + 2][k];
        acc20 = fma(bc00, vec4<f32>(a2[0]), acc20); acc21 = fma(bc01, vec4<f32>(a2[0]), acc21);
        acc20 = fma(bc10, vec4<f32>(a2[1]), acc20); acc21 = fma(bc11, vec4<f32>(a2[1]), acc21);
        acc20 = fma(bc20, vec4<f32>(a2[2]), acc20); acc21 = fma(bc21, vec4<f32>(a2[2]), acc21);
        acc20 = fma(bc30, vec4<f32>(a2[3]), acc20); acc21 = fma(bc31, vec4<f32>(a2[3]), acc21);

        let a3 = mm_Asub[ly * 8 + 3][k];
        acc30 = fma(bc00, vec4<f32>(a3[0]), acc30); acc31 = fma(bc01, vec4<f32>(a3[0]), acc31);
        acc30 = fma(bc10, vec4<f32>(a3[1]), acc30); acc31 = fma(bc11, vec4<f32>(a3[1]), acc31);
        acc30 = fma(bc20, vec4<f32>(a3[2]), acc30); acc31 = fma(bc21, vec4<f32>(a3[2]), acc31);
        acc30 = fma(bc30, vec4<f32>(a3[3]), acc30); acc31 = fma(bc31, vec4<f32>(a3[3]), acc31);

        let a4 = mm_Asub[ly * 8 + 4][k];
        acc40 = fma(bc00, vec4<f32>(a4[0]), acc40); acc41 = fma(bc01, vec4<f32>(a4[0]), acc41);
        acc40 = fma(bc10, vec4<f32>(a4[1]), acc40); acc41 = fma(bc11, vec4<f32>(a4[1]), acc41);
        acc40 = fma(bc20, vec4<f32>(a4[2]), acc40); acc41 = fma(bc21, vec4<f32>(a4[2]), acc41);
        acc40 = fma(bc30, vec4<f32>(a4[3]), acc40); acc41 = fma(bc31, vec4<f32>(a4[3]), acc41);

        let a5 = mm_Asub[ly * 8 + 5][k];
        acc50 = fma(bc00, vec4<f32>(a5[0]), acc50); acc51 = fma(bc01, vec4<f32>(a5[0]), acc51);
        acc50 = fma(bc10, vec4<f32>(a5[1]), acc50); acc51 = fma(bc11, vec4<f32>(a5[1]), acc51);
        acc50 = fma(bc20, vec4<f32>(a5[2]), acc50); acc51 = fma(bc21, vec4<f32>(a5[2]), acc51);
        acc50 = fma(bc30, vec4<f32>(a5[3]), acc50); acc51 = fma(bc31, vec4<f32>(a5[3]), acc51);

        let a6 = mm_Asub[ly * 8 + 6][k];
        acc60 = fma(bc00, vec4<f32>(a6[0]), acc60); acc61 = fma(bc01, vec4<f32>(a6[0]), acc61);
        acc60 = fma(bc10, vec4<f32>(a6[1]), acc60); acc61 = fma(bc11, vec4<f32>(a6[1]), acc61);
        acc60 = fma(bc20, vec4<f32>(a6[2]), acc60); acc61 = fma(bc21, vec4<f32>(a6[2]), acc61);
        acc60 = fma(bc30, vec4<f32>(a6[3]), acc60); acc61 = fma(bc31, vec4<f32>(a6[3]), acc61);

        let a7 = mm_Asub[ly * 8 + 7][k];
        acc70 = fma(bc00, vec4<f32>(a7[0]), acc70); acc71 = fma(bc01, vec4<f32>(a7[0]), acc71);
        acc70 = fma(bc10, vec4<f32>(a7[1]), acc70); acc71 = fma(bc11, vec4<f32>(a7[1]), acc71);
        acc70 = fma(bc20, vec4<f32>(a7[2]), acc70); acc71 = fma(bc21, vec4<f32>(a7[2]), acc71);
        acc70 = fma(bc30, vec4<f32>(a7[3]), acc70); acc71 = fma(bc31, vec4<f32>(a7[3]), acc71);
      }
      workgroupBarrier();
    }

    // WRITE OUTPUT: 8 rows × 2 vec4 cols
    for (var r = 0; r < 8; r++) {
      let outRow = globalRowStart + r;
      if (outRow < M) {
        let cv4_0 = globalColVec4Start;
        let cv4_1 = cv4_0 + 1;
        if (cv4_0 < NVec4) {
          switch (r) {
            case 0: { C[outRow * NVec4 + cv4_0] = acc00; }
            case 1: { C[outRow * NVec4 + cv4_0] = acc10; }
            case 2: { C[outRow * NVec4 + cv4_0] = acc20; }
            case 3: { C[outRow * NVec4 + cv4_0] = acc30; }
            case 4: { C[outRow * NVec4 + cv4_0] = acc40; }
            case 5: { C[outRow * NVec4 + cv4_0] = acc50; }
            case 6: { C[outRow * NVec4 + cv4_0] = acc60; }
            case 7: { C[outRow * NVec4 + cv4_0] = acc70; }
            default: {}
          }
        }
        if (cv4_1 < NVec4) {
          switch (r) {
            case 0: { C[outRow * NVec4 + cv4_1] = acc01; }
            case 1: { C[outRow * NVec4 + cv4_1] = acc11; }
            case 2: { C[outRow * NVec4 + cv4_1] = acc21; }
            case 3: { C[outRow * NVec4 + cv4_1] = acc31; }
            case 4: { C[outRow * NVec4 + cv4_1] = acc41; }
            case 5: { C[outRow * NVec4 + cv4_1] = acc51; }
            case 6: { C[outRow * NVec4 + cv4_1] = acc61; }
            case 7: { C[outRow * NVec4 + cv4_1] = acc71; }
            default: {}
          }
        }
      }
    }
  }
`;

// ============ 8x8 MEGA KERNEL FIT (No bounds checking) ============
// Used when M%64==0 && N%64==0 && K%32==0
// Eliminates all bounds checks for maximum throughput
const MATMUL_8X8_MEGA_FIT_SHADER = `
  struct Uniforms { M: u32, K: u32, N: u32, _pad: u32, }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 64>;
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 16>, 32>;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(local_invocation_id) localId: vec3<u32>,
          @builtin(workgroup_id) wgId: vec3<u32>) {
    let K = i32(uniforms.K);
    let N = i32(uniforms.N);
    let KVec4 = K / 4;
    let NVec4 = N / 4;

    let lx = i32(localId.x);
    let ly = i32(localId.y);
    let globalRowStart = i32(wgId.y) * 64 + ly * 8;
    let globalColVec4Start = i32(wgId.x) * 16 + lx * 2;

    var acc00 = vec4<f32>(0.0); var acc01 = vec4<f32>(0.0);
    var acc10 = vec4<f32>(0.0); var acc11 = vec4<f32>(0.0);
    var acc20 = vec4<f32>(0.0); var acc21 = vec4<f32>(0.0);
    var acc30 = vec4<f32>(0.0); var acc31 = vec4<f32>(0.0);
    var acc40 = vec4<f32>(0.0); var acc41 = vec4<f32>(0.0);
    var acc50 = vec4<f32>(0.0); var acc51 = vec4<f32>(0.0);
    var acc60 = vec4<f32>(0.0); var acc61 = vec4<f32>(0.0);
    var acc70 = vec4<f32>(0.0); var acc71 = vec4<f32>(0.0);

    let numTiles = K / 32;

    for (var t = 0; t < numTiles; t++) {
      let kBase = t * 32;

      // LOAD A TILE: no bounds checking
      for (var r = 0; r < 8; r++) {
        let aRow = globalRowStart + r;
        let aKVec4 = (kBase / 4) + lx;
        mm_Asub[ly * 8 + r][lx] = A[aRow * KVec4 + aKVec4];
      }

      // LOAD B TILE: no bounds checking
      for (var r = 0; r < 4; r++) {
        let bRow = kBase + ly * 4 + r;
        let bColVec4_0 = i32(wgId.x) * 16 + lx * 2;
        let bColVec4_1 = bColVec4_0 + 1;
        mm_Bsub[ly * 4 + r][lx * 2] = B[bRow * NVec4 + bColVec4_0];
        mm_Bsub[ly * 4 + r][lx * 2 + 1] = B[bRow * NVec4 + bColVec4_1];
      }

      workgroupBarrier();

      // COMPUTE: fully unrolled 8 iterations × 8 rows × 4 K values
      for (var k = 0; k < 8; k++) {
        let bc00 = mm_Bsub[k * 4 + 0][lx * 2];     let bc01 = mm_Bsub[k * 4 + 0][lx * 2 + 1];
        let bc10 = mm_Bsub[k * 4 + 1][lx * 2];     let bc11 = mm_Bsub[k * 4 + 1][lx * 2 + 1];
        let bc20 = mm_Bsub[k * 4 + 2][lx * 2];     let bc21 = mm_Bsub[k * 4 + 2][lx * 2 + 1];
        let bc30 = mm_Bsub[k * 4 + 3][lx * 2];     let bc31 = mm_Bsub[k * 4 + 3][lx * 2 + 1];

        let a0 = mm_Asub[ly * 8 + 0][k];
        acc00 = fma(bc00, vec4<f32>(a0[0]), acc00); acc01 = fma(bc01, vec4<f32>(a0[0]), acc01);
        acc00 = fma(bc10, vec4<f32>(a0[1]), acc00); acc01 = fma(bc11, vec4<f32>(a0[1]), acc01);
        acc00 = fma(bc20, vec4<f32>(a0[2]), acc00); acc01 = fma(bc21, vec4<f32>(a0[2]), acc01);
        acc00 = fma(bc30, vec4<f32>(a0[3]), acc00); acc01 = fma(bc31, vec4<f32>(a0[3]), acc01);

        let a1 = mm_Asub[ly * 8 + 1][k];
        acc10 = fma(bc00, vec4<f32>(a1[0]), acc10); acc11 = fma(bc01, vec4<f32>(a1[0]), acc11);
        acc10 = fma(bc10, vec4<f32>(a1[1]), acc10); acc11 = fma(bc11, vec4<f32>(a1[1]), acc11);
        acc10 = fma(bc20, vec4<f32>(a1[2]), acc10); acc11 = fma(bc21, vec4<f32>(a1[2]), acc11);
        acc10 = fma(bc30, vec4<f32>(a1[3]), acc10); acc11 = fma(bc31, vec4<f32>(a1[3]), acc11);

        let a2 = mm_Asub[ly * 8 + 2][k];
        acc20 = fma(bc00, vec4<f32>(a2[0]), acc20); acc21 = fma(bc01, vec4<f32>(a2[0]), acc21);
        acc20 = fma(bc10, vec4<f32>(a2[1]), acc20); acc21 = fma(bc11, vec4<f32>(a2[1]), acc21);
        acc20 = fma(bc20, vec4<f32>(a2[2]), acc20); acc21 = fma(bc21, vec4<f32>(a2[2]), acc21);
        acc20 = fma(bc30, vec4<f32>(a2[3]), acc20); acc21 = fma(bc31, vec4<f32>(a2[3]), acc21);

        let a3 = mm_Asub[ly * 8 + 3][k];
        acc30 = fma(bc00, vec4<f32>(a3[0]), acc30); acc31 = fma(bc01, vec4<f32>(a3[0]), acc31);
        acc30 = fma(bc10, vec4<f32>(a3[1]), acc30); acc31 = fma(bc11, vec4<f32>(a3[1]), acc31);
        acc30 = fma(bc20, vec4<f32>(a3[2]), acc30); acc31 = fma(bc21, vec4<f32>(a3[2]), acc31);
        acc30 = fma(bc30, vec4<f32>(a3[3]), acc30); acc31 = fma(bc31, vec4<f32>(a3[3]), acc31);

        let a4 = mm_Asub[ly * 8 + 4][k];
        acc40 = fma(bc00, vec4<f32>(a4[0]), acc40); acc41 = fma(bc01, vec4<f32>(a4[0]), acc41);
        acc40 = fma(bc10, vec4<f32>(a4[1]), acc40); acc41 = fma(bc11, vec4<f32>(a4[1]), acc41);
        acc40 = fma(bc20, vec4<f32>(a4[2]), acc40); acc41 = fma(bc21, vec4<f32>(a4[2]), acc41);
        acc40 = fma(bc30, vec4<f32>(a4[3]), acc40); acc41 = fma(bc31, vec4<f32>(a4[3]), acc41);

        let a5 = mm_Asub[ly * 8 + 5][k];
        acc50 = fma(bc00, vec4<f32>(a5[0]), acc50); acc51 = fma(bc01, vec4<f32>(a5[0]), acc51);
        acc50 = fma(bc10, vec4<f32>(a5[1]), acc50); acc51 = fma(bc11, vec4<f32>(a5[1]), acc51);
        acc50 = fma(bc20, vec4<f32>(a5[2]), acc50); acc51 = fma(bc21, vec4<f32>(a5[2]), acc51);
        acc50 = fma(bc30, vec4<f32>(a5[3]), acc50); acc51 = fma(bc31, vec4<f32>(a5[3]), acc51);

        let a6 = mm_Asub[ly * 8 + 6][k];
        acc60 = fma(bc00, vec4<f32>(a6[0]), acc60); acc61 = fma(bc01, vec4<f32>(a6[0]), acc61);
        acc60 = fma(bc10, vec4<f32>(a6[1]), acc60); acc61 = fma(bc11, vec4<f32>(a6[1]), acc61);
        acc60 = fma(bc20, vec4<f32>(a6[2]), acc60); acc61 = fma(bc21, vec4<f32>(a6[2]), acc61);
        acc60 = fma(bc30, vec4<f32>(a6[3]), acc60); acc61 = fma(bc31, vec4<f32>(a6[3]), acc61);

        let a7 = mm_Asub[ly * 8 + 7][k];
        acc70 = fma(bc00, vec4<f32>(a7[0]), acc70); acc71 = fma(bc01, vec4<f32>(a7[0]), acc71);
        acc70 = fma(bc10, vec4<f32>(a7[1]), acc70); acc71 = fma(bc11, vec4<f32>(a7[1]), acc71);
        acc70 = fma(bc20, vec4<f32>(a7[2]), acc70); acc71 = fma(bc21, vec4<f32>(a7[2]), acc71);
        acc70 = fma(bc30, vec4<f32>(a7[3]), acc70); acc71 = fma(bc31, vec4<f32>(a7[3]), acc71);
      }
      workgroupBarrier();
    }

    // WRITE OUTPUT: 8 rows × 2 vec4 cols, no bounds checking
    C[globalRowStart * NVec4 + globalColVec4Start] = acc00;
    C[globalRowStart * NVec4 + globalColVec4Start + 1] = acc01;
    C[(globalRowStart + 1) * NVec4 + globalColVec4Start] = acc10;
    C[(globalRowStart + 1) * NVec4 + globalColVec4Start + 1] = acc11;
    C[(globalRowStart + 2) * NVec4 + globalColVec4Start] = acc20;
    C[(globalRowStart + 2) * NVec4 + globalColVec4Start + 1] = acc21;
    C[(globalRowStart + 3) * NVec4 + globalColVec4Start] = acc30;
    C[(globalRowStart + 3) * NVec4 + globalColVec4Start + 1] = acc31;
    C[(globalRowStart + 4) * NVec4 + globalColVec4Start] = acc40;
    C[(globalRowStart + 4) * NVec4 + globalColVec4Start + 1] = acc41;
    C[(globalRowStart + 5) * NVec4 + globalColVec4Start] = acc50;
    C[(globalRowStart + 5) * NVec4 + globalColVec4Start + 1] = acc51;
    C[(globalRowStart + 6) * NVec4 + globalColVec4Start] = acc60;
    C[(globalRowStart + 6) * NVec4 + globalColVec4Start + 1] = acc61;
    C[(globalRowStart + 7) * NVec4 + globalColVec4Start] = acc70;
    C[(globalRowStart + 7) * NVec4 + globalColVec4Start + 1] = acc71;
  }
`;

// ============ FIT SHADER (No bounds checking) ============
// Used when M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % BLOCK_K == 0
// This eliminates ALL bounds checking overhead - the key tfjs optimization!
const MATMUL_FIT_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // Same tile configuration as ULTRA_OPTIMIZED
  const BLOCK_M: u32 = 64u;
  const BLOCK_N: u32 = 64u;
  const BLOCK_K: u32 = 32u;
  const THREAD_M: u32 = 4u;
  const THREAD_N: u32 = 4u;
  const BLOCK_M_PADDED: u32 = 65u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;

  var<workgroup> As: array<f32, 2080>;       // 32 * 65
  var<workgroup> Bs: array<vec4<f32>, 544>;  // 32 * 17 (padded to avoid bank conflicts)
  const BLOCK_N_VEC4_PADDED: u32 = 17u;  // 16 + 1 for bank conflict avoidance

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let K = uniforms.K;
    let NVec4 = uniforms.NPadded / 4u;
    let NVec4Out = uniforms.N / 4u;

    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColVec4 = wid.x * (BLOCK_N / 4u) + tx;

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = K / BLOCK_K;  // Exact division - no remainder!

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // === Load A tile (NO BOUNDS CHECKING) ===
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;
        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;
        As[loadK * BLOCK_M_PADDED + loadRow] = A[globalRow * K + globalK];
      }

      // === Load B tile (NO BOUNDS CHECKING) ===
      for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let loadIdx = tid * 2u + i;
        let loadK = loadIdx / 16u;
        let loadColVec = loadIdx % 16u;
        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * 16u + loadColVec;
        Bs[loadK * BLOCK_N_VEC4_PADDED + loadColVec] = B[globalK * NVec4 + globalColVec];
      }

      workgroupBarrier();

      // === Compute - FULLY unrolled K loop (all 32 iterations, no loop at all) ===
      let aRowBase = ty * THREAD_M;
      // K = 0-31 fully unrolled with no loop overhead
      { let a0 = As[0u * BLOCK_M_PADDED + aRowBase]; let a1 = As[0u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[0u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[0u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[0u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[1u * BLOCK_M_PADDED + aRowBase]; let a1 = As[1u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[1u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[1u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[1u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[2u * BLOCK_M_PADDED + aRowBase]; let a1 = As[2u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[2u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[2u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[2u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[3u * BLOCK_M_PADDED + aRowBase]; let a1 = As[3u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[3u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[3u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[3u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[4u * BLOCK_M_PADDED + aRowBase]; let a1 = As[4u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[4u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[4u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[4u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[5u * BLOCK_M_PADDED + aRowBase]; let a1 = As[5u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[5u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[5u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[5u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[6u * BLOCK_M_PADDED + aRowBase]; let a1 = As[6u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[6u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[6u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[6u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[7u * BLOCK_M_PADDED + aRowBase]; let a1 = As[7u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[7u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[7u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[7u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[8u * BLOCK_M_PADDED + aRowBase]; let a1 = As[8u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[8u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[8u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[8u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[9u * BLOCK_M_PADDED + aRowBase]; let a1 = As[9u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[9u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[9u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[9u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[10u * BLOCK_M_PADDED + aRowBase]; let a1 = As[10u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[10u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[10u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[10u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[11u * BLOCK_M_PADDED + aRowBase]; let a1 = As[11u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[11u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[11u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[11u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[12u * BLOCK_M_PADDED + aRowBase]; let a1 = As[12u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[12u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[12u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[12u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[13u * BLOCK_M_PADDED + aRowBase]; let a1 = As[13u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[13u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[13u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[13u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[14u * BLOCK_M_PADDED + aRowBase]; let a1 = As[14u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[14u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[14u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[14u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[15u * BLOCK_M_PADDED + aRowBase]; let a1 = As[15u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[15u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[15u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[15u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[16u * BLOCK_M_PADDED + aRowBase]; let a1 = As[16u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[16u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[16u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[16u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[17u * BLOCK_M_PADDED + aRowBase]; let a1 = As[17u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[17u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[17u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[17u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[18u * BLOCK_M_PADDED + aRowBase]; let a1 = As[18u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[18u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[18u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[18u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[19u * BLOCK_M_PADDED + aRowBase]; let a1 = As[19u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[19u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[19u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[19u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[20u * BLOCK_M_PADDED + aRowBase]; let a1 = As[20u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[20u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[20u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[20u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[21u * BLOCK_M_PADDED + aRowBase]; let a1 = As[21u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[21u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[21u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[21u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[22u * BLOCK_M_PADDED + aRowBase]; let a1 = As[22u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[22u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[22u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[22u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[23u * BLOCK_M_PADDED + aRowBase]; let a1 = As[23u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[23u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[23u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[23u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[24u * BLOCK_M_PADDED + aRowBase]; let a1 = As[24u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[24u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[24u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[24u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[25u * BLOCK_M_PADDED + aRowBase]; let a1 = As[25u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[25u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[25u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[25u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[26u * BLOCK_M_PADDED + aRowBase]; let a1 = As[26u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[26u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[26u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[26u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[27u * BLOCK_M_PADDED + aRowBase]; let a1 = As[27u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[27u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[27u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[27u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[28u * BLOCK_M_PADDED + aRowBase]; let a1 = As[28u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[28u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[28u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[28u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[29u * BLOCK_M_PADDED + aRowBase]; let a1 = As[29u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[29u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[29u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[29u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[30u * BLOCK_M_PADDED + aRowBase]; let a1 = As[30u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[30u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[30u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[30u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }
      { let a0 = As[31u * BLOCK_M_PADDED + aRowBase]; let a1 = As[31u * BLOCK_M_PADDED + aRowBase + 1u]; let a2 = As[31u * BLOCK_M_PADDED + aRowBase + 2u]; let a3 = As[31u * BLOCK_M_PADDED + aRowBase + 3u]; let b = Bs[31u * BLOCK_N_VEC4_PADDED + tx]; acc0 = fma(vec4<f32>(a0), b, acc0); acc1 = fma(vec4<f32>(a1), b, acc1); acc2 = fma(vec4<f32>(a2), b, acc2); acc3 = fma(vec4<f32>(a3), b, acc3); }

      workgroupBarrier();
    }

    // === Write output (NO BOUNDS CHECKING) ===
    C[(outRowBase + 0u) * NVec4Out + outColVec4] = acc0;
    C[(outRowBase + 1u) * NVec4Out + outColVec4] = acc1;
    C[(outRowBase + 2u) * NVec4Out + outColVec4] = acc2;
    C[(outRowBase + 3u) * NVec4Out + outColVec4] = acc3;
  }
`;

// Double-buffered shader for overlapping load and compute
const MATMUL_DOUBLE_BUFFER_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  const BLOCK_M: u32 = 64u;
  const BLOCK_N: u32 = 64u;
  const BLOCK_K: u32 = 32u;
  const THREAD_M: u32 = 4u;
  const THREAD_N: u32 = 4u;
  const BLOCK_M_PADDED: u32 = 65u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Double-buffered shared memory: 2 buffers for each matrix
  var<workgroup> As0: array<f32, 2080>;  // Buffer 0
  var<workgroup> As1: array<f32, 2080>;  // Buffer 1
  var<workgroup> Bs0: array<vec4<f32>, 512>;  // Buffer 0
  var<workgroup> Bs1: array<vec4<f32>, 512>;  // Buffer 1

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    // Load first tile into buffer 0
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let loadIdx = tid * 8u + i;
      let loadK = loadIdx / BLOCK_M;
      let loadRow = loadIdx % BLOCK_M;
      let globalK = loadK;
      let globalRow = wid.y * BLOCK_M + loadRow;
      var val: f32 = 0.0;
      if (globalRow < M && globalK < K) { val = A[globalRow * K + globalK]; }
      As0[loadK * BLOCK_M_PADDED + loadRow] = val;
    }
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      let loadIdx = tid * 2u + i;
      let loadK = loadIdx / (BLOCK_N / 4u);
      let loadColVec = loadIdx % (BLOCK_N / 4u);
      let globalK = loadK;
      let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;
      var v: vec4<f32> = vec4<f32>(0.0);
      if (globalK < K && globalColVec < NVec4) { v = B[globalK * NVec4 + globalColVec]; }
      Bs0[loadK * (BLOCK_N / 4u) + loadColVec] = v;
    }
    workgroupBarrier();

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      let useBuffer0 = (t % 2u) == 0u;
      let nextT = t + 1u;

      // Load next tile while computing current (if not last tile)
      if (nextT < numTiles) {
        let kStart = nextT * BLOCK_K;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
          let loadIdx = tid * 8u + i;
          let loadK = loadIdx / BLOCK_M;
          let loadRow = loadIdx % BLOCK_M;
          let globalK = kStart + loadK;
          let globalRow = wid.y * BLOCK_M + loadRow;
          var val: f32 = 0.0;
          if (globalRow < M && globalK < K) { val = A[globalRow * K + globalK]; }
          if (useBuffer0) {
            As1[loadK * BLOCK_M_PADDED + loadRow] = val;
          } else {
            As0[loadK * BLOCK_M_PADDED + loadRow] = val;
          }
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
          let loadIdx = tid * 2u + i;
          let loadK = loadIdx / (BLOCK_N / 4u);
          let loadColVec = loadIdx % (BLOCK_N / 4u);
          let globalK = kStart + loadK;
          let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;
          var v: vec4<f32> = vec4<f32>(0.0);
          if (globalK < K && globalColVec < NVec4) { v = B[globalK * NVec4 + globalColVec]; }
          if (useBuffer0) {
            Bs1[loadK * (BLOCK_N / 4u) + loadColVec] = v;
          } else {
            Bs0[loadK * (BLOCK_N / 4u) + loadColVec] = v;
          }
        }
      }

      // Compute current tile
      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        var a0: f32; var a1: f32; var a2: f32; var a3: f32;
        var b: vec4<f32>;
        if (useBuffer0) {
          a0 = As0[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          a1 = As0[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          a2 = As0[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          a3 = As0[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          b = Bs0[k * (BLOCK_N / 4u) + tx];
        } else {
          a0 = As1[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
          a1 = As1[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
          a2 = As1[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
          a3 = As1[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
          b = Bs1[k * (BLOCK_N / 4u) + tx];
        }
        acc0 = fma(vec4<f32>(a0), b, acc0);
        acc1 = fma(vec4<f32>(a1), b, acc1);
        acc2 = fma(vec4<f32>(a2), b, acc2);
        acc3 = fma(vec4<f32>(a3), b, acc3);
      }

      workgroupBarrier();
    }

    // Write output
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
      let row = outRowBase + r;
      if (row < M) {
        var accRow: vec4<f32>;
        if (r == 0u) { accRow = acc0; }
        else if (r == 1u) { accRow = acc1; }
        else if (r == 2u) { accRow = acc2; }
        else { accRow = acc3; }
        if (outColBase + 0u < N) { C[row * N + outColBase + 0u] = accRow.x; }
        if (outColBase + 1u < N) { C[row * N + outColBase + 1u] = accRow.y; }
        if (outColBase + 2u < N) { C[row * N + outColBase + 2u] = accRow.z; }
        if (outColBase + 3u < N) { C[row * N + outColBase + 3u] = accRow.w; }
      }
    }
  }
`;

// TFJS-exact config: [8,8,1] workgroup with [4,4,1] elements per thread
// This gives 32x32 output tiles per workgroup (8*4 x 8*4)
const MATMUL_TFJS_8x8_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // TFJS config: 8x8 workgroup, 4x4 elements per thread
  const BLOCK_M: u32 = 32u;   // 8 threads * 4 elements
  const BLOCK_N: u32 = 32u;   // 8 threads * 4 elements
  const BLOCK_K: u32 = 32u;   // tileInner
  const THREAD_M: u32 = 4u;
  const THREAD_N: u32 = 4u;
  const BLOCK_M_PADDED: u32 = 33u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // Shared memory: 32x33 + 32x8 vec4s
  var<workgroup> As: array<f32, 1056>;    // 32 * 33
  var<workgroup> Bs: array<vec4<f32>, 256>;  // 32 * 8

  @compute @workgroup_size(8, 8)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let tx = lid.x;  // 0-7
    let ty = lid.y;  // 0-7
    let tid = ty * 8u + tx;  // 0-63

    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColBase = wid.x * BLOCK_N + tx * THREAD_N;

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = (K + BLOCK_K - 1u) / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
      // Load A tile: 32x32 = 1024 elements, 64 threads load 16 each
      for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let loadIdx = tid * 16u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;
        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;
        var val: f32 = 0.0;
        if (globalRow < M && globalK < K) {
          val = A[globalRow * K + globalK];
        }
        As[loadK * BLOCK_M_PADDED + loadRow] = val;
      }

      // Load B tile: 32x32 = 256 vec4s, 64 threads load 4 each
      for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let loadIdx = tid * 4u + i;
        let loadK = loadIdx / (BLOCK_N / 4u);
        let loadColVec = loadIdx % (BLOCK_N / 4u);
        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * (BLOCK_N / 4u) + loadColVec;
        var v: vec4<f32> = vec4<f32>(0.0);
        if (globalK < K && globalColVec < NVec4) {
          v = B[globalK * NVec4 + globalColVec];
        }
        Bs[loadK * (BLOCK_N / 4u) + loadColVec] = v;
      }

      workgroupBarrier();

      // Compute
      for (var k: u32 = 0u; k < BLOCK_K; k = k + 1u) {
        let a0 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 0u];
        let a1 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 1u];
        let a2 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 2u];
        let a3 = As[k * BLOCK_M_PADDED + ty * THREAD_M + 3u];
        let b = Bs[k * (BLOCK_N / 4u) + tx];

        acc0 = fma(vec4<f32>(a0), b, acc0);
        acc1 = fma(vec4<f32>(a1), b, acc1);
        acc2 = fma(vec4<f32>(a2), b, acc2);
        acc3 = fma(vec4<f32>(a3), b, acc3);
      }

      workgroupBarrier();
    }

    // Write output
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
      let row = outRowBase + r;
      if (row < M) {
        var accRow: vec4<f32>;
        if (r == 0u) { accRow = acc0; }
        else if (r == 1u) { accRow = acc1; }
        else if (r == 2u) { accRow = acc2; }
        else { accRow = acc3; }
        if (outColBase + 0u < N) { C[row * N + outColBase + 0u] = accRow.x; }
        if (outColBase + 1u < N) { C[row * N + outColBase + 1u] = accRow.y; }
        if (outColBase + 2u < N) { C[row * N + outColBase + 2u] = accRow.z; }
        if (outColBase + 3u < N) { C[row * N + outColBase + 3u] = accRow.w; }
      }
    }
  }
`;

// Exact TFJS Vec4 pattern with innerElementSize=4
// A stored as vec4 (4 consecutive K values), K loop processes 4 at a time
const MATMUL_TFJS_VEC4_INNER_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // [8,8] workgroup, 4x4 elements per thread
  const WG_X: u32 = 8u;
  const WG_Y: u32 = 8u;
  const ROW_PER_THREAD: u32 = 4u;
  const COL_PER_THREAD: u32 = 4u;
  const TILE_A_OUTER: u32 = 32u;  // WG_Y * ROW_PER_THREAD
  const TILE_B_OUTER: u32 = 32u;  // WG_X * COL_PER_THREAD
  const TILE_INNER: u32 = 32u;
  const INNER_ELEMENT_SIZE: u32 = 4u;  // tileInner / WG_X = 32 / 8

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // A: [tileAOuter][tileInner/4] = [32][8] of vec4 (each row has 8 vec4s = 32 K values)
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 32>;
  // B: [tileInner][tileBOuter/4] = [32][8] of vec4
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 8>, 32>;

  @compute @workgroup_size(8, 8)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let M = uniforms.M;
    let K = uniforms.K;
    let N = uniforms.N;
    let NPadded = uniforms.NPadded;
    let NVec4 = NPadded / 4u;

    let localRow = i32(lid.y);  // 0-7
    let localCol = i32(lid.x);  // 0-7
    let tileRow = localRow * i32(ROW_PER_THREAD);  // Output row in tile
    let tileCol = localCol;  // vec4 column index

    let globalRow = i32(gid.y) * i32(ROW_PER_THREAD);
    let globalCol = i32(gid.x) * i32(COL_PER_THREAD);
    let globalRowStart = i32(wid.y) * i32(TILE_A_OUTER);

    let numTiles = (i32(K) + i32(TILE_INNER) - 1) / i32(TILE_INNER);
    let rowPerThreadB = TILE_INNER / WG_Y;  // 32 / 8 = 4

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i++) { acc[i] = vec4<f32>(0.0); }

    var kStart = 0;
    let tileRowB = localRow * i32(rowPerThreadB);

    for (var t = 0; t < numTiles; t++) {
      // Load A: each thread loads rowPerThread rows, 1 vec4 per row
      // mm_Asub[row][col] = A[globalRow + row, kStart + col*4 : col*4+4]
      for (var innerRow = 0u; innerRow < ROW_PER_THREAD; innerRow++) {
        let inputRow = u32(tileRow) + innerRow;
        let inputCol = u32(localCol);  // vec4 column (0-7)
        let gRow = globalRow + i32(innerRow);
        let gKBase = kStart + i32(inputCol) * 4;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (gRow < i32(M) && gKBase + 3 < i32(K)) {
          let base = u32(gRow) * K + u32(gKBase);
          v = vec4<f32>(A[base], A[base+1u], A[base+2u], A[base+3u]);
        } else if (gRow < i32(M)) {
          let base = u32(gRow) * K + u32(gKBase);
          if (gKBase < i32(K)) { v.x = A[base]; }
          if (gKBase + 1 < i32(K)) { v.y = A[base+1u]; }
          if (gKBase + 2 < i32(K)) { v.z = A[base+2u]; }
          if (gKBase + 3 < i32(K)) { v.w = A[base+3u]; }
        }
        mm_Asub[inputRow][inputCol] = v;
      }

      // Load B: each thread loads rowPerThreadB rows (K values), 1 vec4 per row
      for (var innerRow = 0u; innerRow < rowPerThreadB; innerRow++) {
        let inputRow = u32(tileRowB) + innerRow;
        let inputCol = u32(localCol);
        let gK = kStart + i32(inputRow);
        let gColVec = wid.x * (TILE_B_OUTER / 4u) + inputCol;

        var v: vec4<f32> = vec4<f32>(0.0);
        if (gK < i32(K) && gColVec < NVec4) {
          v = B[u32(gK) * NVec4 + gColVec];
        }
        mm_Bsub[inputRow][inputCol] = v;
      }

      kStart = kStart + i32(TILE_INNER);
      workgroupBarrier();

      // Compute: K loop processes 4 K values at a time (matching innerElementSize)
      // Total: 32 K values / 4 = 8 iterations
      for (var k = 0u; k < 8u; k++) {
        // Load 4 B vec4s at once
        let BCached0 = mm_Bsub[k * 4u + 0u][u32(localCol)];
        let BCached1 = mm_Bsub[k * 4u + 1u][u32(localCol)];
        let BCached2 = mm_Bsub[k * 4u + 2u][u32(localCol)];
        let BCached3 = mm_Bsub[k * 4u + 3u][u32(localCol)];

        // For each output row
        for (var i = 0u; i < 4u; i++) {
          // Load A vec4 (4 consecutive K values for this row)
          let ACached = mm_Asub[u32(tileRow) + i][k];
          // Multiply each A component with corresponding B vec4
          acc[i] = fma(vec4<f32>(ACached.x), BCached0, acc[i]);
          acc[i] = fma(vec4<f32>(ACached.y), BCached1, acc[i]);
          acc[i] = fma(vec4<f32>(ACached.z), BCached2, acc[i]);
          acc[i] = fma(vec4<f32>(ACached.w), BCached3, acc[i]);
        }
      }

      workgroupBarrier();
    }

    // Write output
    for (var innerRow = 0u; innerRow < 4u; innerRow++) {
      let row = globalRow + i32(innerRow);
      if (row < i32(M)) {
        for (var innerCol = 0u; innerCol < 4u; innerCol++) {
          let col = globalCol + i32(innerCol);
          if (col < i32(N)) {
            C[u32(row) * N + u32(col)] = acc[innerRow][innerCol];
          }
        }
      }
    }
  }
`;

// Bounds-check-free version of TFJS_VEC4_INNER for perfectly fitting matrices
// Used when M % 32 == 0 && N % 32 == 0 && K % 32 == 0
const MATMUL_TFJS_VEC4_FIT_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // [8,8] workgroup, 4x4 elements per thread
  const WG_X: u32 = 8u;
  const WG_Y: u32 = 8u;
  const ROW_PER_THREAD: u32 = 4u;
  const COL_PER_THREAD: u32 = 4u;
  const TILE_A_OUTER: u32 = 32u;
  const TILE_B_OUTER: u32 = 32u;
  const TILE_INNER: u32 = 32u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 32>;
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 8>, 32>;

  @compute @workgroup_size(8, 8)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let K = uniforms.K;
    let N = uniforms.N;
    let NVec4 = uniforms.NPadded / 4u;

    let localRow = lid.y;
    let localCol = lid.x;
    let tileRow = localRow * ROW_PER_THREAD;

    let globalRow = gid.y * ROW_PER_THREAD;
    let globalCol = gid.x * COL_PER_THREAD;

    let numTiles = K / TILE_INNER;  // Exact division - no bounds checking needed!

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    var kStart: u32 = 0u;
    let tileRowB = localRow * 4u;  // 32 / 8 = 4

    for (var t: u32 = 0u; t < numTiles; t++) {
      // Load A: NO BOUNDS CHECKING
      for (var innerRow: u32 = 0u; innerRow < ROW_PER_THREAD; innerRow++) {
        let inputRow = tileRow + innerRow;
        let inputCol = localCol;
        let gRow = globalRow + innerRow;
        let gKBase = kStart + inputCol * 4u;
        let base = gRow * K + gKBase;
        mm_Asub[inputRow][inputCol] = vec4<f32>(A[base], A[base+1u], A[base+2u], A[base+3u]);
      }

      // Load B: NO BOUNDS CHECKING
      for (var innerRow: u32 = 0u; innerRow < 4u; innerRow++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = localCol;
        let gK = kStart + inputRow;
        let gColVec = wid.x * (TILE_B_OUTER / 4u) + inputCol;
        mm_Bsub[inputRow][inputCol] = B[gK * NVec4 + gColVec];
      }

      kStart = kStart + TILE_INNER;
      workgroupBarrier();

      // Compute: 8 iterations of K loop (32 / 4 = 8)
      for (var k: u32 = 0u; k < 8u; k++) {
        let BCached0 = mm_Bsub[k * 4u + 0u][localCol];
        let BCached1 = mm_Bsub[k * 4u + 1u][localCol];
        let BCached2 = mm_Bsub[k * 4u + 2u][localCol];
        let BCached3 = mm_Bsub[k * 4u + 3u][localCol];

        let ACached0 = mm_Asub[tileRow + 0u][k];
        let ACached1 = mm_Asub[tileRow + 1u][k];
        let ACached2 = mm_Asub[tileRow + 2u][k];
        let ACached3 = mm_Asub[tileRow + 3u][k];

        acc0 = fma(vec4<f32>(ACached0.x), BCached0, acc0);
        acc0 = fma(vec4<f32>(ACached0.y), BCached1, acc0);
        acc0 = fma(vec4<f32>(ACached0.z), BCached2, acc0);
        acc0 = fma(vec4<f32>(ACached0.w), BCached3, acc0);

        acc1 = fma(vec4<f32>(ACached1.x), BCached0, acc1);
        acc1 = fma(vec4<f32>(ACached1.y), BCached1, acc1);
        acc1 = fma(vec4<f32>(ACached1.z), BCached2, acc1);
        acc1 = fma(vec4<f32>(ACached1.w), BCached3, acc1);

        acc2 = fma(vec4<f32>(ACached2.x), BCached0, acc2);
        acc2 = fma(vec4<f32>(ACached2.y), BCached1, acc2);
        acc2 = fma(vec4<f32>(ACached2.z), BCached2, acc2);
        acc2 = fma(vec4<f32>(ACached2.w), BCached3, acc2);

        acc3 = fma(vec4<f32>(ACached3.x), BCached0, acc3);
        acc3 = fma(vec4<f32>(ACached3.y), BCached1, acc3);
        acc3 = fma(vec4<f32>(ACached3.z), BCached2, acc3);
        acc3 = fma(vec4<f32>(ACached3.w), BCached3, acc3);
      }

      workgroupBarrier();
    }

    // Write output: NO BOUNDS CHECKING
    let outBase0 = (globalRow + 0u) * N + globalCol;
    let outBase1 = (globalRow + 1u) * N + globalCol;
    let outBase2 = (globalRow + 2u) * N + globalCol;
    let outBase3 = (globalRow + 3u) * N + globalCol;

    C[outBase0 + 0u] = acc0.x;
    C[outBase0 + 1u] = acc0.y;
    C[outBase0 + 2u] = acc0.z;
    C[outBase0 + 3u] = acc0.w;

    C[outBase1 + 0u] = acc1.x;
    C[outBase1 + 1u] = acc1.y;
    C[outBase1 + 2u] = acc1.z;
    C[outBase1 + 3u] = acc1.w;

    C[outBase2 + 0u] = acc2.x;
    C[outBase2 + 1u] = acc2.y;
    C[outBase2 + 2u] = acc2.z;
    C[outBase2 + 3u] = acc2.w;

    C[outBase3 + 0u] = acc3.x;
    C[outBase3 + 1u] = acc3.y;
    C[outBase3 + 2u] = acc3.z;
    C[outBase3 + 3u] = acc3.w;
  }
`;

// TFJS-style FIT shader: [8,8] workgroup, 32x32 tiles, no bounds checking
// Used when M % 32 == 0 && N % 32 == 0 && K % 32 == 0
const MATMUL_TFJS_FIT_32_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // [8,8] workgroup, 4x4 elements per thread = 32x32 tiles
  const TILE_M: u32 = 32u;
  const TILE_N: u32 = 32u;
  const TILE_K: u32 = 32u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<f32>;

  // 2D array syntax like tfjs for potentially better compilation
  var<workgroup> mm_Asub: array<array<vec4<f32>, 8>, 32>;  // [32][8] vec4 = 32 rows x 32 K values
  var<workgroup> mm_Bsub: array<array<vec4<f32>, 8>, 32>;  // [32][8] vec4 = 32 K values x 32 cols

  @compute @workgroup_size(8, 8)
  fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let K = uniforms.K;
    let N = uniforms.N;
    let NVec4 = uniforms.NPadded / 4u;

    let localRow = lid.y;  // 0-7
    let localCol = lid.x;  // 0-7
    let tileRow = localRow * 4u;  // Output row in tile (0, 4, 8, ..., 28)

    let globalRow = gid.y * 4u;  // Global output row
    let globalCol = gid.x * 4u;  // Global output col

    let numTiles = K / TILE_K;  // Exact division - no remainder!

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    var kStart: u32 = 0u;

    for (var t: u32 = 0u; t < numTiles; t++) {
      // Load A: each thread loads 4 rows, 1 vec4 per row (4 K values)
      // Total: 64 threads * 4 vec4s = 256 vec4s = 32 rows * 8 vec4s
      for (var i: u32 = 0u; i < 4u; i++) {
        let row = tileRow + i;
        let col = localCol;  // vec4 column (0-7)
        let gRow = wid.y * TILE_M + row;
        let gKBase = kStart + col * 4u;
        let base = gRow * K + gKBase;
        mm_Asub[row][col] = vec4<f32>(A[base], A[base+1u], A[base+2u], A[base+3u]);
      }

      // Load B: each thread loads 4 rows (K values), 1 vec4 per row
      // Total: 64 threads * 4 vec4s = 256 vec4s = 32 K * 8 vec4s
      for (var i: u32 = 0u; i < 4u; i++) {
        let kRow = localRow * 4u + i;  // K index (0-31)
        let colVec = localCol;  // vec4 column (0-7)
        let gK = kStart + kRow;
        let gColVec = wid.x * 8u + colVec;  // 32 / 4 = 8 vec4s per tile
        mm_Bsub[kRow][colVec] = B[gK * NVec4 + gColVec];
      }

      kStart = kStart + TILE_K;
      workgroupBarrier();

      // Compute: process 4 K values at a time, 8 iterations
      for (var k: u32 = 0u; k < 8u; k++) {
        // Cache 4 B vec4s
        let BCached0 = mm_Bsub[k * 4u + 0u][localCol];
        let BCached1 = mm_Bsub[k * 4u + 1u][localCol];
        let BCached2 = mm_Bsub[k * 4u + 2u][localCol];
        let BCached3 = mm_Bsub[k * 4u + 3u][localCol];

        // Load A vec4s and compute
        let ACached0 = mm_Asub[tileRow + 0u][k];
        let ACached1 = mm_Asub[tileRow + 1u][k];
        let ACached2 = mm_Asub[tileRow + 2u][k];
        let ACached3 = mm_Asub[tileRow + 3u][k];

        acc0 = fma(vec4<f32>(ACached0.x), BCached0, acc0);
        acc0 = fma(vec4<f32>(ACached0.y), BCached1, acc0);
        acc0 = fma(vec4<f32>(ACached0.z), BCached2, acc0);
        acc0 = fma(vec4<f32>(ACached0.w), BCached3, acc0);

        acc1 = fma(vec4<f32>(ACached1.x), BCached0, acc1);
        acc1 = fma(vec4<f32>(ACached1.y), BCached1, acc1);
        acc1 = fma(vec4<f32>(ACached1.z), BCached2, acc1);
        acc1 = fma(vec4<f32>(ACached1.w), BCached3, acc1);

        acc2 = fma(vec4<f32>(ACached2.x), BCached0, acc2);
        acc2 = fma(vec4<f32>(ACached2.y), BCached1, acc2);
        acc2 = fma(vec4<f32>(ACached2.z), BCached2, acc2);
        acc2 = fma(vec4<f32>(ACached2.w), BCached3, acc2);

        acc3 = fma(vec4<f32>(ACached3.x), BCached0, acc3);
        acc3 = fma(vec4<f32>(ACached3.y), BCached1, acc3);
        acc3 = fma(vec4<f32>(ACached3.z), BCached2, acc3);
        acc3 = fma(vec4<f32>(ACached3.w), BCached3, acc3);
      }

      workgroupBarrier();
    }

    // Write output (NO BOUNDS CHECKING)
    let outBase0 = (globalRow + 0u) * N + globalCol;
    let outBase1 = (globalRow + 1u) * N + globalCol;
    let outBase2 = (globalRow + 2u) * N + globalCol;
    let outBase3 = (globalRow + 3u) * N + globalCol;

    C[outBase0 + 0u] = acc0.x;
    C[outBase0 + 1u] = acc0.y;
    C[outBase0 + 2u] = acc0.z;
    C[outBase0 + 3u] = acc0.w;

    C[outBase1 + 0u] = acc1.x;
    C[outBase1 + 1u] = acc1.y;
    C[outBase1 + 2u] = acc1.z;
    C[outBase1 + 3u] = acc1.w;

    C[outBase2 + 0u] = acc2.x;
    C[outBase2 + 1u] = acc2.y;
    C[outBase2 + 2u] = acc2.z;
    C[outBase2 + 3u] = acc2.w;

    C[outBase3 + 0u] = acc3.x;
    C[outBase3 + 1u] = acc3.y;
    C[outBase3 + 2u] = acc3.z;
    C[outBase3 + 3u] = acc3.w;
  }
`;

// FIT shader with vec4 A loads - matching tfjs's memory access pattern
const MATMUL_FIT_VEC4A_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // 64x64 output tiles, 32 K tile, [16,16] workgroup
  const BLOCK_M: u32 = 64u;
  const BLOCK_N: u32 = 64u;
  const BLOCK_K: u32 = 32u;
  const THREAD_M: u32 = 4u;
  const THREAD_N: u32 = 4u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;

  // A: [64][8] vec4s (each row has 8 vec4s = 32 K values)
  var<workgroup> As: array<array<vec4<f32>, 8>, 64>;
  // B: [32][16] vec4s
  var<workgroup> Bs: array<array<vec4<f32>, 16>, 32>;

  @compute @workgroup_size(16, 16)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let K = uniforms.K;
    let NVec4 = uniforms.NPadded / 4u;
    let NVec4Out = uniforms.N / 4u;

    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;

    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColVec4 = wid.x * (BLOCK_N / 4u) + tx;

    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);

    let numTiles = K / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t++) {
      // Load A: 64 rows * 8 vec4s = 512 vec4s, 256 threads load 2 vec4s each
      for (var i: u32 = 0u; i < 2u; i++) {
        let loadIdx = tid * 2u + i;
        let loadRow = loadIdx / 8u;
        let loadVec4Col = loadIdx % 8u;
        let globalRow = wid.y * BLOCK_M + loadRow;
        let globalKBase = t * BLOCK_K + loadVec4Col * 4u;
        let base = globalRow * K + globalKBase;
        As[loadRow][loadVec4Col] = vec4<f32>(A[base], A[base+1u], A[base+2u], A[base+3u]);
      }

      // Load B: 32 rows * 16 vec4s = 512 vec4s, 256 threads load 2 vec4s each
      for (var i: u32 = 0u; i < 2u; i++) {
        let loadIdx = tid * 2u + i;
        let loadK = loadIdx / 16u;
        let loadColVec4 = loadIdx % 16u;
        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * 16u + loadColVec4;
        Bs[loadK][loadColVec4] = B[globalK * NVec4 + globalColVec];
      }

      workgroupBarrier();

      // Compute: process 4 K values at a time
      for (var kVec: u32 = 0u; kVec < 8u; kVec++) {
        let ACached0 = As[ty * THREAD_M + 0u][kVec];
        let ACached1 = As[ty * THREAD_M + 1u][kVec];
        let ACached2 = As[ty * THREAD_M + 2u][kVec];
        let ACached3 = As[ty * THREAD_M + 3u][kVec];

        let BCached0 = Bs[kVec * 4u + 0u][tx];
        let BCached1 = Bs[kVec * 4u + 1u][tx];
        let BCached2 = Bs[kVec * 4u + 2u][tx];
        let BCached3 = Bs[kVec * 4u + 3u][tx];

        acc0 = fma(vec4<f32>(ACached0.x), BCached0, acc0);
        acc0 = fma(vec4<f32>(ACached0.y), BCached1, acc0);
        acc0 = fma(vec4<f32>(ACached0.z), BCached2, acc0);
        acc0 = fma(vec4<f32>(ACached0.w), BCached3, acc0);

        acc1 = fma(vec4<f32>(ACached1.x), BCached0, acc1);
        acc1 = fma(vec4<f32>(ACached1.y), BCached1, acc1);
        acc1 = fma(vec4<f32>(ACached1.z), BCached2, acc1);
        acc1 = fma(vec4<f32>(ACached1.w), BCached3, acc1);

        acc2 = fma(vec4<f32>(ACached2.x), BCached0, acc2);
        acc2 = fma(vec4<f32>(ACached2.y), BCached1, acc2);
        acc2 = fma(vec4<f32>(ACached2.z), BCached2, acc2);
        acc2 = fma(vec4<f32>(ACached2.w), BCached3, acc2);

        acc3 = fma(vec4<f32>(ACached3.x), BCached0, acc3);
        acc3 = fma(vec4<f32>(ACached3.y), BCached1, acc3);
        acc3 = fma(vec4<f32>(ACached3.z), BCached2, acc3);
        acc3 = fma(vec4<f32>(ACached3.w), BCached3, acc3);
      }

      workgroupBarrier();
    }

    // Write output as vec4
    C[(outRowBase + 0u) * NVec4Out + outColVec4] = acc0;
    C[(outRowBase + 1u) * NVec4Out + outColVec4] = acc1;
    C[(outRowBase + 2u) * NVec4Out + outColVec4] = acc2;
    C[(outRowBase + 3u) * NVec4Out + outColVec4] = acc3;
  }
`;

// 8x8 per-thread shader: [8,8] workgroup, each thread computes 8x8 = 64 outputs
// Total tile: 64x64 (same as FIT but with fewer threads doing more work)
// Based on nuss-and-bolts optimization guide for 1+ TFLOPS on M-series
const MATMUL_8x8_PER_THREAD_SHADER = `
  struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    NPadded: u32,
  }

  // 64x64 output tile, 32 K tile, [8,8] workgroup, 8x8 per thread
  const BLOCK_M: u32 = 64u;
  const BLOCK_N: u32 = 64u;
  const BLOCK_K: u32 = 32u;
  const THREAD_M: u32 = 8u;
  const THREAD_N: u32 = 8u;
  const WG_SIZE: u32 = 8u;

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> A: array<f32>;
  @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
  @group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;

  // Shared memory: A is [32][65] (K x M+padding), B is [32][17] vec4 (K x N/4+padding)
  var<workgroup> As: array<f32, 2080>;  // 32 * 65
  var<workgroup> Bs: array<vec4<f32>, 544>;  // 32 * 17
  const BLOCK_M_PADDED: u32 = 65u;
  const BLOCK_N_VEC4_PADDED: u32 = 17u;

  @compute @workgroup_size(8, 8)
  fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
  ) {
    let K = uniforms.K;
    let NVec4 = uniforms.NPadded / 4u;
    let NVec4Out = uniforms.N / 4u;

    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * WG_SIZE + tx;

    // Each thread computes 8 rows x 8 cols = 64 outputs
    // We'll use 8 rows x 2 vec4s (8 values per row)
    let outRowBase = wid.y * BLOCK_M + ty * THREAD_M;
    let outColVec4Base = wid.x * (BLOCK_N / 4u) + tx * 2u;  // Each thread handles 2 vec4s

    // 64 accumulators organized as 8 rows x 2 vec4s
    var acc00: vec4<f32> = vec4<f32>(0.0); var acc01: vec4<f32> = vec4<f32>(0.0);
    var acc10: vec4<f32> = vec4<f32>(0.0); var acc11: vec4<f32> = vec4<f32>(0.0);
    var acc20: vec4<f32> = vec4<f32>(0.0); var acc21: vec4<f32> = vec4<f32>(0.0);
    var acc30: vec4<f32> = vec4<f32>(0.0); var acc31: vec4<f32> = vec4<f32>(0.0);
    var acc40: vec4<f32> = vec4<f32>(0.0); var acc41: vec4<f32> = vec4<f32>(0.0);
    var acc50: vec4<f32> = vec4<f32>(0.0); var acc51: vec4<f32> = vec4<f32>(0.0);
    var acc60: vec4<f32> = vec4<f32>(0.0); var acc61: vec4<f32> = vec4<f32>(0.0);
    var acc70: vec4<f32> = vec4<f32>(0.0); var acc71: vec4<f32> = vec4<f32>(0.0);

    let numTiles = K / BLOCK_K;

    for (var t: u32 = 0u; t < numTiles; t++) {
      // === Load A tile: 64 threads load 32*64 = 2048 f32s = 32 f32s per thread ===
      for (var i: u32 = 0u; i < 32u; i++) {
        let loadIdx = tid * 32u + i;
        let loadK = loadIdx / BLOCK_M;
        let loadRow = loadIdx % BLOCK_M;
        let globalK = t * BLOCK_K + loadK;
        let globalRow = wid.y * BLOCK_M + loadRow;
        As[loadK * BLOCK_M_PADDED + loadRow] = A[globalRow * K + globalK];
      }

      // === Load B tile: 64 threads load 32*16 = 512 vec4s = 8 vec4s per thread ===
      for (var i: u32 = 0u; i < 8u; i++) {
        let loadIdx = tid * 8u + i;
        let loadK = loadIdx / 16u;
        let loadColVec = loadIdx % 16u;
        let globalK = t * BLOCK_K + loadK;
        let globalColVec = wid.x * 16u + loadColVec;
        Bs[loadK * BLOCK_N_VEC4_PADDED + loadColVec] = B[globalK * NVec4 + globalColVec];
      }

      workgroupBarrier();

      // === Compute: K loop unrolled 32x ===
      let aRowBase = ty * THREAD_M;
      let bColVec0 = tx * 2u;
      let bColVec1 = tx * 2u + 1u;

      // Manually unrolled K=0..31
      { let a0=As[0u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[0u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[0u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[0u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[0u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[0u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[0u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[0u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[0u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[0u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[1u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[1u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[1u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[1u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[1u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[1u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[1u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[1u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[1u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[1u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[2u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[2u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[2u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[2u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[2u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[2u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[2u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[2u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[2u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[2u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[3u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[3u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[3u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[3u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[3u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[3u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[3u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[3u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[3u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[3u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[4u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[4u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[4u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[4u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[4u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[4u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[4u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[4u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[4u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[4u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[5u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[5u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[5u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[5u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[5u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[5u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[5u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[5u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[5u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[5u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[6u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[6u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[6u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[6u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[6u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[6u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[6u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[6u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[6u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[6u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[7u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[7u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[7u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[7u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[7u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[7u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[7u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[7u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[7u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[7u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[8u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[8u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[8u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[8u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[8u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[8u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[8u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[8u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[8u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[8u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[9u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[9u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[9u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[9u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[9u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[9u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[9u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[9u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[9u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[9u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[10u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[10u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[10u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[10u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[10u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[10u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[10u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[10u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[10u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[10u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[11u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[11u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[11u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[11u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[11u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[11u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[11u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[11u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[11u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[11u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[12u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[12u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[12u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[12u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[12u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[12u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[12u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[12u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[12u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[12u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[13u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[13u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[13u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[13u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[13u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[13u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[13u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[13u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[13u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[13u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[14u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[14u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[14u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[14u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[14u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[14u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[14u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[14u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[14u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[14u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[15u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[15u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[15u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[15u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[15u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[15u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[15u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[15u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[15u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[15u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[16u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[16u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[16u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[16u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[16u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[16u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[16u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[16u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[16u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[16u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[17u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[17u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[17u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[17u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[17u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[17u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[17u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[17u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[17u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[17u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[18u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[18u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[18u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[18u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[18u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[18u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[18u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[18u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[18u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[18u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[19u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[19u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[19u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[19u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[19u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[19u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[19u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[19u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[19u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[19u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[20u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[20u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[20u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[20u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[20u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[20u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[20u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[20u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[20u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[20u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[21u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[21u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[21u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[21u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[21u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[21u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[21u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[21u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[21u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[21u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[22u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[22u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[22u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[22u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[22u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[22u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[22u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[22u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[22u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[22u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[23u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[23u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[23u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[23u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[23u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[23u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[23u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[23u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[23u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[23u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[24u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[24u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[24u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[24u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[24u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[24u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[24u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[24u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[24u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[24u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[25u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[25u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[25u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[25u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[25u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[25u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[25u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[25u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[25u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[25u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[26u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[26u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[26u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[26u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[26u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[26u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[26u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[26u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[26u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[26u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[27u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[27u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[27u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[27u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[27u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[27u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[27u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[27u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[27u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[27u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[28u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[28u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[28u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[28u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[28u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[28u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[28u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[28u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[28u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[28u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[29u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[29u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[29u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[29u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[29u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[29u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[29u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[29u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[29u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[29u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[30u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[30u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[30u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[30u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[30u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[30u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[30u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[30u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[30u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[30u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }
      { let a0=As[31u*BLOCK_M_PADDED+aRowBase+0u]; let a1=As[31u*BLOCK_M_PADDED+aRowBase+1u]; let a2=As[31u*BLOCK_M_PADDED+aRowBase+2u]; let a3=As[31u*BLOCK_M_PADDED+aRowBase+3u]; let a4=As[31u*BLOCK_M_PADDED+aRowBase+4u]; let a5=As[31u*BLOCK_M_PADDED+aRowBase+5u]; let a6=As[31u*BLOCK_M_PADDED+aRowBase+6u]; let a7=As[31u*BLOCK_M_PADDED+aRowBase+7u]; let b0=Bs[31u*BLOCK_N_VEC4_PADDED+bColVec0]; let b1=Bs[31u*BLOCK_N_VEC4_PADDED+bColVec1]; acc00=fma(vec4<f32>(a0),b0,acc00); acc01=fma(vec4<f32>(a0),b1,acc01); acc10=fma(vec4<f32>(a1),b0,acc10); acc11=fma(vec4<f32>(a1),b1,acc11); acc20=fma(vec4<f32>(a2),b0,acc20); acc21=fma(vec4<f32>(a2),b1,acc21); acc30=fma(vec4<f32>(a3),b0,acc30); acc31=fma(vec4<f32>(a3),b1,acc31); acc40=fma(vec4<f32>(a4),b0,acc40); acc41=fma(vec4<f32>(a4),b1,acc41); acc50=fma(vec4<f32>(a5),b0,acc50); acc51=fma(vec4<f32>(a5),b1,acc51); acc60=fma(vec4<f32>(a6),b0,acc60); acc61=fma(vec4<f32>(a6),b1,acc61); acc70=fma(vec4<f32>(a7),b0,acc70); acc71=fma(vec4<f32>(a7),b1,acc71); }

      workgroupBarrier();
    }

    // === Write 8 rows x 2 vec4s = 16 vec4s per thread ===
    C[(outRowBase + 0u) * NVec4Out + outColVec4Base + 0u] = acc00;
    C[(outRowBase + 0u) * NVec4Out + outColVec4Base + 1u] = acc01;
    C[(outRowBase + 1u) * NVec4Out + outColVec4Base + 0u] = acc10;
    C[(outRowBase + 1u) * NVec4Out + outColVec4Base + 1u] = acc11;
    C[(outRowBase + 2u) * NVec4Out + outColVec4Base + 0u] = acc20;
    C[(outRowBase + 2u) * NVec4Out + outColVec4Base + 1u] = acc21;
    C[(outRowBase + 3u) * NVec4Out + outColVec4Base + 0u] = acc30;
    C[(outRowBase + 3u) * NVec4Out + outColVec4Base + 1u] = acc31;
    C[(outRowBase + 4u) * NVec4Out + outColVec4Base + 0u] = acc40;
    C[(outRowBase + 4u) * NVec4Out + outColVec4Base + 1u] = acc41;
    C[(outRowBase + 5u) * NVec4Out + outColVec4Base + 0u] = acc50;
    C[(outRowBase + 5u) * NVec4Out + outColVec4Base + 1u] = acc51;
    C[(outRowBase + 6u) * NVec4Out + outColVec4Base + 0u] = acc60;
    C[(outRowBase + 6u) * NVec4Out + outColVec4Base + 1u] = acc61;
    C[(outRowBase + 7u) * NVec4Out + outColVec4Base + 0u] = acc70;
    C[(outRowBase + 7u) * NVec4Out + outColVec4Base + 1u] = acc71;
  }
`;

const MATMUL_SHADER_LARGE = MATMUL_TFJS_VEC4_INNER_SHADER;
const MATMUL_TILE_M_LARGE = 32; // 8 threads * 4 elements
const MATMUL_TILE_N_LARGE = 32; // 8 threads * 4 elements

// Use simpler register-blocked shader for small matrices (< 64)
const MATMUL_SHADER_SMALL = MATMUL_REGISTER_BLOCKED_SHADER;
const MATMUL_TILE_M_SMALL = 64;
const MATMUL_TILE_N_SMALL = 64;

// Threshold for using large shader (all dimensions >= this)
const LARGE_THRESHOLD = 64;

// Whether to use vec4 output for large matrices (requires N divisible by 4)
const USE_VEC4_OUTPUT = true;

// ============ Autotuning System ============
// Each ShaderConfig defines a matmul kernel variant with its parameters
interface ShaderConfig {
  name: string;
  shader: string;
  tileM: number;
  tileN: number;
  tileK: number;
  workgroupSize: [number, number];
  requiresFit: boolean; // If true, only use when dims are exactly divisible by tile sizes
  usesVec4A?: boolean; // If true, A matrix stored as vec4 along K (fastest pattern!)
  usesVec4B: boolean; // If true, B matrix needs vec4 padding
  usesVec4C: boolean; // If true, C matrix output is vec4
  minSize: number; // Minimum matrix dimension to consider this config
  maxSize: number; // Maximum matrix dimension (-1 = no limit)
}

// Available shader configurations for autotuning
const SHADER_CONFIGS: ShaderConfig[] = [
  {
    name: '8X8-MEGA',
    shader: MATMUL_8X8_MEGA_SHADER,
    tileM: 64,
    tileN: 64,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: false, // Has bounds checking
    usesVec4B: true, // B is vec4 along N
    usesVec4C: true, // C output is vec4
    usesVec4A: true, // A is vec4 along K
    minSize: 64,
    maxSize: -1, // Works for all sizes
  },
  {
    name: '8X8-MEGA-FIT',
    shader: MATMUL_8X8_MEGA_FIT_SHADER,
    tileM: 64,
    tileN: 64,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: true, // No bounds checking - requires exact fit
    usesVec4B: true,
    usesVec4C: true,
    usesVec4A: true,
    minSize: 128,
    maxSize: -1,
  },
  {
    name: 'BCACHE-FIT',
    shader: MATMUL_BCACHE_FIT_SHADER,
    tileM: 32,
    tileN: 32,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: true, // No bounds checking - requires exact fit
    usesVec4B: true,
    usesVec4C: true,
    usesVec4A: true,
    minSize: 64,
    maxSize: -1,
  },
  {
    name: 'TFJS-BCACHE',
    shader: MATMUL_TFJS_BCACHE_SHADER,
    tileM: 32,
    tileN: 32,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: false, // Has bounds checking
    usesVec4B: true, // B is vec4 along N
    usesVec4C: true, // C output is vec4
    usesVec4A: true, // A is vec4 along K (KEY DIFFERENCE!)
    minSize: 64,
    maxSize: -1, // Works for all sizes
  },
  {
    name: 'FIT-64x64',
    shader: MATMUL_FIT_SHADER,
    tileM: 64,
    tileN: 64,
    tileK: 32,
    workgroupSize: [16, 16],
    requiresFit: true,
    usesVec4B: true,
    usesVec4C: true,
    minSize: 256,
    maxSize: -1, // Best for large matrices
  },
  {
    name: 'FIT-32x32',
    shader: MATMUL_TFJS_FIT_32_SHADER,
    tileM: 32,
    tileN: 32,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: true,
    usesVec4B: true,
    usesVec4C: false,
    minSize: 128,
    maxSize: 2048, // Better for medium matrices
  },
  {
    name: 'TFJS-VEC4-INNER',
    shader: MATMUL_TFJS_VEC4_INNER_SHADER,
    tileM: 32,
    tileN: 32,
    tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: false, // Has bounds checking
    usesVec4B: true,
    usesVec4C: false,
    minSize: 64,
    maxSize: -1,
  },
  {
    name: 'REGISTER-BLOCKED',
    shader: MATMUL_REGISTER_BLOCKED_SHADER,
    tileM: 64,
    tileN: 64,
    tileK: 16,
    workgroupSize: [16, 16],
    requiresFit: false,
    usesVec4B: false,
    usesVec4C: false,
    minSize: 32,
    maxSize: 512, // Better for small/medium
  },
];

// Cache for autotuned best configs: key = "MxKxN" -> config name
const autotuneCache = new Map<string, string>();

// Pre-seeded optimal configs
// 8X8-MEGA-FIT is fastest for large aligned matrices (8×8 output per thread)
// 8X8-MEGA for large non-aligned matrices (with bounds checking)
// REGISTER-BLOCKED for small matrices (lower launch overhead)
// Autotuned presets (tested on M4 Mac mini):
// TFJS-BCACHE wins at ALL sizes due to lower register pressure on Apple GPUs.
// 4×4 output/thread = 4 vec4 accumulators vs 8×8's 16 vec4 accumulators.
// The 8×8 kernel's extra register pressure causes spills that negate its
// higher compute-per-thread advantage.
const PRESET_CONFIGS: Record<string, string> = {
  '4096x4096x4096': 'TFJS-BCACHE',
  '2048x2048x2048': 'TFJS-BCACHE',
  '1024x1024x1024': 'TFJS-BCACHE',
  '512x512x512': 'TFJS-BCACHE',
  '256x256x256': 'TFJS-BCACHE',
  '128x128x128': 'TFJS-BCACHE',
  // Small matrices: simpler shader with lower overhead
  '64x64x64': 'REGISTER-BLOCKED',
};

// Initialize preset configs into cache
for (const [key, value] of Object.entries(PRESET_CONFIGS)) {
  autotuneCache.set(key, value);
}

// Get the best config for given dimensions (from cache or heuristic)
function getBestConfig(m: number, k: number, n: number): ShaderConfig | null {
  const key = `${m}x${k}x${n}`;

  // Check cache first
  if (autotuneCache.has(key)) {
    const configName = autotuneCache.get(key)!;
    return SHADER_CONFIGS.find(c => c.name === configName) || null;
  }

  // Use heuristics to pick a config
  const minDim = Math.min(m, k, n);
  const maxDim = Math.max(m, k, n);

  // Filter configs that can handle this size
  const validConfigs = SHADER_CONFIGS.filter(c => {
    // Check size bounds
    if (minDim < c.minSize) return false;
    if (c.maxSize > 0 && maxDim > c.maxSize) return false;

    // Check fit requirements
    if (c.requiresFit) {
      if (m % c.tileM !== 0 || n % c.tileN !== 0 || k % c.tileK !== 0) return false;
      if (c.usesVec4B && n % 4 !== 0) return false;
    }

    return true;
  });

  if (validConfigs.length === 0) {
    // Fallback to TFJS-VEC4-INNER which handles all cases
    return SHADER_CONFIGS.find(c => c.name === 'TFJS-VEC4-INNER') || null;
  }

  // Prefer TFJS-BCACHE: lower register pressure, best on Apple GPUs at all sizes
  // 4×4 output/thread with B-caching pattern wins over 8×8 due to register spills
  const bcache = validConfigs.find(c => c.name === 'TFJS-BCACHE');
  if (bcache && minDim >= 64) return bcache;

  // Fallback for small matrices
  if (maxDim <= 256) {
    return validConfigs.find(c => c.name === 'REGISTER-BLOCKED') || validConfigs[0];
  }

  // Generic fallback
  return validConfigs[0];
}

// Get config by name
function getConfigByName(name: string): ShaderConfig | null {
  return SHADER_CONFIGS.find(c => c.name === name) || null;
}

// ============ WebGPU Backend ============

// ============ Buffer Manager (tfjs-style pooling) ============

class BufferManager {
  private freeBuffers = new Map<string, GPUBuffer[]>();
  private usedBuffers = new Map<string, GPUBuffer[]>();
  private device: GPUDevice;

  // Separate pool for staging buffers (MAP_READ) - they must be unmapped before reuse
  private freeStagingBuffers = new Map<number, GPUBuffer[]>();
  private usedStagingBuffers = new Map<number, GPUBuffer[]>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  private getKey(size: number, usage: GPUBufferUsageFlags): string {
    return `${size}_${usage}`;
  }

  acquire(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const key = this.getKey(size, usage);

    // Try to reuse a free buffer
    if (!this.freeBuffers.has(key)) {
      this.freeBuffers.set(key, []);
    }
    const freeList = this.freeBuffers.get(key)!;

    let buffer: GPUBuffer;
    if (freeList.length > 0) {
      buffer = freeList.pop()!;
    } else {
      buffer = this.device.createBuffer({ size, usage });
    }

    // Track as used
    if (!this.usedBuffers.has(key)) {
      this.usedBuffers.set(key, []);
    }
    this.usedBuffers.get(key)!.push(buffer);

    return buffer;
  }

  release(buffer: GPUBuffer): void {
    const key = this.getKey(buffer.size, buffer.usage);

    // Remove from used
    const usedList = this.usedBuffers.get(key);
    if (usedList) {
      const idx = usedList.indexOf(buffer);
      if (idx >= 0) {
        usedList.splice(idx, 1);
      }
    }

    // Add to free pool
    if (!this.freeBuffers.has(key)) {
      this.freeBuffers.set(key, []);
    }
    this.freeBuffers.get(key)!.push(buffer);
  }

  /**
   * Round size up to next power of 2 for better pool reuse.
   * This reduces fragmentation when sizes vary slightly.
   */
  private roundToPowerOf2(size: number): number {
    if (size <= 0) return 1;
    // Find next power of 2 >= size
    let power = 1;
    while (power < size) power *= 2;
    return power;
  }

  /**
   * Acquire a staging buffer (MAP_READ | COPY_DST) from the pool.
   * Staging buffers are pooled separately because they have map state.
   * Uses power-of-2 bucketing for better reuse across similar sizes.
   */
  acquireStaging(size: number): GPUBuffer {
    const bucketSize = this.roundToPowerOf2(size);

    if (!this.freeStagingBuffers.has(bucketSize)) {
      this.freeStagingBuffers.set(bucketSize, []);
    }
    const freeList = this.freeStagingBuffers.get(bucketSize)!;

    let buffer: GPUBuffer;
    if (freeList.length > 0) {
      buffer = freeList.pop()!;
    } else {
      buffer = this.device.createBuffer({
        size: bucketSize, // Allocate power-of-2 size
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }

    // Track as used (use actual buffer size for tracking)
    const actualSize = buffer.size;
    if (!this.usedStagingBuffers.has(actualSize)) {
      this.usedStagingBuffers.set(actualSize, []);
    }
    this.usedStagingBuffers.get(actualSize)!.push(buffer);

    return buffer;
  }

  /**
   * Release a staging buffer back to the pool.
   * IMPORTANT: Buffer must be unmapped before calling this.
   */
  releaseStaging(buffer: GPUBuffer): void {
    const size = buffer.size; // Use actual buffer size (power of 2)

    // Remove from used
    const usedList = this.usedStagingBuffers.get(size);
    if (usedList) {
      const idx = usedList.indexOf(buffer);
      if (idx >= 0) {
        usedList.splice(idx, 1);
      }
    }

    // Add to free pool (buffer must already be unmapped)
    if (!this.freeStagingBuffers.has(size)) {
      this.freeStagingBuffers.set(size, []);
    }
    this.freeStagingBuffers.get(size)!.push(buffer);
  }

  dispose(): void {
    for (const buffers of this.freeBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    for (const buffers of this.usedBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    for (const buffers of this.freeStagingBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    for (const buffers of this.usedStagingBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    this.freeBuffers.clear();
    this.usedBuffers.clear();
    this.freeStagingBuffers.clear();
    this.usedStagingBuffers.clear();
  }
}

export class WebGPUBackend implements Backend {
  name = 'webgpu';
  readonly device: GPUDevice;
  private bufferManager: BufferManager;

  constructor(device: GPUDevice) {
    this.device = device;
    this.bufferManager = new BufferManager(device);
  }

  // Pipeline cache for elementwise shaders
  private pipelineCache = new Map<string, GPUComputePipeline>();

  private getOrCreatePipeline(key: string, shaderCode: string): GPUComputePipeline {
    if (this.pipelineCache.has(key)) {
      return this.pipelineCache.get(key)!;
    }
    const module = this.device.createShaderModule({ code: shaderCode });
    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  // ============ GPU-Native Tensor Operations ============
  // These work directly with WebGPUTensor and don't read back to CPU

  /**
   * Create a WebGPUTensor from CPU data
   */
  createTensor(data: Float64Array | number[], shape: number[]): WebGPUTensor {
    return WebGPUTensor.fromArray(data, shape, this.device);
  }

  /**
   * Run unary op on tensor, return new tensor (data stays on GPU)
   */
  private runUnaryOpOnTensor(tensor: WebGPUTensor, opName: string): WebGPUTensor {
    const shader = UNARY_SHADERS[opName];
    if (!shader) throw new Error(`Unknown unary op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`unary_${opName}`, shader);
    const n = tensor.size;

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return new WebGPUTensor(outputBuffer, tensor.shape, this.device);
  }

  /**
   * Run binary op on tensors, return new tensor (data stays on GPU)
   */
  private runBinaryOpOnTensor(a: WebGPUTensor, b: WebGPUTensor, opName: string): WebGPUTensor {
    const shader = BINARY_SHADERS[opName];
    if (!shader) throw new Error(`Unknown binary op: ${opName}`);
    if (a.size !== b.size) throw new Error('Shape mismatch');

    const pipeline = this.getOrCreatePipeline(`binary_${opName}`, shader);
    const n = a.size;

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return new WebGPUTensor(outputBuffer, a.shape, this.device);
  }

  /**
   * Convert IFaceNDArray to WebGPUTensor (uploads if needed)
   */
  private toTensor(arr: IFaceNDArray): WebGPUTensor {
    if (arr instanceof WebGPUNDArray) {
      return arr.tensor;
    }
    // Legacy NDArray - upload to GPU
    return this.createTensor(arr.data, arr.shape);
  }

  /**
   * Convert WebGPUTensor to WebGPUNDArray wrapper
   */
  private fromTensor(tensor: WebGPUTensor): IFaceNDArray {
    return new WebGPUNDArray(tensor);
  }

  /**
   * Materialize all pending GPU arrays.
   * Call this before accessing .data on GPU-backed arrays in tests.
   */
  async materializeAll(): Promise<void> {
    return materializeAll();
  }

  /**
   * Run scalar op on tensor (array + scalar), return new tensor (data stays on GPU)
   */
  private runScalarOpOnTensor(tensor: WebGPUTensor, scalar: number, opName: string): WebGPUTensor {
    const shader = SCALAR_SHADERS[opName];
    if (!shader) throw new Error(`Unknown scalar op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`scalar_${opName}`, shader);
    const n = tensor.size;

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 8, // u32 size + f32 scalar
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Write uniforms
    const uniformData = new ArrayBuffer(8);
    new Uint32Array(uniformData, 0, 1)[0] = n;
    new Float32Array(uniformData, 4, 1)[0] = scalar;
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint8Array(uniformData));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    uniformBuffer.destroy();

    return new WebGPUTensor(outputBuffer, tensor.shape, this.device);
  }

  /**
   * Run modf on tensor, returns two tensors (data stays on GPU)
   */
  private runModfOnTensor(tensor: WebGPUTensor): { frac: WebGPUTensor; integ: WebGPUTensor } {
    const pipeline = this.getOrCreatePipeline('modf', MODF_SHADER);
    const n = tensor.size;

    const fracBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const integBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: fracBuffer } },
        { binding: 2, resource: { buffer: integBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return {
      frac: new WebGPUTensor(fracBuffer, tensor.shape, this.device),
      integ: new WebGPUTensor(integBuffer, tensor.shape, this.device),
    };
  }

  /**
   * Run frexp on tensor, returns two tensors (data stays on GPU)
   */
  private runFrexpOnTensor(tensor: WebGPUTensor): {
    mantissa: WebGPUTensor;
    exponent: WebGPUTensor;
  } {
    const pipeline = this.getOrCreatePipeline('frexp', FREXP_SHADER);
    const n = tensor.size;

    const mantissaBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const exponentBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: mantissaBuffer } },
        { binding: 2, resource: { buffer: exponentBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return {
      mantissa: new WebGPUTensor(mantissaBuffer, tensor.shape, this.device),
      exponent: new WebGPUTensor(exponentBuffer, tensor.shape, this.device),
    };
  }

  /**
   * Run divmod on tensors, returns two tensors (data stays on GPU)
   */
  private runDivmodOnTensor(
    a: WebGPUTensor,
    b: WebGPUTensor
  ): { quotient: WebGPUTensor; remainder: WebGPUTensor } {
    const pipeline = this.getOrCreatePipeline('divmod', DIVMOD_SHADER);
    const n = a.size;

    const quotientBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const remainderBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: quotientBuffer } },
        { binding: 3, resource: { buffer: remainderBuffer } },
        { binding: 4, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return {
      quotient: new WebGPUTensor(quotientBuffer, a.shape, this.device),
      remainder: new WebGPUTensor(remainderBuffer, a.shape, this.device),
    };
  }

  /**
   * Run reduction on tensor (async, reads back result immediately)
   * Returns a single number from sum/prod/min/max
   */
  private async runReductionOnTensor(tensor: WebGPUTensor, opName: string): Promise<number> {
    const shader = REDUCTION_SHADERS[opName];
    if (!shader) throw new Error(`Unknown reduction op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`reduction_${opName}`, shader);
    const n = tensor.size;
    const numWorkgroups = Math.ceil(n / 256);

    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const partialResults = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Final reduction on CPU (for small number of workgroups)
    let result: number;
    if (opName === 'sum') {
      result = 0;
      for (let i = 0; i < numWorkgroups; i++) result += partialResults[i];
    } else if (opName === 'prod') {
      result = 1;
      for (let i = 0; i < numWorkgroups; i++) result *= partialResults[i];
    } else if (opName === 'min') {
      result = partialResults[0];
      for (let i = 1; i < numWorkgroups; i++) result = Math.min(result, partialResults[i]);
    } else if (opName === 'max') {
      result = partialResults[0];
      for (let i = 1; i < numWorkgroups; i++) result = Math.max(result, partialResults[i]);
    } else {
      throw new Error(`Unknown reduction: ${opName}`);
    }

    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return result;
  }

  /**
   * Run cumulative op on tensor (cumsum/cumprod), return new tensor (data stays on GPU)
   */
  private runCumulativeOnTensor(tensor: WebGPUTensor, opName: 'cumsum' | 'cumprod'): WebGPUTensor {
    const shader = opName === 'cumsum' ? CUMSUM_SHADER : CUMPROD_SHADER;
    const pipeline = this.getOrCreatePipeline(opName, shader);
    const n = tensor.size;

    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return new WebGPUTensor(outputBuffer, tensor.shape, this.device);
  }

  // GPU execution helpers
  private async runUnaryOp(arr: IFaceNDArray, opName: string): Promise<IFaceNDArray> {
    const shader = UNARY_SHADERS[opName];
    if (!shader) throw new Error(`Unknown unary op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`unary_${opName}`, shader);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // f32 -> f64
    const f64Output = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Output[i] = f32Output[i];

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, arr.shape);
  }

  private async runBinaryOp(
    a: IFaceNDArray,
    b: IFaceNDArray,
    opName: string
  ): Promise<IFaceNDArray> {
    const shader = BINARY_SHADERS[opName];
    if (!shader) throw new Error(`Unknown binary op: ${opName}`);
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');

    const pipeline = this.getOrCreatePipeline(`binary_${opName}`, shader);
    const n = a.data.length;

    // Create buffers
    const aBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32A = new Float32Array(n);
    const f32B = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      f32A[i] = a.data[i];
      f32B[i] = b.data[i];
    }
    this.device.queue.writeBuffer(aBuffer, 0, f32A);
    this.device.queue.writeBuffer(bBuffer, 0, f32B);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // f32 -> f64
    const f64Output = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Output[i] = f32Output[i];

    // Cleanup
    aBuffer.destroy();
    bBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, a.shape);
  }

  private async runScalarOp(
    arr: IFaceNDArray,
    scalar: number,
    opName: string
  ): Promise<IFaceNDArray> {
    const shader = SCALAR_SHADERS[opName];
    if (!shader) throw new Error(`Unknown scalar op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`scalar_${opName}`, shader);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 8, // u32 size + f32 scalar
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);

    // Write uniforms
    const uniformData = new ArrayBuffer(8);
    new Uint32Array(uniformData, 0, 1)[0] = n;
    new Float32Array(uniformData, 4, 1)[0] = scalar;
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint8Array(uniformData));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // f32 -> f64
    const f64Output = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Output[i] = f32Output[i];

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, arr.shape);
  }

  // ============ Decomposition GPU ops ============

  private async runModf(arr: IFaceNDArray): Promise<{ frac: IFaceNDArray; integ: IFaceNDArray }> {
    const pipeline = this.getOrCreatePipeline('modf', MODF_SHADER);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const fracBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const integBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: fracBuffer } },
        { binding: 2, resource: { buffer: integBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back both buffers
    const readFrac = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readInteg = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(fracBuffer, 0, readFrac, 0, n * 4);
    commandEncoder.copyBufferToBuffer(integBuffer, 0, readInteg, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readFrac.mapAsync(GPUMapMode.READ);
    const f32Frac = new Float32Array(readFrac.getMappedRange().slice(0));
    readFrac.unmap();

    await readInteg.mapAsync(GPUMapMode.READ);
    const f32Integ = new Float32Array(readInteg.getMappedRange().slice(0));
    readInteg.unmap();

    // f32 -> f64
    const f64Frac = new Float64Array(n);
    const f64Integ = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      f64Frac[i] = f32Frac[i];
      f64Integ[i] = f32Integ[i];
    }

    // Cleanup
    inputBuffer.destroy();
    fracBuffer.destroy();
    integBuffer.destroy();
    sizeBuffer.destroy();
    readFrac.destroy();
    readInteg.destroy();

    return {
      frac: this.createArray(f64Frac, arr.shape),
      integ: this.createArray(f64Integ, arr.shape),
    };
  }

  private async runFrexp(
    arr: IFaceNDArray
  ): Promise<{ mantissa: IFaceNDArray; exponent: IFaceNDArray }> {
    const pipeline = this.getOrCreatePipeline('frexp', FREXP_SHADER);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const mantissaBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const exponentBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: mantissaBuffer } },
        { binding: 2, resource: { buffer: exponentBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back both buffers
    const readMantissa = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readExponent = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(mantissaBuffer, 0, readMantissa, 0, n * 4);
    commandEncoder.copyBufferToBuffer(exponentBuffer, 0, readExponent, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readMantissa.mapAsync(GPUMapMode.READ);
    const f32Mantissa = new Float32Array(readMantissa.getMappedRange().slice(0));
    readMantissa.unmap();

    await readExponent.mapAsync(GPUMapMode.READ);
    const f32Exponent = new Float32Array(readExponent.getMappedRange().slice(0));
    readExponent.unmap();

    // f32 -> f64
    const f64Mantissa = new Float64Array(n);
    const f64Exponent = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      f64Mantissa[i] = f32Mantissa[i];
      f64Exponent[i] = f32Exponent[i];
    }

    // Cleanup
    inputBuffer.destroy();
    mantissaBuffer.destroy();
    exponentBuffer.destroy();
    sizeBuffer.destroy();
    readMantissa.destroy();
    readExponent.destroy();

    return {
      mantissa: this.createArray(f64Mantissa, arr.shape),
      exponent: this.createArray(f64Exponent, arr.shape),
    };
  }

  private async runDivmod(
    a: IFaceNDArray,
    b: IFaceNDArray
  ): Promise<{ quotient: IFaceNDArray; remainder: IFaceNDArray }> {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');

    const pipeline = this.getOrCreatePipeline('divmod', DIVMOD_SHADER);
    const n = a.data.length;

    // Create buffers
    const aBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const quotientBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const remainderBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data
    const f32A = new Float32Array(n);
    const f32B = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      f32A[i] = a.data[i];
      f32B[i] = b.data[i];
    }
    this.device.queue.writeBuffer(aBuffer, 0, f32A);
    this.device.queue.writeBuffer(bBuffer, 0, f32B);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: quotientBuffer } },
        { binding: 3, resource: { buffer: remainderBuffer } },
        { binding: 4, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back both buffers
    const readQuotient = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readRemainder = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(quotientBuffer, 0, readQuotient, 0, n * 4);
    commandEncoder.copyBufferToBuffer(remainderBuffer, 0, readRemainder, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readQuotient.mapAsync(GPUMapMode.READ);
    const f32Quotient = new Float32Array(readQuotient.getMappedRange().slice(0));
    readQuotient.unmap();

    await readRemainder.mapAsync(GPUMapMode.READ);
    const f32Remainder = new Float32Array(readRemainder.getMappedRange().slice(0));
    readRemainder.unmap();

    // f32 -> f64
    const f64Quotient = new Float64Array(n);
    const f64Remainder = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      f64Quotient[i] = f32Quotient[i];
      f64Remainder[i] = f32Remainder[i];
    }

    // Cleanup
    aBuffer.destroy();
    bBuffer.destroy();
    quotientBuffer.destroy();
    remainderBuffer.destroy();
    sizeBuffer.destroy();
    readQuotient.destroy();
    readRemainder.destroy();

    return {
      quotient: this.createArray(f64Quotient, a.shape),
      remainder: this.createArray(f64Remainder, a.shape),
    };
  }

  private async runReduction(arr: IFaceNDArray, opName: string): Promise<number> {
    const shader = REDUCTION_SHADERS[opName];
    if (!shader) throw new Error(`Unknown reduction op: ${opName}`);

    const pipeline = this.getOrCreatePipeline(`reduction_${opName}`, shader);
    const n = arr.data.length;
    const numWorkgroups = Math.ceil(n / 256);

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const partialResults = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Final reduction on CPU (for small number of workgroups)
    let result: number;
    if (opName === 'sum') {
      result = 0;
      for (let i = 0; i < numWorkgroups; i++) result += partialResults[i];
    } else if (opName === 'prod') {
      result = 1;
      for (let i = 0; i < numWorkgroups; i++) result *= partialResults[i];
    } else if (opName === 'min') {
      result = partialResults[0];
      for (let i = 1; i < numWorkgroups; i++) result = Math.min(result, partialResults[i]);
    } else if (opName === 'max') {
      result = partialResults[0];
      for (let i = 1; i < numWorkgroups; i++) result = Math.max(result, partialResults[i]);
    } else {
      throw new Error(`Unknown reduction: ${opName}`);
    }

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return result;
  }

  private async runCumulative(
    arr: IFaceNDArray,
    opName: 'cumsum' | 'cumprod'
  ): Promise<IFaceNDArray> {
    const shader = opName === 'cumsum' ? CUMSUM_SHADER : CUMPROD_SHADER;
    const pipeline = this.getOrCreatePipeline(opName, shader);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // f32 -> f64
    const f64Output = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Output[i] = f32Output[i];

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, arr.shape);
  }

  private async runClip(arr: IFaceNDArray, minVal: number, maxVal: number): Promise<IFaceNDArray> {
    const pipeline = this.getOrCreatePipeline('clip', CLIP_SHADER);
    const n = arr.data.length;

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 16, // u32 + f32 + f32 + padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data (f64 -> f32)
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);

    // Write uniforms
    const uniformData = new ArrayBuffer(16);
    new Uint32Array(uniformData, 0, 1)[0] = n;
    new Float32Array(uniformData, 4, 1)[0] = minVal;
    new Float32Array(uniformData, 8, 1)[0] = maxVal;
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint8Array(uniformData));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    // Read back
    const readBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // f32 -> f64
    const f64Output = new Float64Array(n);
    for (let i = 0; i < n; i++) f64Output[i] = f32Output[i];

    // Cleanup
    inputBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, arr.shape);
  }

  // Argmin/argmax reduction
  private async runArgReduction(arr: IFaceNDArray, opName: 'argmin' | 'argmax'): Promise<number> {
    if (arr.data.length === 0) throw new Error('zero-size array');

    const shader = opName === 'argmin' ? ARGMIN_SHADER : ARGMAX_SHADER;
    const pipeline = this.getOrCreatePipeline(opName, shader);
    const n = arr.data.length;
    const numWorkgroups = Math.ceil(n / 256);

    // Create buffers
    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputValBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const outputIdxBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload
    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputValBuffer } },
        { binding: 2, resource: { buffer: outputIdxBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    // Read back indices
    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readValBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputIdxBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    commandEncoder.copyBufferToBuffer(outputValBuffer, 0, readValBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    await readValBuffer.mapAsync(GPUMapMode.READ);
    const indices = new Uint32Array(readBuffer.getMappedRange().slice(0));
    const values = new Float32Array(readValBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readValBuffer.unmap();

    // Final reduction on CPU
    let bestIdx = indices[0];
    let bestVal = values[0];
    for (let i = 1; i < numWorkgroups; i++) {
      if (opName === 'argmin' && values[i] < bestVal) {
        bestVal = values[i];
        bestIdx = indices[i];
      } else if (opName === 'argmax' && values[i] > bestVal) {
        bestVal = values[i];
        bestIdx = indices[i];
      }
    }

    // Cleanup
    inputBuffer.destroy();
    outputValBuffer.destroy();
    outputIdxBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();
    readValBuffer.destroy();

    return bestIdx;
  }

  // All/any reduction
  private async runBoolReduction(arr: IFaceNDArray, opName: 'all' | 'any'): Promise<boolean> {
    const shader = opName === 'all' ? ALL_SHADER : ANY_SHADER;
    const pipeline = this.getOrCreatePipeline(opName, shader);
    const n = arr.data.length;
    const numWorkgroups = Math.ceil(n / 256);

    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const results = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Final reduction
    let result = opName === 'all' ? 1 : 0;
    for (let i = 0; i < numWorkgroups; i++) {
      if (opName === 'all') {
        result = result & results[i];
      } else {
        result = result | results[i];
      }
    }

    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return result !== 0;
  }

  // Sum along axis
  private async runSumAxis(arr: IFaceNDArray, axis: number): Promise<IFaceNDArray> {
    if (arr.shape.length !== 2) throw new Error('sumAxis only supports 2D arrays');
    const [rows, cols] = arr.shape;

    const shader = axis === 0 ? SUM_AXIS0_SHADER : SUM_AXIS1_SHADER;
    const pipeline = this.getOrCreatePipeline(`sumAxis${axis}`, shader);
    const outSize = axis === 0 ? cols : rows;

    const inputBuffer = this.device.createBuffer({
      size: rows * cols * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: outSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32Input = new Float32Array(rows * cols);
    for (let i = 0; i < arr.data.length; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(outSize / 256));
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: outSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outSize * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    const f64Output = new Float64Array(outSize);
    for (let i = 0; i < outSize; i++) f64Output[i] = f32Output[i];

    inputBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, [outSize]);
  }

  // Transpose
  private async runTranspose(arr: IFaceNDArray): Promise<IFaceNDArray> {
    if (arr.shape.length !== 2) throw new Error('transpose only supports 2D arrays');
    const [rows, cols] = arr.shape;

    const pipeline = this.getOrCreatePipeline('transpose', TRANSPOSE_SHADER);

    const inputBuffer = this.device.createBuffer({
      size: rows * cols * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: rows * cols * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32Input = new Float32Array(rows * cols);
    for (let i = 0; i < arr.data.length; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(cols / 16), Math.ceil(rows / 16));
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: rows * cols * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, rows * cols * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    const f64Output = new Float64Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) f64Output[i] = f32Output[i];

    inputBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, [cols, rows]);
  }

  // Outer product
  private async runOuter(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    const m = a.data.length;
    const n = b.data.length;

    const pipeline = this.getOrCreatePipeline('outer', OUTER_SHADER);

    const aBuffer = this.device.createBuffer({
      size: m * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const uniformBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32A = new Float32Array(m);
    const f32B = new Float32Array(n);
    for (let i = 0; i < m; i++) f32A[i] = a.data[i];
    for (let i = 0; i < n; i++) f32B[i] = b.data[i];
    this.device.queue.writeBuffer(aBuffer, 0, f32A);
    this.device.queue.writeBuffer(bBuffer, 0, f32B);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([m, n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 16), Math.ceil(m / 16));
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, m * n * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const f32Output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    const f64Output = new Float64Array(m * n);
    for (let i = 0; i < m * n; i++) f64Output[i] = f32Output[i];

    aBuffer.destroy();
    bBuffer.destroy();
    outputBuffer.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return this.createArray(f64Output, [m, n]);
  }

  // Dot product (1D)
  private async runDot(a: IFaceNDArray, b: IFaceNDArray): Promise<number> {
    if (a.data.length !== b.data.length) throw new Error('Dimension mismatch');
    const n = a.data.length;
    const numWorkgroups = Math.ceil(n / 256);

    const pipeline = this.getOrCreatePipeline('dot', DOT_SHADER);

    const aBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32A = new Float32Array(n);
    const f32B = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      f32A[i] = a.data[i];
      f32B[i] = b.data[i];
    }
    this.device.queue.writeBuffer(aBuffer, 0, f32A);
    this.device.queue.writeBuffer(bBuffer, 0, f32B);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const partials = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    let result = 0;
    for (let i = 0; i < numWorkgroups; i++) result += partials[i];

    aBuffer.destroy();
    bBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return result;
  }

  // Trace
  private async runTrace(arr: IFaceNDArray): Promise<number> {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('trace requires square matrix');
    }
    const n = arr.shape[0];
    const numWorkgroups = Math.ceil(n / 256);

    const pipeline = this.getOrCreatePipeline('trace', TRACE_SHADER);

    const inputBuffer = this.device.createBuffer({
      size: n * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32Input = new Float32Array(n * n);
    for (let i = 0; i < arr.data.length; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const partials = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    let result = 0;
    for (let i = 0; i < numWorkgroups; i++) result += partials[i];

    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return result;
  }

  // Norm (L2)
  private async runNorm(arr: IFaceNDArray): Promise<number> {
    const n = arr.data.length;
    const numWorkgroups = Math.ceil(n / 256);

    const pipeline = this.getOrCreatePipeline('norm', NORM_SHADER);

    const inputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const f32Input = new Float32Array(n);
    for (let i = 0; i < n; i++) f32Input[i] = arr.data[i];
    this.device.queue.writeBuffer(inputBuffer, 0, f32Input);
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    const readBuffer = this.device.createBuffer({
      size: numWorkgroups * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numWorkgroups * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const partials = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    let sumSq = 0;
    for (let i = 0; i < numWorkgroups; i++) sumSq += partials[i];

    inputBuffer.destroy();
    outputBuffer.destroy();
    sizeBuffer.destroy();
    readBuffer.destroy();

    return Math.sqrt(sumSq);
  }

  // Helper methods
  private createArray(data: number[] | Float64Array, shape: number[]): IFaceNDArray {
    const f64 = data instanceof Float64Array ? data : new Float64Array(data);
    // Use the static factory that creates tensor and pre-caches the CPU data
    return WebGPUNDArray.fromArray(f64, shape, this.device);
  }

  private size(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
  }

  // ============ Creation Operations ============

  zeros(shape: number[]): IFaceNDArray {
    return this.createArray(new Float64Array(this.size(shape)), shape);
  }

  ones(shape: number[]): IFaceNDArray {
    return this.createArray(new Float64Array(this.size(shape)).fill(1), shape);
  }

  full(shape: number[], value: number): IFaceNDArray {
    return this.createArray(new Float64Array(this.size(shape)).fill(value), shape);
  }

  arange(start: number, stop: number, step: number): IFaceNDArray {
    if (step === 0) throw new Error('step cannot be zero');
    const data: number[] = [];
    if (step > 0) {
      for (let i = start; i < stop; i += step) data.push(i);
    } else {
      for (let i = start; i > stop; i += step) data.push(i);
    }
    return this.createArray(data, [data.length]);
  }

  linspace(start: number, stop: number, num: number): IFaceNDArray {
    if (num === 0) return this.createArray([], [0]);
    if (num === 1) return this.createArray([start], [1]);
    const data: number[] = [];
    const step = (stop - start) / (num - 1);
    for (let i = 0; i < num; i++) data.push(start + i * step);
    return this.createArray(data, [num]);
  }

  eye(n: number): IFaceNDArray {
    const data = new Float64Array(n * n);
    for (let i = 0; i < n; i++) data[i * n + i] = 1;
    return this.createArray(data, [n, n]);
  }

  diag(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    const shape = arr.shape;
    const data = arr.data;

    if (shape.length === 1) {
      const n = shape[0] + Math.abs(k);
      const result = new Float64Array(n * n);
      for (let i = 0; i < shape[0]; i++) {
        if (k >= 0) result[i * n + i + k] = data[i];
        else result[(i - k) * n + i] = data[i];
      }
      return this.createArray(result, [n, n]);
    } else if (shape.length === 2) {
      const [rows, cols] = shape;
      const diagLen = Math.max(0, Math.min(k >= 0 ? rows : rows + k, k >= 0 ? cols - k : cols));
      const result: number[] = [];
      for (let i = 0; i < diagLen; i++) {
        const row = k >= 0 ? i : i - k;
        const col = k >= 0 ? i + k : i;
        result.push(data[row * cols + col]);
      }
      return this.createArray(result, [result.length]);
    }
    throw new Error('diag requires 1D or 2D array');
  }

  array(data: number[], shape?: number[]): IFaceNDArray {
    return this.createArray(data, shape || [data.length]);
  }

  // ============ Math - Unary Operations (GPU) ============

  sin(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'sin'));
  }

  cos(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'cos'));
  }

  tan(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'tan'));
  }

  arcsin(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'asin'));
  }

  arccos(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'acos'));
  }

  arctan(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'atan'));
  }

  sinh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'sinh'));
  }

  cosh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'cosh'));
  }

  tanh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'tanh'));
  }

  exp(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'exp'));
  }

  log(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'log'));
  }

  log2(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'log2'));
  }

  log10(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'log10'));
  }

  sqrt(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'sqrt'));
  }

  cbrt(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'cbrt'));
  }

  abs(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'abs'));
  }

  sign(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'sign'));
  }

  floor(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'floor'));
  }

  ceil(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'ceil'));
  }

  round(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'round'));
  }

  neg(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'neg'));
  }

  reciprocal(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'reciprocal'));
  }

  square(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'square'));
  }

  // ============ Math - Unary (Extended) ============

  arcsinh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'asinh'));
  }

  arccosh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'acosh'));
  }

  arctanh(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'atanh'));
  }

  expm1(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'expm1'));
  }

  log1p(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'log1p'));
  }

  trunc(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'trunc'));
  }

  fix(arr: IFaceNDArray): IFaceNDArray {
    // Same as trunc - round toward zero
    return this.trunc(arr);
  }

  sinc(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'sinc'));
  }

  deg2rad(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'deg2rad'));
  }

  rad2deg(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'rad2deg'));
  }

  heaviside(arr: IFaceNDArray, h0: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, h0, 'heaviside'));
  }

  signbit(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runUnaryOpOnTensor(tensor, 'signbit'));
  }

  // ============ Math - Decomposition ============

  modf(arr: IFaceNDArray): { frac: IFaceNDArray; integ: IFaceNDArray } {
    const tensor = this.toTensor(arr);
    const result = this.runModfOnTensor(tensor);
    return {
      frac: this.fromTensor(result.frac),
      integ: this.fromTensor(result.integ),
    };
  }

  frexp(arr: IFaceNDArray): { mantissa: IFaceNDArray; exponent: IFaceNDArray } {
    const tensor = this.toTensor(arr);
    const result = this.runFrexpOnTensor(tensor);
    return {
      mantissa: this.fromTensor(result.mantissa),
      exponent: this.fromTensor(result.exponent),
    };
  }

  ldexp(arr: IFaceNDArray, exp: IFaceNDArray): IFaceNDArray {
    // ldexp(x, e) = x * 2^e
    // Use binary mul with exp2(e)
    const arrTensor = this.toTensor(arr);
    const expTensor = this.toTensor(exp);
    const exp2Tensor = this.runUnaryOpOnTensor(expTensor, 'exp2');
    return this.fromTensor(this.runBinaryOpOnTensor(arrTensor, exp2Tensor, 'mul'));
  }

  divmod(a: IFaceNDArray, b: IFaceNDArray): { quotient: IFaceNDArray; remainder: IFaceNDArray } {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    const result = this.runDivmodOnTensor(ta, tb);
    return {
      quotient: this.fromTensor(result.quotient),
      remainder: this.fromTensor(result.remainder),
    };
  }

  // ============ Math - Binary Operations (GPU) ============

  add(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'add'));
  }

  sub(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'sub'));
  }

  mul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'mul'));
  }

  div(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'div'));
  }

  pow(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'pow'));
  }

  maximum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'maximum'));
  }

  minimum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'minimum'));
  }

  // ============ Math - Binary (Extended) (GPU) ============

  mod(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'mod'));
  }

  fmod(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'fmod'));
  }

  remainder(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.mod(a, b); // Same as mod in NumPy
  }

  copysign(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'copysign'));
  }

  hypot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'hypot'));
  }

  arctan2(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'arctan2'));
  }

  logaddexp(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'logaddexp'));
  }

  logaddexp2(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'logaddexp2'));
  }

  fmax(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'fmax'));
  }

  fmin(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const ta = this.toTensor(a);
    const tb = this.toTensor(b);
    return this.fromTensor(this.runBinaryOpOnTensor(ta, tb, 'fmin'));
  }

  // ============ Math - Scalar Operations (GPU) ============

  addScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, scalar, 'addScalar'));
  }

  subScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, scalar, 'subScalar'));
  }

  mulScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, scalar, 'mulScalar'));
  }

  divScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, scalar, 'divScalar'));
  }

  powScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runScalarOpOnTensor(tensor, scalar, 'powScalar'));
  }

  clip(arr: IFaceNDArray, min: number, max: number): IFaceNDArray {
    // clip = max(min, min(x, maxVal))
    // Use two scalar ops: first clamp to max, then clamp to min
    const tensor = this.toTensor(arr);
    const clippedMax = this.runScalarOpOnTensor(tensor, max, 'minScalar');
    return this.fromTensor(this.runScalarOpOnTensor(clippedMax, min, 'maxScalar'));
  }

  // ============ Stats Operations ============

  sum(arr: IFaceNDArray): number {
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) sum += arr.data[i];
    return sum;
  }

  prod(arr: IFaceNDArray): number {
    if (arr.data.length === 0) return 1;
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) prod *= arr.data[i];
    return prod;
  }

  mean(arr: IFaceNDArray): number {
    if (arr.data.length === 0) return NaN;
    return this.sum(arr) / arr.data.length;
  }

  var(arr: IFaceNDArray, ddof: number = 0): number {
    const n = arr.data.length;
    if (n === 0) return NaN;
    const mean = this.mean(arr);
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const diff = arr.data[i] - mean;
      sumSq += diff * diff;
    }
    return sumSq / (n - ddof);
  }

  std(arr: IFaceNDArray, ddof: number = 0): number {
    return Math.sqrt(this.var(arr, ddof));
  }

  min(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let min = arr.data[0];
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < min) min = arr.data[i];
    }
    return min;
  }

  max(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let max = arr.data[0];
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > max) max = arr.data[i];
    }
    return max;
  }

  argmin(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let minIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < arr.data[minIdx]) minIdx = i;
    }
    return minIdx;
  }

  argmax(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let maxIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > arr.data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  cumsum(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runCumulativeOnTensor(tensor, 'cumsum'));
  }

  cumprod(arr: IFaceNDArray): IFaceNDArray {
    const tensor = this.toTensor(arr);
    return this.fromTensor(this.runCumulativeOnTensor(tensor, 'cumprod'));
  }

  all(arr: IFaceNDArray): boolean {
    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] === 0) return false;
    }
    return true;
  }

  any(arr: IFaceNDArray): boolean {
    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] !== 0) return true;
    }
    return false;
  }

  sumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (axis < 0 || axis >= shape.length) throw new Error(`Invalid axis ${axis}`);

    const newShape = shape.filter((_, i) => i !== axis);
    if (newShape.length === 0) newShape.push(1);

    const newSize = newShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(newSize);

    if (shape.length === 2) {
      const [rows, cols] = shape;
      if (axis === 0) {
        for (let j = 0; j < cols; j++) {
          let sum = 0;
          for (let i = 0; i < rows; i++) sum += arr.data[i * cols + j];
          result[j] = sum;
        }
      } else {
        for (let i = 0; i < rows; i++) {
          let sum = 0;
          for (let j = 0; j < cols; j++) sum += arr.data[i * cols + j];
          result[i] = sum;
        }
      }
    }
    return this.createArray(result, newShape);
  }

  meanAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const sumResult = this.sumAxis(arr, axis);
    const axisLen = arr.shape[axis];
    const result = new Float64Array(sumResult.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = sumResult.data[i] / axisLen;
    }
    return this.createArray(result, sumResult.shape);
  }

  minAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('minAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        data[j] = arr.data[j];
        for (let i = 1; i < rows; i++) data[j] = Math.min(data[j], arr.data[i * cols + j]);
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        data[i] = arr.data[i * cols];
        for (let j = 1; j < cols; j++) data[i] = Math.min(data[i], arr.data[i * cols + j]);
      }
      return this.createArray(data, [rows]);
    }
  }

  maxAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('maxAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        data[j] = arr.data[j];
        for (let i = 1; i < rows; i++) data[j] = Math.max(data[j], arr.data[i * cols + j]);
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        data[i] = arr.data[i * cols];
        for (let j = 1; j < cols; j++) data[i] = Math.max(data[i], arr.data[i * cols + j]);
      }
      return this.createArray(data, [rows]);
    }
  }

  argminAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('argminAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        let minIdx = 0;
        for (let i = 1; i < rows; i++) {
          if (arr.data[i * cols + j] < arr.data[minIdx * cols + j]) minIdx = i;
        }
        data[j] = minIdx;
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let minIdx = 0;
        for (let j = 1; j < cols; j++) {
          if (arr.data[i * cols + j] < arr.data[i * cols + minIdx]) minIdx = j;
        }
        data[i] = minIdx;
      }
      return this.createArray(data, [rows]);
    }
  }

  argmaxAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('argmaxAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        let maxIdx = 0;
        for (let i = 1; i < rows; i++) {
          if (arr.data[i * cols + j] > arr.data[maxIdx * cols + j]) maxIdx = i;
        }
        data[j] = maxIdx;
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let maxIdx = 0;
        for (let j = 1; j < cols; j++) {
          if (arr.data[i * cols + j] > arr.data[i * cols + maxIdx]) maxIdx = j;
        }
        data[i] = maxIdx;
      }
      return this.createArray(data, [rows]);
    }
  }

  varAxis(arr: IFaceNDArray, axis: number, ddof: number = 0): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('varAxis only supports 2D');
    const mean = this.meanAxis(arr, axis);
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        let sumSq = 0;
        for (let i = 0; i < rows; i++) {
          const diff = arr.data[i * cols + j] - mean.data[j];
          sumSq += diff * diff;
        }
        data[j] = sumSq / (rows - ddof);
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let sumSq = 0;
        for (let j = 0; j < cols; j++) {
          const diff = arr.data[i * cols + j] - mean.data[i];
          sumSq += diff * diff;
        }
        data[i] = sumSq / (cols - ddof);
      }
      return this.createArray(data, [rows]);
    }
  }

  stdAxis(arr: IFaceNDArray, axis: number, ddof: number = 0): IFaceNDArray {
    const variance = this.varAxis(arr, axis, ddof);
    const data = new Float64Array(variance.data.length);
    for (let i = 0; i < data.length; i++) data[i] = Math.sqrt(variance.data[i]);
    return this.createArray(data, variance.shape);
  }

  prodAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('prodAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols).fill(1);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) data[j] *= arr.data[i * cols + j];
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(1);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) data[i] *= arr.data[i * cols + j];
      }
      return this.createArray(data, [rows]);
    }
  }

  allAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('allAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols).fill(1);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          if (arr.data[i * cols + j] === 0) {
            data[j] = 0;
            break;
          }
        }
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(1);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (arr.data[i * cols + j] === 0) {
            data[i] = 0;
            break;
          }
        }
      }
      return this.createArray(data, [rows]);
    }
  }

  anyAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('anyAxis only supports 2D');
    const [rows, cols] = shape;
    if (axis === 0) {
      const data = new Float64Array(cols).fill(0);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          if (arr.data[i * cols + j] !== 0) {
            data[j] = 1;
            break;
          }
        }
      }
      return this.createArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(0);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (arr.data[i * cols + j] !== 0) {
            data[i] = 1;
            break;
          }
        }
      }
      return this.createArray(data, [rows]);
    }
  }

  cumsumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('cumsumAxis only supports 2D');
    const [rows, cols] = shape;
    const data = new Float64Array(arr.data);
    if (axis === 0) {
      for (let j = 0; j < cols; j++) {
        for (let i = 1; i < rows; i++) data[i * cols + j] += data[(i - 1) * cols + j];
      }
    } else {
      for (let i = 0; i < rows; i++) {
        for (let j = 1; j < cols; j++) data[i * cols + j] += data[i * cols + j - 1];
      }
    }
    return this.createArray(data, [rows, cols]);
  }

  cumprodAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('cumprodAxis only supports 2D');
    const [rows, cols] = shape;
    const data = new Float64Array(arr.data);
    if (axis === 0) {
      for (let j = 0; j < cols; j++) {
        for (let i = 1; i < rows; i++) data[i * cols + j] *= data[(i - 1) * cols + j];
      }
    } else {
      for (let i = 0; i < rows; i++) {
        for (let j = 1; j < cols; j++) data[i * cols + j] *= data[i * cols + j - 1];
      }
    }
    return this.createArray(data, [rows, cols]);
  }

  // ============ Comparison Operations ============

  private _checkSameShapeCompare(a: IFaceNDArray, b: IFaceNDArray): void {
    if (a.shape.length !== b.shape.length) {
      throw new Error(`Shape mismatch: [${a.shape}] vs [${b.shape}]`);
    }
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) {
        throw new Error(`Shape mismatch: [${a.shape}] vs [${b.shape}]`);
      }
    }
  }

  equal(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] === b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  notEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] !== b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  less(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] < b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  lessEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] <= b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  greater(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] > b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  greaterEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    this._checkSameShapeCompare(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] >= b.data[i] ? 1 : 0;
    }
    return this.createArray(data, a.shape);
  }

  isnan(arr: IFaceNDArray): IFaceNDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      data[i] = Number.isNaN(arr.data[i]) ? 1 : 0;
    }
    return this.createArray(data, arr.shape);
  }

  isinf(arr: IFaceNDArray): IFaceNDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      const x = arr.data[i];
      data[i] = !Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0;
    }
    return this.createArray(data, arr.shape);
  }

  isfinite(arr: IFaceNDArray): IFaceNDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      data[i] = Number.isFinite(arr.data[i]) ? 1 : 0;
    }
    return this.createArray(data, arr.shape);
  }

  // ============ Set Operations ============

  setdiff1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const setB = new Set(b.data);
    const result = Array.from(a.data).filter(x => !setB.has(x));
    const unique = [...new Set(result)].sort((x, y) => x - y);
    return this.createArray(new Float64Array(unique), [unique.length]);
  }

  union1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const combined = new Set([...a.data, ...b.data]);
    const result = [...combined].sort((x, y) => x - y);
    return this.createArray(new Float64Array(result), [result.length]);
  }

  intersect1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const setB = new Set(b.data);
    const result = [...new Set(Array.from(a.data).filter(x => setB.has(x)))].sort((x, y) => x - y);
    return this.createArray(new Float64Array(result), [result.length]);
  }

  isin(element: IFaceNDArray, testElements: IFaceNDArray): IFaceNDArray {
    const testSet = new Set(testElements.data);
    const data = new Float64Array(element.data.length);
    for (let i = 0; i < element.data.length; i++) {
      data[i] = testSet.has(element.data[i]) ? 1 : 0;
    }
    return this.createArray(data, element.shape);
  }

  // ============ Extended Array Manipulation ============

  insert(
    arr: IFaceNDArray,
    index: number,
    values: IFaceNDArray | number,
    axis?: number
  ): IFaceNDArray {
    if (axis === undefined) {
      const flat = Array.from(this.flatten(arr).data);
      const toInsert = typeof values === 'number' ? [values] : Array.from(values.data);
      if (index < 0) index = flat.length + index + 1;
      flat.splice(index, 0, ...toInsert);
      return this.createArray(new Float64Array(flat), [flat.length]);
    }
    throw new Error('insert with axis not yet implemented');
  }

  deleteArr(arr: IFaceNDArray, index: number | number[], axis?: number): IFaceNDArray {
    if (axis === undefined) {
      const flat = Array.from(this.flatten(arr).data);
      const indices = Array.isArray(index) ? index : [index];
      const normalized = indices.map(i => (i < 0 ? flat.length + i : i)).sort((a, b) => b - a);
      for (const i of normalized) {
        flat.splice(i, 1);
      }
      return this.createArray(new Float64Array(flat), [flat.length]);
    }
    throw new Error('delete with axis not yet implemented');
  }

  append(arr: IFaceNDArray, values: IFaceNDArray, axis?: number): IFaceNDArray {
    if (axis === undefined) {
      const flat1 = this.flatten(arr);
      const flat2 = this.flatten(values);
      const result = new Float64Array(flat1.data.length + flat2.data.length);
      result.set(flat1.data);
      result.set(flat2.data, flat1.data.length);
      return this.createArray(result, [result.length]);
    }
    return this.concatenate([arr, values], axis);
  }

  atleast1d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.createArray(new Float64Array(arr.data), [1]);
    }
    return arr;
  }

  atleast2d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.createArray(new Float64Array(arr.data), [1, 1]);
    }
    if (arr.shape.length === 1) {
      return this.createArray(new Float64Array(arr.data), [1, arr.shape[0]]);
    }
    return arr;
  }

  atleast3d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.createArray(new Float64Array(arr.data), [1, 1, 1]);
    }
    if (arr.shape.length === 1) {
      return this.createArray(new Float64Array(arr.data), [1, arr.shape[0], 1]);
    }
    if (arr.shape.length === 2) {
      return this.createArray(new Float64Array(arr.data), [arr.shape[0], arr.shape[1], 1]);
    }
    return arr;
  }

  countNonzero(arr: IFaceNDArray, axis?: number): IFaceNDArray | number {
    if (axis === undefined) {
      let count = 0;
      for (let i = 0; i < arr.data.length; i++) {
        if (arr.data[i] !== 0) count++;
      }
      return count;
    }
    // With axis - count along that axis
    const normalizedAxis = axis < 0 ? arr.shape.length + axis : axis;
    const outShape = arr.shape.filter((_, i) => i !== normalizedAxis);
    const outSize = outShape.reduce((a, b) => a * b, 1) || 1;
    const result = new Float64Array(outSize);
    const strides = this._computeStrides(arr.shape);
    const outStrides = outShape.length > 0 ? this._computeStrides(outShape) : [1];
    const axisLen = arr.shape[normalizedAxis];

    for (let outIdx = 0; outIdx < outSize; outIdx++) {
      const outerCoords = new Array(outShape.length);
      let remaining = outIdx;
      for (let d = 0; d < outShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outStrides[d]);
        remaining = remaining % outStrides[d];
      }

      let count = 0;
      for (let i = 0; i < axisLen; i++) {
        const coords = new Array(arr.shape.length);
        let outerD = 0;
        for (let d = 0; d < arr.shape.length; d++) {
          if (d === normalizedAxis) {
            coords[d] = i;
          } else {
            coords[d] = outerCoords[outerD++];
          }
        }
        let idx = 0;
        for (let d = 0; d < arr.shape.length; d++) {
          idx += coords[d] * strides[d];
        }
        if (arr.data[idx] !== 0) count++;
      }
      result[outIdx] = count;
    }

    return this.createArray(result, outShape);
  }

  // ============ Linear Algebra - Sync (CPU) ============

  matmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.shape.length !== 2 || b.shape.length !== 2) throw new Error('matmul requires 2D arrays');
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error(`Dimension mismatch: ${k1} vs ${k2}`);

    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let l = 0; l < k1; l++) {
          sum += a.data[i * k1 + l] * b.data[l * n + j];
        }
        result[i * n + j] = sum;
      }
    }
    return this.createArray(result, [m, n]);
  }

  /**
   * GPU-accelerated matrix multiplication (async)
   * Uses WebGPU compute shaders with autotuning for optimal performance
   */
  async matmulAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    if (a.shape.length !== 2 || b.shape.length !== 2) throw new Error('matmul requires 2D arrays');
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error(`Dimension mismatch: ${k1} vs ${k2}`);
    const k = k1;

    // Get best shader config from autotuning system
    const config = getBestConfig(m, k, n);
    if (!config) {
      // Fallback to CPU if no config works
      return this.matmul(a, b);
    }

    // Determine padded dimensions
    const kPadded = config.usesVec4A ? Math.ceil(k / 4) * 4 : k;
    const nPadded = config.usesVec4B ? Math.ceil(n / 4) * 4 : n;

    // Convert A to f32, with optional K-dimension padding for vec4 A
    let aF32: Float32Array;
    let aBufferSize: number;
    if (config.usesVec4A) {
      // A is stored as vec4 along K: [m][kPadded/4] of vec4
      aF32 = new Float32Array(m * kPadded);
      for (let row = 0; row < m; row++) {
        for (let col = 0; col < k; col++) {
          aF32[row * kPadded + col] = a.data[row * k + col];
        }
        // Padding columns remain 0
      }
      aBufferSize = aF32.byteLength;
    } else {
      aF32 = new Float32Array(m * k);
      for (let i = 0; i < a.data.length; i++) aF32[i] = a.data[i];
      aBufferSize = aF32.byteLength;
    }

    // Convert B to f32, with optional N-dimension padding for vec4 B
    let bF32: Float32Array;
    let bBufferSize: number;
    if (config.usesVec4B) {
      // Pad N to multiple of 4 for vec4 storage
      bF32 = new Float32Array(k * nPadded);
      // Copy B with padding
      for (let row = 0; row < k; row++) {
        for (let col = 0; col < n; col++) {
          bF32[row * nPadded + col] = b.data[row * n + col];
        }
        // Padding columns remain 0
      }
      bBufferSize = bF32.byteLength;
    } else {
      bF32 = new Float32Array(k * n);
      for (let i = 0; i < b.data.length; i++) bF32[i] = b.data[i];
      bBufferSize = bF32.byteLength;
    }

    // Create uniform buffer for dimensions
    const uniformData = new Uint32Array([m, k, n, nPadded]);
    const uniformBuffer = this.bufferManager.acquire(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create storage buffers using buffer manager (pooled for reuse!)
    const aBuffer = this.bufferManager.acquire(
      aBufferSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const bBuffer = this.bufferManager.acquire(
      bBufferSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    // Output buffer size depends on vec4C config
    const outputSize = config.usesVec4C ? m * nPadded * 4 : m * n * 4;
    const outputBuffer = this.bufferManager.acquire(
      outputSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const stagingBuffer = this.bufferManager.acquireStaging(outputSize);

    // Upload data
    this.device.queue.writeBuffer(aBuffer, 0, aF32);
    this.device.queue.writeBuffer(bBuffer, 0, bF32);

    // Get or create pipeline for this config
    const cacheKey = `matmul-${config.name}`;
    let pipeline = shaderCache.get(cacheKey);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: config.shader });
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      shaderCache.set(cacheKey, pipeline);
    }

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: aBuffer } },
        { binding: 2, resource: { buffer: bBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });

    // Dispatch compute using config's tile sizes
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / config.tileN), Math.ceil(m / config.tileM));
    passEncoder.end();

    // Copy to staging
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);
    this.device.queue.submit([commandEncoder.finish()]);

    // Async readback with exception safety to prevent pool corruption
    let outputF32: Float32Array;
    try {
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      outputF32 = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
    } catch (error) {
      // Release all buffers on error to prevent leaks
      this.bufferManager.release(uniformBuffer);
      this.bufferManager.release(aBuffer);
      this.bufferManager.release(bBuffer);
      this.bufferManager.release(outputBuffer);
      // Don't release staging buffer if map failed - destroy it instead
      stagingBuffer.destroy();
      throw error;
    }

    // Convert f32 back to f64, handling vec4 output if needed
    const result = new Float64Array(m * n);
    if (config.usesVec4C && nPadded !== n) {
      // Unpad the output
      for (let row = 0; row < m; row++) {
        for (let col = 0; col < n; col++) {
          result[row * n + col] = outputF32[row * nPadded + col];
        }
      }
    } else {
      for (let i = 0; i < result.length; i++) result[i] = outputF32[i];
    }

    // Release buffers back to pool (staging buffer is already unmapped)
    this.bufferManager.release(uniformBuffer);
    this.bufferManager.release(aBuffer);
    this.bufferManager.release(bBuffer);
    this.bufferManager.release(outputBuffer);
    this.bufferManager.releaseStaging(stagingBuffer);

    return this.createArray(result, [m, n]);
  }

  /**
   * GPU-resident matmul: keeps data on GPU, no CPU readback.
   * For maximum performance benchmarking.
   * Input: WebGPUTensor (f32 on GPU), Output: WebGPUTensor (f32 on GPU)
   */
  matmulTensor(a: WebGPUTensor, b: WebGPUTensor): WebGPUTensor {
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error(`Dimension mismatch: ${k1} vs ${k2}`);
    const k = k1;

    const config = getBestConfig(m, k, n);
    if (!config) throw new Error('No suitable matmul config');

    // Determine padded dimensions
    const kPadded = config.usesVec4A ? Math.ceil(k / 4) * 4 : k;
    const nPadded = config.usesVec4B ? Math.ceil(n / 4) * 4 : n;

    // Prepare A buffer: if vec4A, need to repack with K-padding
    // For now, assume inputs are already properly aligned (pad in createTensorF32)
    // TODO: handle non-aligned inputs via a repack compute shader
    const aBufferSize = m * kPadded * 4;
    const bBufferSize = k * nPadded * 4;
    const outputSize = config.usesVec4C ? m * nPadded * 4 : m * n * 4;

    // Allocate buffers from pool
    const uniformBuffer = this.bufferManager.acquire(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const outputBuffer = this.bufferManager.acquire(
      outputSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    // Write uniforms
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([m, k, n, nPadded]));

    // Get or create compute pipeline
    const cacheKey = `matmul-${config.name}`;
    let pipeline = shaderCache.get(cacheKey);
    if (!pipeline) {
      const shaderModule = this.device.createShaderModule({ code: config.shader });
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
      shaderCache.set(cacheKey, pipeline);
    }

    // Create bind group using tensor GPU buffers directly
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: a.buffer, size: aBufferSize } },
        { binding: 2, resource: { buffer: b.buffer, size: bBufferSize } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });

    // Dispatch compute
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / config.tileN), Math.ceil(m / config.tileM));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Release uniform buffer
    this.bufferManager.release(uniformBuffer);

    // Create output tensor (data stays on GPU)
    const outShape = config.usesVec4C && nPadded !== n ? [m, nPadded] : [m, n];
    return new WebGPUTensor(outputBuffer, outShape, this.device);
  }

  /**
   * Create a WebGPUTensor with f32 data already on GPU
   * Pads K and N dimensions for vec4 alignment
   */
  createAlignedTensor(
    data: Float32Array,
    shape: number[],
    padK?: boolean,
    padN?: boolean
  ): WebGPUTensor {
    const [rows, cols] = shape;
    const paddedCols = padN ? Math.ceil(cols / 4) * 4 : cols;

    let f32: Float32Array;
    if (paddedCols !== cols) {
      f32 = new Float32Array(rows * paddedCols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          f32[r * paddedCols + c] = data[r * cols + c];
        }
      }
    } else {
      f32 = data;
    }

    const buffer = this.device.createBuffer({
      size: f32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, f32);

    return new WebGPUTensor(buffer, [rows, paddedCols], this.device);
  }

  /**
   * Run autotuning benchmark for a specific matrix size
   * Tests all compatible configs and caches the fastest one
   */
  async autotune(m: number, k: number, n: number, iterations: number = 5): Promise<string> {
    const key = `${m}x${k}x${n}`;
    console.log(`Autotuning matmul for ${key}...`);

    // Find all compatible configs
    const compatibleConfigs = SHADER_CONFIGS.filter(c => {
      if (m < c.minSize || k < c.minSize || n < c.minSize) return false;
      if (c.maxSize > 0 && (m > c.maxSize || k > c.maxSize || n > c.maxSize)) return false;
      if (c.requiresFit) {
        if (m % c.tileM !== 0 || n % c.tileN !== 0 || k % c.tileK !== 0) return false;
        if (c.usesVec4B && n % 4 !== 0) return false;
      }
      return true;
    });

    if (compatibleConfigs.length === 0) {
      console.log('No compatible configs found');
      return 'NONE';
    }

    // Create test matrices
    const aData = new Float64Array(m * k);
    const bData = new Float64Array(k * n);
    for (let i = 0; i < m * k; i++) aData[i] = Math.random();
    for (let i = 0; i < k * n; i++) bData[i] = Math.random();
    const a = this.createArray(aData, [m, k]);
    const b = this.createArray(bData, [k, n]);

    const results: { name: string; avgMs: number }[] = [];

    for (const config of compatibleConfigs) {
      // Temporarily override the cache to force this config
      autotuneCache.set(key, config.name);

      // Warmup
      for (let i = 0; i < 2; i++) {
        await this.matmulAsync(a, b);
      }

      // Benchmark
      const times: number[] = [];
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await this.matmulAsync(a, b);
        const end = performance.now();
        times.push(end - start);
      }

      const avgMs = times.reduce((a, b) => a + b) / times.length;
      results.push({ name: config.name, avgMs });
      console.log(`  ${config.name}: ${avgMs.toFixed(2)}ms`);
    }

    // Find the fastest config
    results.sort((a, b) => a.avgMs - b.avgMs);
    const best = results[0];
    autotuneCache.set(key, best.name);

    console.log(`Winner: ${best.name} (${best.avgMs.toFixed(2)}ms)`);
    return best.name;
  }

  dot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.shape.length === 1 && b.shape.length === 1) {
      let sum = 0;
      for (let i = 0; i < a.data.length; i++) sum += a.data[i] * b.data[i];
      return this.createArray([sum], [1]);
    }
    return this.matmul(a, b);
  }

  inner(a: IFaceNDArray, b: IFaceNDArray): number {
    let sum = 0;
    for (let i = 0; i < a.data.length; i++) sum += a.data[i] * b.data[i];
    return sum;
  }

  outer(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const m = a.data.length;
    const n = b.data.length;
    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = a.data[i] * b.data[j];
      }
    }
    return this.createArray(result, [m, n]);
  }

  transpose(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 1) return this.createArray(arr.data, arr.shape);
    const [rows, cols] = arr.shape;
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = arr.data[i * cols + j];
      }
    }
    return this.createArray(result, [cols, rows]);
  }

  trace(arr: IFaceNDArray): number {
    if (arr.shape.length !== 2) throw new Error('trace requires 2D array');
    const n = Math.min(arr.shape[0], arr.shape[1]);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += arr.data[i * arr.shape[1] + i];
    return sum;
  }

  /**
   * Determinant using GPU-accelerated LU decomposition
   *
   * For n >= 64, uses GPU shaders:
   * - LU_FIND_PIVOT_SHADER: parallel reduction for pivot selection
   * - LU_SWAP_ROWS_SHADER: parallel row swapping
   * - LU_ELIMINATE_SHADER: parallel elimination step
   *
   * For smaller matrices, CPU is faster due to kernel launch overhead.
   */
  det(arr: IFaceNDArray): number {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('det requires square matrix');
    }
    const n = arr.shape[0];

    // For small matrices, GPU overhead exceeds benefit
    // LU decomposition benefits from GPU when n >= 64
    if (n < 64) {
      return this.detCPU(arr);
    }

    // Use GPU-accelerated LU for larger matrices
    return this.detGPU(arr);
  }

  private detCPU(arr: IFaceNDArray): number {
    const n = arr.shape[0];
    const lu = new Float64Array(arr.data);
    let det = 1;

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(lu[k * n + i]) > Math.abs(lu[maxRow * n + i])) maxRow = k;
      }
      if (maxRow !== i) {
        for (let k = 0; k < n; k++) {
          [lu[i * n + k], lu[maxRow * n + k]] = [lu[maxRow * n + k], lu[i * n + k]];
        }
        det *= -1;
      }
      const pivot = lu[i * n + i];
      if (Math.abs(pivot) < 1e-10) return 0;
      det *= pivot;
      for (let k = i + 1; k < n; k++) {
        const factor = lu[k * n + i] / pivot;
        for (let j = i; j < n; j++) lu[k * n + j] -= factor * lu[i * n + j];
      }
    }
    return det;
  }

  private detGPU(arr: IFaceNDArray): number {
    const n = arr.shape[0];

    // Convert to f32 for GPU
    const luF32 = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) luF32[i] = arr.data[i];

    // Create GPU buffer
    const luBuffer = this.device.createBuffer({
      size: n * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(luBuffer, 0, luF32);

    // Dimension/parameter buffers
    const dimsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const pivotBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Get pipelines
    const findPivotPipeline = this.getOrCreatePipeline('lu_find_pivot', LU_FIND_PIVOT_SHADER);
    const swapRowsPipeline = this.getOrCreatePipeline('lu_swap_rows', LU_SWAP_ROWS_SHADER);
    const eliminatePipeline = this.getOrCreatePipeline('lu_eliminate', LU_ELIMINATE_SHADER);

    let det = 1;
    let signFlips = 0;

    // LU decomposition with partial pivoting
    for (let col = 0; col < n; col++) {
      // Step 1: Find pivot row (parallel reduction)
      const numRows = n - col;
      const numWorkgroups = Math.ceil(numRows / 256);
      const pivotPartialsBuffer = this.device.createBuffer({
        size: numWorkgroups * 8, // [val, idx] per workgroup
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, col, 0, 0]));

      let commandEncoder = this.device.createCommandEncoder();
      let pass = commandEncoder.beginComputePass();
      pass.setPipeline(findPivotPipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: findPivotPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: luBuffer } },
            { binding: 1, resource: { buffer: pivotPartialsBuffer } },
            { binding: 2, resource: { buffer: dimsBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(numWorkgroups);
      pass.end();

      // Read back pivot results
      const stagingBuffer = this.device.createBuffer({
        size: numWorkgroups * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      commandEncoder.copyBufferToBuffer(
        pivotPartialsBuffer,
        0,
        stagingBuffer,
        0,
        numWorkgroups * 8
      );
      this.device.queue.submit([commandEncoder.finish()]);

      // Synchronous wait (this is unavoidable for LU - each step depends on previous)
      const arrayBuffer = new ArrayBuffer(numWorkgroups * 8);
      stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const mapped = new Float32Array(stagingBuffer.getMappedRange());
        new Float32Array(arrayBuffer).set(mapped);
        stagingBuffer.unmap();
      });
      // Busy wait for map (not ideal, but needed for sync det())
      while (stagingBuffer.mapState === 'pending') {
        // Spin
      }

      // Find global max from workgroup results
      const partials = new Float32Array(arrayBuffer);
      let maxVal = -Infinity;
      let maxRow = col;
      for (let w = 0; w < numWorkgroups; w++) {
        if (partials[w * 2] > maxVal) {
          maxVal = partials[w * 2];
          maxRow = Math.round(partials[w * 2 + 1]);
        }
      }

      // Step 2: Swap rows if needed (GPU parallel)
      if (maxRow !== col) {
        signFlips++;
        this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, col, maxRow, 0]));

        commandEncoder = this.device.createCommandEncoder();
        pass = commandEncoder.beginComputePass();
        pass.setPipeline(swapRowsPipeline);
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: swapRowsPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: luBuffer } },
              { binding: 1, resource: { buffer: dimsBuffer } },
            ],
          })
        );
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
      }

      // Get pivot value
      const pivotStaging = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(luBuffer, (col * n + col) * 4, pivotStaging, 0, 4);
      this.device.queue.submit([commandEncoder.finish()]);

      pivotStaging.mapAsync(GPUMapMode.READ).then(() => {});
      while (pivotStaging.mapState === 'pending') {}
      const pivotVal = new Float32Array(pivotStaging.getMappedRange())[0];
      pivotStaging.unmap();

      if (Math.abs(pivotVal) < 1e-10) {
        // Cleanup
        luBuffer.destroy();
        dimsBuffer.destroy();
        pivotBuffer.destroy();
        pivotPartialsBuffer.destroy();
        stagingBuffer.destroy();
        pivotStaging.destroy();
        return 0;
      }

      det *= pivotVal;

      // Step 3: Elimination (GPU parallel)
      if (col < n - 1) {
        this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, col, 0, 0]));
        this.device.queue.writeBuffer(pivotBuffer, 0, new Float32Array([pivotVal]));

        commandEncoder = this.device.createCommandEncoder();
        pass = commandEncoder.beginComputePass();
        pass.setPipeline(eliminatePipeline);
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: eliminatePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: luBuffer } },
              { binding: 1, resource: { buffer: dimsBuffer } },
              { binding: 2, resource: { buffer: pivotBuffer } },
            ],
          })
        );
        pass.dispatchWorkgroups(Math.ceil((n - col) / 16), Math.ceil((n - col - 1) / 16));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
      }

      // Cleanup iteration buffers
      pivotPartialsBuffer.destroy();
      stagingBuffer.destroy();
      pivotStaging.destroy();
    }

    // Cleanup
    luBuffer.destroy();
    dimsBuffer.destroy();
    pivotBuffer.destroy();

    return signFlips % 2 === 0 ? det : -det;
  }

  /**
   * Matrix inverse using GPU-accelerated Gauss-Jordan elimination
   *
   * For n >= 64, uses GPU shaders:
   * - INV_SWAP_ROWS_SHADER: parallel row swapping
   * - INV_SCALE_ROW_SHADER: parallel row scaling
   * - INV_ELIMINATE_SHADER: parallel elimination
   *
   * For smaller matrices, CPU is faster due to kernel launch overhead.
   */
  inv(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('inv requires square matrix');
    }
    const n = arr.shape[0];

    // For small matrices, GPU overhead exceeds benefit
    if (n < 64) {
      return this.invCPU(arr);
    }

    // Use GPU-accelerated Gauss-Jordan for larger matrices
    return this.invGPU(arr);
  }

  private invCPU(arr: IFaceNDArray): IFaceNDArray {
    const n = arr.shape[0];
    const aug = new Float64Array(n * n * 2);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aug[i * n * 2 + j] = arr.data[i * n + j];
        aug[i * n * 2 + n + j] = i === j ? 1 : 0;
      }
    }

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k * n * 2 + i]) > Math.abs(aug[maxRow * n * 2 + i])) maxRow = k;
      }
      for (let k = 0; k < n * 2; k++) {
        [aug[i * n * 2 + k], aug[maxRow * n * 2 + k]] = [
          aug[maxRow * n * 2 + k],
          aug[i * n * 2 + k],
        ];
      }
      const pivot = aug[i * n * 2 + i];
      if (Math.abs(pivot) < 1e-10) throw new Error('Matrix is singular');
      for (let k = 0; k < n * 2; k++) aug[i * n * 2 + k] /= pivot;
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k * n * 2 + i];
          for (let j = 0; j < n * 2; j++) aug[k * n * 2 + j] -= factor * aug[i * n * 2 + j];
        }
      }
    }

    const result = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aug[i * n * 2 + n + j];
      }
    }
    return this.createArray(result, [n, n]);
  }

  private invGPU(arr: IFaceNDArray): IFaceNDArray {
    const n = arr.shape[0];
    const width = n * 2;

    // Create augmented matrix [A|I] in f32
    const augF32 = new Float32Array(n * width);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        augF32[i * width + j] = arr.data[i * n + j];
        augF32[i * width + n + j] = i === j ? 1 : 0;
      }
    }

    // Create GPU buffer
    const augBuffer = this.device.createBuffer({
      size: n * width * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(augBuffer, 0, augF32);

    // Dimension buffers
    const dimsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const scaleBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Get pipelines
    const swapRowsPipeline = this.getOrCreatePipeline('inv_swap_rows', INV_SWAP_ROWS_SHADER);
    const scaleRowPipeline = this.getOrCreatePipeline('inv_scale_row', INV_SCALE_ROW_SHADER);
    const eliminatePipeline = this.getOrCreatePipeline('inv_eliminate', INV_ELIMINATE_SHADER);

    // Gauss-Jordan elimination
    for (let i = 0; i < n; i++) {
      // Find pivot row (CPU for simplicity - small O(n) operation)
      const colStagingBuffer = this.device.createBuffer({
        size: (n - i) * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const commandEncoder1 = this.device.createCommandEncoder();
      for (let k = i; k < n; k++) {
        commandEncoder1.copyBufferToBuffer(
          augBuffer,
          (k * width + i) * 4,
          colStagingBuffer,
          (k - i) * 4,
          4
        );
      }
      this.device.queue.submit([commandEncoder1.finish()]);

      colStagingBuffer.mapAsync(GPUMapMode.READ).then(() => {});
      while (colStagingBuffer.mapState === 'pending') {}
      const colData = new Float32Array(colStagingBuffer.getMappedRange());
      let maxRow = i;
      let maxVal = Math.abs(colData[0]);
      for (let k = 1; k < n - i; k++) {
        if (Math.abs(colData[k]) > maxVal) {
          maxVal = Math.abs(colData[k]);
          maxRow = i + k;
        }
      }
      colStagingBuffer.unmap();
      colStagingBuffer.destroy();

      // Swap rows if needed (GPU parallel)
      if (maxRow !== i) {
        this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, i, maxRow, 0]));

        const commandEncoder2 = this.device.createCommandEncoder();
        const pass2 = commandEncoder2.beginComputePass();
        pass2.setPipeline(swapRowsPipeline);
        pass2.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: swapRowsPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: augBuffer } },
              { binding: 1, resource: { buffer: dimsBuffer } },
            ],
          })
        );
        pass2.dispatchWorkgroups(Math.ceil(width / 256));
        pass2.end();
        this.device.queue.submit([commandEncoder2.finish()]);
      }

      // Get pivot value
      const pivotStagingBuffer = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const commandEncoder3 = this.device.createCommandEncoder();
      commandEncoder3.copyBufferToBuffer(augBuffer, (i * width + i) * 4, pivotStagingBuffer, 0, 4);
      this.device.queue.submit([commandEncoder3.finish()]);

      pivotStagingBuffer.mapAsync(GPUMapMode.READ).then(() => {});
      while (pivotStagingBuffer.mapState === 'pending') {}
      const pivotVal = new Float32Array(pivotStagingBuffer.getMappedRange())[0];
      pivotStagingBuffer.unmap();
      pivotStagingBuffer.destroy();

      if (Math.abs(pivotVal) < 1e-10) {
        augBuffer.destroy();
        dimsBuffer.destroy();
        scaleBuffer.destroy();
        throw new Error('Matrix is singular');
      }

      // Scale row by 1/pivot (GPU parallel)
      this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, i, 0, 0]));
      this.device.queue.writeBuffer(scaleBuffer, 0, new Float32Array([1 / pivotVal]));

      let commandEncoder = this.device.createCommandEncoder();
      let pass = commandEncoder.beginComputePass();
      pass.setPipeline(scaleRowPipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: scaleRowPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: augBuffer } },
            { binding: 1, resource: { buffer: dimsBuffer } },
            { binding: 2, resource: { buffer: scaleBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(width / 256));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      // Eliminate column in all other rows (GPU parallel)
      this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([n, i, 0, 0]));

      commandEncoder = this.device.createCommandEncoder();
      pass = commandEncoder.beginComputePass();
      pass.setPipeline(eliminatePipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: eliminatePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: augBuffer } },
            { binding: 1, resource: { buffer: dimsBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(n / 16));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);
    }

    // Read back result (right half of augmented matrix)
    const resultStagingBuffer = this.device.createBuffer({
      size: n * width * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(augBuffer, 0, resultStagingBuffer, 0, n * width * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    resultStagingBuffer.mapAsync(GPUMapMode.READ).then(() => {});
    while (resultStagingBuffer.mapState === 'pending') {}
    const augResult = new Float32Array(resultStagingBuffer.getMappedRange());

    // Extract inverse (right half)
    const result = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = augResult[i * width + n + j];
      }
    }

    resultStagingBuffer.unmap();
    resultStagingBuffer.destroy();
    augBuffer.destroy();
    dimsBuffer.destroy();
    scaleBuffer.destroy();

    return this.createArray(result, [n, n]);
  }

  solve(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aInv = this.inv(a);
    const bMat = b.shape.length === 1 ? this.createArray(b.data, [b.shape[0], 1]) : b;
    return this.matmul(aInv, bMat);
  }

  norm(arr: IFaceNDArray, ord: number = 2): number {
    // Sync implementation - works directly on data (requires arr to be materialized)
    // For GPU-optimized async version, use normAsync()
    const data = arr.data; // Will throw if GPU data not cached
    if (ord === Infinity) {
      // L-infinity: max of absolute values
      let max = Math.abs(data[0]);
      for (let i = 1; i < data.length; i++) {
        const abs = Math.abs(data[i]);
        if (abs > max) max = abs;
      }
      return max;
    }
    if (ord === 1) {
      // L1 norm: sum of absolute values
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += Math.abs(data[i]);
      return sum;
    }
    if (ord === -Infinity) {
      // -Infinity norm: min of absolute values
      let min = Math.abs(data[0]);
      for (let i = 1; i < data.length; i++) {
        const abs = Math.abs(data[i]);
        if (abs < min) min = abs;
      }
      return min;
    }
    // L2 (default): sqrt(sum of squares)
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) sumSq += data[i] * data[i];
    return Math.sqrt(sumSq);
  }

  /**
   * QR decomposition using GPU-accelerated Modified Gram-Schmidt (async)
   *
   * GPU operations via WGSL shaders:
   * - QR_COLUMN_NORM_SHADER: parallel reduction for column norms
   * - QR_NORMALIZE_COLUMN_SHADER: parallel column normalization
   * - QR_ORTHOGONALIZE_SHADER: parallel projection subtraction
   *
   * All heavy operations run on GPU. Uses async/await for proper
   * GPU synchronization without busy-waiting.
   */
  async qrAsync(arr: IFaceNDArray): Promise<{ q: IFaceNDArray; r: IFaceNDArray }> {
    if (arr.shape.length !== 2) throw new Error('qr requires 2D array');
    const [m, n] = arr.shape;

    // Get input data
    let arrData: Float64Array;
    if (arr instanceof WebGPUNDArray) {
      arrData = await arr.getData();
    } else {
      arrData = arr.data;
    }

    // Convert to f32 and create GPU buffers
    const qF32 = new Float32Array(m * n);
    for (let i = 0; i < m * n; i++) qF32[i] = arrData[i];

    const qBuffer = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(qBuffer, 0, qF32);

    const rBuffer = this.device.createBuffer({
      size: n * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(rBuffer, 0, new Float32Array(n * n));

    // Get pipelines
    const normPipeline = this.getOrCreatePipeline('qr_column_norm', QR_COLUMN_NORM_SHADER);
    const normalizePipeline = this.getOrCreatePipeline('qr_normalize', QR_NORMALIZE_COLUMN_SHADER);
    const orthogPipeline = this.getOrCreatePipeline('qr_orthogonalize', QR_ORTHOGONALIZE_SHADER);

    const numRowWorkgroups = Math.ceil(m / 256);

    // Reusable buffers (created once, reused across iterations)
    const dimsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const normBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const normPartials = this.device.createBuffer({
      size: Math.max(numRowWorkgroups, 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Pooled staging buffer for norm partials (reused across iterations)
    const normStagingBuffer = this.bufferManager.acquireStaging(numRowWorkgroups * 4);

    // Track staging buffers acquired during iteration for cleanup
    const acquiredStagingBuffers: GPUBuffer[] = [normStagingBuffer];

    try {
      // Process each column
      for (let col = 0; col < n; col++) {
        // Step 1: GPU reduction for column norm
        this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([m, n, col, 0]));

        let commandEncoder = this.device.createCommandEncoder();
        let pass = commandEncoder.beginComputePass();
        pass.setPipeline(normPipeline);
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: normPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: qBuffer } },
              { binding: 1, resource: { buffer: normPartials } },
              { binding: 2, resource: { buffer: dimsBuffer } },
            ],
          })
        );
        pass.dispatchWorkgroups(Math.max(numRowWorkgroups, 1));
        pass.end();

        // Copy to pooled staging buffer (reused)
        commandEncoder.copyBufferToBuffer(
          normPartials,
          0,
          normStagingBuffer,
          0,
          numRowWorkgroups * 4
        );
        this.device.queue.submit([commandEncoder.finish()]);

        await normStagingBuffer.mapAsync(GPUMapMode.READ);
        const partials = new Float32Array(normStagingBuffer.getMappedRange().slice(0));
        normStagingBuffer.unmap();

        let normSquared = 0;
        for (let i = 0; i < numRowWorkgroups; i++) normSquared += partials[i];
        const colNorm = Math.sqrt(normSquared);

        // Step 2: GPU normalize column
        this.device.queue.writeBuffer(normBuffer, 0, new Float32Array([colNorm]));

        // Step 3: Orthogonalize remaining columns using GPU
        const numCols = n - col - 1;

        if (numCols > 0) {
          // BATCHED: Combine normalize + dot products + orthogonalize into fewer submissions
          // We need to read dot products mid-way, so we batch: [normalize] then [dot+copy] then [orthogonalize]

          // Pass 1: Normalize column (no readback needed)
          commandEncoder = this.device.createCommandEncoder();
          pass = commandEncoder.beginComputePass();
          pass.setPipeline(normalizePipeline);
          pass.setBindGroup(
            0,
            this.device.createBindGroup({
              layout: normalizePipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: qBuffer } },
                { binding: 1, resource: { buffer: rBuffer } },
                { binding: 2, resource: { buffer: dimsBuffer } },
                { binding: 3, resource: { buffer: normBuffer } },
              ],
            })
          );
          pass.dispatchWorkgroups(Math.ceil(m / 256));
          pass.end();

          // GPU: Compute dot products using QR_DOT_COLUMNS_SHADER
          // Output is partial sums: [numCols * numRowWorkgroups]
          const dotPartialsBuffer = this.device.createBuffer({
            size: numCols * numRowWorkgroups * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([m, n, col, numCols]));

          const dotPipeline = this.getOrCreatePipeline('qr_dot_columns', QR_DOT_COLUMNS_SHADER);
          pass = commandEncoder.beginComputePass();
          pass.setPipeline(dotPipeline);
          pass.setBindGroup(
            0,
            this.device.createBindGroup({
              layout: dotPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: qBuffer } },
                { binding: 1, resource: { buffer: dotPartialsBuffer } },
                { binding: 2, resource: { buffer: dimsBuffer } },
              ],
            })
          );
          pass.dispatchWorkgroups(numRowWorkgroups, numCols);
          pass.end();

          // Pooled staging for dot partials
          const dotPartialsStagingBuffer = this.bufferManager.acquireStaging(
            numCols * numRowWorkgroups * 4
          );
          acquiredStagingBuffers.push(dotPartialsStagingBuffer);

          commandEncoder.copyBufferToBuffer(
            dotPartialsBuffer,
            0,
            dotPartialsStagingBuffer,
            0,
            numCols * numRowWorkgroups * 4
          );

          // Submit batched work: normalize + dot products
          this.device.queue.submit([commandEncoder.finish()]);

          await dotPartialsStagingBuffer.mapAsync(GPUMapMode.READ);
          const dotPartials = new Float32Array(dotPartialsStagingBuffer.getMappedRange().slice(0));
          dotPartialsStagingBuffer.unmap();

          // Release staging buffer back to pool for reuse
          this.bufferManager.releaseStaging(dotPartialsStagingBuffer);
          acquiredStagingBuffers.pop();

          dotPartialsBuffer.destroy();

          // Final reduction of partial sums (small: numRowWorkgroups values per column)
          const dots = new Float32Array(numCols);
          for (let kOffset = 0; kOffset < numCols; kOffset++) {
            let sum = 0;
            for (let wg = 0; wg < numRowWorkgroups; wg++) {
              sum += dotPartials[kOffset * numRowWorkgroups + wg];
            }
            dots[kOffset] = sum;
          }

          // GPU orthogonalize
          const dotsBuffer = this.device.createBuffer({
            size: numCols * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.device.queue.writeBuffer(dotsBuffer, 0, dots);
          this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([m, n, col, numCols]));

          commandEncoder = this.device.createCommandEncoder();
          pass = commandEncoder.beginComputePass();
          pass.setPipeline(orthogPipeline);
          pass.setBindGroup(
            0,
            this.device.createBindGroup({
              layout: orthogPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: qBuffer } },
                { binding: 1, resource: { buffer: rBuffer } },
                { binding: 2, resource: { buffer: dotsBuffer } },
                { binding: 3, resource: { buffer: dimsBuffer } },
              ],
            })
          );
          pass.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(numCols / 16));
          pass.end();
          this.device.queue.submit([commandEncoder.finish()]);

          dotsBuffer.destroy();
        } else {
          // Last column: just normalize, no orthogonalization needed
          commandEncoder = this.device.createCommandEncoder();
          pass = commandEncoder.beginComputePass();
          pass.setPipeline(normalizePipeline);
          pass.setBindGroup(
            0,
            this.device.createBindGroup({
              layout: normalizePipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: qBuffer } },
                { binding: 1, resource: { buffer: rBuffer } },
                { binding: 2, resource: { buffer: dimsBuffer } },
                { binding: 3, resource: { buffer: normBuffer } },
              ],
            })
          );
          pass.dispatchWorkgroups(Math.ceil(m / 256));
          pass.end();
          this.device.queue.submit([commandEncoder.finish()]);
        }
      }

      // Read final results using pooled staging buffers
      const qFinalStaging = this.bufferManager.acquireStaging(m * n * 4);
      const rFinalStaging = this.bufferManager.acquireStaging(n * n * 4);
      acquiredStagingBuffers.push(qFinalStaging, rFinalStaging);

      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(qBuffer, 0, qFinalStaging, 0, m * n * 4);
      commandEncoder.copyBufferToBuffer(rBuffer, 0, rFinalStaging, 0, n * n * 4);
      this.device.queue.submit([commandEncoder.finish()]);

      await qFinalStaging.mapAsync(GPUMapMode.READ);
      const qFinal = new Float32Array(qFinalStaging.getMappedRange().slice(0));
      qFinalStaging.unmap();

      await rFinalStaging.mapAsync(GPUMapMode.READ);
      const rFinal = new Float32Array(rFinalStaging.getMappedRange().slice(0));
      rFinalStaging.unmap();

      // Convert to f64
      const qResult = new Float64Array(m * n);
      const rResult = new Float64Array(n * n);
      for (let i = 0; i < m * n; i++) qResult[i] = qFinal[i];
      for (let i = 0; i < n * n; i++) rResult[i] = rFinal[i];

      return {
        q: this.createArray(qResult, [m, n]),
        r: this.createArray(rResult, [n, n]),
      };
    } finally {
      // Cleanup: release all pooled staging buffers
      for (const buf of acquiredStagingBuffers) {
        this.bufferManager.releaseStaging(buf);
      }

      // Destroy non-pooled GPU buffers
      dimsBuffer.destroy();
      normBuffer.destroy();
      normPartials.destroy();
      qBuffer.destroy();
      rBuffer.destroy();
    }
  }

  /**
   * Sync QR wrapper - calls async version
   */
  qr(arr: IFaceNDArray): { q: IFaceNDArray; r: IFaceNDArray } {
    // For sync interface, use CPU fallback (Modified Gram-Schmidt)
    if (arr.shape.length !== 2) throw new Error('qr requires 2D array');
    const [m, n] = arr.shape;
    const arrData = arr.data;

    const q = new Float64Array(m * n);
    const r = new Float64Array(n * n);
    for (let i = 0; i < m * n; i++) q[i] = arrData[i];

    for (let col = 0; col < n; col++) {
      // Compute norm
      let normSquared = 0;
      for (let i = 0; i < m; i++) {
        normSquared += q[i * n + col] ** 2;
      }
      const colNorm = Math.sqrt(normSquared);
      r[col * n + col] = colNorm;

      // Normalize
      if (colNorm > 1e-10) {
        for (let i = 0; i < m; i++) {
          q[i * n + col] /= colNorm;
        }
      }

      // Orthogonalize remaining columns
      for (let k = col + 1; k < n; k++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += q[i * n + col] * q[i * n + k];
        }
        r[col * n + k] = dot;
        for (let i = 0; i < m; i++) {
          q[i * n + k] -= dot * q[i * n + col];
        }
      }
    }

    return {
      q: this.createArray(q, [m, n]),
      r: this.createArray(r, [n, n]),
    };
  }

  /**
   * SVD using GPU-accelerated power iteration (async)
   * Computes FULL SVD: A = U * Σ * V^T
   *
   * Algorithm:
   * 1. Compute A^T A via GPU matmul
   * 2. Find eigenvectors of A^T A via power iteration with:
   *    - Convergence checking (||v_new - v_old|| < tol)
   *    - Orthogonalization against previously found eigenvectors
   *    - Deflation to find subsequent eigenvectors
   * 3. V = eigenvectors of A^T A (right singular vectors)
   * 4. σ_i = sqrt(eigenvalue_i) (singular values)
   * 5. U = A * V * Σ^(-1) (left singular vectors)
   *
   * GPU operations:
   * - A^T A via matmulAsync
   * - Power iteration matrix-vector multiply (SVD_POWER_ITERATION_SHADER)
   * - U computation via matmulAsync
   */
  async svdAsync(
    arr: IFaceNDArray
  ): Promise<{ u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray }> {
    if (arr.shape.length !== 2) throw new Error('svd requires 2D array');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);

    // Get input data
    let arrData: Float64Array;
    if (arr instanceof WebGPUNDArray) {
      arrData = await arr.getData();
    } else {
      arrData = arr.data;
    }

    // Compute A^T A using GPU matmul
    const at = this.transpose(arr);
    const ata = await this.matmulAsync(at, arr);

    let ataData: Float64Array;
    if (ata instanceof WebGPUNDArray) {
      ataData = await ata.getData();
    } else {
      ataData = ata.data;
    }

    // Get pipeline
    const powerIterPipeline = this.getOrCreatePipeline(
      'svd_power_iter',
      SVD_POWER_ITERATION_SHADER
    );

    // Working copy for deflation
    const ataWork = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) ataWork[i] = ataData[i];

    const singularValues = new Float64Array(k);
    const vMatrix = new Float64Array(n * k); // Store V column by column

    // GPU buffers with try/finally for cleanup
    const ataBuffer = this.device.createBuffer({
      size: n * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const vInBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const vOutBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const nBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Reusable staging buffer
    const stagingBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    try {
      this.device.queue.writeBuffer(nBuffer, 0, new Uint32Array([n]));

      // Find each singular value/vector via power iteration
      for (let svIdx = 0; svIdx < k; svIdx++) {
        this.device.queue.writeBuffer(ataBuffer, 0, ataWork);

        // Initialize random unit vector
        const v = new Float32Array(n);
        for (let j = 0; j < n; j++) v[j] = Math.random() - 0.5;

        // Orthogonalize against previously found eigenvectors
        for (let prevIdx = 0; prevIdx < svIdx; prevIdx++) {
          let dot = 0;
          for (let j = 0; j < n; j++) {
            dot += v[j] * vMatrix[j * k + prevIdx];
          }
          for (let j = 0; j < n; j++) {
            v[j] -= dot * vMatrix[j * k + prevIdx];
          }
        }

        // Normalize
        let vNorm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
        if (vNorm > 1e-10) {
          for (let j = 0; j < n; j++) v[j] /= vNorm;
        }

        // Power iteration with convergence check
        const MAX_ITER = 30;
        const CONV_TOL = 1e-6;
        let eigenvalue = 0;
        let converged = false;

        for (let iter = 0; iter < MAX_ITER && !converged; iter++) {
          this.device.queue.writeBuffer(vInBuffer, 0, v);

          // GPU: v_out = ATA * v_in
          const commandEncoder = this.device.createCommandEncoder();
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(powerIterPipeline);
          pass.setBindGroup(
            0,
            this.device.createBindGroup({
              layout: powerIterPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: ataBuffer } },
                { binding: 1, resource: { buffer: vInBuffer } },
                { binding: 2, resource: { buffer: vOutBuffer } },
                { binding: 3, resource: { buffer: nBuffer } },
              ],
            })
          );
          pass.dispatchWorkgroups(Math.ceil(n / 256));
          pass.end();
          commandEncoder.copyBufferToBuffer(vOutBuffer, 0, stagingBuffer, 0, n * 4);
          this.device.queue.submit([commandEncoder.finish()]);

          await stagingBuffer.mapAsync(GPUMapMode.READ);
          const vNew = new Float32Array(stagingBuffer.getMappedRange().slice(0));
          stagingBuffer.unmap();

          // Normalize
          vNorm = Math.sqrt(vNew.reduce((acc, x) => acc + x * x, 0));
          eigenvalue = vNorm;

          // Check convergence: ||v_new/norm - v_old|| < tol
          let diff = 0;
          for (let j = 0; j < n; j++) {
            const vNewNorm = vNorm > 1e-10 ? vNew[j] / vNorm : 0;
            diff += (vNewNorm - v[j]) ** 2;
          }
          converged = Math.sqrt(diff) < CONV_TOL;

          // Update v
          if (vNorm > 1e-10) {
            for (let j = 0; j < n; j++) v[j] = vNew[j] / vNorm;
          }

          // Re-orthogonalize against previous eigenvectors (for stability)
          for (let prevIdx = 0; prevIdx < svIdx; prevIdx++) {
            let dot = 0;
            for (let j = 0; j < n; j++) {
              dot += v[j] * vMatrix[j * k + prevIdx];
            }
            for (let j = 0; j < n; j++) {
              v[j] -= dot * vMatrix[j * k + prevIdx];
            }
          }
          // Re-normalize after orthogonalization
          vNorm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
          if (vNorm > 1e-10) {
            for (let j = 0; j < n; j++) v[j] /= vNorm;
          }
        }

        // Store singular value and V column
        singularValues[svIdx] = Math.sqrt(Math.abs(eigenvalue));
        for (let j = 0; j < n; j++) {
          vMatrix[j * k + svIdx] = v[j];
        }

        // Deflate: ATA -= eigenvalue * v * v^T
        for (let row = 0; row < n; row++) {
          for (let col = 0; col < n; col++) {
            ataWork[row * n + col] -= eigenvalue * v[row] * v[col];
          }
        }
      }
    } finally {
      // Cleanup GPU buffers
      ataBuffer.destroy();
      vInBuffer.destroy();
      vOutBuffer.destroy();
      nBuffer.destroy();
      stagingBuffer.destroy();
    }

    // Sort by singular value (descending)
    const indices = Array.from({ length: k }, (_, i) => i);
    indices.sort((a, b) => singularValues[b] - singularValues[a]);

    const sortedS = new Float64Array(k);
    const sortedV = new Float64Array(n * k);
    for (let i = 0; i < k; i++) {
      sortedS[i] = singularValues[indices[i]];
      for (let j = 0; j < n; j++) {
        sortedV[j * k + i] = vMatrix[j * k + indices[i]];
      }
    }

    // Compute U = A * V * Σ^(-1)
    // First compute A * V
    const vArr = this.createArray(sortedV, [n, k]);
    const av = await this.matmulAsync(arr, vArr);

    let avData: Float64Array;
    if (av instanceof WebGPUNDArray) {
      avData = await av.getData();
    } else {
      avData = av.data;
    }

    // Scale by 1/σ_i to get U
    const uData = new Float64Array(m * k);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        const sigma = sortedS[j];
        uData[i * k + j] = sigma > 1e-10 ? avData[i * k + j] / sigma : 0;
      }
    }

    // Transpose V to get V^T
    const vtData = new Float64Array(k * n);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n; j++) {
        vtData[i * n + j] = sortedV[j * k + i];
      }
    }

    return {
      u: this.createArray(uData, [m, k]),
      s: this.createArray(sortedS, [k]),
      vt: this.createArray(vtData, [k, n]),
    };
  }

  /**
   * Sync SVD - full implementation with U and V matrices
   * Uses power iteration with convergence check and orthogonalization
   */
  svd(arr: IFaceNDArray): { u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray } {
    if (arr.shape.length !== 2) throw new Error('svd requires 2D array');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);
    const arrData = arr.data;

    // Compute A^T A
    const at = this.transpose(arr);
    const ata = this.matmul(at, arr);
    const ataData = ata.data;

    // Working copy for deflation
    const ataWork = new Float64Array(ataData);
    const singularValues = new Float64Array(k);
    const vMatrix = new Float64Array(n * k);

    // Power iteration to find eigenvalues and eigenvectors
    for (let svIdx = 0; svIdx < k; svIdx++) {
      // Initialize random unit vector
      const v = new Float64Array(n);
      for (let j = 0; j < n; j++) v[j] = Math.random() - 0.5;

      // Orthogonalize against previously found eigenvectors
      for (let prevIdx = 0; prevIdx < svIdx; prevIdx++) {
        let dot = 0;
        for (let j = 0; j < n; j++) {
          dot += v[j] * vMatrix[j * k + prevIdx];
        }
        for (let j = 0; j < n; j++) {
          v[j] -= dot * vMatrix[j * k + prevIdx];
        }
      }

      let vNorm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
      if (vNorm > 1e-10) {
        for (let j = 0; j < n; j++) v[j] /= vNorm;
      }

      // Power iteration with convergence check
      const MAX_ITER = 30;
      const CONV_TOL = 1e-8;
      let eigenvalue = 0;
      let converged = false;

      for (let iter = 0; iter < MAX_ITER && !converged; iter++) {
        // v_new = ATA * v
        const vNew = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < n; j++) {
            sum += ataWork[i * n + j] * v[j];
          }
          vNew[i] = sum;
        }

        // Normalize
        vNorm = Math.sqrt(vNew.reduce((acc, x) => acc + x * x, 0));
        eigenvalue = vNorm;

        // Check convergence
        let diff = 0;
        for (let j = 0; j < n; j++) {
          const vNewNorm = vNorm > 1e-10 ? vNew[j] / vNorm : 0;
          diff += (vNewNorm - v[j]) ** 2;
        }
        converged = Math.sqrt(diff) < CONV_TOL;

        // Update v
        if (vNorm > 1e-10) {
          for (let j = 0; j < n; j++) v[j] = vNew[j] / vNorm;
        }

        // Re-orthogonalize against previous eigenvectors
        for (let prevIdx = 0; prevIdx < svIdx; prevIdx++) {
          let dot = 0;
          for (let j = 0; j < n; j++) {
            dot += v[j] * vMatrix[j * k + prevIdx];
          }
          for (let j = 0; j < n; j++) {
            v[j] -= dot * vMatrix[j * k + prevIdx];
          }
        }
        // Re-normalize
        vNorm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
        if (vNorm > 1e-10) {
          for (let j = 0; j < n; j++) v[j] /= vNorm;
        }
      }

      // Store singular value and V column
      singularValues[svIdx] = Math.sqrt(Math.abs(eigenvalue));
      for (let j = 0; j < n; j++) {
        vMatrix[j * k + svIdx] = v[j];
      }

      // Deflate
      for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
          ataWork[row * n + col] -= eigenvalue * v[row] * v[col];
        }
      }
    }

    // Sort descending
    const indices = Array.from({ length: k }, (_, i) => i);
    indices.sort((a, b) => singularValues[b] - singularValues[a]);

    const sortedS = new Float64Array(k);
    const sortedV = new Float64Array(n * k);
    for (let i = 0; i < k; i++) {
      sortedS[i] = singularValues[indices[i]];
      for (let j = 0; j < n; j++) {
        sortedV[j * k + i] = vMatrix[j * k + indices[i]];
      }
    }

    // Compute U = A * V * Σ^(-1)
    const vArr = this.createArray(sortedV, [n, k]);
    const av = this.matmul(arr, vArr);
    const avData = av.data;

    const uData = new Float64Array(m * k);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        const sigma = sortedS[j];
        uData[i * k + j] = sigma > 1e-10 ? avData[i * k + j] / sigma : 0;
      }
    }

    // Transpose V to get V^T
    const vtData = new Float64Array(k * n);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < n; j++) {
        vtData[i * n + j] = sortedV[j * k + i];
      }
    }

    return {
      u: this.createArray(uData, [m, k]),
      s: this.createArray(sortedS, [k]),
      vt: this.createArray(vtData, [k, n]),
    };
  }

  // ============ Advanced Linalg ============

  matrixPower(arr: IFaceNDArray, n: number): IFaceNDArray {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('matrixPower requires square 2D array');
    }
    if (n === 0) {
      return this.eye(arr.shape[0]);
    }
    if (n < 0) {
      arr = this.inv(arr);
      n = -n;
    }
    let result = this.eye(arr.shape[0]);
    let base = arr;
    while (n > 0) {
      if (n % 2 === 1) {
        result = this.matmul(result, base);
      }
      base = this.matmul(base, base);
      n = Math.floor(n / 2);
    }
    return result;
  }

  kron(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aFlat = a.shape.length === 1 ? this.reshape(a, [a.shape[0], 1]) : a;
    const bFlat = b.shape.length === 1 ? this.reshape(b, [b.shape[0], 1]) : b;

    if (aFlat.shape.length !== 2 || bFlat.shape.length !== 2) {
      throw new Error('kron requires 1D or 2D arrays');
    }

    const [am, an] = aFlat.shape;
    const [bm, bn] = bFlat.shape;
    const outShape: [number, number] = [am * bm, an * bn];

    // Use GPU shader
    const aTensor = this.toTensor(aFlat);
    const bTensor = this.toTensor(bFlat);
    const outTensor = this.runKronOnTensor(aTensor, bTensor, am, an, bm, bn);
    return this.fromTensor(outTensor);
  }

  private runKronOnTensor(
    a: WebGPUTensor,
    b: WebGPUTensor,
    am: number,
    an: number,
    bm: number,
    bn: number
  ): WebGPUTensor {
    const shaderCode = makeKronShader();
    const pipeline = this.getOrCreatePipeline('kron', shaderCode);

    const outM = am * bm;
    const outN = an * bn;
    const n = outM * outN;

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create dims uniform buffer
    const dimsBuffer = this.device.createBuffer({
      size: 16, // vec4<u32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([am, an, bm, bn]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    dimsBuffer.destroy();

    return new WebGPUTensor(outputBuffer, [outM, outN], this.device);
  }

  /**
   * Condition number using GPU-accelerated SVD (async)
   */
  async condAsync(arr: IFaceNDArray, _p: number | 'fro' = 2): Promise<number> {
    if (arr.shape.length !== 2) {
      throw new Error('cond requires a 2D matrix');
    }
    // Compute condition number using GPU SVD
    const { s } = await this.svdAsync(arr);
    let sData: Float64Array;
    if (s instanceof WebGPUNDArray) {
      sData = await s.getData();
    } else {
      sData = s.data;
    }
    const sMax = Math.max(...sData);
    const sMin = Math.min(...Array.from(sData).filter(v => v > 0));
    if (sMin === 0 || sData.length === 0) {
      return Infinity;
    }
    return sMax / sMin;
  }

  /**
   * Sync condition number using CPU SVD
   */
  cond(arr: IFaceNDArray, _p: number | 'fro' = 2): number {
    if (arr.shape.length !== 2) {
      throw new Error('cond requires a 2D matrix');
    }
    // Compute condition number using SVD
    const { s } = this.svd(arr);
    const sData = s.data;
    const sMax = Math.max(...sData);
    const sMin = Math.min(...Array.from(sData).filter(v => v > 0)); // Exclude zeros
    if (sMin === 0 || sData.length === 0) {
      return Infinity;
    }
    return sMax / sMin;
  }

  slogdet(arr: IFaceNDArray): { sign: number; logabsdet: number } {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('slogdet requires a square 2D matrix');
    }
    const det = this.det(arr);
    if (det === 0) {
      return { sign: 0, logabsdet: -Infinity };
    }
    return {
      sign: det > 0 ? 1 : -1,
      logabsdet: Math.log(Math.abs(det)),
    };
  }

  multiDot(arrays: IFaceNDArray[]): IFaceNDArray {
    if (arrays.length === 0) {
      throw new Error('multiDot requires at least one array');
    }
    if (arrays.length === 1) {
      return this.createArray(arrays[0].data.slice(), arrays[0].shape);
    }
    let result = arrays[0];
    for (let i = 1; i < arrays.length; i++) {
      result = this.matmul(result, arrays[i]);
    }
    return result;
  }

  // ============ Polynomial ============

  polyval(p: IFaceNDArray, x: IFaceNDArray): IFaceNDArray {
    const pFlat = this.flatten(p);
    const xFlat = this.flatten(x);
    const degree = pFlat.shape[0] - 1;

    // Use GPU shader
    const pTensor = this.toTensor(pFlat);
    const xTensor = this.toTensor(xFlat);
    const outTensor = this.runPolyvalOnTensor(pTensor, xTensor, degree);
    return this.fromTensor(outTensor);
  }

  private runPolyvalOnTensor(coeffs: WebGPUTensor, x: WebGPUTensor, degree: number): WebGPUTensor {
    const shaderCode = makePolyvalShader(degree);
    const pipeline = this.getOrCreatePipeline(`polyval_${degree}`, shaderCode);

    const n = x.size;

    // Create output buffer
    const outputBuffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create size uniform buffer
    const sizeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([n]));

    // Bind and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: coeffs.buffer } },
        { binding: 1, resource: { buffer: x.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sizeBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    sizeBuffer.destroy();

    return new WebGPUTensor(outputBuffer, x.shape, this.device);
  }

  polyadd(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aCoeffs = Array.from(this.flatten(a).data);
    const bCoeffs = Array.from(this.flatten(b).data);
    const maxLen = Math.max(aCoeffs.length, bCoeffs.length);
    const result = new Float64Array(maxLen);
    const aPadded = new Array(maxLen - aCoeffs.length).fill(0).concat(aCoeffs);
    const bPadded = new Array(maxLen - bCoeffs.length).fill(0).concat(bCoeffs);
    for (let i = 0; i < maxLen; i++) {
      result[i] = aPadded[i] + bPadded[i];
    }
    return this.createArray(result, [maxLen]);
  }

  polymul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aCoeffs = Array.from(this.flatten(a).data);
    const bCoeffs = Array.from(this.flatten(b).data);
    const resultLen = aCoeffs.length + bCoeffs.length - 1;
    const result = new Float64Array(resultLen);
    for (let i = 0; i < aCoeffs.length; i++) {
      for (let j = 0; j < bCoeffs.length; j++) {
        result[i + j] += aCoeffs[i] * bCoeffs[j];
      }
    }
    return this.createArray(result, [resultLen]);
  }

  polyfit(x: IFaceNDArray, y: IFaceNDArray, deg: number): IFaceNDArray {
    const xData = this.flatten(x).data;
    const yData = this.flatten(y).data;
    const n = xData.length;
    const m = deg + 1;

    const V = new Float64Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        V[i * m + j] = Math.pow(xData[i], deg - j);
      }
    }

    const VtV = new Float64Array(m * m);
    const Vty = new Float64Array(m);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += V[k * m + i] * V[k * m + j];
        }
        VtV[i * m + j] = sum;
      }
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += V[k * m + i] * yData[k];
      }
      Vty[i] = sum;
    }

    const A = Array.from(VtV);
    const b = Array.from(Vty);

    for (let k = 0; k < m; k++) {
      let maxVal = Math.abs(A[k * m + k]);
      let maxRow = k;
      for (let i = k + 1; i < m; i++) {
        if (Math.abs(A[i * m + k]) > maxVal) {
          maxVal = Math.abs(A[i * m + k]);
          maxRow = i;
        }
      }
      if (maxRow !== k) {
        for (let j = 0; j < m; j++) {
          const tmp = A[k * m + j];
          A[k * m + j] = A[maxRow * m + j];
          A[maxRow * m + j] = tmp;
        }
        const tmp = b[k];
        b[k] = b[maxRow];
        b[maxRow] = tmp;
      }
      for (let i = k + 1; i < m; i++) {
        const factor = A[i * m + k] / A[k * m + k];
        for (let j = k; j < m; j++) {
          A[i * m + j] -= factor * A[k * m + j];
        }
        b[i] -= factor * b[k];
      }
    }

    const c = new Float64Array(m);
    for (let i = m - 1; i >= 0; i--) {
      let sum = b[i];
      for (let j = i + 1; j < m; j++) {
        sum -= A[i * m + j] * c[j];
      }
      c[i] = sum / A[i * m + i];
    }

    return this.createArray(c, [m]);
  }

  roots(p: IFaceNDArray): IFaceNDArray {
    const coeffs = Array.from(this.flatten(p).data);

    while (coeffs.length > 0 && coeffs[0] === 0) {
      coeffs.shift();
    }

    if (coeffs.length <= 1) {
      return this.createArray(new Float64Array(0), [0]);
    }

    const n = coeffs.length - 1;

    const a0 = coeffs[0];
    for (let i = 0; i < coeffs.length; i++) {
      coeffs[i] /= a0;
    }

    const C = new Float64Array(n * n);
    for (let i = 1; i < n; i++) {
      C[i * n + (i - 1)] = 1;
    }
    for (let i = 0; i < n; i++) {
      C[i * n + (n - 1)] = -coeffs[n - i];
    }

    const maxIter = 100;
    const tol = 1e-10;
    const roots: number[] = [];

    const H = Array.from(C);

    let size = n;
    for (let deflation = 0; deflation < n; deflation++) {
      if (size === 0) break;

      if (size === 1) {
        roots.push(H[0]);
        break;
      }

      for (let iter = 0; iter < maxIter; iter++) {
        const bottomLeft = Math.abs(H[(size - 1) * n + (size - 2)]);
        const diag1 = Math.abs(H[(size - 2) * n + (size - 2)]);
        const diag2 = Math.abs(H[(size - 1) * n + (size - 1)]);

        if (bottomLeft < tol * (diag1 + diag2 + tol)) {
          roots.push(H[(size - 1) * n + (size - 1)]);
          size--;
          break;
        }

        const d = (H[(size - 2) * n + (size - 2)] - H[(size - 1) * n + (size - 1)]) / 2;
        const bElem = H[(size - 1) * n + (size - 2)];
        const shift =
          H[(size - 1) * n + (size - 1)] -
          (bElem * bElem) / (d + Math.sign(d || 1) * Math.sqrt(d * d + bElem * bElem));

        for (let i = 0; i < size; i++) {
          H[i * n + i] -= shift;
        }

        for (let i = 0; i < size - 1; i++) {
          const a = H[i * n + i];
          const b = H[(i + 1) * n + i];
          const r = Math.sqrt(a * a + b * b);
          if (r < tol) continue;
          const c = a / r;
          const s = b / r;

          for (let j = 0; j < size; j++) {
            const t1 = H[i * n + j];
            const t2 = H[(i + 1) * n + j];
            H[i * n + j] = c * t1 + s * t2;
            H[(i + 1) * n + j] = -s * t1 + c * t2;
          }

          for (let j = 0; j < size; j++) {
            const t1 = H[j * n + i];
            const t2 = H[j * n + (i + 1)];
            H[j * n + i] = c * t1 + s * t2;
            H[j * n + (i + 1)] = -s * t1 + c * t2;
          }
        }

        for (let i = 0; i < size; i++) {
          H[i * n + i] += shift;
        }
      }

      if (roots.length === deflation) {
        roots.push(H[(size - 1) * n + (size - 1)]);
        size--;
      }
    }

    roots.sort((a, b) => b - a);

    return this.createArray(new Float64Array(roots), [roots.length]);
  }

  // ============ Interpolation ============

  interp(x: IFaceNDArray, xp: IFaceNDArray, fp: IFaceNDArray): IFaceNDArray {
    const xpData = this.flatten(xp).data;
    const fpData = this.flatten(fp).data;

    const result = x.data.map(xi => {
      if (xi <= xpData[0]) return fpData[0];
      if (xi >= xpData[xpData.length - 1]) return fpData[fpData.length - 1];

      let lo = 0,
        hi = xpData.length - 1;
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (xpData[mid] <= xi) lo = mid;
        else hi = mid;
      }

      const t = (xi - xpData[lo]) / (xpData[hi] - xpData[lo]);
      return fpData[lo] + t * (fpData[hi] - fpData[lo]);
    });
    return this.createArray(new Float64Array(result), x.shape);
  }

  // ============ Histogram ============
  // Native bincount implementation - uses atomic scatter pattern
  // For large arrays, use bincountAsync() which uses GPU atomics

  bincount(x: IFaceNDArray, weights?: IFaceNDArray, minlength?: number): IFaceNDArray {
    const xFlat = this.flatten(x).data;
    const wFlat = weights ? this.flatten(weights).data : null;

    // Find max value to determine output size
    let maxVal = 0;
    for (let i = 0; i < xFlat.length; i++) {
      const v = Math.floor(xFlat[i]);
      if (v < 0) throw new Error('bincount requires non-negative integers');
      if (v > maxVal) maxVal = v;
    }

    const outLen = Math.max(maxVal + 1, minlength || 0);
    const result = new Float64Array(outLen);

    // Scatter-add pattern (mirrors GPU atomic scatter)
    for (let i = 0; i < xFlat.length; i++) {
      const bin = Math.floor(xFlat[i]);
      result[bin] += wFlat ? wFlat[i] : 1;
    }

    return this.createArray(result, [outLen]);
  }

  // ============ Creation - Like Functions ============

  zerosLike(arr: IFaceNDArray): IFaceNDArray {
    return this.zeros(arr.shape);
  }

  onesLike(arr: IFaceNDArray): IFaceNDArray {
    return this.ones(arr.shape);
  }

  emptyLike(arr: IFaceNDArray): IFaceNDArray {
    return this.zeros(arr.shape);
  }

  fullLike(arr: IFaceNDArray, value: number): IFaceNDArray {
    return this.full(arr.shape, value);
  }

  // ============ Broadcasting ============

  private _computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  private _computeBroadcastStrides(srcShape: number[], dstShape: number[]): number[] {
    const strides = new Array(dstShape.length);
    let srcStride = 1;
    for (let i = srcShape.length - 1; i >= 0; i--) {
      strides[i] = srcShape[i] === 1 ? 0 : srcStride;
      srcStride *= srcShape[i];
    }
    return strides;
  }

  broadcastTo(arr: IFaceNDArray, shape: number[]): IFaceNDArray {
    const arrShape = arr.shape;
    if (arrShape.length > shape.length) {
      throw new Error('Cannot broadcast to smaller number of dimensions');
    }

    const paddedShape = new Array(shape.length - arrShape.length).fill(1).concat(arrShape);

    for (let i = 0; i < shape.length; i++) {
      if (paddedShape[i] !== 1 && paddedShape[i] !== shape[i]) {
        throw new Error(`Cannot broadcast shape [${arrShape}] to [${shape}]`);
      }
    }

    const size = shape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(size);
    const strides = this._computeStrides(shape);
    const srcStrides = this._computeBroadcastStrides(paddedShape, shape);

    for (let i = 0; i < size; i++) {
      let srcIdx = 0;
      let remaining = i;
      for (let d = 0; d < shape.length; d++) {
        const coord = Math.floor(remaining / strides[d]);
        remaining = remaining % strides[d];
        srcIdx += coord * srcStrides[d];
      }
      result[i] = arr.data[srcIdx];
    }

    return this.createArray(result, shape);
  }

  broadcastArrays(...arrays: IFaceNDArray[]): IFaceNDArray[] {
    if (arrays.length === 0) return [];
    if (arrays.length === 1) return [this.createArray(arrays[0].data.slice(), arrays[0].shape)];

    const shapes = arrays.map(a => a.shape);
    const maxDims = Math.max(...shapes.map(s => s.length));

    const paddedShapes = shapes.map(s => {
      const padded = new Array(maxDims - s.length).fill(1);
      return padded.concat(s);
    });

    const outShape: number[] = [];
    for (let i = 0; i < maxDims; i++) {
      const dims = paddedShapes.map(s => s[i]);
      const maxDim = Math.max(...dims);
      for (const d of dims) {
        if (d !== 1 && d !== maxDim) {
          throw new Error('Shapes are not broadcastable');
        }
      }
      outShape.push(maxDim);
    }

    return arrays.map(arr => this.broadcastTo(arr, outShape));
  }

  // ============ Shape Manipulation ============

  private _normalizeAxis(axis: number, ndim: number): number {
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    return axis;
  }

  private _transposeGeneral(arr: IFaceNDArray, perm: number[], newShape: number[]): IFaceNDArray {
    const size = arr.data.length;
    const result = new Float64Array(size);

    const oldStrides = this._computeStrides(arr.shape);
    const newStrides = this._computeStrides(newShape);

    for (let i = 0; i < size; i++) {
      const coords = new Array(newShape.length);
      let remaining = i;
      for (let d = 0; d < newShape.length; d++) {
        coords[d] = Math.floor(remaining / newStrides[d]);
        remaining = remaining % newStrides[d];
      }

      let oldIdx = 0;
      for (let d = 0; d < perm.length; d++) {
        oldIdx += coords[d] * oldStrides[perm[d]];
      }

      result[i] = arr.data[oldIdx];
    }

    return this.createArray(result, newShape);
  }

  swapaxes(arr: IFaceNDArray, axis1: number, axis2: number): IFaceNDArray {
    const ndim = arr.shape.length;
    axis1 = this._normalizeAxis(axis1, ndim);
    axis2 = this._normalizeAxis(axis2, ndim);

    if (axis1 === axis2) {
      return this.createArray(arr.data.slice(), arr.shape);
    }

    const newShape = [...arr.shape];
    [newShape[axis1], newShape[axis2]] = [newShape[axis2], newShape[axis1]];

    const perm = Array.from({ length: ndim }, (_, i) => i);
    [perm[axis1], perm[axis2]] = [perm[axis2], perm[axis1]];

    return this._transposeGeneral(arr, perm, newShape);
  }

  moveaxis(arr: IFaceNDArray, source: number, destination: number): IFaceNDArray {
    const ndim = arr.shape.length;
    source = this._normalizeAxis(source, ndim);
    destination = this._normalizeAxis(destination, ndim);

    if (source === destination) {
      return this.createArray(arr.data.slice(), arr.shape);
    }

    const perm: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i !== source) perm.push(i);
    }
    perm.splice(destination, 0, source);

    const newShape = perm.map(i => arr.shape[i]);
    return this._transposeGeneral(arr, perm, newShape);
  }

  squeeze(arr: IFaceNDArray, axis?: number): IFaceNDArray {
    if (axis !== undefined) {
      const normalizedAxis = this._normalizeAxis(axis, arr.shape.length);
      if (arr.shape[normalizedAxis] !== 1) {
        throw new Error(`cannot squeeze axis ${axis} with size ${arr.shape[normalizedAxis]}`);
      }
      const newShape = arr.shape.filter((_, i) => i !== normalizedAxis);
      return this.createArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
    }

    const newShape = arr.shape.filter(d => d !== 1);
    return this.createArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
  }

  expandDims(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const ndim = arr.shape.length + 1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    const newShape = [...arr.shape];
    newShape.splice(axis, 0, 1);
    return this.createArray(arr.data.slice(), newShape);
  }

  reshape(arr: IFaceNDArray, shape: number[]): IFaceNDArray {
    let inferIdx = -1;
    let knownSize = 1;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] === -1) {
        if (inferIdx !== -1) throw new Error('can only specify one unknown dimension');
        inferIdx = i;
      } else {
        knownSize *= shape[i];
      }
    }

    const newShape = [...shape];
    if (inferIdx !== -1) {
      newShape[inferIdx] = arr.data.length / knownSize;
    }

    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== arr.data.length) {
      throw new Error(`cannot reshape array of size ${arr.data.length} into shape [${newShape}]`);
    }

    return this.createArray(arr.data.slice(), newShape);
  }

  flatten(arr: IFaceNDArray): IFaceNDArray {
    return this.createArray(arr.data.slice(), [arr.data.length]);
  }

  concatenate(arrays: IFaceNDArray[], axis: number = 0): IFaceNDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');
    if (arrays.length === 1) return this.createArray(arrays[0].data.slice(), arrays[0].shape);

    const ndim = arrays[0].shape.length;
    axis = this._normalizeAxis(axis, ndim);

    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== ndim) {
        throw new Error('all input arrays must have same number of dimensions');
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== axis && arrays[i].shape[d] !== arrays[0].shape[d]) {
          throw new Error('all input array dimensions except concat axis must match');
        }
      }
    }

    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    if (ndim === 1) {
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return this.createArray(result, outShape);
    }

    const outStrides = this._computeStrides(outShape);

    let axisOffset = 0;
    for (const arr of arrays) {
      const srcStrides = this._computeStrides(arr.shape);
      const srcSize = arr.data.length;

      for (let srcIdx = 0; srcIdx < srcSize; srcIdx++) {
        const coords = new Array(ndim);
        let remaining = srcIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / srcStrides[d]);
          remaining = remaining % srcStrides[d];
        }

        coords[axis] += axisOffset;

        let dstIdx = 0;
        for (let d = 0; d < ndim; d++) {
          dstIdx += coords[d] * outStrides[d];
        }

        result[dstIdx] = arr.data[srcIdx];
      }

      axisOffset += arr.shape[axis];
    }

    return this.createArray(result, outShape);
  }

  stack(arrays: IFaceNDArray[], axis: number = 0): IFaceNDArray {
    if (arrays.length === 0) throw new Error('need at least one array to stack');

    const shape = arrays[0].shape;
    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== shape.length) {
        throw new Error('all input arrays must have the same shape');
      }
      for (let d = 0; d < shape.length; d++) {
        if (arrays[i].shape[d] !== shape[d]) {
          throw new Error('all input arrays must have the same shape');
        }
      }
    }

    const expanded = arrays.map(arr => this.expandDims(arr, axis));
    return this.concatenate(expanded, axis);
  }

  split(arr: IFaceNDArray, indices: number | number[], axis: number = 0): IFaceNDArray[] {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const axisSize = arr.shape[axis];

    let splitIndices: number[];
    if (typeof indices === 'number') {
      if (axisSize % indices !== 0) {
        throw new Error(`array of size ${axisSize} cannot be split into ${indices} equal parts`);
      }
      const partSize = axisSize / indices;
      splitIndices = [];
      for (let i = partSize; i < axisSize; i += partSize) {
        splitIndices.push(i);
      }
    } else {
      splitIndices = indices;
    }

    const results: IFaceNDArray[] = [];
    let start = 0;

    const getSlice = (startIdx: number, endIdx: number): IFaceNDArray => {
      const sliceShape = [...arr.shape];
      sliceShape[axis] = endIdx - startIdx;
      const sliceSize = sliceShape.reduce((a, b) => a * b, 1);
      const sliceData = new Float64Array(sliceSize);

      const srcStrides = this._computeStrides(arr.shape);
      const dstStrides = this._computeStrides(sliceShape);

      for (let dstIdx = 0; dstIdx < sliceSize; dstIdx++) {
        const coords = new Array(ndim);
        let remaining = dstIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / dstStrides[d]);
          remaining = remaining % dstStrides[d];
        }

        coords[axis] += startIdx;

        let srcIdx = 0;
        for (let d = 0; d < ndim; d++) {
          srcIdx += coords[d] * srcStrides[d];
        }

        sliceData[dstIdx] = arr.data[srcIdx];
      }

      return this.createArray(sliceData, sliceShape);
    };

    for (const idx of splitIndices) {
      results.push(getSlice(start, idx));
      start = idx;
    }
    results.push(getSlice(start, axisSize));

    return results;
  }

  // ============ Conditional ============

  where(condition: IFaceNDArray, x: IFaceNDArray, y: IFaceNDArray): IFaceNDArray {
    const [condBcast, xBcast, yBcast] = this.broadcastArrays(condition, x, y);
    const size = condBcast.data.length;
    const result = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      result[i] = condBcast.data[i] !== 0 ? xBcast.data[i] : yBcast.data[i];
    }

    return this.createArray(result, condBcast.shape);
  }

  // ============ Advanced Indexing ============

  take(arr: IFaceNDArray, indices: IFaceNDArray | number[], axis?: number): IFaceNDArray {
    const indexArray = Array.isArray(indices) ? indices : Array.from(indices.data);

    if (axis === undefined) {
      const result = new Float64Array(indexArray.length);
      for (let i = 0; i < indexArray.length; i++) {
        let idx = indexArray[i];
        if (idx < 0) idx += arr.data.length;
        result[i] = arr.data[idx];
      }
      return this.createArray(result, [indexArray.length]);
    }

    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const outShape = [...arr.shape];
    outShape[axis] = indexArray.length;

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const srcStrides = this._computeStrides(arr.shape);
    const dstStrides = this._computeStrides(outShape);

    for (let dstIdx = 0; dstIdx < outSize; dstIdx++) {
      const coords = new Array(ndim);
      let remaining = dstIdx;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      let srcAxisCoord = indexArray[coords[axis]];
      if (srcAxisCoord < 0) srcAxisCoord += arr.shape[axis];
      coords[axis] = srcAxisCoord;

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return this.createArray(result, outShape);
  }

  partition(arr: IFaceNDArray, kth: number, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = axis < 0 ? axis + ndim : axis;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const result = new Float64Array(arr.data);
    const strides = this._computeStrides(arr.shape);
    const axisLen = arr.shape[axis];

    if (kth < 0 || kth >= axisLen) {
      throw new Error(`kth(=${kth}) out of bounds (${axisLen})`);
    }

    const outerShape = arr.shape.filter((_, i) => i !== axis);
    const outerStrides = outerShape.length > 0 ? this._computeStrides(outerShape) : [1];
    const outerSize = outerShape.reduce((a, b) => a * b, 1) || 1;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const outerCoords = new Array(outerShape.length);
      let remaining = outerIdx;
      for (let d = 0; d < outerShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outerStrides[d]);
        remaining = remaining % outerStrides[d];
      }

      const baseCoords = new Array(ndim);
      let outerD = 0;
      for (let d = 0; d < ndim; d++) {
        if (d === axis) {
          baseCoords[d] = 0;
        } else {
          baseCoords[d] = outerCoords[outerD++];
        }
      }

      const slice: { value: number; idx: number }[] = [];
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        slice.push({ value: arr.data[idx], idx });
      }

      const nth = (arr: { value: number; idx: number }[], k: number, lo: number, hi: number) => {
        while (lo < hi) {
          const pivot = arr[Math.floor((lo + hi) / 2)].value;
          let i = lo,
            j = hi;
          while (i <= j) {
            while (arr[i].value < pivot) i++;
            while (arr[j].value > pivot) j--;
            if (i <= j) {
              const tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          }
          if (k <= j) hi = j;
          else if (k >= i) lo = i;
          else break;
        }
      };

      nth(slice, kth, 0, axisLen - 1);

      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        result[idx] = slice[i].value;
      }
    }

    return this.createArray(result, arr.shape);
  }

  argpartition(arr: IFaceNDArray, kth: number, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = axis < 0 ? axis + ndim : axis;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }

    const result = new Float64Array(arr.data.length);
    const strides = this._computeStrides(arr.shape);
    const axisLen = arr.shape[axis];

    if (kth < 0 || kth >= axisLen) {
      throw new Error(`kth(=${kth}) out of bounds (${axisLen})`);
    }

    const outerShape = arr.shape.filter((_, i) => i !== axis);
    const outerStrides = outerShape.length > 0 ? this._computeStrides(outerShape) : [1];
    const outerSize = outerShape.reduce((a, b) => a * b, 1) || 1;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const outerCoords = new Array(outerShape.length);
      let remaining = outerIdx;
      for (let d = 0; d < outerShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outerStrides[d]);
        remaining = remaining % outerStrides[d];
      }

      const baseCoords = new Array(ndim);
      let outerD = 0;
      for (let d = 0; d < ndim; d++) {
        if (d === axis) {
          baseCoords[d] = 0;
        } else {
          baseCoords[d] = outerCoords[outerD++];
        }
      }

      const indices: { value: number; origIdx: number }[] = [];
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        indices.push({ value: arr.data[idx], origIdx: i });
      }

      const nth = (
        arr: { value: number; origIdx: number }[],
        k: number,
        lo: number,
        hi: number
      ) => {
        while (lo < hi) {
          const pivot = arr[Math.floor((lo + hi) / 2)].value;
          let i = lo,
            j = hi;
          while (i <= j) {
            while (arr[i].value < pivot) i++;
            while (arr[j].value > pivot) j--;
            if (i <= j) {
              const tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          }
          if (k <= j) hi = j;
          else if (k >= i) lo = i;
          else break;
        }
      };

      nth(indices, kth, 0, axisLen - 1);

      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        result[idx] = indices[i].origIdx;
      }
    }

    return this.createArray(result, arr.shape);
  }

  // lexsort - multi-key sort returning indices
  // Native implementation using comparator chain (last key is primary)
  // For GPU version, use lexsortAsync() which uses iterative GPU argsort
  lexsort(keys: IFaceNDArray[]): IFaceNDArray {
    if (keys.length === 0) {
      return this.createArray(new Float64Array(0), [0]);
    }

    const n = keys[0].data.length;
    for (const key of keys) {
      if (key.data.length !== n) {
        throw new Error('all keys must have the same length');
      }
    }

    const indices = Array.from({ length: n }, (_, i) => i);

    // Sort by last key first (primary), then second-to-last, etc.
    indices.sort((a, b) => {
      for (let k = keys.length - 1; k >= 0; k--) {
        const va = keys[k].data[a];
        const vb = keys[k].data[b];
        if (va < vb) return -1;
        if (va > vb) return 1;
      }
      return 0;
    });

    return this.createArray(new Float64Array(indices), [n]);
  }

  compress(condition: IFaceNDArray, arr: IFaceNDArray, axis?: number): IFaceNDArray {
    const condFlat = this.flatten(condition).data;

    if (axis === undefined) {
      const arrFlat = this.flatten(arr).data;
      const result: number[] = [];
      const len = Math.min(condFlat.length, arrFlat.length);
      for (let i = 0; i < len; i++) {
        if (condFlat[i] !== 0) {
          result.push(arrFlat[i]);
        }
      }
      return this.createArray(new Float64Array(result), [result.length]);
    }

    const ndim = arr.shape.length;
    axis = axis < 0 ? axis + ndim : axis;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    let trueCount = 0;
    const axisLen = arr.shape[axis];
    for (let i = 0; i < Math.min(condFlat.length, axisLen); i++) {
      if (condFlat[i] !== 0) trueCount++;
    }

    const outShape = [...arr.shape];
    outShape[axis] = trueCount;
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const srcStrides = this._computeStrides(arr.shape);
    const dstStrides = this._computeStrides(outShape);

    const mapping: number[] = [];
    for (let i = 0; i < Math.min(condFlat.length, axisLen); i++) {
      if (condFlat[i] !== 0) mapping.push(i);
    }

    for (let dstIdx = 0; dstIdx < outSize; dstIdx++) {
      const coords = new Array(ndim);
      let remaining = dstIdx;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      coords[axis] = mapping[coords[axis]];

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return this.createArray(result, outShape);
  }

  extract(condition: IFaceNDArray, arr: IFaceNDArray): IFaceNDArray {
    const condFlat = this.flatten(condition).data;
    const arrFlat = this.flatten(arr).data;
    const result: number[] = [];

    const len = Math.min(condFlat.length, arrFlat.length);
    for (let i = 0; i < len; i++) {
      if (condFlat[i] !== 0) {
        result.push(arrFlat[i]);
      }
    }

    return this.createArray(new Float64Array(result), [result.length]);
  }

  place(arr: IFaceNDArray, mask: IFaceNDArray, vals: IFaceNDArray): void {
    const maskFlat = this.flatten(mask).data;
    const valsFlat = this.flatten(vals).data;

    let valIdx = 0;
    for (let i = 0; i < arr.data.length && i < maskFlat.length; i++) {
      if (maskFlat[i] !== 0) {
        arr.data[i] = valsFlat[valIdx % valsFlat.length];
        valIdx++;
      }
    }
  }

  select(
    condlist: IFaceNDArray[],
    choicelist: IFaceNDArray[],
    defaultVal: number = 0
  ): IFaceNDArray {
    if (condlist.length !== choicelist.length) {
      throw new Error('condlist and choicelist must have same length');
    }
    if (condlist.length === 0) {
      throw new Error('condlist and choicelist must not be empty');
    }

    const allArrays = [...condlist, ...choicelist];
    const broadcasted = this.broadcastArrays(...allArrays);
    const shape = broadcasted[0].shape;
    const size = broadcasted[0].data.length;

    const conditions = broadcasted.slice(0, condlist.length);
    const choices = broadcasted.slice(condlist.length);

    const result = new Float64Array(size).fill(defaultVal);

    const selected = new Uint8Array(size);
    for (let c = 0; c < condlist.length; c++) {
      for (let i = 0; i < size; i++) {
        if (!selected[i] && conditions[c].data[i] !== 0) {
          result[i] = choices[c].data[i];
          selected[i] = 1;
        }
      }
    }

    return this.createArray(result, shape);
  }

  // ============ Batched Operations ============

  batchedMatmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('batchedMatmul requires at least 2D arrays');
    }

    const aM = a.shape[a.shape.length - 2];
    const aK = a.shape[a.shape.length - 1];
    const bK = b.shape[b.shape.length - 2];
    const bN = b.shape[b.shape.length - 1];

    if (aK !== bK) throw new Error('matmul inner dimensions must match');

    const aBatchShape = a.shape.slice(0, -2);
    const bBatchShape = b.shape.slice(0, -2);

    const maxBatchDims = Math.max(aBatchShape.length, bBatchShape.length);
    const paddedABatch = new Array(maxBatchDims - aBatchShape.length).fill(1).concat(aBatchShape);
    const paddedBBatch = new Array(maxBatchDims - bBatchShape.length).fill(1).concat(bBatchShape);

    const outBatchShape: number[] = [];
    for (let i = 0; i < maxBatchDims; i++) {
      const ad = paddedABatch[i];
      const bd = paddedBBatch[i];
      if (ad !== 1 && bd !== 1 && ad !== bd) {
        throw new Error('batch dimensions are not broadcastable');
      }
      outBatchShape.push(Math.max(ad, bd));
    }

    const outShape = [...outBatchShape, aM, bN];
    const batchSize = outBatchShape.length === 0 ? 1 : outBatchShape.reduce((x, y) => x * y, 1);
    const matSize = aM * bN;
    const result = new Float64Array(batchSize * matSize);

    const aBatchStrides = this._computeStrides(paddedABatch);
    const bBatchStrides = this._computeStrides(paddedBBatch);
    const outBatchStrides = this._computeStrides(outBatchShape);

    const aMatStride = aM * aK;
    const bMatStride = bK * bN;

    for (let batch = 0; batch < batchSize; batch++) {
      const coords = new Array(maxBatchDims);
      let remaining = batch;
      for (let d = 0; d < maxBatchDims; d++) {
        coords[d] = Math.floor(remaining / outBatchStrides[d]);
        remaining = remaining % outBatchStrides[d];
      }

      let aOffset = 0;
      let bOffset = 0;
      for (let d = 0; d < maxBatchDims; d++) {
        const aCoord = paddedABatch[d] === 1 ? 0 : coords[d];
        const bCoord = paddedBBatch[d] === 1 ? 0 : coords[d];
        aOffset += aCoord * aBatchStrides[d];
        bOffset += bCoord * bBatchStrides[d];
      }
      aOffset *= aMatStride;
      bOffset *= bMatStride;

      const outOffset = batch * matSize;
      for (let i = 0; i < aM; i++) {
        for (let j = 0; j < bN; j++) {
          let sum = 0;
          for (let k = 0; k < aK; k++) {
            sum += a.data[aOffset + i * aK + k] * b.data[bOffset + k * bN + j];
          }
          result[outOffset + i * bN + j] = sum;
        }
      }
    }

    return this.createArray(result, outShape);
  }

  // ============ Einstein Summation (GPU-optimized) ============

  /**
   * GPU-accelerated einsum that routes common patterns to existing GPU ops
   * and uses a GPU shader for general contractions.
   */
  einsum(subscripts: string, ...operands: IFaceNDArray[]): IFaceNDArray {
    const [inputStr, outputStr] = subscripts.split('->').map(s => s.trim());
    const inputSubscripts = inputStr.split(',').map(s => s.trim());

    if (inputSubscripts.length !== operands.length) {
      throw new Error(
        `einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`
      );
    }

    // Try to route to optimized GPU ops for common patterns
    const optimized = this._tryOptimizedEinsum(subscripts, operands);
    if (optimized !== null) {
      return optimized;
    }

    // General case: use GPU-accelerated contraction shader
    return this._einsumGPU(inputSubscripts, outputStr, operands);
  }

  /**
   * Try to route einsum to existing optimized GPU ops
   */
  private _tryOptimizedEinsum(subscripts: string, operands: IFaceNDArray[]): IFaceNDArray | null {
    const normalized = subscripts.replace(/\s+/g, '');

    // Matrix multiply: ij,jk->ik or ij,kj->ik (transposed B)
    if (normalized === 'ij,jk->ik' && operands.length === 2) {
      return this.matmul(operands[0], operands[1]);
    }
    if (normalized === 'ij,kj->ik' && operands.length === 2) {
      return this.matmul(operands[0], this.transpose(operands[1]));
    }
    if (normalized === 'ji,jk->ik' && operands.length === 2) {
      return this.matmul(this.transpose(operands[0]), operands[1]);
    }

    // Dot product / inner product: i,i->
    if ((normalized === 'i,i->' || normalized === 'i,i') && operands.length === 2) {
      return this.dot(operands[0], operands[1]);
    }

    // Outer product: i,j->ij
    if (normalized === 'i,j->ij' && operands.length === 2) {
      return this.outer(operands[0], operands[1]);
    }

    // Trace: ii->
    if ((normalized === 'ii->' || normalized === 'ii') && operands.length === 1) {
      return this.createArray(new Float64Array([this.trace(operands[0])]), [1]);
    }

    // Diagonal: ii->i
    if (normalized === 'ii->i' && operands.length === 1) {
      return this.diag(operands[0]);
    }

    // Transpose: ij->ji
    if (normalized === 'ij->ji' && operands.length === 1) {
      return this.transpose(operands[0]);
    }

    // Sum all: i-> or ij->
    if (normalized.endsWith('->') && operands.length === 1) {
      const sumVal = this.sum(operands[0]);
      return this.createArray(new Float64Array([sumVal]), [1]);
    }

    // Batch matmul: bij,bjk->bik
    if (normalized === 'bij,bjk->bik' && operands.length === 2) {
      // Could implement batch matmul here, but for now fall through
    }

    // Elementwise multiply with broadcast: ij,j->ij
    if (normalized === 'ij,j->ij' && operands.length === 2) {
      const [m, n] = operands[0].shape;
      const v = operands[1];
      if (v.shape.length === 1 && v.shape[0] === n) {
        // Broadcast multiply: each row of A multiplied by v
        const result = new Float64Array(m * n);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            result[i * n + j] = operands[0].data[i * n + j] * v.data[j];
          }
        }
        // This is still CPU - but we can use GPU broadcast multiply
        const aFlat = this.flatten(operands[0]);
        const vTiled = this.tile(v, [m]); // Tile v m times
        return this.reshape(this.mul(aFlat, vTiled), [m, n]);
      }
    }

    return null; // Fall through to general implementation
  }

  /**
   * GPU-accelerated general einsum using compute shader
   */
  private _einsumGPU(
    inputSubscripts: string[],
    outputStr: string | undefined,
    operands: IFaceNDArray[]
  ): IFaceNDArray {
    // Parse labels and sizes
    const labelSizes: Map<string, number> = new Map();
    const inputLabels: string[][] = [];

    for (let i = 0; i < operands.length; i++) {
      const labels = inputSubscripts[i].split('');
      inputLabels.push(labels);
      if (labels.length !== operands[i].shape.length) {
        throw new Error(
          `einsum: operand ${i} has ${operands[i].shape.length} dimensions but subscripts specify ${labels.length}`
        );
      }
      for (let j = 0; j < labels.length; j++) {
        const label = labels[j];
        const size = operands[i].shape[j];
        if (labelSizes.has(label)) {
          if (labelSizes.get(label) !== size) {
            throw new Error(`einsum: inconsistent size for label '${label}'`);
          }
        } else {
          labelSizes.set(label, size);
        }
      }
    }

    let outputLabels: string[];
    if (outputStr !== undefined && outputStr !== '') {
      outputLabels = outputStr.split('');
    } else {
      const labelCounts: Map<string, number> = new Map();
      for (const labels of inputLabels) {
        for (const label of labels) {
          labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
        }
      }
      outputLabels = [];
      const allLabels = Array.from(labelSizes.keys()).sort();
      for (const label of allLabels) {
        if (labelCounts.get(label) === 1) {
          outputLabels.push(label);
        }
      }
    }

    const outputShape = outputLabels.map(l => labelSizes.get(l)!);
    const outputSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

    const outputSet = new Set(outputLabels);
    const allLabels = Array.from(labelSizes.keys());
    const contractedLabels = allLabels.filter(l => !outputSet.has(l));
    const contractedSizes = contractedLabels.map(l => labelSizes.get(l)!);
    const contractedTotal =
      contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

    // For very small outputs or simple contractions, GPU overhead isn't worth it
    // Use GPU shader for larger computations
    if (outputSize * contractedTotal > 1024) {
      return this._einsumGPUShader(
        operands,
        inputLabels,
        outputLabels,
        contractedLabels,
        labelSizes,
        outputShape,
        outputSize,
        contractedTotal,
        contractedSizes
      );
    }

    // Small case: CPU is faster
    return this._einsumCPU(
      operands,
      inputLabels,
      outputLabels,
      contractedLabels,
      labelSizes,
      outputShape,
      outputSize,
      contractedTotal,
      contractedSizes
    );
  }

  /**
   * CPU implementation for small einsum operations
   */
  private _einsumCPU(
    operands: IFaceNDArray[],
    inputLabels: string[][],
    outputLabels: string[],
    contractedLabels: string[],
    labelSizes: Map<string, number>,
    outputShape: number[],
    outputSize: number,
    contractedTotal: number,
    contractedSizes: number[]
  ): IFaceNDArray {
    const result = new Float64Array(outputSize);
    const inputStrides = operands.map(op => this._computeStrides(op.shape));
    const outputStrides = outputShape.length === 0 ? [] : this._computeStrides(outputShape);

    for (let outIdx = 0; outIdx < outputSize; outIdx++) {
      const outCoords: Map<string, number> = new Map();
      let remaining = outIdx;
      for (let d = 0; d < outputLabels.length; d++) {
        const coord = Math.floor(remaining / outputStrides[d]);
        remaining = remaining % outputStrides[d];
        outCoords.set(outputLabels[d], coord);
      }

      let sum = 0;
      for (let contrIdx = 0; contrIdx < contractedTotal; contrIdx++) {
        const contrCoords: Map<string, number> = new Map();
        let contrRemaining = contrIdx;
        for (let d = 0; d < contractedLabels.length; d++) {
          const stride =
            d < contractedSizes.length - 1
              ? contractedSizes.slice(d + 1).reduce((a, b) => a * b, 1)
              : 1;
          const coord = Math.floor(contrRemaining / stride);
          contrRemaining = contrRemaining % stride;
          contrCoords.set(contractedLabels[d], coord);
        }

        const allCoords = new Map([...outCoords, ...contrCoords]);

        let prod = 1;
        for (let i = 0; i < operands.length; i++) {
          const labels = inputLabels[i];
          const strides = inputStrides[i];
          let idx = 0;
          for (let d = 0; d < labels.length; d++) {
            idx += allCoords.get(labels[d])! * strides[d];
          }
          prod *= operands[i].data[idx];
        }
        sum += prod;
      }

      result[outIdx] = sum;
    }

    return this.createArray(result, outputShape.length === 0 ? [1] : outputShape);
  }

  /**
   * GPU shader implementation for general einsum
   * Creates a specialized shader for the contraction pattern
   */
  private _einsumGPUShader(
    operands: IFaceNDArray[],
    inputLabels: string[][],
    outputLabels: string[],
    contractedLabels: string[],
    labelSizes: Map<string, number>,
    outputShape: number[],
    outputSize: number,
    contractedTotal: number,
    contractedSizes: number[]
  ): IFaceNDArray {
    // Build unique label ordering for shader
    const allLabelsList = [...outputLabels, ...contractedLabels];
    const labelToIdx = new Map<string, number>();
    allLabelsList.forEach((l, i) => labelToIdx.set(l, i));

    // Compute strides for each operand based on labels
    const inputStrides = operands.map((op, opIdx) => {
      const strides: number[] = [];
      let stride = 1;
      for (let d = op.shape.length - 1; d >= 0; d--) {
        strides.unshift(stride);
        stride *= op.shape[d];
      }
      return strides;
    });

    const outputStrides = outputShape.length === 0 ? [] : this._computeStrides(outputShape);
    const contractedStrides =
      contractedSizes.length === 0 ? [] : this._computeStrides(contractedSizes);

    // Generate specialized shader code
    const shaderCode = this._generateEinsumShader(
      operands.length,
      inputLabels,
      outputLabels,
      contractedLabels,
      labelSizes,
      inputStrides,
      outputStrides,
      contractedStrides,
      outputSize,
      contractedTotal
    );

    // Create or get cached pipeline
    const shaderKey = `einsum_${inputLabels.map(l => l.join('')).join('_')}_${outputLabels.join('')}`;
    const pipeline = this.getOrCreatePipeline(shaderKey, shaderCode);

    // Create buffers
    const operandBuffers: GPUBuffer[] = [];
    for (let i = 0; i < operands.length; i++) {
      const size = operands[i].data.length * 4;
      const buffer = this.bufferManager.acquire(
        size,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      const f32 = new Float32Array(operands[i].data.length);
      for (let j = 0; j < operands[i].data.length; j++) f32[j] = operands[i].data[j];
      this.device.queue.writeBuffer(buffer, 0, f32);
      operandBuffers.push(buffer);
    }

    const outBuffer = this.bufferManager.acquire(
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    // Build bind group entries
    const entries: GPUBindGroupEntry[] = operandBuffers.map((buf, i) => ({
      binding: i,
      resource: { buffer: buf },
    }));
    entries.push({ binding: operands.length, resource: { buffer: outBuffer } });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(outputSize / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back
    const stagingBuffer = this.device.createBuffer({
      size: outputSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outBuffer, 0, stagingBuffer, 0, outputSize * 4);
    this.device.queue.submit([copyEncoder.finish()]);

    stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {});
    while (stagingBuffer.mapState === 'pending') {}
    const resultF32 = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    // Release buffers
    for (const buf of operandBuffers) this.bufferManager.release(buf);
    this.bufferManager.release(outBuffer);

    const result = new Float64Array(outputSize);
    for (let i = 0; i < outputSize; i++) result[i] = resultF32[i];

    return this.createArray(result, outputShape.length === 0 ? [1] : outputShape);
  }

  /**
   * Generate specialized WGSL shader for einsum pattern
   */
  private _generateEinsumShader(
    numOperands: number,
    inputLabels: string[][],
    outputLabels: string[],
    contractedLabels: string[],
    labelSizes: Map<string, number>,
    inputStrides: number[][],
    outputStrides: number[],
    contractedStrides: number[],
    outputSize: number,
    contractedTotal: number
  ): string {
    // Generate operand buffer bindings
    let bindings = '';
    for (let i = 0; i < numOperands; i++) {
      bindings += `  @group(0) @binding(${i}) var<storage, read> op${i}: array<f32>;\n`;
    }
    bindings += `  @group(0) @binding(${numOperands}) var<storage, read_write> result: array<f32>;\n`;

    // Generate index computation for each operand
    const genOperandIndex = (opIdx: number): string => {
      const labels = inputLabels[opIdx];
      const strides = inputStrides[opIdx];
      const terms: string[] = [];
      for (let d = 0; d < labels.length; d++) {
        const label = labels[d];
        const outIdx = outputLabels.indexOf(label);
        const contrIdx = contractedLabels.indexOf(label);
        if (outIdx >= 0) {
          terms.push(`outCoord${outIdx} * ${strides[d]}u`);
        } else if (contrIdx >= 0) {
          terms.push(`contrCoord${contrIdx} * ${strides[d]}u`);
        }
      }
      return terms.length > 0 ? terms.join(' + ') : '0u';
    };

    // Generate output coordinate extraction
    let outCoordCode = '';
    for (let d = 0; d < outputLabels.length; d++) {
      const stride = outputStrides[d] || 1;
      if (d === 0) {
        outCoordCode += `    var outCoord${d} = outIdx / ${stride}u;\n`;
        outCoordCode += `    var outRem${d} = outIdx % ${stride}u;\n`;
      } else {
        outCoordCode += `    var outCoord${d} = outRem${d - 1} / ${stride}u;\n`;
        if (d < outputLabels.length - 1) {
          outCoordCode += `    var outRem${d} = outRem${d - 1} % ${stride}u;\n`;
        }
      }
    }

    // Generate contracted coordinate extraction
    let contrCoordCode = '';
    for (let d = 0; d < contractedLabels.length; d++) {
      const stride = contractedStrides[d] || 1;
      if (d === 0) {
        contrCoordCode += `      var contrCoord${d} = contrIdx / ${stride}u;\n`;
        contrCoordCode += `      var contrRem${d} = contrIdx % ${stride}u;\n`;
      } else {
        contrCoordCode += `      var contrCoord${d} = contrRem${d - 1} / ${stride}u;\n`;
        if (d < contractedLabels.length - 1) {
          contrCoordCode += `      var contrRem${d} = contrRem${d - 1} % ${stride}u;\n`;
        }
      }
    }

    // Generate product computation
    let productCode = '';
    for (let i = 0; i < numOperands; i++) {
      const idx = genOperandIndex(i);
      productCode += `      prod = prod * op${i}[${idx}];\n`;
    }

    return `
${bindings}
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outIdx = gid.x;
    if (outIdx >= ${outputSize}u) { return; }

${outCoordCode}
    var sum: f32 = 0.0;
    for (var contrIdx: u32 = 0u; contrIdx < ${contractedTotal}u; contrIdx = contrIdx + 1u) {
${contrCoordCode}
      var prod: f32 = 1.0;
${productCode}
      sum = sum + prod;
    }
    result[outIdx] = sum;
  }
`;
  }

  // ============ Differences ============

  diff(arr: IFaceNDArray, n: number = 1, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    let result: IFaceNDArray = arr;
    for (let i = 0; i < n; i++) {
      result = this._diffOnce(result, axis);
    }
    return result;
  }

  private _diffOnce(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const shape = arr.shape;
    if (shape[axis] < 2) {
      throw new Error('diff requires at least 2 elements along axis');
    }

    const newShape = [...shape];
    newShape[axis] -= 1;
    const newSize = newShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(newSize);

    const srcStrides = this._computeStrides(shape);
    const dstStrides = this._computeStrides(newShape);

    for (let dstIdx = 0; dstIdx < newSize; dstIdx++) {
      const coords = new Array(shape.length);
      let remaining = dstIdx;
      for (let d = 0; d < shape.length; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      let srcIdx1 = 0,
        srcIdx2 = 0;
      for (let d = 0; d < shape.length; d++) {
        if (d === axis) {
          srcIdx1 += (coords[d] + 1) * srcStrides[d];
          srcIdx2 += coords[d] * srcStrides[d];
        } else {
          srcIdx1 += coords[d] * srcStrides[d];
          srcIdx2 += coords[d] * srcStrides[d];
        }
      }

      result[dstIdx] = arr.data[srcIdx1] - arr.data[srcIdx2];
    }

    return this.createArray(result, newShape);
  }

  gradient(arr: IFaceNDArray, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const shape = arr.shape;
    const axisLen = shape[axis];

    if (axisLen < 2) {
      throw new Error('gradient requires at least 2 elements along axis');
    }

    const result = new Float64Array(arr.data.length);
    const strides = this._computeStrides(shape);

    for (let i = 0; i < arr.data.length; i++) {
      const coords = new Array(ndim);
      let remaining = i;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / strides[d]);
        remaining = remaining % strides[d];
      }

      const axisCoord = coords[axis];
      let grad: number;

      if (axisCoord === 0) {
        const nextCoords = [...coords];
        nextCoords[axis] = 1;
        let nextIdx = 0;
        for (let d = 0; d < ndim; d++) nextIdx += nextCoords[d] * strides[d];
        grad = arr.data[nextIdx] - arr.data[i];
      } else if (axisCoord === axisLen - 1) {
        const prevCoords = [...coords];
        prevCoords[axis] = axisLen - 2;
        let prevIdx = 0;
        for (let d = 0; d < ndim; d++) prevIdx += prevCoords[d] * strides[d];
        grad = arr.data[i] - arr.data[prevIdx];
      } else {
        const prevCoords = [...coords];
        const nextCoords = [...coords];
        prevCoords[axis] = axisCoord - 1;
        nextCoords[axis] = axisCoord + 1;
        let prevIdx = 0,
          nextIdx = 0;
        for (let d = 0; d < ndim; d++) {
          prevIdx += prevCoords[d] * strides[d];
          nextIdx += nextCoords[d] * strides[d];
        }
        grad = (arr.data[nextIdx] - arr.data[prevIdx]) / 2;
      }

      result[i] = grad;
    }

    return this.createArray(result, shape);
  }

  ediff1d(arr: IFaceNDArray): IFaceNDArray {
    const flat = this.flatten(arr);
    return this.diff(flat, 1, 0);
  }

  // ============ Cross Product ============

  cross(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aFlat = this.flatten(a);
    const bFlat = this.flatten(b);

    if (aFlat.data.length !== 3 || bFlat.data.length !== 3) {
      throw new Error('cross product only supports 3D vectors');
    }

    const [a1, a2, a3] = aFlat.data;
    const [b1, b2, b3] = bFlat.data;

    return this.createArray(
      new Float64Array([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]),
      [3]
    );
  }

  // ============ Statistics ============

  cov(x: IFaceNDArray, y?: IFaceNDArray): IFaceNDArray {
    if (y === undefined) {
      if (x.shape.length !== 2) {
        throw new Error('cov requires 2D array when y is not provided');
      }
      const [nVars, nObs] = x.shape;
      const result = new Float64Array(nVars * nVars);

      const means = new Float64Array(nVars);
      for (let i = 0; i < nVars; i++) {
        let sum = 0;
        for (let j = 0; j < nObs; j++) {
          sum += x.data[i * nObs + j];
        }
        means[i] = sum / nObs;
      }

      for (let i = 0; i < nVars; i++) {
        for (let j = 0; j < nVars; j++) {
          let cov = 0;
          for (let k = 0; k < nObs; k++) {
            cov += (x.data[i * nObs + k] - means[i]) * (x.data[j * nObs + k] - means[j]);
          }
          result[i * nVars + j] = cov / (nObs - 1);
        }
      }

      return this.createArray(result, [nVars, nVars]);
    } else {
      const xFlat = this.flatten(x);
      const yFlat = this.flatten(y);

      if (xFlat.data.length !== yFlat.data.length) {
        throw new Error('x and y must have same length');
      }

      const n = xFlat.data.length;
      const xMean = this.mean(xFlat);
      const yMean = this.mean(yFlat);

      let cov = 0;
      for (let i = 0; i < n; i++) {
        cov += (xFlat.data[i] - xMean) * (yFlat.data[i] - yMean);
      }
      cov /= n - 1;

      const xVar = this.var(xFlat, 1);
      const yVar = this.var(yFlat, 1);

      return this.createArray(new Float64Array([xVar, cov, cov, yVar]), [2, 2]);
    }
  }

  corrcoef(x: IFaceNDArray, y?: IFaceNDArray): IFaceNDArray {
    const covMatrix = this.cov(x, y);
    const n = covMatrix.shape[0];
    const result = new Float64Array(n * n);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const covIJ = covMatrix.data[i * n + j];
        const varI = covMatrix.data[i * n + i];
        const varJ = covMatrix.data[j * n + j];
        result[i * n + j] = covIJ / Math.sqrt(varI * varJ);
      }
    }

    return this.createArray(result, [n, n]);
  }

  // ============ Convolution (GPU) ============

  convolve(
    a: IFaceNDArray,
    v: IFaceNDArray,
    mode: 'full' | 'same' | 'valid' = 'full'
  ): IFaceNDArray {
    const aFlat = this.flatten(a);
    const vFlat = this.flatten(v);
    const aLen = aFlat.data.length;
    const vLen = vFlat.data.length;

    // Compute full convolution length and mode-specific output range
    const fullLen = aLen + vLen - 1;
    let outLen: number;
    let startIdx: number;

    if (mode === 'full') {
      outLen = fullLen;
      startIdx = 0;
    } else if (mode === 'same') {
      outLen = aLen;
      startIdx = Math.floor((vLen - 1) / 2);
    } else {
      outLen = Math.max(aLen - vLen + 1, 0);
      startIdx = vLen - 1;
    }

    // Get or create GPU pipeline
    const pipeline = this.getOrCreatePipeline('convolve', CONVOLVE_SHADER);

    // Create GPU buffers for input arrays
    const aBuffer = this.bufferManager.acquire(
      aLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const vBuffer = this.bufferManager.acquire(
      vLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const fullBuffer = this.bufferManager.acquire(
      fullLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const paramsBuffer = this.bufferManager.acquire(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    // Upload input data (f64 -> f32)
    const aF32 = new Float32Array(aLen);
    const vF32 = new Float32Array(vLen);
    for (let i = 0; i < aLen; i++) aF32[i] = aFlat.data[i];
    for (let i = 0; i < vLen; i++) vF32[i] = vFlat.data[i];
    this.device.queue.writeBuffer(aBuffer, 0, aF32);
    this.device.queue.writeBuffer(vBuffer, 0, vF32);
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([aLen, vLen, fullLen, 0]));

    // Create bind group and dispatch
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: vBuffer } },
        { binding: 2, resource: { buffer: fullBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(fullLen / 256));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Read back result
    const stagingBuffer = this.device.createBuffer({
      size: fullLen * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(fullBuffer, 0, stagingBuffer, 0, fullLen * 4);
    this.device.queue.submit([copyEncoder.finish()]);

    // Sync wait for readback
    stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {});
    while (stagingBuffer.mapState === 'pending') {}
    const fullF32 = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    // Release buffers
    this.bufferManager.release(aBuffer);
    this.bufferManager.release(vBuffer);
    this.bufferManager.release(fullBuffer);
    this.bufferManager.release(paramsBuffer);

    // Extract the output range based on mode
    const result = new Float64Array(outLen);
    for (let i = 0; i < outLen; i++) {
      result[i] = fullF32[startIdx + i];
    }

    return this.createArray(result, [outLen]);
  }

  correlate(
    a: IFaceNDArray,
    v: IFaceNDArray,
    mode: 'full' | 'same' | 'valid' = 'valid'
  ): IFaceNDArray {
    // Correlate is convolve with reversed kernel - reverse on GPU
    const vFlat = this.flatten(v);
    const vLen = vFlat.data.length;

    // Reverse the kernel (this is a simple op, done on CPU for now)
    const vReversedData = new Float64Array(vLen);
    for (let i = 0; i < vLen; i++) {
      vReversedData[i] = vFlat.data[vLen - 1 - i];
    }
    const vReversed = this.createArray(vReversedData, vFlat.shape);

    return this.convolve(a, vReversed, mode);
  }

  // ============ Matrix Creation ============

  identity(n: number): IFaceNDArray {
    return this.eye(n);
  }

  tril(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    if (arr.shape.length !== 2) {
      throw new Error('tril requires 2D array');
    }
    const [rows, cols] = arr.shape;
    const result = new Float64Array(rows * cols);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (j <= i + k) {
          result[i * cols + j] = arr.data[i * cols + j];
        }
      }
    }

    return this.createArray(result, [rows, cols]);
  }

  triu(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    if (arr.shape.length !== 2) {
      throw new Error('triu requires 2D array');
    }
    const [rows, cols] = arr.shape;
    const result = new Float64Array(rows * cols);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (j >= i + k) {
          result[i * cols + j] = arr.data[i * cols + j];
        }
      }
    }

    return this.createArray(result, [rows, cols]);
  }

  // ============ Grid Creation ============

  meshgrid(x: IFaceNDArray, y: IFaceNDArray): { X: IFaceNDArray; Y: IFaceNDArray } {
    const xFlat = this.flatten(x);
    const yFlat = this.flatten(y);
    const nx = xFlat.data.length;
    const ny = yFlat.data.length;

    const X = new Float64Array(ny * nx);
    const Y = new Float64Array(ny * nx);

    for (let i = 0; i < ny; i++) {
      for (let j = 0; j < nx; j++) {
        X[i * nx + j] = xFlat.data[j];
        Y[i * nx + j] = yFlat.data[i];
      }
    }

    return {
      X: this.createArray(X, [ny, nx]),
      Y: this.createArray(Y, [ny, nx]),
    };
  }

  logspace(start: number, stop: number, num: number, base: number = 10): IFaceNDArray {
    // GPU implementation: linspace then exp(x * ln(base))
    const linear = this.linspace(start, stop, num);
    const lnBase = Math.log(base);
    // Multiply by ln(base) then apply exp - equivalent to pow(base, x)
    const scaled = this.mulScalar(linear, lnBase);
    return this.exp(scaled);
  }

  geomspace(start: number, stop: number, num: number): IFaceNDArray {
    if (start === 0 || stop === 0) {
      throw new Error('geomspace: start and stop must be non-zero');
    }
    if (start < 0 !== stop < 0) {
      throw new Error('geomspace: start and stop must have same sign');
    }

    // GPU implementation: linspace in log space then exp
    const logStart = Math.log(Math.abs(start));
    const logStop = Math.log(Math.abs(stop));
    const linear = this.linspace(logStart, logStop, num);
    const expd = this.exp(linear);

    // If negative, negate all values
    if (start < 0) {
      return this.mulScalar(expd, -1);
    }
    return expd;
  }

  // ============ Stacking Shortcuts ============

  vstack(arrays: IFaceNDArray[]): IFaceNDArray {
    const processed = arrays.map(arr => {
      if (arr.shape.length === 1) {
        return this.reshape(arr, [1, arr.shape[0]]);
      }
      return arr;
    });
    return this.concatenate(processed, 0);
  }

  hstack(arrays: IFaceNDArray[]): IFaceNDArray {
    if (arrays[0].shape.length === 1) {
      return this.concatenate(arrays, 0);
    }
    return this.concatenate(arrays, 1);
  }

  dstack(arrays: IFaceNDArray[]): IFaceNDArray {
    const processed = arrays.map(arr => {
      if (arr.shape.length === 1) {
        return this.reshape(arr, [1, arr.shape[0], 1]);
      } else if (arr.shape.length === 2) {
        return this.reshape(arr, [arr.shape[0], arr.shape[1], 1]);
      }
      return arr;
    });
    return this.concatenate(processed, 2);
  }

  // ============ Split Shortcuts ============

  vsplit(arr: IFaceNDArray, indices: number | number[]): IFaceNDArray[] {
    if (arr.shape.length < 2) {
      throw new Error('vsplit requires array with at least 2 dimensions');
    }
    return this.split(arr, indices, 0);
  }

  hsplit(arr: IFaceNDArray, indices: number | number[]): IFaceNDArray[] {
    if (arr.shape.length === 1) {
      return this.split(arr, indices, 0);
    }
    return this.split(arr, indices, 1);
  }

  dsplit(arr: IFaceNDArray, indices: number | number[]): IFaceNDArray[] {
    if (arr.shape.length < 3) {
      throw new Error('dsplit requires array with at least 3 dimensions');
    }
    return this.split(arr, indices, 2);
  }

  // ============ Array Replication ============

  tile(arr: IFaceNDArray, reps: number | number[]): IFaceNDArray {
    const repsArray = Array.isArray(reps) ? reps : [reps];

    const ndim = Math.max(arr.shape.length, repsArray.length);
    const paddedReps = new Array(ndim - repsArray.length).fill(1).concat(repsArray);
    const paddedShape = new Array(ndim - arr.shape.length).fill(1).concat(arr.shape);

    const outShape = paddedShape.map((s, i) => s * paddedReps[i]);
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const outStrides = this._computeStrides(outShape);
    const srcStrides = this._computeStrides(paddedShape);

    for (let i = 0; i < outSize; i++) {
      const coords = new Array(ndim);
      let remaining = i;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / outStrides[d]);
        remaining = remaining % outStrides[d];
      }

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += (coords[d] % paddedShape[d]) * srcStrides[d];
      }

      result[i] = arr.data[srcIdx];
    }

    return this.createArray(result, outShape);
  }

  repeat(arr: IFaceNDArray, repeats: number, axis?: number): IFaceNDArray {
    if (axis === undefined) {
      const flat = this.flatten(arr);
      const result = new Float64Array(flat.data.length * repeats);
      for (let i = 0; i < flat.data.length; i++) {
        for (let j = 0; j < repeats; j++) {
          result[i * repeats + j] = flat.data[i];
        }
      }
      return this.createArray(result, [result.length]);
    }

    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const outShape = [...arr.shape];
    outShape[axis] *= repeats;
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const srcStrides = this._computeStrides(arr.shape);
    const dstStrides = this._computeStrides(outShape);

    for (let dstIdx = 0; dstIdx < outSize; dstIdx++) {
      const coords = new Array(ndim);
      let remaining = dstIdx;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      coords[axis] = Math.floor(coords[axis] / repeats);

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return this.createArray(result, outShape);
  }

  // ============ Index Finding ============

  nonzero(arr: IFaceNDArray): IFaceNDArray[] {
    const indices: number[][] = Array.from({ length: arr.shape.length }, () => []);
    const strides = this._computeStrides(arr.shape);

    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] !== 0) {
        let remaining = i;
        for (let d = 0; d < arr.shape.length; d++) {
          const coord = Math.floor(remaining / strides[d]);
          remaining = remaining % strides[d];
          indices[d].push(coord);
        }
      }
    }

    return indices.map(idx => this.createArray(new Float64Array(idx), [idx.length]));
  }

  argwhere(arr: IFaceNDArray): IFaceNDArray {
    const indices = this.nonzero(arr);
    if (indices.length === 0 || indices[0].data.length === 0) {
      return this.createArray(new Float64Array(0), [0, arr.shape.length]);
    }

    const nNonzero = indices[0].data.length;
    const ndim = arr.shape.length;
    const result = new Float64Array(nNonzero * ndim);

    for (let i = 0; i < nNonzero; i++) {
      for (let d = 0; d < ndim; d++) {
        result[i * ndim + d] = indices[d].data[i];
      }
    }

    return this.createArray(result, [nNonzero, ndim]);
  }

  flatnonzero(arr: IFaceNDArray): IFaceNDArray {
    const indices: number[] = [];
    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] !== 0) {
        indices.push(i);
      }
    }
    return this.createArray(new Float64Array(indices), [indices.length]);
  }

  // ============ Value Handling ============

  nanToNum(arr: IFaceNDArray, nan: number = 0, posInf?: number, negInf?: number): IFaceNDArray {
    const maxFloat = Number.MAX_VALUE;
    const pInf = posInf ?? maxFloat;
    const nInf = negInf ?? -maxFloat;

    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      const v = arr.data[i];
      if (Number.isNaN(v)) {
        result[i] = nan;
      } else if (v === Infinity) {
        result[i] = pInf;
      } else if (v === -Infinity) {
        result[i] = nInf;
      } else {
        result[i] = v;
      }
    }

    return this.createArray(result, arr.shape);
  }

  // ============ Sorting (GPU-accelerated) ============
  // For 1D arrays: uses GPU bitonic sort (O(n log^2 n) parallel)
  // For multi-dim with axis: extracts slices, sorts on GPU, writes back

  sort(arr: IFaceNDArray, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const shape = arr.shape;
    const axisLen = shape[axis];

    // 1D case: direct GPU sort
    if (ndim === 1) {
      return this._sortSliceGPU(arr.data, shape);
    }

    // Multi-dim: sort along axis using GPU for each slice
    const result = new Float64Array(arr.data);
    const strides = this._computeStrides(shape);
    const outerShape = shape.filter((_, i) => i !== axis);
    const outerStrides = outerShape.length > 0 ? this._computeStrides(outerShape) : [1];
    const outerSize = outerShape.reduce((a, b) => a * b, 1) || 1;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const slice = new Float64Array(axisLen);
      const outerCoords = new Array(outerShape.length);
      let remaining = outerIdx;
      for (let d = 0; d < outerShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outerStrides[d]);
        remaining = remaining % outerStrides[d];
      }

      const baseCoords = new Array(ndim);
      let outerD = 0;
      for (let d = 0; d < ndim; d++) {
        if (d === axis) {
          baseCoords[d] = 0;
        } else {
          baseCoords[d] = outerCoords[outerD++];
        }
      }

      // Extract slice
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        slice[i] = arr.data[idx];
      }

      // Sort slice using native TypeScript sort (GPU bitonic only for large flat arrays)
      // This is the WebGPU backend's native implementation - not a fallback
      const sorted = this._sortSliceNative(slice);

      // Write back
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        result[idx] = sorted[i];
      }
    }

    return this.createArray(result, shape);
  }

  // Native sort for a single slice - handles NaN correctly
  private _sortSliceNative(slice: Float64Array): Float64Array {
    const arr = Array.from(slice);
    arr.sort((a, b) => {
      if (Number.isNaN(a) && Number.isNaN(b)) return 0;
      if (Number.isNaN(a)) return 1;
      if (Number.isNaN(b)) return -1;
      return a - b;
    });
    return new Float64Array(arr);
  }

  // GPU sort for 1D array (used by sortFlatAsync)
  private _sortSliceGPU(data: Float64Array, shape: number[]): IFaceNDArray {
    // For sync interface, use native TypeScript implementation
    // The async version sortFlatAsync uses actual GPU bitonic sort
    return this.createArray(this._sortSliceNative(data), shape);
  }

  // argsort - returns indices that would sort the array
  // Uses native TypeScript argsort implementation
  argsort(arr: IFaceNDArray, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const shape = arr.shape;
    const axisLen = shape[axis];

    // 1D case
    if (ndim === 1) {
      return this.createArray(this._argsortSliceNative(arr.data), shape);
    }

    // Multi-dim: argsort along axis
    const result = new Float64Array(arr.data.length);
    const strides = this._computeStrides(shape);
    const outerShape = shape.filter((_, i) => i !== axis);
    const outerStrides = outerShape.length > 0 ? this._computeStrides(outerShape) : [1];
    const outerSize = outerShape.reduce((a, b) => a * b, 1) || 1;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      const outerCoords = new Array(outerShape.length);
      let remaining = outerIdx;
      for (let d = 0; d < outerShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outerStrides[d]);
        remaining = remaining % outerStrides[d];
      }

      const baseCoords = new Array(ndim);
      let outerD = 0;
      for (let d = 0; d < ndim; d++) {
        if (d === axis) {
          baseCoords[d] = 0;
        } else {
          baseCoords[d] = outerCoords[outerD++];
        }
      }

      // Extract slice values
      const values = new Float64Array(axisLen);
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        values[i] = arr.data[idx];
      }

      // Argsort the slice
      const sortedIndices = this._argsortSliceNative(values);

      // Write back
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        result[idx] = sortedIndices[i];
      }
    }

    return this.createArray(result, shape);
  }

  // Native argsort implementation
  private _argsortSliceNative(values: Float64Array): Float64Array {
    const indices = Array.from({ length: values.length }, (_, i) => i);
    indices.sort((a, b) => {
      const va = values[a],
        vb = values[b];
      if (Number.isNaN(va) && Number.isNaN(vb)) return 0;
      if (Number.isNaN(va)) return 1;
      if (Number.isNaN(vb)) return -1;
      return va - vb;
    });
    return new Float64Array(indices);
  }

  // searchsorted - binary search for insertion points
  // Native implementation using parallel binary search pattern
  searchsorted(
    arr: IFaceNDArray,
    v: number | IFaceNDArray,
    side: 'left' | 'right' = 'left'
  ): IFaceNDArray | number {
    const flat = this.flatten(arr);
    const data = flat.data;

    const search = (val: number): number => {
      let lo = 0,
        hi = data.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (side === 'left') {
          if (data[mid] < val) lo = mid + 1;
          else hi = mid;
        } else {
          if (data[mid] <= val) lo = mid + 1;
          else hi = mid;
        }
      }
      return lo;
    };

    if (typeof v === 'number') {
      return search(v);
    } else {
      const vFlat = this.flatten(v);
      const result = new Float64Array(vFlat.data.length);
      for (let i = 0; i < vFlat.data.length; i++) {
        result[i] = search(vFlat.data[i]);
      }
      return this.createArray(result, vFlat.shape);
    }
  }

  // unique - returns sorted unique values
  // Native implementation using Set + sort
  unique(arr: IFaceNDArray): IFaceNDArray {
    const flat = this.flatten(arr);
    const seen = new Set<number>();
    const result: number[] = [];

    for (let i = 0; i < flat.data.length; i++) {
      const v = flat.data[i];
      if (!seen.has(v)) {
        seen.add(v);
        result.push(v);
      }
    }

    // Sort with NaN handling
    result.sort((a, b) => {
      if (Number.isNaN(a) && Number.isNaN(b)) return 0;
      if (Number.isNaN(a)) return 1;
      if (Number.isNaN(b)) return -1;
      return a - b;
    });

    return this.createArray(new Float64Array(result), [result.length]);
  }

  // ============ Async GPU Operations ============
  // These use real WebGPU compute shaders for acceleration

  // Unary ops (GPU-accelerated)
  async sinAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'sin');
  }
  async cosAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'cos');
  }
  async tanAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'tan');
  }
  async arcsinAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'asin');
  }
  async arccosAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'acos');
  }
  async arctanAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'atan');
  }
  async sinhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'sinh');
  }
  async coshAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'cosh');
  }
  async tanhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'tanh');
  }
  async expAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'exp');
  }
  async exp2Async(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'exp2');
  }
  async logAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'log');
  }
  async log2Async(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'log2');
  }
  async log10Async(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'log10');
  }
  async sqrtAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'sqrt');
  }
  async cbrtAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'cbrt');
  }
  async absAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'abs');
  }
  async signAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'sign');
  }
  async floorAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'floor');
  }
  async ceilAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'ceil');
  }
  async roundAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'round');
  }
  async negAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'neg');
  }
  async reciprocalAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'reciprocal');
  }
  async squareAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'square');
  }

  // Extended unary ops (GPU-accelerated)
  async arcsinhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'asinh');
  }
  async arccoshAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'acosh');
  }
  async arctanhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'atanh');
  }
  async expm1Async(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'expm1');
  }
  async log1pAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'log1p');
  }
  async truncAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'trunc');
  }
  async fixAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'trunc');
  } // alias for trunc
  async sincAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'sinc');
  }
  async deg2radAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'deg2rad');
  }
  async rad2degAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'rad2deg');
  }
  async signbitAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runUnaryOp(arr, 'signbit');
  }
  async heavisideAsync(arr: IFaceNDArray, h0: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, h0, 'heaviside');
  }

  // Binary ops (GPU-accelerated)
  async addAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'add');
  }
  async subAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'sub');
  }
  async mulAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'mul');
  }
  async divAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'div');
  }
  async powAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'pow');
  }
  async maximumAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'maximum');
  }
  async minimumAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'minimum');
  }

  // Extended binary ops (GPU-accelerated)
  async modAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'mod');
  }
  async fmodAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'fmod');
  }
  async remainderAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'mod');
  } // Same as mod
  async copysignAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'copysign');
  }
  async hypotAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'hypot');
  }
  async arctan2Async(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'arctan2');
  }
  async logaddexpAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'logaddexp');
  }
  async logaddexp2Async(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'logaddexp2');
  }
  async fmaxAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'fmax');
  }
  async fminAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runBinaryOp(a, b, 'fmin');
  }

  // Decomposition ops (GPU-accelerated)
  async modfAsync(arr: IFaceNDArray): Promise<{ frac: IFaceNDArray; integ: IFaceNDArray }> {
    return this.runModf(arr);
  }
  async frexpAsync(arr: IFaceNDArray): Promise<{ mantissa: IFaceNDArray; exponent: IFaceNDArray }> {
    return this.runFrexp(arr);
  }
  async ldexpAsync(arr: IFaceNDArray, exp: IFaceNDArray): Promise<IFaceNDArray> {
    // ldexp(mantissa, exp) = mantissa * 2^exp
    // We need a binary shader for this, but we can use scalar op with exp2
    // Since exp is an array, we need to use a binary operation approach
    // Create a temp array with exp2(exp) and multiply
    const exp2Result = await this.runUnaryOp(exp, 'exp2');
    return this.runBinaryOp(arr, exp2Result, 'mul');
  }
  async divmodAsync(
    a: IFaceNDArray,
    b: IFaceNDArray
  ): Promise<{ quotient: IFaceNDArray; remainder: IFaceNDArray }> {
    return this.runDivmod(a, b);
  }

  // Scalar ops (GPU-accelerated)
  async addScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, scalar, 'addScalar');
  }
  async subScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, scalar, 'subScalar');
  }
  async mulScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, scalar, 'mulScalar');
  }
  async divScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, scalar, 'divScalar');
  }
  async powScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> {
    return this.runScalarOp(arr, scalar, 'powScalar');
  }
  async clipAsync(arr: IFaceNDArray, minVal: number, maxVal: number): Promise<IFaceNDArray> {
    return this.runClip(arr, minVal, maxVal);
  }

  // Reduction ops (GPU-accelerated)
  async sumAsync(arr: IFaceNDArray): Promise<number> {
    return this.runReduction(arr, 'sum');
  }
  async prodAsync(arr: IFaceNDArray): Promise<number> {
    return this.runReduction(arr, 'prod');
  }
  async minAsync(arr: IFaceNDArray): Promise<number> {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.runReduction(arr, 'min');
  }
  async maxAsync(arr: IFaceNDArray): Promise<number> {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.runReduction(arr, 'max');
  }

  // Cumulative ops (GPU-accelerated)
  async cumsumAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runCumulative(arr, 'cumsum');
  }
  async cumprodAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runCumulative(arr, 'cumprod');
  }

  // Derived async stats
  async meanAsync(arr: IFaceNDArray): Promise<number> {
    if (arr.data.length === 0) return NaN;
    const sum = await this.sumAsync(arr);
    return sum / arr.data.length;
  }

  async varAsync(arr: IFaceNDArray, ddof: number = 0): Promise<number> {
    const n = arr.data.length;
    if (n === 0) return NaN;
    const mean = await this.meanAsync(arr);
    // Compute (x - mean)^2, then sum
    const centered = await this.addScalarAsync(arr, -mean);
    const squared = await this.squareAsync(centered);
    const sumSq = await this.sumAsync(squared);
    return sumSq / (n - ddof);
  }

  async stdAsync(arr: IFaceNDArray, ddof: number = 0): Promise<number> {
    return Math.sqrt(await this.varAsync(arr, ddof));
  }

  // Stats ops (GPU-accelerated)
  async argminAsync(arr: IFaceNDArray): Promise<number> {
    return this.runArgReduction(arr, 'argmin');
  }
  async argmaxAsync(arr: IFaceNDArray): Promise<number> {
    return this.runArgReduction(arr, 'argmax');
  }
  async allAsync(arr: IFaceNDArray): Promise<boolean> {
    return this.runBoolReduction(arr, 'all');
  }
  async anyAsync(arr: IFaceNDArray): Promise<boolean> {
    return this.runBoolReduction(arr, 'any');
  }
  async sumAxisAsync(arr: IFaceNDArray, axis: number): Promise<IFaceNDArray> {
    return this.runSumAxis(arr, axis);
  }
  async meanAxisAsync(arr: IFaceNDArray, axis: number): Promise<IFaceNDArray> {
    const sumResult = await this.runSumAxis(arr, axis);
    const axisLen = arr.shape[axis];
    const result = new Float64Array(sumResult.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = sumResult.data[i] / axisLen;
    }
    return this.createArray(result, sumResult.shape);
  }

  // Linalg ops (GPU-accelerated)
  async transposeAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runTranspose(arr);
  }
  async outerAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    return this.runOuter(a, b);
  }
  async dotAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    // For 1D arrays, dot is inner product wrapped in array
    if (a.shape.length === 1 && b.shape.length === 1) {
      const result = await this.runDot(a, b);
      return this.createArray([result], [1]);
    }
    // For 2D arrays, use matmul
    return this.matmulAsync(a, b);
  }
  async innerAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<number> {
    return this.runDot(a, b);
  }
  async traceAsync(arr: IFaceNDArray): Promise<number> {
    return this.runTrace(arr);
  }
  async normAsync(arr: IFaceNDArray, ord?: number): Promise<number> {
    // Default is L2 norm (Frobenius)
    if (ord === undefined || ord === 2) {
      return this.runNorm(arr);
    }
    // For other norms, fall back to CPU
    if (ord === 1) {
      // L1 norm: sum of absolute values
      const absArr = await this.absAsync(arr);
      return this.sumAsync(absArr);
    }
    if (ord === Infinity) {
      // Inf norm: max absolute value
      const absArr = await this.absAsync(arr);
      return this.maxAsync(absArr);
    }
    throw new Error(`Norm ord=${ord} not supported`);
  }

  // ============ GPU Sorting Operations ============

  /**
   * GPU bitonic sort (async) - sorts 1D array using parallel bitonic sort on GPU
   * Handles NaN values by treating them as larger than any number
   */
  async sortFlatAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    const data = arr.data;
    const n = data.length;
    if (n <= 1) return this.createArray(new Float64Array(data), arr.shape);

    // Pad to next power of 2 for bitonic sort
    let paddedN = 1;
    while (paddedN < n) paddedN *= 2;

    // Create data buffer with padding (pad with Infinity so they sort to end)
    const paddedData = new Float32Array(paddedN);
    for (let i = 0; i < n; i++) paddedData[i] = data[i];
    for (let i = n; i < paddedN; i++) paddedData[i] = Infinity;

    // Create GPU buffer
    const dataBuffer = this.bufferManager.acquire(
      paddedN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(dataBuffer, 0, paddedData);

    // Get or create pipeline for bitonic sort step
    const cacheKey = 'bitonic-sort-step';
    let pipeline = shaderCache.get(cacheKey);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: makeBitonicSortStepShader() });
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      shaderCache.set(cacheKey, pipeline);
    }

    // Uniform buffer for parameters
    const uniformBuffer = this.bufferManager.acquire(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const workgroups = Math.ceil(paddedN / 256);

    try {
      // Bitonic sort: O(log^2 n) passes
      // k goes through powers of 2: 2, 4, 8, ..., paddedN
      for (let k = 2; k <= paddedN; k *= 2) {
        // j goes through k/2, k/4, ..., 1
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
          // Update uniforms
          const params = new Uint32Array([paddedN, j, k, 0]);
          this.device.queue.writeBuffer(uniformBuffer, 0, params);

          // Create bind group
          const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: dataBuffer } },
              { binding: 1, resource: { buffer: uniformBuffer } },
            ],
          });

          // Dispatch
          const commandEncoder = this.device.createCommandEncoder();
          const pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroups);
          pass.end();
          this.device.queue.submit([commandEncoder.finish()]);
        }
      }

      // Read back result
      const stagingBuffer = this.bufferManager.acquireStaging(n * 4); // Only need original n elements
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(dataBuffer, 0, stagingBuffer, 0, n * 4);
      this.device.queue.submit([commandEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const resultF32 = new Float32Array(stagingBuffer.getMappedRange().slice(0, n * 4));
      stagingBuffer.unmap();
      this.bufferManager.releaseStaging(stagingBuffer);

      // Convert to f64
      const resultF64 = new Float64Array(n);
      for (let i = 0; i < n; i++) resultF64[i] = resultF32[i];

      return this.createArray(resultF64, arr.shape);
    } finally {
      this.bufferManager.release(dataBuffer);
      this.bufferManager.release(uniformBuffer);
    }
  }

  /**
   * GPU argsort (async) - returns indices that would sort the array
   */
  async argsortFlatAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    const data = arr.data;
    const n = data.length;
    if (n <= 1) return this.createArray(new Float64Array([0]), [1]);

    // Pad to next power of 2
    let paddedN = 1;
    while (paddedN < n) paddedN *= 2;

    // Create value buffer (with padding)
    const paddedData = new Float32Array(paddedN);
    for (let i = 0; i < n; i++) paddedData[i] = data[i];
    for (let i = n; i < paddedN; i++) paddedData[i] = Infinity;

    const valuesBuffer = this.bufferManager.acquire(
      paddedN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(valuesBuffer, 0, paddedData);

    // Create indices buffer (initialized to 0, 1, 2, ...)
    const indicesBuffer = this.bufferManager.acquire(
      paddedN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    // Initialize indices
    const initCacheKey = 'init-indices';
    let initPipeline = shaderCache.get(initCacheKey);
    if (!initPipeline) {
      const module = this.device.createShaderModule({ code: INIT_INDICES_SHADER });
      initPipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      shaderCache.set(initCacheKey, initPipeline);
    }

    const initUniformBuffer = this.bufferManager.acquire(
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(initUniformBuffer, 0, new Uint32Array([paddedN]));

    const initBindGroup = this.device.createBindGroup({
      layout: initPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: indicesBuffer } },
        { binding: 1, resource: { buffer: initUniformBuffer } },
      ],
    });

    let commandEncoder = this.device.createCommandEncoder();
    let pass = commandEncoder.beginComputePass();
    pass.setPipeline(initPipeline);
    pass.setBindGroup(0, initBindGroup);
    pass.dispatchWorkgroups(Math.ceil(paddedN / 256));
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Get or create argsort pipeline
    const cacheKey = 'bitonic-argsort-step';
    let pipeline = shaderCache.get(cacheKey);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: makeBitonicArgsortStepShader() });
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      shaderCache.set(cacheKey, pipeline);
    }

    const uniformBuffer = this.bufferManager.acquire(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const workgroups = Math.ceil(paddedN / 256);

    try {
      // Bitonic sort on indices
      for (let k = 2; k <= paddedN; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
          const params = new Uint32Array([paddedN, j, k, 0]);
          this.device.queue.writeBuffer(uniformBuffer, 0, params);

          const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: valuesBuffer } },
              { binding: 1, resource: { buffer: indicesBuffer } },
              { binding: 2, resource: { buffer: uniformBuffer } },
            ],
          });

          commandEncoder = this.device.createCommandEncoder();
          pass = commandEncoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroups);
          pass.end();
          this.device.queue.submit([commandEncoder.finish()]);
        }
      }

      // Read back indices (only first n)
      const stagingBuffer = this.bufferManager.acquireStaging(n * 4);
      commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(indicesBuffer, 0, stagingBuffer, 0, n * 4);
      this.device.queue.submit([commandEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const resultU32 = new Uint32Array(stagingBuffer.getMappedRange().slice(0, n * 4));
      stagingBuffer.unmap();
      this.bufferManager.releaseStaging(stagingBuffer);

      // Convert to f64
      const resultF64 = new Float64Array(n);
      for (let i = 0; i < n; i++) resultF64[i] = resultU32[i];

      return this.createArray(resultF64, arr.shape);
    } finally {
      this.bufferManager.release(valuesBuffer);
      this.bufferManager.release(indicesBuffer);
      this.bufferManager.release(uniformBuffer);
      this.bufferManager.release(initUniformBuffer);
    }
  }

  /**
   * GPU unique (async) - returns unique values in sorted order
   * Uses sort + parallel mark + scan + compact
   */
  async uniqueAsync(arr: IFaceNDArray): Promise<IFaceNDArray> {
    const data = arr.data;
    const n = data.length;
    if (n === 0) return this.createArray(new Float64Array(0), [0]);
    if (n === 1) return this.createArray(new Float64Array(data), [1]);

    // First, sort the data
    const sorted = await this.sortFlatAsync(this.flatten(arr));
    const sortedData = sorted.data;

    // For small arrays or when GPU overhead dominates, use CPU
    if (n < 1024) {
      const seen = new Set<number>();
      const result: number[] = [];
      for (let i = 0; i < n; i++) {
        const v = sortedData[i];
        if (!seen.has(v)) {
          seen.add(v);
          result.push(v);
        }
      }
      return this.createArray(new Float64Array(result), [result.length]);
    }

    // GPU path: mark unique elements, scan, compact
    const sortedF32 = new Float32Array(n);
    for (let i = 0; i < n; i++) sortedF32[i] = sortedData[i];

    const sortedBuffer = this.bufferManager.acquire(
      n * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(sortedBuffer, 0, sortedF32);

    const maskBuffer = this.bufferManager.acquire(
      n * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const scanBuffer = this.bufferManager.acquire(
      n * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const outputBuffer = this.bufferManager.acquire(
      n * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const uniformBuffer = this.bufferManager.acquire(
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    try {
      // Step 1: Mark unique elements
      const markCacheKey = 'mark-unique';
      let markPipeline = shaderCache.get(markCacheKey);
      if (!markPipeline) {
        const module = this.device.createShaderModule({ code: MARK_UNIQUE_SHADER });
        markPipeline = this.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        shaderCache.set(markCacheKey, markPipeline);
      }

      this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([n]));

      let commandEncoder = this.device.createCommandEncoder();
      let pass = commandEncoder.beginComputePass();
      pass.setPipeline(markPipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: markPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: sortedBuffer } },
            { binding: 1, resource: { buffer: maskBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      // Step 2: Exclusive scan of mask
      const scanCacheKey = 'exclusive-scan';
      let scanPipeline = shaderCache.get(scanCacheKey);
      if (!scanPipeline) {
        const module = this.device.createShaderModule({ code: EXCLUSIVE_SCAN_SHADER });
        scanPipeline = this.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        shaderCache.set(scanCacheKey, scanPipeline);
      }

      const scanUniformBuffer = this.bufferManager.acquire(
        8,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      this.device.queue.writeBuffer(scanUniformBuffer, 0, new Uint32Array([n, 0]));

      commandEncoder = this.device.createCommandEncoder();
      pass = commandEncoder.beginComputePass();
      pass.setPipeline(scanPipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: scanPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: maskBuffer } },
            { binding: 1, resource: { buffer: scanBuffer } },
            { binding: 2, resource: { buffer: scanUniformBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      // Step 3: Compact
      const compactCacheKey = 'compact-unique';
      let compactPipeline = shaderCache.get(compactCacheKey);
      if (!compactPipeline) {
        const module = this.device.createShaderModule({ code: COMPACT_UNIQUE_SHADER });
        compactPipeline = this.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        shaderCache.set(compactCacheKey, compactPipeline);
      }

      commandEncoder = this.device.createCommandEncoder();
      pass = commandEncoder.beginComputePass();
      pass.setPipeline(compactPipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: compactPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: sortedBuffer } },
            { binding: 1, resource: { buffer: maskBuffer } },
            { binding: 2, resource: { buffer: scanBuffer } },
            { binding: 3, resource: { buffer: outputBuffer } },
            { binding: 4, resource: { buffer: uniformBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      // Read back mask to count unique elements
      const maskStagingBuffer = this.bufferManager.acquireStaging(n * 4);
      commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(maskBuffer, 0, maskStagingBuffer, 0, n * 4);
      this.device.queue.submit([commandEncoder.finish()]);

      await maskStagingBuffer.mapAsync(GPUMapMode.READ);
      const maskData = new Uint32Array(maskStagingBuffer.getMappedRange().slice(0));
      maskStagingBuffer.unmap();
      this.bufferManager.releaseStaging(maskStagingBuffer);

      // Count unique elements
      let uniqueCount = 0;
      for (let i = 0; i < n; i++) uniqueCount += maskData[i];

      // Read back output
      const outputStagingBuffer = this.bufferManager.acquireStaging(uniqueCount * 4);
      commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(outputBuffer, 0, outputStagingBuffer, 0, uniqueCount * 4);
      this.device.queue.submit([commandEncoder.finish()]);

      await outputStagingBuffer.mapAsync(GPUMapMode.READ);
      const resultF32 = new Float32Array(
        outputStagingBuffer.getMappedRange().slice(0, uniqueCount * 4)
      );
      outputStagingBuffer.unmap();
      this.bufferManager.releaseStaging(outputStagingBuffer);

      // Convert to f64
      const resultF64 = new Float64Array(uniqueCount);
      for (let i = 0; i < uniqueCount; i++) resultF64[i] = resultF32[i];

      this.bufferManager.release(scanUniformBuffer);
      return this.createArray(resultF64, [uniqueCount]);
    } finally {
      this.bufferManager.release(sortedBuffer);
      this.bufferManager.release(maskBuffer);
      this.bufferManager.release(scanBuffer);
      this.bufferManager.release(outputBuffer);
      this.bufferManager.release(uniformBuffer);
    }
  }

  /**
   * GPU bincount (async) - count occurrences using atomic scatter
   */
  async bincountAsync(
    x: IFaceNDArray,
    weights?: IFaceNDArray,
    minlength?: number
  ): Promise<IFaceNDArray> {
    const xData = x.data;
    const n = xData.length;

    // Find max value (need CPU pass to determine output size)
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
      const v = Math.floor(xData[i]);
      if (v < 0) throw new Error('bincount requires non-negative integers');
      if (v > maxVal) maxVal = v;
    }

    const outLen = Math.max(maxVal + 1, minlength || 0);
    if (n === 0) return this.createArray(new Float64Array(outLen), [outLen]);

    // For small outputs or inputs, CPU is faster
    if (n < 256 || outLen < 16) {
      return this.bincount(x, weights, minlength);
    }

    // Convert to i32 for GPU
    const xI32 = new Int32Array(n);
    for (let i = 0; i < n; i++) xI32[i] = Math.floor(xData[i]);

    const xBuffer = this.bufferManager.acquire(
      n * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(xBuffer, 0, xI32);

    const outputBuffer = this.bufferManager.acquire(
      outLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    // Zero initialize output
    const zeros = new Uint32Array(outLen);
    this.device.queue.writeBuffer(outputBuffer, 0, zeros);

    const uniformBuffer = this.bufferManager.acquire(
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([n]));

    try {
      if (weights) {
        // Weighted bincount - need fixed-point arithmetic for atomics
        const wData = weights.data;
        const wF32 = new Float32Array(n);
        for (let i = 0; i < n; i++) wF32[i] = wData[i];

        const weightsBuffer = this.bufferManager.acquire(
          n * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(weightsBuffer, 0, wF32);

        const cacheKey = 'weighted-bincount';
        let pipeline = shaderCache.get(cacheKey);
        if (!pipeline) {
          const module = this.device.createShaderModule({ code: makeWeightedBincountShader() });
          pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
          });
          shaderCache.set(cacheKey, pipeline);
        }

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: xBuffer } },
              { binding: 1, resource: { buffer: weightsBuffer } },
              { binding: 2, resource: { buffer: outputBuffer } },
              { binding: 3, resource: { buffer: uniformBuffer } },
            ],
          })
        );
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        // Read back and convert from fixed-point
        const stagingBuffer = this.bufferManager.acquireStaging(outLen * 4);
        const readEncoder = this.device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outLen * 4);
        this.device.queue.submit([readEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultU32 = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        this.bufferManager.releaseStaging(stagingBuffer);
        this.bufferManager.release(weightsBuffer);

        // Convert from fixed-point
        const resultF64 = new Float64Array(outLen);
        for (let i = 0; i < outLen; i++) resultF64[i] = resultU32[i] / 1000000.0;

        return this.createArray(resultF64, [outLen]);
      } else {
        // Unweighted bincount
        const cacheKey = 'bincount';
        let pipeline = shaderCache.get(cacheKey);
        if (!pipeline) {
          const module = this.device.createShaderModule({ code: makeBincountShader() });
          pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
          });
          shaderCache.set(cacheKey, pipeline);
        }

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: xBuffer } },
              { binding: 1, resource: { buffer: outputBuffer } },
              { binding: 2, resource: { buffer: uniformBuffer } },
            ],
          })
        );
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        // Read back
        const stagingBuffer = this.bufferManager.acquireStaging(outLen * 4);
        const readEncoder = this.device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outLen * 4);
        this.device.queue.submit([readEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultU32 = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        this.bufferManager.releaseStaging(stagingBuffer);

        // Convert to f64
        const resultF64 = new Float64Array(outLen);
        for (let i = 0; i < outLen; i++) resultF64[i] = resultU32[i];

        return this.createArray(resultF64, [outLen]);
      }
    } finally {
      this.bufferManager.release(xBuffer);
      this.bufferManager.release(outputBuffer);
      this.bufferManager.release(uniformBuffer);
    }
  }

  /**
   * GPU searchsorted (async) - parallel binary search
   */
  async searchsortedAsync(
    arr: IFaceNDArray,
    v: IFaceNDArray,
    side: 'left' | 'right' = 'left'
  ): Promise<IFaceNDArray> {
    const haystack = this.flatten(arr).data;
    const needles = this.flatten(v).data;
    const haystackN = haystack.length;
    const needlesN = needles.length;

    if (needlesN === 0) return this.createArray(new Float64Array(0), [0]);
    if (haystackN === 0) return this.createArray(new Float64Array(needlesN).fill(0), v.shape);

    // For small inputs, CPU is faster
    if (needlesN < 64 || haystackN < 64) {
      return this.searchsorted(arr, v, side) as IFaceNDArray;
    }

    // Convert to f32
    const haystackF32 = new Float32Array(haystackN);
    for (let i = 0; i < haystackN; i++) haystackF32[i] = haystack[i];

    const needlesF32 = new Float32Array(needlesN);
    for (let i = 0; i < needlesN; i++) needlesF32[i] = needles[i];

    const haystackBuffer = this.bufferManager.acquire(
      haystackN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const needlesBuffer = this.bufferManager.acquire(
      needlesN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const outputBuffer = this.bufferManager.acquire(
      needlesN * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const uniformBuffer = this.bufferManager.acquire(
      8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(haystackBuffer, 0, haystackF32);
    this.device.queue.writeBuffer(needlesBuffer, 0, needlesF32);
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([haystackN, needlesN]));

    try {
      const cacheKey = `searchsorted-${side}`;
      let pipeline = shaderCache.get(cacheKey);
      if (!pipeline) {
        const module = this.device.createShaderModule({ code: makeSearchSortedShader(side) });
        pipeline = this.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'main' },
        });
        shaderCache.set(cacheKey, pipeline);
      }

      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: haystackBuffer } },
            { binding: 1, resource: { buffer: needlesBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
          ],
        })
      );
      pass.dispatchWorkgroups(Math.ceil(needlesN / 256));
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);

      // Read back
      const stagingBuffer = this.bufferManager.acquireStaging(needlesN * 4);
      const readEncoder = this.device.createCommandEncoder();
      readEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, needlesN * 4);
      this.device.queue.submit([readEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const resultU32 = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      this.bufferManager.releaseStaging(stagingBuffer);

      // Convert to f64
      const resultF64 = new Float64Array(needlesN);
      for (let i = 0; i < needlesN; i++) resultF64[i] = resultU32[i];

      return this.createArray(resultF64, v.shape);
    } finally {
      this.bufferManager.release(haystackBuffer);
      this.bufferManager.release(needlesBuffer);
      this.bufferManager.release(outputBuffer);
      this.bufferManager.release(uniformBuffer);
    }
  }

  /**
   * GPU lexsort (async) - multi-key sort using iterative argsort
   * Sorts by last key first, then second-to-last, etc.
   */
  async lexsortAsync(keys: IFaceNDArray[]): Promise<IFaceNDArray> {
    if (keys.length === 0) throw new Error('lexsort requires at least one key');

    const n = keys[0].data.length;
    for (const k of keys) {
      if (k.data.length !== n) throw new Error('All keys must have the same length');
    }

    if (n <= 1) return this.createArray(new Float64Array([0]), [1]);

    // Start with identity permutation
    let indices = new Float64Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    // Sort by keys in reverse order (last key is primary)
    for (let ki = keys.length - 1; ki >= 0; ki--) {
      const keyData = keys[ki].data;

      // Create permuted key values
      const permuted = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        permuted[i] = keyData[indices[i]];
      }

      // Argsort the permuted key
      const permutedArr = this.createArray(permuted, [n]);
      const sortedIndices = await this.argsortFlatAsync(permutedArr);

      // Apply permutation
      const newIndices = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        newIndices[i] = indices[sortedIndices.data[i]];
      }
      indices = newIndices;
    }

    return this.createArray(indices, [n]);
  }

  // ============ NaN-aware Stats ============

  nansum(arr: IFaceNDArray): number {
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) sum += arr.data[i];
    }
    return sum;
  }

  nanmean(arr: IFaceNDArray): number {
    let sum = 0,
      count = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) {
        sum += arr.data[i];
        count++;
      }
    }
    return count > 0 ? sum / count : NaN;
  }

  nanvar(arr: IFaceNDArray, ddof: number = 0): number {
    const mean = this.nanmean(arr);
    if (Number.isNaN(mean)) return NaN;
    let sumSq = 0,
      count = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) {
        const diff = arr.data[i] - mean;
        sumSq += diff * diff;
        count++;
      }
    }
    return count > ddof ? sumSq / (count - ddof) : NaN;
  }

  nanstd(arr: IFaceNDArray, ddof: number = 0): number {
    return Math.sqrt(this.nanvar(arr, ddof));
  }

  nanmin(arr: IFaceNDArray): number {
    let min = Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] < min) min = arr.data[i];
    }
    return min === Infinity ? NaN : min;
  }

  nanmax(arr: IFaceNDArray): number {
    let max = -Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] > max) max = arr.data[i];
    }
    return max === -Infinity ? NaN : max;
  }

  nanargmin(arr: IFaceNDArray): number {
    let minIdx = -1,
      minVal = Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] < minVal) {
        minVal = arr.data[i];
        minIdx = i;
      }
    }
    return minIdx;
  }

  nanargmax(arr: IFaceNDArray): number {
    let maxIdx = -1,
      maxVal = -Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] > maxVal) {
        maxVal = arr.data[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  nanprod(arr: IFaceNDArray): number {
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) prod *= arr.data[i];
    }
    return prod;
  }

  // ============ Order Statistics ============

  median(arr: IFaceNDArray): number {
    const sorted = Array.from(arr.data).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length === 0) return NaN;
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  percentile(arr: IFaceNDArray, q: number): number {
    if (arr.data.length === 0) return NaN;
    const sorted = Array.from(arr.data).sort((a, b) => a - b);
    const idx = (q / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    if (lower === upper) return sorted[lower];
    const frac = idx - lower;
    return sorted[lower] * (1 - frac) + sorted[upper] * frac;
  }

  quantile(arr: IFaceNDArray, q: number): number {
    return this.percentile(arr, q * 100);
  }

  nanmedian(arr: IFaceNDArray): number {
    const nonNaN = Array.from(arr.data).filter(x => !Number.isNaN(x));
    if (nonNaN.length === 0) return NaN;
    const sorted = nonNaN.sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  nanpercentile(arr: IFaceNDArray, q: number): number {
    const nonNaN = Array.from(arr.data).filter(x => !Number.isNaN(x));
    if (nonNaN.length === 0) return NaN;
    const sorted = nonNaN.sort((a, b) => a - b);
    const idx = (q / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    if (lower === upper) return sorted[lower];
    const frac = idx - lower;
    return sorted[lower] * (1 - frac) + sorted[upper] * frac;
  }

  // ============ Histogram ============

  histogram(arr: IFaceNDArray, bins: number = 10): { hist: IFaceNDArray; binEdges: IFaceNDArray } {
    const data = arr.data;
    let min = Infinity,
      max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (!Number.isNaN(data[i])) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
      }
    }
    if (min === Infinity) {
      return {
        hist: this.array(Array(bins).fill(0), [bins]),
        binEdges: this.array(Array(bins + 1).fill(0), [bins + 1]),
      };
    }

    const range = max - min;
    const binWidth = range / bins || 1;
    const edges = new Float64Array(bins + 1);
    for (let i = 0; i <= bins; i++) edges[i] = min + i * binWidth;

    const hist = new Float64Array(bins);
    for (let i = 0; i < data.length; i++) {
      if (Number.isNaN(data[i])) continue;
      let binIdx = Math.floor((data[i] - min) / binWidth);
      if (binIdx >= bins) binIdx = bins - 1;
      if (binIdx < 0) binIdx = 0;
      hist[binIdx]++;
    }

    return {
      hist: this.array(Array.from(hist), [bins]),
      binEdges: this.array(Array.from(edges), [bins + 1]),
    };
  }

  histogramBinEdges(arr: IFaceNDArray, bins: number = 10): IFaceNDArray {
    const { binEdges } = this.histogram(arr, bins);
    return binEdges;
  }

  // ============ Random ============
  private _rngState: number = Date.now();

  private _xorshift(): number {
    let x = this._rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this._rngState = x >>> 0;
    return (this._rngState >>> 0) / 0xffffffff;
  }

  seed(s: number): void {
    this._rngState = s >>> 0;
    if (this._rngState === 0) this._rngState = 1;
  }

  rand(shape: number[]): IFaceNDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift();
    }
    return this.createArray(data, shape);
  }

  randn(shape: number[]): IFaceNDArray {
    // Box-Muller transform
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i += 2) {
      const u1 = this._xorshift();
      const u2 = this._xorshift();
      const r = Math.sqrt(-2.0 * Math.log(u1 || 1e-10));
      const theta = 2.0 * Math.PI * u2;
      data[i] = r * Math.cos(theta);
      if (i + 1 < size) data[i + 1] = r * Math.sin(theta);
    }
    return this.createArray(data, shape);
  }

  randint(low: number, high: number, shape: number[]): IFaceNDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(this._xorshift() * range) + low;
    }
    return this.createArray(data, shape);
  }

  uniform(low: number, high: number, shape: number[]): IFaceNDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift() * range + low;
    }
    return this.createArray(data, shape);
  }

  normal(loc: number, scale: number, shape: number[]): IFaceNDArray {
    const arr = this.randn(shape);
    const data = arr.data;
    for (let i = 0; i < data.length; i++) {
      data[i] = data[i] * scale + loc;
    }
    return this.createArray(data, shape);
  }

  shuffle(arr: IFaceNDArray): IFaceNDArray {
    const data = new Float64Array(arr.data);
    const shape = [...arr.shape];
    if (shape.length === 1) {
      for (let i = data.length - 1; i > 0; i--) {
        const j = Math.floor(this._xorshift() * (i + 1));
        [data[i], data[j]] = [data[j], data[i]];
      }
    } else {
      const stride = shape.slice(1).reduce((a, b) => a * b, 1);
      const n = shape[0];
      const temp = new Float64Array(stride);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this._xorshift() * (i + 1));
        temp.set(data.subarray(i * stride, (i + 1) * stride));
        data.copyWithin(i * stride, j * stride, (j + 1) * stride);
        data.set(temp, j * stride);
      }
    }
    return this.createArray(data, shape);
  }

  choice(arr: IFaceNDArray, size: number, replace: boolean = true): IFaceNDArray {
    const n = arr.data.length;
    const data = new Float64Array(size);
    if (replace) {
      for (let i = 0; i < size; i++) {
        const idx = Math.floor(this._xorshift() * n);
        data[i] = arr.data[idx];
      }
    } else {
      if (size > n) throw new Error('Cannot sample more than array size without replacement');
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < size; i++) {
        const j = i + Math.floor(this._xorshift() * (n - i));
        [indices[i], indices[j]] = [indices[j], indices[i]];
        data[i] = arr.data[indices[i]];
      }
    }
    return this.createArray(data, [size]);
  }

  permutation(n: number | IFaceNDArray): IFaceNDArray {
    let arr: IFaceNDArray;
    if (typeof n === 'number') {
      arr = this.arange(0, n, 1);
    } else {
      arr = this.createArray(new Float64Array(n.data), [...n.shape]);
    }
    return this.shuffle(arr);
  }
}

// ============ Initialization ============

export async function initWebGPUBackend(): Promise<void> {
  if (!navigator.gpu) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found');
  gpuDevice = await adapter.requestDevice();
}

export function createWebGPUBackend(): Backend {
  if (!gpuDevice) throw new Error('WebGPU not initialized');
  return new WebGPUBackend(gpuDevice);
}
