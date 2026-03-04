/**
 * WebGPU Backend for rumpy-ts tests
 *
 * This backend uses WebGPU compute shaders for GPU-accelerated operations.
 *
 * Architecture:
 * - Sync operations (Backend interface) use CPU for test compatibility
 * - matmulAsync uses real GPU compute shaders for benchmarking
 * - GPU path: f64 -> f32 (upload) -> GPU compute -> f32 -> f64 (download)
 *
 * The sync/async split is necessary because:
 * - WebGPU buffer readback is inherently async
 * - JavaScript can't block on promises
 * - Test interface requires sync methods
 */

import { Backend, NDArray as IFaceNDArray } from './test-utils';

// ============ GPU Device & Shader Cache ============

let gpuDevice: GPUDevice | null = null;
const shaderCache = new Map<string, GPUComputePipeline>();

// ============ NDArray Implementation ============

class WebGPUNDArray implements IFaceNDArray {
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
  // Note: WGSL round() uses banker's rounding (round half to even)
  // To match JS Math.round (round half up), we use floor(x + 0.5)
  round: makeUnaryShader('floor(x + 0.5)'),
  neg: makeUnaryShader('-x'),
  reciprocal: makeUnaryShader('1.0 / x'),
  square: makeUnaryShader('x * x'),
  // cbrt and log10 need special handling (not in WGSL)
  cbrt: makeUnaryShader('sign(x) * pow(abs(x), 0.333333333333)'),
  log10: makeUnaryShader('log(x) / 2.302585093'),  // log10(e) = 1/ln(10)
};

// Binary shader definitions
const BINARY_SHADERS: Record<string, string> = {
  add: makeBinaryShader('av + bv'),
  sub: makeBinaryShader('av - bv'),
  mul: makeBinaryShader('av * bv'),
  div: makeBinaryShader('av / bv'),
  pow: makeBinaryShader('pow(av, bv)'),
  maximum: makeBinaryShader('max(av, bv)'),
  minimum: makeBinaryShader('min(av, bv)'),
};

// Scalar shader definitions
const SCALAR_SHADERS: Record<string, string> = {
  addScalar: makeScalarShader('x + s'),
  subScalar: makeScalarShader('x - s'),
  mulScalar: makeScalarShader('x * s'),
  divScalar: makeScalarShader('x / s'),
  powScalar: makeScalarShader('pow(x, s)'),
};

// Reduction shader definitions
const REDUCTION_SHADERS: Record<string, string> = {
  sum: makeReductionShader('0.0f', '$a + $b'),
  prod: makeReductionShader('1.0f', '$a * $b'),
  min: makeReductionShader('3.40282e+38f', 'min($a, $b)'),  // f32 max
  max: makeReductionShader('-3.40282e+38f', 'max($a, $b)'), // f32 min
};

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
const MATMUL_TILE_M_LARGE = 32;  // 8 threads * 4 elements
const MATMUL_TILE_N_LARGE = 32;  // 8 threads * 4 elements

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
  requiresFit: boolean;  // If true, only use when dims are exactly divisible by tile sizes
  usesVec4B: boolean;    // If true, B matrix needs vec4 padding
  usesVec4C: boolean;    // If true, C matrix output is vec4
  minSize: number;       // Minimum matrix dimension to consider this config
  maxSize: number;       // Maximum matrix dimension (-1 = no limit)
}

// Available shader configurations for autotuning
const SHADER_CONFIGS: ShaderConfig[] = [
  {
    name: 'FIT-64x64',
    shader: MATMUL_FIT_SHADER,
    tileM: 64, tileN: 64, tileK: 32,
    workgroupSize: [16, 16],
    requiresFit: true,
    usesVec4B: true,
    usesVec4C: true,
    minSize: 256,
    maxSize: -1,  // Best for large matrices
  },
  {
    name: 'FIT-32x32',
    shader: MATMUL_TFJS_FIT_32_SHADER,
    tileM: 32, tileN: 32, tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: true,
    usesVec4B: true,
    usesVec4C: false,
    minSize: 128,
    maxSize: 2048,  // Better for medium matrices
  },
  {
    name: 'TFJS-VEC4-INNER',
    shader: MATMUL_TFJS_VEC4_INNER_SHADER,
    tileM: 32, tileN: 32, tileK: 32,
    workgroupSize: [8, 8],
    requiresFit: false,  // Has bounds checking
    usesVec4B: true,
    usesVec4C: false,
    minSize: 64,
    maxSize: -1,
  },
  {
    name: 'REGISTER-BLOCKED',
    shader: MATMUL_REGISTER_BLOCKED_SHADER,
    tileM: 64, tileN: 64, tileK: 16,
    workgroupSize: [16, 16],
    requiresFit: false,
    usesVec4B: false,
    usesVec4C: false,
    minSize: 32,
    maxSize: 512,  // Better for small/medium
  },
];

// Cache for autotuned best configs: key = "MxKxN" -> config name
const autotuneCache = new Map<string, string>();

// Pre-seeded optimal configs based on benchmarking (M4 Pro)
// These are used without running autotuning when dims match
const PRESET_CONFIGS: Record<string, string> = {
  // Large matrices: FIT-64x64 is optimal
  '4096x4096x4096': 'FIT-64x64',
  '2048x2048x2048': 'FIT-64x64',
  // Medium matrices: 32x32 tiles work better
  '1024x1024x1024': 'FIT-32x32',
  // Small matrices: register blocked
  '512x512x512': 'REGISTER-BLOCKED',
  '256x256x256': 'REGISTER-BLOCKED',
  '128x128x128': 'REGISTER-BLOCKED',
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

  // Prefer FIT shaders when dimensions align (they're fastest)
  const fitConfig = validConfigs.find(c => c.requiresFit);
  if (fitConfig) return fitConfig;

  // Otherwise pick based on size
  if (maxDim >= 2048) {
    return validConfigs.find(c => c.name === 'TFJS-VEC4-INNER') || validConfigs[0];
  } else if (maxDim >= 512) {
    return validConfigs.find(c => c.name === 'FIT-32x32') || validConfigs[0];
  } else {
    return validConfigs.find(c => c.name === 'REGISTER-BLOCKED') || validConfigs[0];
  }
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

  dispose(): void {
    for (const buffers of this.freeBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    for (const buffers of this.usedBuffers.values()) {
      for (const buf of buffers) buf.destroy();
    }
    this.freeBuffers.clear();
    this.usedBuffers.clear();
  }
}

export class WebGPUBackend implements Backend {
  name = 'webgpu';
  private device: GPUDevice;
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

  private async runBinaryOp(a: IFaceNDArray, b: IFaceNDArray, opName: string): Promise<IFaceNDArray> {
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

  private async runScalarOp(arr: IFaceNDArray, scalar: number, opName: string): Promise<IFaceNDArray> {
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
      size: 8,  // u32 size + f32 scalar
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

  private async runCumulative(arr: IFaceNDArray, opName: 'cumsum' | 'cumprod'): Promise<IFaceNDArray> {
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
      size: 16,  // u32 + f32 + f32 + padding
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
    return new WebGPUNDArray(f64, shape);
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
      const diagLen = Math.max(0, Math.min(
        k >= 0 ? rows : rows + k,
        k >= 0 ? cols - k : cols
      ));
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

  // ============ Math - Unary Operations ============

  sin(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.sin(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  cos(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.cos(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  tan(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.tan(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  arcsin(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.asin(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  arccos(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.acos(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  arctan(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.atan(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  sinh(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.sinh(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  cosh(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.cosh(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  tanh(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.tanh(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  exp(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.exp(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  log(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.log(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  log2(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.log2(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  log10(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.log10(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  sqrt(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.sqrt(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  cbrt(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.cbrt(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  abs(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.abs(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  sign(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.sign(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  floor(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.floor(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  ceil(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.ceil(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  round(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.round(arr.data[i]);
    return this.createArray(result, arr.shape);
  }

  neg(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = -arr.data[i];
    return this.createArray(result, arr.shape);
  }

  reciprocal(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = 1 / arr.data[i];
    return this.createArray(result, arr.shape);
  }

  square(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = arr.data[i] * arr.data[i];
    return this.createArray(result, arr.shape);
  }

  // ============ Math - Binary Operations ============

  add(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = a.data[i] + b.data[i];
    return this.createArray(result, a.shape);
  }

  sub(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = a.data[i] - b.data[i];
    return this.createArray(result, a.shape);
  }

  mul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = a.data[i] * b.data[i];
    return this.createArray(result, a.shape);
  }

  div(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = a.data[i] / b.data[i];
    return this.createArray(result, a.shape);
  }

  pow(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = Math.pow(a.data[i], b.data[i]);
    return this.createArray(result, a.shape);
  }

  maximum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = Math.max(a.data[i], b.data[i]);
    return this.createArray(result, a.shape);
  }

  minimum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) result[i] = Math.min(a.data[i], b.data[i]);
    return this.createArray(result, a.shape);
  }

  // ============ Math - Scalar Operations ============

  addScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = arr.data[i] + scalar;
    return this.createArray(result, arr.shape);
  }

  subScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = arr.data[i] - scalar;
    return this.createArray(result, arr.shape);
  }

  mulScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = arr.data[i] * scalar;
    return this.createArray(result, arr.shape);
  }

  divScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = arr.data[i] / scalar;
    return this.createArray(result, arr.shape);
  }

  powScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) result[i] = Math.pow(arr.data[i], scalar);
    return this.createArray(result, arr.shape);
  }

  clip(arr: IFaceNDArray, min: number, max: number): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      result[i] = Math.min(Math.max(arr.data[i], min), max);
    }
    return this.createArray(result, arr.shape);
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
    const result = new Float64Array(arr.data.length);
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      sum += arr.data[i];
      result[i] = sum;
    }
    return this.createArray(result, arr.shape);
  }

  cumprod(arr: IFaceNDArray): IFaceNDArray {
    const result = new Float64Array(arr.data.length);
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      prod *= arr.data[i];
      result[i] = prod;
    }
    return this.createArray(result, arr.shape);
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

    // Convert f64 to f32 for GPU
    const aF32 = new Float32Array(m * k);
    for (let i = 0; i < a.data.length; i++) aF32[i] = a.data[i];

    // For vec4 B storage, we need N to be padded to multiple of 4
    let bF32: Float32Array;
    let nPadded: number;
    let bBufferSize: number;

    if (config.usesVec4B) {
      // Pad N to multiple of 4 for vec4 storage
      nPadded = Math.ceil(n / 4) * 4;
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
      nPadded = n;
      bF32 = new Float32Array(k * n);
      for (let i = 0; i < b.data.length; i++) bF32[i] = b.data[i];
      bBufferSize = bF32.byteLength;
    }

    // Create uniform buffer for dimensions
    const uniformData = new Uint32Array([m, k, n, nPadded]);
    const uniformBuffer = this.bufferManager.acquire(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create storage buffers using buffer manager (pooled for reuse!)
    const aBuffer = this.bufferManager.acquire(aF32.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bBuffer = this.bufferManager.acquire(bBufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    // Output buffer size depends on vec4C config
    const outputSize = config.usesVec4C ? m * nPadded * 4 : m * n * 4;
    const outputBuffer = this.bufferManager.acquire(outputSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const stagingBuffer = this.device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

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

    // Async readback
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const outputF32 = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();

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

    // Release buffers back to pool
    this.bufferManager.release(uniformBuffer);
    this.bufferManager.release(aBuffer);
    this.bufferManager.release(bBuffer);
    this.bufferManager.release(outputBuffer);
    stagingBuffer.destroy();

    return this.createArray(result, [m, n]);
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

  det(arr: IFaceNDArray): number {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('det requires square matrix');
    }
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

  inv(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length !== 2 || arr.shape[0] !== arr.shape[1]) {
      throw new Error('inv requires square matrix');
    }
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
        [aug[i * n * 2 + k], aug[maxRow * n * 2 + k]] = [aug[maxRow * n * 2 + k], aug[i * n * 2 + k]];
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

  solve(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const aInv = this.inv(a);
    const bMat = b.shape.length === 1
      ? this.createArray(b.data, [b.shape[0], 1])
      : b;
    return this.matmul(aInv, bMat);
  }

  norm(arr: IFaceNDArray, ord: number = 2): number {
    if (ord === Infinity) {
      return Math.max(...Array.from(arr.data).map(Math.abs));
    }
    if (ord === 1) {
      return Array.from(arr.data).reduce((acc, x) => acc + Math.abs(x), 0);
    }
    return Math.sqrt(Array.from(arr.data).reduce((acc, x) => acc + x * x, 0));
  }

  qr(_arr: IFaceNDArray): { q: IFaceNDArray; r: IFaceNDArray } {
    throw new Error('QR decomposition not yet implemented');
  }

  svd(_arr: IFaceNDArray): { u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray } {
    throw new Error('SVD not yet implemented');
  }

  // ============ Async GPU Operations ============
  // These use real WebGPU compute shaders for acceleration

  // Unary ops (GPU-accelerated)
  async sinAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'sin'); }
  async cosAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'cos'); }
  async tanAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'tan'); }
  async arcsinAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'asin'); }
  async arccosAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'acos'); }
  async arctanAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'atan'); }
  async sinhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'sinh'); }
  async coshAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'cosh'); }
  async tanhAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'tanh'); }
  async expAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'exp'); }
  async exp2Async(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'exp2'); }
  async logAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'log'); }
  async log2Async(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'log2'); }
  async log10Async(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'log10'); }
  async sqrtAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'sqrt'); }
  async cbrtAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'cbrt'); }
  async absAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'abs'); }
  async signAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'sign'); }
  async floorAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'floor'); }
  async ceilAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'ceil'); }
  async roundAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'round'); }
  async negAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'neg'); }
  async reciprocalAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'reciprocal'); }
  async squareAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runUnaryOp(arr, 'square'); }

  // Binary ops (GPU-accelerated)
  async addAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'add'); }
  async subAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'sub'); }
  async mulAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'mul'); }
  async divAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'div'); }
  async powAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'pow'); }
  async maximumAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'maximum'); }
  async minimumAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runBinaryOp(a, b, 'minimum'); }

  // Scalar ops (GPU-accelerated)
  async addScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> { return this.runScalarOp(arr, scalar, 'addScalar'); }
  async subScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> { return this.runScalarOp(arr, scalar, 'subScalar'); }
  async mulScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> { return this.runScalarOp(arr, scalar, 'mulScalar'); }
  async divScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> { return this.runScalarOp(arr, scalar, 'divScalar'); }
  async powScalarAsync(arr: IFaceNDArray, scalar: number): Promise<IFaceNDArray> { return this.runScalarOp(arr, scalar, 'powScalar'); }
  async clipAsync(arr: IFaceNDArray, minVal: number, maxVal: number): Promise<IFaceNDArray> { return this.runClip(arr, minVal, maxVal); }

  // Reduction ops (GPU-accelerated)
  async sumAsync(arr: IFaceNDArray): Promise<number> { return this.runReduction(arr, 'sum'); }
  async prodAsync(arr: IFaceNDArray): Promise<number> { return this.runReduction(arr, 'prod'); }
  async minAsync(arr: IFaceNDArray): Promise<number> {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.runReduction(arr, 'min');
  }
  async maxAsync(arr: IFaceNDArray): Promise<number> {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.runReduction(arr, 'max');
  }

  // Cumulative ops (GPU-accelerated)
  async cumsumAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runCumulative(arr, 'cumsum'); }
  async cumprodAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runCumulative(arr, 'cumprod'); }

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
  async argminAsync(arr: IFaceNDArray): Promise<number> { return this.runArgReduction(arr, 'argmin'); }
  async argmaxAsync(arr: IFaceNDArray): Promise<number> { return this.runArgReduction(arr, 'argmax'); }
  async allAsync(arr: IFaceNDArray): Promise<boolean> { return this.runBoolReduction(arr, 'all'); }
  async anyAsync(arr: IFaceNDArray): Promise<boolean> { return this.runBoolReduction(arr, 'any'); }
  async sumAxisAsync(arr: IFaceNDArray, axis: number): Promise<IFaceNDArray> { return this.runSumAxis(arr, axis); }
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
  async transposeAsync(arr: IFaceNDArray): Promise<IFaceNDArray> { return this.runTranspose(arr); }
  async outerAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> { return this.runOuter(a, b); }
  async dotAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<IFaceNDArray> {
    // For 1D arrays, dot is inner product wrapped in array
    if (a.shape.length === 1 && b.shape.length === 1) {
      const result = await this.runDot(a, b);
      return this.createArray([result], [1]);
    }
    // For 2D arrays, use matmul
    return this.matmulAsync(a, b);
  }
  async innerAsync(a: IFaceNDArray, b: IFaceNDArray): Promise<number> { return this.runDot(a, b); }
  async traceAsync(arr: IFaceNDArray): Promise<number> { return this.runTrace(arr); }
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
