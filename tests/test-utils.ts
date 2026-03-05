/**
 * Test utilities - matching rumpy-tests/src/lib.rs
 */

/** Check if two f64 values are approximately equal */
export function approxEq(a: number, b: number, tol: number): boolean {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (!Number.isFinite(a) && !Number.isFinite(b)) {
    return Math.sign(a) === Math.sign(b);
  }
  return Math.abs(a - b) < tol;
}

/** Check if two arrays are approximately equal */
export function arraysApproxEq(
  a: Float64Array | number[],
  b: Float64Array | number[],
  tol: number
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!approxEq(a[i], b[i], tol)) return false;
  }
  return true;
}

/** Default tolerance for floating point comparisons (f64) */
export const DEFAULT_TOL = 1e-10;

/** Relaxed tolerance for operations with accumulated error */
export const RELAXED_TOL = 1e-6;

/** WebGPU tolerance - f32 has ~6-7 decimal digits of precision */
export const WEBGPU_TOL = 1e-5;

/** SVD tolerance - power iteration SVD has limited precision */
export const SVD_TOL = 1e-4;

/** Get tolerance appropriate for backend precision */
export function getTol(backend: Backend, relaxed: boolean = false): number {
  if (backend.name === 'webgpu') {
    // WebGPU uses f32 which has ~6-7 decimal digits precision
    return relaxed ? 1e-4 : WEBGPU_TOL;
  }
  return relaxed ? RELAXED_TOL : DEFAULT_TOL;
}

/**
 * Get array data, materializing GPU tensors if needed.
 * For non-GPU backends, returns data immediately.
 * For GPU backends, awaits materialization first.
 */
export async function getData(arr: NDArray, backend: Backend): Promise<number[]> {
  if (backend.materializeAll) {
    await backend.materializeAll();
  }
  return arr.toArray();
}

/**
 * Same as getData but for multiple arrays at once (more efficient for GPU)
 */
export async function getDataMany(arrays: NDArray[], backend: Backend): Promise<number[][]> {
  if (backend.materializeAll) {
    await backend.materializeAll();
  }
  return arrays.map(arr => arr.toArray());
}

/** NDArray interface */
export interface NDArray {
  shape: number[];
  data: Float64Array;
  toArray(): number[];
}

/** Backend interface that all backends must implement */
export interface Backend {
  name: string;

  // ============ Creation ============
  zeros(shape: number[]): NDArray;
  ones(shape: number[]): NDArray;
  full(shape: number[], value: number): NDArray;
  arange(start: number, stop: number, step: number): NDArray;
  linspace(start: number, stop: number, num: number): NDArray;
  eye(n: number): NDArray;
  diag(arr: NDArray, k?: number): NDArray;
  array(data: number[], shape?: number[]): NDArray;

  // ============ Math - Unary ============
  sin(arr: NDArray): NDArray;
  cos(arr: NDArray): NDArray;
  tan(arr: NDArray): NDArray;
  arcsin(arr: NDArray): NDArray;
  arccos(arr: NDArray): NDArray;
  arctan(arr: NDArray): NDArray;
  sinh(arr: NDArray): NDArray;
  cosh(arr: NDArray): NDArray;
  tanh(arr: NDArray): NDArray;
  exp(arr: NDArray): NDArray;
  log(arr: NDArray): NDArray;
  log2(arr: NDArray): NDArray;
  log10(arr: NDArray): NDArray;
  sqrt(arr: NDArray): NDArray;
  cbrt(arr: NDArray): NDArray;
  abs(arr: NDArray): NDArray;
  sign(arr: NDArray): NDArray;
  floor(arr: NDArray): NDArray;
  ceil(arr: NDArray): NDArray;
  round(arr: NDArray): NDArray;
  neg(arr: NDArray): NDArray;
  reciprocal(arr: NDArray): NDArray;
  square(arr: NDArray): NDArray;

  // ============ Math - Unary (Extended) ============
  arcsinh(arr: NDArray): NDArray;
  arccosh(arr: NDArray): NDArray;
  arctanh(arr: NDArray): NDArray;
  expm1(arr: NDArray): NDArray;
  log1p(arr: NDArray): NDArray;
  trunc(arr: NDArray): NDArray;
  fix(arr: NDArray): NDArray;  // alias for trunc (round toward zero)
  sinc(arr: NDArray): NDArray;
  deg2rad(arr: NDArray): NDArray;
  rad2deg(arr: NDArray): NDArray;
  heaviside(arr: NDArray, h0: number): NDArray;
  signbit(arr: NDArray): NDArray;  // 1.0 if sign bit set, 0.0 otherwise

  // ============ Math - Decomposition ============
  modf(arr: NDArray): { frac: NDArray; integ: NDArray };  // fractional and integral parts
  frexp(arr: NDArray): { mantissa: NDArray; exponent: NDArray };  // mantissa and exponent
  ldexp(arr: NDArray, exp: NDArray): NDArray;  // mantissa * 2^exp
  divmod(a: NDArray, b: NDArray): { quotient: NDArray; remainder: NDArray };  // floor division and mod

  // ============ Math - Binary (Extended) ============
  mod(a: NDArray, b: NDArray): NDArray;
  fmod(a: NDArray, b: NDArray): NDArray;
  remainder(a: NDArray, b: NDArray): NDArray;
  copysign(a: NDArray, b: NDArray): NDArray;
  hypot(a: NDArray, b: NDArray): NDArray;
  arctan2(a: NDArray, b: NDArray): NDArray;
  logaddexp(a: NDArray, b: NDArray): NDArray;
  logaddexp2(a: NDArray, b: NDArray): NDArray;
  fmax(a: NDArray, b: NDArray): NDArray;
  fmin(a: NDArray, b: NDArray): NDArray;

  // ============ Comparison ============
  equal(a: NDArray, b: NDArray): NDArray;
  notEqual(a: NDArray, b: NDArray): NDArray;
  less(a: NDArray, b: NDArray): NDArray;
  lessEqual(a: NDArray, b: NDArray): NDArray;
  greater(a: NDArray, b: NDArray): NDArray;
  greaterEqual(a: NDArray, b: NDArray): NDArray;
  isnan(arr: NDArray): NDArray;
  isinf(arr: NDArray): NDArray;
  isfinite(arr: NDArray): NDArray;

  // ============ Set Operations ============
  setdiff1d(a: NDArray, b: NDArray): NDArray;
  union1d(a: NDArray, b: NDArray): NDArray;
  intersect1d(a: NDArray, b: NDArray): NDArray;
  isin(element: NDArray, testElements: NDArray): NDArray;

  // ============ Array Manipulation (Extended) ============
  insert(arr: NDArray, index: number, values: NDArray | number, axis?: number): NDArray;
  deleteArr(arr: NDArray, index: number | number[], axis?: number): NDArray;
  append(arr: NDArray, values: NDArray, axis?: number): NDArray;
  atleast1d(arr: NDArray): NDArray;
  atleast2d(arr: NDArray): NDArray;
  atleast3d(arr: NDArray): NDArray;
  countNonzero(arr: NDArray, axis?: number): NDArray | number;

  // ============ Advanced Linalg ============
  matrixPower(arr: NDArray, n: number): NDArray;
  kron(a: NDArray, b: NDArray): NDArray;
  cond(arr: NDArray, p?: number | 'fro'): number;
  slogdet(arr: NDArray): { sign: number; logabsdet: number };
  multiDot(arrays: NDArray[]): NDArray;

  // ============ Polynomial ============
  polyval(p: NDArray, x: NDArray): NDArray;
  polyadd(a: NDArray, b: NDArray): NDArray;
  polymul(a: NDArray, b: NDArray): NDArray;
  polyfit(x: NDArray, y: NDArray, deg: number): NDArray;
  roots(p: NDArray): NDArray;

  // ============ Interpolation ============
  interp(x: NDArray, xp: NDArray, fp: NDArray): NDArray;

  // ============ Histogram ============
  bincount(x: NDArray, weights?: NDArray, minlength?: number): NDArray;

  // ============ Advanced Indexing ============
  partition(arr: NDArray, kth: number, axis?: number): NDArray;
  argpartition(arr: NDArray, kth: number, axis?: number): NDArray;
  lexsort(keys: NDArray[]): NDArray;
  compress(condition: NDArray, arr: NDArray, axis?: number): NDArray;
  extract(condition: NDArray, arr: NDArray): NDArray;
  place(arr: NDArray, mask: NDArray, vals: NDArray): void;
  select(condlist: NDArray[], choicelist: NDArray[], defaultVal?: number): NDArray;

  // ============ Math - Binary ============
  add(a: NDArray, b: NDArray): NDArray;
  sub(a: NDArray, b: NDArray): NDArray;
  mul(a: NDArray, b: NDArray): NDArray;
  div(a: NDArray, b: NDArray): NDArray;
  pow(a: NDArray, b: NDArray): NDArray;
  maximum(a: NDArray, b: NDArray): NDArray;
  minimum(a: NDArray, b: NDArray): NDArray;

  // ============ Math - Scalar ============
  addScalar(arr: NDArray, scalar: number): NDArray;
  subScalar(arr: NDArray, scalar: number): NDArray;
  mulScalar(arr: NDArray, scalar: number): NDArray;
  divScalar(arr: NDArray, scalar: number): NDArray;
  powScalar(arr: NDArray, scalar: number): NDArray;
  clip(arr: NDArray, min: number, max: number): NDArray;

  // ============ Stats ============
  sum(arr: NDArray): number;
  prod(arr: NDArray): number;
  mean(arr: NDArray): number;
  var(arr: NDArray, ddof?: number): number;
  std(arr: NDArray, ddof?: number): number;
  min(arr: NDArray): number;
  max(arr: NDArray): number;
  argmin(arr: NDArray): number;
  argmax(arr: NDArray): number;
  cumsum(arr: NDArray): NDArray;
  cumprod(arr: NDArray): NDArray;
  all(arr: NDArray): boolean;
  any(arr: NDArray): boolean;
  sumAxis(arr: NDArray, axis: number): NDArray;
  meanAxis(arr: NDArray, axis: number): NDArray;
  minAxis(arr: NDArray, axis: number): NDArray;
  maxAxis(arr: NDArray, axis: number): NDArray;
  argminAxis(arr: NDArray, axis: number): NDArray;
  argmaxAxis(arr: NDArray, axis: number): NDArray;
  varAxis(arr: NDArray, axis: number, ddof?: number): NDArray;
  stdAxis(arr: NDArray, axis: number, ddof?: number): NDArray;
  prodAxis(arr: NDArray, axis: number): NDArray;
  allAxis(arr: NDArray, axis: number): NDArray;
  anyAxis(arr: NDArray, axis: number): NDArray;
  cumsumAxis(arr: NDArray, axis: number): NDArray;
  cumprodAxis(arr: NDArray, axis: number): NDArray;

  // ============ NaN-aware Stats ============
  nansum(arr: NDArray): number;
  nanmean(arr: NDArray): number;
  nanstd(arr: NDArray, ddof?: number): number;
  nanvar(arr: NDArray, ddof?: number): number;
  nanmin(arr: NDArray): number;
  nanmax(arr: NDArray): number;
  nanargmin(arr: NDArray): number;
  nanargmax(arr: NDArray): number;
  nanprod(arr: NDArray): number;

  // ============ Linalg ============
  matmul(a: NDArray, b: NDArray): NDArray;
  dot(a: NDArray, b: NDArray): NDArray;
  inner(a: NDArray, b: NDArray): number;
  outer(a: NDArray, b: NDArray): NDArray;
  transpose(arr: NDArray): NDArray;
  trace(arr: NDArray): number;
  det(arr: NDArray): number;
  inv(arr: NDArray): NDArray;
  solve(a: NDArray, b: NDArray): NDArray;
  norm(arr: NDArray, ord?: number): number;
  qr(arr: NDArray): { q: NDArray; r: NDArray };
  svd(arr: NDArray): { u: NDArray; s: NDArray; vt: NDArray };

  // ============ Creation - Like Functions ============
  zerosLike(arr: NDArray): NDArray;
  onesLike(arr: NDArray): NDArray;
  emptyLike(arr: NDArray): NDArray;
  fullLike(arr: NDArray, value: number): NDArray;

  // ============ Broadcasting ============
  broadcastTo(arr: NDArray, shape: number[]): NDArray;
  broadcastArrays(...arrays: NDArray[]): NDArray[];

  // ============ Shape Manipulation ============
  swapaxes(arr: NDArray, axis1: number, axis2: number): NDArray;
  moveaxis(arr: NDArray, source: number, destination: number): NDArray;
  squeeze(arr: NDArray, axis?: number): NDArray;
  expandDims(arr: NDArray, axis: number): NDArray;
  reshape(arr: NDArray, shape: number[]): NDArray;
  flatten(arr: NDArray): NDArray;
  concatenate(arrays: NDArray[], axis?: number): NDArray;
  stack(arrays: NDArray[], axis?: number): NDArray;
  split(arr: NDArray, indices: number | number[], axis?: number): NDArray[];

  // ============ Conditional ============
  where(condition: NDArray, x: NDArray, y: NDArray): NDArray;

  // ============ Advanced Indexing ============
  take(arr: NDArray, indices: NDArray | number[], axis?: number): NDArray;

  // ============ Batched Operations ============
  batchedMatmul(a: NDArray, b: NDArray): NDArray;

  // ============ Einstein Summation ============
  einsum(subscripts: string, ...operands: NDArray[]): NDArray;

  // ============ Differences ============
  diff(arr: NDArray, n?: number, axis?: number): NDArray;
  gradient(arr: NDArray, axis?: number): NDArray;
  ediff1d(arr: NDArray): NDArray;

  // ============ Cross Product ============
  cross(a: NDArray, b: NDArray): NDArray;

  // ============ Statistics ============
  cov(x: NDArray, y?: NDArray): NDArray;
  corrcoef(x: NDArray, y?: NDArray): NDArray;

  // ============ Convolution ============
  convolve(a: NDArray, v: NDArray, mode?: 'full' | 'same' | 'valid'): NDArray;
  correlate(a: NDArray, v: NDArray, mode?: 'full' | 'same' | 'valid'): NDArray;

  // ============ Matrix Creation ============
  identity(n: number): NDArray;
  tril(arr: NDArray, k?: number): NDArray;
  triu(arr: NDArray, k?: number): NDArray;

  // ============ Grid Creation ============
  meshgrid(x: NDArray, y: NDArray): { X: NDArray; Y: NDArray };
  logspace(start: number, stop: number, num: number, base?: number): NDArray;
  geomspace(start: number, stop: number, num: number): NDArray;

  // ============ Stacking Shortcuts ============
  vstack(arrays: NDArray[]): NDArray;
  hstack(arrays: NDArray[]): NDArray;
  dstack(arrays: NDArray[]): NDArray;

  // ============ Split Shortcuts ============
  vsplit(arr: NDArray, indices: number | number[]): NDArray[];
  hsplit(arr: NDArray, indices: number | number[]): NDArray[];
  dsplit(arr: NDArray, indices: number | number[]): NDArray[];

  // ============ Array Replication ============
  tile(arr: NDArray, reps: number | number[]): NDArray;
  repeat(arr: NDArray, repeats: number, axis?: number): NDArray;

  // ============ Index Finding ============
  nonzero(arr: NDArray): NDArray[];
  argwhere(arr: NDArray): NDArray;
  flatnonzero(arr: NDArray): NDArray;

  // ============ Value Handling ============
  nanToNum(arr: NDArray, nan?: number, posInf?: number, negInf?: number): NDArray;

  // ============ Sorting ============
  sort(arr: NDArray, axis?: number): NDArray;
  argsort(arr: NDArray, axis?: number): NDArray;
  searchsorted(arr: NDArray, v: number | NDArray, side?: 'left' | 'right'): NDArray | number;
  unique(arr: NDArray): NDArray;

  // ============ Random ============
  seed(s: number): void;
  rand(shape: number[]): NDArray;
  randn(shape: number[]): NDArray;
  randint(low: number, high: number, shape: number[]): NDArray;
  uniform(low: number, high: number, shape: number[]): NDArray;
  normal(loc: number, scale: number, shape: number[]): NDArray;
  shuffle(arr: NDArray): NDArray;
  choice(arr: NDArray, size: number, replace?: boolean): NDArray;
  permutation(n: number | NDArray): NDArray;

  // ============ GPU Materialization ============
  // For GPU backends: batch-read all pending GPU data before sync access
  // For non-GPU backends: no-op
  materializeAll?(): Promise<void>;
}
