/**
 * Core types for numpyjs
 *
 * NDArray and Backend interfaces - the canonical type definitions
 * used by all backends (JS, WebGPU, WASM).
 */

// ============ DType System ============

/** Supported data types, matching NumPy's dtype system */
export type DType = 'float64' | 'float32' | 'int32' | 'int16' | 'int8' | 'uint8' | 'bool';

/** Union of all typed array types used by NDArray */
export type AnyTypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint8Array;

/** Get the TypedArray constructor for a given dtype */
export function dtypeArrayConstructor(dtype: DType): {
  new (n: number): AnyTypedArray;
  new (buf: ArrayLike<number>): AnyTypedArray;
} {
  switch (dtype) {
    case 'float64':
      return Float64Array as any;
    case 'float32':
      return Float32Array as any;
    case 'int32':
      return Int32Array as any;
    case 'int16':
      return Int16Array as any;
    case 'int8':
      return Int8Array as any;
    case 'uint8':
    case 'bool':
      return Uint8Array as any;
  }
}

/** Create an empty typed array for a given dtype */
export function createTypedArray(dtype: DType, length: number): AnyTypedArray {
  return new (dtypeArrayConstructor(dtype))(length);
}

/** Create a typed array from existing data */
export function createTypedArrayFrom(dtype: DType, data: ArrayLike<number>): AnyTypedArray {
  return new (dtypeArrayConstructor(dtype))(data as any);
}

/** Default dtype */
export const DEFAULT_DTYPE: DType = 'float64';

/** Promote two dtypes to their common type (NumPy-style) */
export function promoteDTypes(a: DType, b: DType): DType {
  if (a === b) return a;
  // Float types dominate
  if (a === 'float64' || b === 'float64') return 'float64';
  if (a === 'float32' || b === 'float32') return 'float32';
  // Integer promotion: wider wins
  const intRank: Record<string, number> = { bool: 0, int8: 1, uint8: 1, int16: 2, int32: 3 };
  return (intRank[a] || 0) >= (intRank[b] || 0) ? a : b;
}

// ============ Histogram Bin Strategies ============

/** Bin selection algorithm strings, matching NumPy */
export type BinStrategy = 'auto' | 'fd' | 'sturges' | 'rice' | 'sqrt' | 'scott' | 'doane';

/** Bins parameter: number of bins, explicit edges, or algorithm name */
export type BinsParam = number | NDArray | BinStrategy;

/** Sort algorithm kind — all map to JS native sort, but signature matches NumPy */
export type SortKind = 'quicksort' | 'mergesort' | 'heapsort' | 'stable';

// ============ Scalar Broadcasting ============

/** An NDArray or a scalar number — used for binary op arguments */
export type ArrayOrScalar = NDArray | number;

// ============ NDArray Interface ============

/** NDArray interface - represents an n-dimensional array */
export interface NDArray {
  shape: number[];
  dtype: DType;
  data: AnyTypedArray;
  /** Number of dimensions (equivalent to shape.length) */
  ndim: number;
  /** Total number of elements */
  size: number;
  /** Transpose shortcut — returns the transposed array */
  T: NDArray;
  toArray(): number[];
  /** Extract scalar from 0-d or 1-element array; throws if more than 1 element */
  item(): number;
}

/** Backend interface that all backends must implement */
export interface Backend {
  name: string;

  // ============ Constants ============
  /** Math.PI */
  pi: number;
  /** Math.E */
  e: number;
  /** Infinity */
  inf: number;
  /** NaN */
  nan: number;
  /** null — used as axis expansion sentinel (np.newaxis) */
  newaxis: null;

  // ============ Creation ============
  zeros(shape: number[], dtype?: DType): NDArray;
  ones(shape: number[], dtype?: DType): NDArray;
  full(shape: number[], value: number, dtype?: DType): NDArray;
  arange(startOrStop: number, stop?: number, step?: number, dtype?: DType): NDArray;
  linspace(
    start: number,
    stop: number,
    num: number,
    endpoint?: boolean | DType,
    dtype?: DType
  ): NDArray;
  eye(n: number, M?: number | DType, k?: number, dtype?: DType): NDArray;
  diag(arr: NDArray, k?: number): NDArray;
  array(data: number[] | number[][] | any[], shape?: number[], dtype?: DType): NDArray;

  // ============ Creation (Additional) ============
  /** np.asarray — convert input to NDArray (no-copy if already NDArray with matching dtype) */
  asarray(a: NDArray | number[] | number, dtype?: DType): NDArray;
  fromfunction(fn: (...coords: number[]) => number, shape: number[], dtype?: DType): NDArray;
  fromiter(iter: Iterable<number>, count?: number, dtype?: DType): NDArray;

  // ============ Complex Helpers ============
  real(arr: NDArray): NDArray;
  imag(arr: NDArray): NDArray;
  conj(arr: NDArray): NDArray;

  // ============ Type Casting ============
  astype(arr: NDArray, dtype: DType): NDArray;

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
  /** np.absolute — alias for abs */
  absolute(arr: NDArray): NDArray;
  sign(arr: NDArray): NDArray;
  floor(arr: NDArray): NDArray;
  ceil(arr: NDArray): NDArray;
  round(arr: NDArray, decimals?: number): NDArray;
  negative(arr: NDArray): NDArray;
  /** @deprecated Use negative() instead */
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
  fix(arr: NDArray): NDArray;
  sinc(arr: NDArray): NDArray;
  deg2rad(arr: NDArray): NDArray;
  rad2deg(arr: NDArray): NDArray;
  heaviside(arr: NDArray, h0: number): NDArray;
  signbit(arr: NDArray): NDArray;

  // ============ Math - Decomposition ============
  modf(arr: NDArray): { frac: NDArray; integ: NDArray };
  frexp(arr: NDArray): { mantissa: NDArray; exponent: NDArray };
  ldexp(arr: NDArray, exp: NDArray): NDArray;
  divmod(a: ArrayOrScalar, b: ArrayOrScalar): { quotient: NDArray; remainder: NDArray };

  // ============ Math - Binary (Extended) ============
  mod(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  fmod(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  remainder(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  copysign(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  hypot(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  arctan2(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  logaddexp(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  logaddexp2(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  fmax(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  fmin(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;

  // ============ Comparison ============
  equal(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  notEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  less(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  lessEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  greater(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  greaterEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
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
  countNonzero(arr: NDArray, axis?: number, keepdims?: boolean): NDArray | number;

  // ============ Advanced Linalg ============
  matrixPower(arr: NDArray, n: number): NDArray | Promise<NDArray>;
  kron(a: NDArray, b: NDArray): NDArray;
  cond(arr: NDArray, p?: number | 'fro'): number;
  slogdet(
    arr: NDArray
  ): { sign: number; logabsdet: number } | Promise<{ sign: number; logabsdet: number }>;
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
  add(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  subtract(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  multiply(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  divide(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  power(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  floorDivide(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  maximum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  minimum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  /** @deprecated Use subtract() instead */
  sub(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  /** @deprecated Use multiply() instead */
  mul(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  /** @deprecated Use divide() instead */
  div(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  /** @deprecated Use power() instead */
  pow(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;

  // ============ Math - Scalar (deprecated, use add/subtract/etc with scalar) ============
  /** @deprecated Use add(arr, scalar) instead */
  addScalar(arr: NDArray, scalar: number): NDArray;
  /** @deprecated Use subtract(arr, scalar) instead */
  subScalar(arr: NDArray, scalar: number): NDArray;
  /** @deprecated Use multiply(arr, scalar) instead */
  mulScalar(arr: NDArray, scalar: number): NDArray;
  /** @deprecated Use divide(arr, scalar) instead */
  divScalar(arr: NDArray, scalar: number): NDArray;
  /** @deprecated Use power(arr, scalar) instead */
  powScalar(arr: NDArray, scalar: number): NDArray;
  clip(arr: NDArray, min: number | null, max: number | null): NDArray;

  // ============ Stats (NumPy-style with optional axis) ============
  sum(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray;
  prod(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray;
  mean(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray;
  /** np.var — variance. var(arr, axis?, ddof?, keepdims?) matches NumPy param order */
  var(arr: NDArray, axis?: number | null, ddof?: number, keepdims?: boolean): number | NDArray;
  /** np.std — standard deviation. std(arr, axis?, ddof?, keepdims?) matches NumPy param order */
  std(arr: NDArray, axis?: number | null, ddof?: number, keepdims?: boolean): number | NDArray;
  min(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  max(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  argmin(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  argmax(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  cumsum(arr: NDArray, axis?: number, dtype?: DType): NDArray;
  cumprod(arr: NDArray, axis?: number, dtype?: DType): NDArray;
  all(arr: NDArray, axis?: number, keepdims?: boolean): boolean | NDArray;
  any(arr: NDArray, axis?: number, keepdims?: boolean): boolean | NDArray;
  /** @deprecated Use sum(arr, axis) instead */
  sumAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use mean(arr, axis) instead */
  meanAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use min(arr, axis) instead */
  minAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use max(arr, axis) instead */
  maxAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use argmin(arr, axis) instead */
  argminAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use argmax(arr, axis) instead */
  argmaxAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use var(arr, ddof, axis) instead */
  varAxis(arr: NDArray, axis: number, ddof?: number): NDArray;
  /** @deprecated Use std(arr, ddof, axis) instead */
  stdAxis(arr: NDArray, axis: number, ddof?: number): NDArray;
  /** @deprecated Use prod(arr, axis) instead */
  prodAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use all(arr, axis) instead */
  allAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use any(arr, axis) instead */
  anyAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use cumsum(arr, axis) instead */
  cumsumAxis(arr: NDArray, axis: number): NDArray;
  /** @deprecated Use cumprod(arr, axis) instead */
  cumprodAxis(arr: NDArray, axis: number): NDArray;

  // ============ NaN-aware Stats ============
  nansum(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  nanmean(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  nanstd(arr: NDArray, axis?: number | null, ddof?: number, keepdims?: boolean): number | NDArray;
  nanvar(arr: NDArray, axis?: number | null, ddof?: number, keepdims?: boolean): number | NDArray;
  nanmin(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  nanmax(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  nanargmin(arr: NDArray, axis?: number): number | NDArray;
  nanargmax(arr: NDArray, axis?: number): number | NDArray;
  nanprod(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;

  // ============ Order Statistics ============
  median(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  percentile(
    arr: NDArray,
    q: number,
    axis?: number,
    keepdims?: boolean,
    method?: 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'
  ): number | NDArray;
  quantile(
    arr: NDArray,
    q: number,
    axis?: number,
    keepdims?: boolean,
    method?: 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'
  ): number | NDArray;
  nanmedian(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  nanpercentile(
    arr: NDArray,
    q: number,
    axis?: number,
    keepdims?: boolean,
    method?: 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'
  ): number | NDArray;

  // ============ Histogram ============
  histogram(
    arr: NDArray,
    bins?: BinsParam,
    range?: [number, number] | null,
    density?: boolean,
    weights?: NDArray
  ): { hist: NDArray; binEdges: NDArray };
  histogramBinEdges(arr: NDArray, bins?: BinsParam): NDArray;

  // ============ Linalg ============
  matmul(a: NDArray, b: NDArray): NDArray;
  dot(a: NDArray, b: NDArray): NDArray;
  inner(a: NDArray, b: NDArray): number | NDArray;
  outer(a: NDArray, b: NDArray): NDArray;
  transpose(arr: NDArray, axes?: number[]): NDArray;
  trace(arr: NDArray): number;
  det(arr: NDArray): number | Promise<number>;
  inv(arr: NDArray): NDArray | Promise<NDArray>;
  solve(a: NDArray, b: NDArray): NDArray | Promise<NDArray>;
  norm(arr: NDArray, ord?: number | 'fro' | 'nuc', axis?: number): number | NDArray;
  qr(arr: NDArray, mode?: 'reduced' | 'complete'): { q: NDArray; r: NDArray };
  svd(arr: NDArray, fullMatrices?: boolean): { u: NDArray; s: NDArray; vt: NDArray };

  // ============ Creation - Like Functions ============
  zerosLike(arr: NDArray, dtype?: DType): NDArray;
  onesLike(arr: NDArray, dtype?: DType): NDArray;
  emptyLike(arr: NDArray, dtype?: DType): NDArray;
  fullLike(arr: NDArray, value: number, dtype?: DType): NDArray;

  // ============ Broadcasting ============
  broadcastTo(arr: NDArray, shape: number[]): NDArray;
  broadcastArrays(...arrays: NDArray[]): NDArray[];

  // ============ Shape Manipulation ============
  swapaxes(arr: NDArray, axis1: number, axis2: number): NDArray;
  moveaxis(arr: NDArray, source: number | number[], destination: number | number[]): NDArray;
  squeeze(arr: NDArray, axis?: number | number[]): NDArray;
  expandDims(arr: NDArray, axis: number): NDArray;
  reshape(arr: NDArray, shape: number[]): NDArray;
  /** np.resize — resize array with repetition to fill new shape */
  resize(arr: NDArray, newShape: number[]): NDArray;
  flatten(arr: NDArray): NDArray;
  concatenate(arrays: NDArray[], axis?: number | null): NDArray;
  stack(arrays: NDArray[], axis?: number): NDArray;
  split(arr: NDArray, indices: number | number[], axis?: number): NDArray[];

  // ============ Conditional ============
  where(condition: NDArray, x?: NDArray, y?: NDArray): NDArray | NDArray[];

  // ============ Advanced Indexing ============
  take(arr: NDArray, indices: NDArray | number[], axis?: number): NDArray;
  /** np.put — set elements at flat indices (mutates arr in place) */
  put(arr: NDArray, ind: number[] | NDArray, v: number | number[]): void;
  /** np.ix_ — construct open mesh from multiple sequences */
  ix_(...args: NDArray[]): NDArray[];

  // ============ Batched Operations ============
  batchedMatmul(a: NDArray, b: NDArray): NDArray;

  // ============ Einstein Summation ============
  einsum(subscripts: string, ...operands: NDArray[]): NDArray | Promise<NDArray>;

  // ============ Differences ============
  diff(
    arr: NDArray,
    n?: number,
    axis?: number,
    prepend?: NDArray | number,
    append?: NDArray | number
  ): NDArray;
  gradient(arr: NDArray, axis?: number, edgeOrder?: 1 | 2): NDArray;
  ediff1d(arr: NDArray): NDArray;

  // ============ Cross Product ============
  cross(a: NDArray, b: NDArray): NDArray;

  // ============ Statistics ============
  cov(x: NDArray, y?: NDArray, rowvar?: boolean, bias?: boolean, ddof?: number | null): NDArray;
  corrcoef(x: NDArray, y?: NDArray, rowvar?: boolean): NDArray;

  // ============ Convolution ============
  convolve(a: NDArray, v: NDArray, mode?: 'full' | 'same' | 'valid'): NDArray | Promise<NDArray>;
  correlate(a: NDArray, v: NDArray, mode?: 'full' | 'same' | 'valid'): NDArray | Promise<NDArray>;

  // ============ Matrix Creation ============
  identity(n: number, dtype?: DType): NDArray;
  tril(arr: NDArray, k?: number): NDArray;
  triu(arr: NDArray, k?: number): NDArray;

  // ============ Grid Creation ============
  meshgrid(...args: (NDArray | 'xy' | 'ij')[]): NDArray[];
  logspace(
    start: number,
    stop: number,
    num: number,
    base?: number,
    endpoint?: boolean,
    dtype?: DType
  ): NDArray;
  geomspace(start: number, stop: number, num: number, endpoint?: boolean, dtype?: DType): NDArray;

  // ============ Stacking Shortcuts ============
  vstack(arrays: NDArray[]): NDArray;
  /** np.row_stack — alias for vstack */
  rowStack(arrays: NDArray[]): NDArray;
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
  sort(arr: NDArray, axis?: number, kind?: SortKind): NDArray;
  argsort(arr: NDArray, axis?: number, kind?: SortKind): NDArray;
  searchsorted(
    arr: NDArray,
    v: number | NDArray,
    side?: 'left' | 'right',
    sorter?: NDArray
  ): NDArray | number;
  unique(
    arr: NDArray,
    returnIndex?: boolean,
    returnInverse?: boolean,
    returnCounts?: boolean
  ): NDArray | { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray };

  // ============ Random ============
  seed(s: number): void;
  rand(shape: number[], dtype?: DType): NDArray;
  randn(shape: number[], dtype?: DType): NDArray;
  randint(low: number, high: number, shape: number[], dtype?: DType): NDArray;
  uniform(low: number, high: number, shape: number[], dtype?: DType): NDArray;
  normal(loc: number, scale: number, shape: number[], dtype?: DType): NDArray;
  shuffle(arr: NDArray): void;
  choice(arr: NDArray, size: number, replace?: boolean, p?: NDArray | number[]): NDArray;
  permutation(n: number | NDArray): NDArray;

  // ============ Logic ============
  logicalAnd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  logicalOr(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  logicalNot(arr: NDArray): NDArray;
  logicalXor(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  isclose(a: NDArray, b: NDArray, rtol?: number, atol?: number): NDArray;
  allclose(a: NDArray, b: NDArray, rtol?: number, atol?: number): boolean;
  arrayEqual(a: NDArray, b: NDArray): boolean;

  // ============ Bitwise ============
  bitwiseAnd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  bitwiseOr(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  bitwiseXor(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  bitwiseNot(arr: NDArray): NDArray;
  leftShift(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  rightShift(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;

  // ============ Array Manipulation (Additional) ============
  copy(arr: NDArray, dtype?: DType): NDArray;
  empty(shape: number[], dtype?: DType): NDArray;
  flip(arr: NDArray, axis?: number): NDArray;
  fliplr(arr: NDArray): NDArray;
  flipud(arr: NDArray): NDArray;
  roll(arr: NDArray, shift: number | number[], axis?: number | number[]): NDArray;
  rot90(arr: NDArray, k?: number, axes?: [number, number]): NDArray;
  ravel(arr: NDArray): NDArray;
  pad(
    arr: NDArray,
    padWidth: number | [number, number],
    mode?:
      | 'constant'
      | 'edge'
      | 'reflect'
      | 'wrap'
      | 'symmetric'
      | 'linear_ramp'
      | 'mean'
      | 'minimum'
      | 'maximum',
    constantValue?: number
  ): NDArray;
  columnStack(arrays: NDArray[]): NDArray;
  arraySplit(arr: NDArray, indices: number | number[], axis?: number): NDArray[];
  putAlongAxis(arr: NDArray, indices: NDArray, values: NDArray, axis: number): NDArray;
  takeAlongAxis(arr: NDArray, indices: NDArray, axis: number): NDArray;

  // ============ Additional Linalg ============
  eig(arr: NDArray): { values: NDArray; vectors: NDArray };
  eigh(arr: NDArray): { values: NDArray; vectors: NDArray };
  eigvals(arr: NDArray): NDArray;
  cholesky(arr: NDArray): NDArray;
  lstsq(
    a: NDArray,
    b: NDArray,
    rcond?: number | null
  ): { x: NDArray; residuals: NDArray; rank: number; singularValues: NDArray };
  pinv(arr: NDArray): NDArray;
  matrixRank(arr: NDArray, tol?: number): number;
  tensordot(a: NDArray, b: NDArray, axes?: number | [number[], number[]]): NDArray;
  vdot(a: NDArray, b: NDArray): number;

  // ============ FFT ============
  fft(arr: NDArray): { real: NDArray; imag: NDArray };
  ifft(real: NDArray, imag: NDArray): { real: NDArray; imag: NDArray };
  fft2(arr: NDArray): { real: NDArray; imag: NDArray };
  ifft2(real: NDArray, imag: NDArray): { real: NDArray; imag: NDArray };
  rfft(arr: NDArray): { real: NDArray; imag: NDArray };
  irfft(real: NDArray, imag: NDArray, n?: number): NDArray;
  fftfreq(n: number, d?: number): NDArray;
  rfftfreq(n: number, d?: number): NDArray;
  fftshift(arr: NDArray): NDArray;
  ifftshift(arr: NDArray): NDArray;

  // ============ Additional Random Distributions ============
  exponential(scale: number, shape: number[]): NDArray;
  poisson(lam: number, shape: number[]): NDArray;
  binomial(n: number, p: number, shape: number[]): NDArray;
  beta(a: number, b: number, shape: number[]): NDArray;
  gamma(shape_param: number, scale: number, size: number[]): NDArray;
  lognormal(mean: number, sigma: number, shape: number[]): NDArray;
  chisquare(df: number, shape: number[]): NDArray;
  standardT(df: number, shape: number[]): NDArray;
  multivariateNormal(mean: NDArray, cov: NDArray, size?: number): NDArray;
  geometric(p: number, shape: number[]): NDArray;
  weibull(a: number, shape: number[]): NDArray;
  /** np.random.standard_normal — alias for randn */
  standardNormal(shape: number[]): NDArray;
  /** np.random.standard_cauchy — standard Cauchy distribution */
  standardCauchy(shape: number[]): NDArray;
  /** np.random.multinomial — multinomial distribution */
  multinomial(n: number, pvals: number[], size?: number): NDArray;
  /** np.random.dirichlet — Dirichlet distribution */
  dirichlet(alpha: number[], size?: number): NDArray;
  /** np.random.random — alias for rand */
  random(shape: number[]): NDArray;
  /** np.random.f — F-distribution */
  f(dfnum: number, dfden: number, shape: number[]): NDArray;
  /** np.random.hypergeometric — hypergeometric distribution */
  hypergeometric(ngood: number, nbad: number, nsample: number, shape: number[]): NDArray;
  /** np.random.negative_binomial — negative binomial distribution */
  negativeBinomial(n: number, p: number, shape: number[]): NDArray;
  /** np.random.pareto — Pareto distribution */
  pareto(a: number, shape: number[]): NDArray;
  /** np.random.rayleigh — Rayleigh distribution */
  rayleigh(scale: number, shape: number[]): NDArray;
  /** np.random.triangular — triangular distribution */
  triangular(left: number, mode: number, right: number, shape: number[]): NDArray;
  /** np.random.vonmises — von Mises distribution */
  vonmises(mu: number, kappa: number, shape: number[]): NDArray;
  /** np.random.wald — Wald (inverse Gaussian) distribution */
  wald(mean: number, scale: number, shape: number[]): NDArray;
  /** np.random.zipf — Zipf distribution */
  zipf(a: number, shape: number[]): NDArray;

  // ============ Additional Stats ============
  average(arr: NDArray, weights?: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  ptp(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray;
  digitize(x: NDArray, bins: NDArray, right?: boolean): NDArray;
  nanquantile(
    arr: NDArray,
    q: number,
    axis?: number,
    keepdims?: boolean,
    method?: 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'
  ): number | NDArray;
  nancumsum(arr: NDArray, axis?: number): NDArray;
  nancumprod(arr: NDArray, axis?: number): NDArray;
  uniqueCounts(arr: NDArray): { values: NDArray; counts: NDArray };
  uniqueInverse(arr: NDArray): { values: NDArray; inverse: NDArray };
  histogram2d(
    x: NDArray,
    y: NDArray,
    bins?: number,
    range?: [[number, number], [number, number]] | null,
    density?: boolean,
    weights?: NDArray
  ): { hist: NDArray; xEdges: NDArray; yEdges: NDArray };

  // ============ Additional Comparison ============
  rint(arr: NDArray): NDArray;
  around(arr: NDArray, decimals?: number): NDArray;

  // ============ Additional Polynomial ============
  polyder(p: NDArray, m?: number): NDArray;
  polyint(p: NDArray, m?: number, k?: number): NDArray;
  polydiv(u: NDArray, v: NDArray): { q: NDArray; r: NDArray };
  polysub(a: NDArray, b: NDArray): NDArray;

  // ============ Integration ============
  trapezoid(y: NDArray, x?: NDArray, dx?: number): number;
  /** @deprecated Use trapezoid() instead — np.trapz alias */
  trapz(y: NDArray, x?: NDArray, dx?: number): number;

  // ============ Index Utilities ============
  /** np.unravel_index — convert flat index to multi-dimensional index */
  unravelIndex(indices: NDArray | number, shape: number[]): NDArray[];
  /** np.ravel_multi_index — convert multi-dimensional index to flat index */
  ravelMultiIndex(multiIndex: NDArray[], shape: number[]): NDArray;

  // ============ Integer Math ============
  /** np.gcd — element-wise greatest common divisor */
  gcd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;
  /** np.lcm — element-wise least common multiple */
  lcm(a: ArrayOrScalar, b: ArrayOrScalar): NDArray;

  // ============ Matrix Utilities ============
  /** np.tri — lower-triangular matrix of ones */
  tri(n: number, m?: number, k?: number, dtype?: DType): NDArray;
  /** np.diagflat — create diagonal matrix from flat input */
  diagflat(v: NDArray, k?: number): NDArray;
  /** np.block — assemble arrays from nested blocks */
  block(arrays: (NDArray | NDArray[])[]): NDArray;
  /** np.fill_diagonal — fill diagonal of matrix with a value */
  fillDiagonal(arr: NDArray, val: number, wrap?: boolean): NDArray;

  // ============ Index Arrays ============
  /** np.indices — return an array representing grid indices */
  indices(dimensions: number[], dtype?: DType): NDArray[];
  /** np.diag_indices — indices for main diagonal */
  diagIndices(n: number, ndim?: number): NDArray[];
  /** np.tril_indices — indices for lower triangle */
  trilIndices(n: number, k?: number, m?: number): [NDArray, NDArray];
  /** np.triu_indices — indices for upper triangle */
  triuIndices(n: number, k?: number, m?: number): [NDArray, NDArray];

  // ============ Window Functions ============
  /** np.bartlett — Bartlett window */
  bartlett(M: number): NDArray;
  /** np.blackman — Blackman window */
  blackman(M: number): NDArray;
  /** np.hamming — Hamming window */
  hamming(M: number): NDArray;
  /** np.hanning — Hanning window */
  hanning(M: number): NDArray;
  /** np.kaiser — Kaiser window */
  kaiser(M: number, beta: number): NDArray;

  // ============ Bit Manipulation ============
  /** np.packbits — pack binary array into uint8 */
  packbits(arr: NDArray, axis?: number, bitorder?: 'big' | 'little'): NDArray;
  /** np.unpackbits — unpack uint8 to binary array */
  unpackbits(arr: NDArray, axis?: number, count?: number, bitorder?: 'big' | 'little'): NDArray;

  // ============ Additional Linalg ============
  /** np.linalg.eigvalsh — eigenvalues of symmetric/Hermitian matrix */
  eigvalsh(arr: NDArray): NDArray;

  // ============ N-dimensional FFT ============
  /** np.fft.fftn — n-dimensional FFT */
  fftn(arr: NDArray, shape?: number[]): { real: NDArray; imag: NDArray };
  /** np.fft.ifftn — n-dimensional inverse FFT */
  ifftn(real: NDArray, imag: NDArray, shape?: number[]): { real: NDArray; imag: NDArray };

  // ============ GPU Materialization ============
  materializeAll?(): Promise<void>;
}
