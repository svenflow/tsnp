/**
 * Pure JavaScript reference backend for testing
 *
 * This implements all Backend operations in pure JS, serving as:
 * 1. A reference implementation for testing
 * 2. A fallback when WebGPU is not available
 * 3. A baseline for performance comparisons
 */

import {
  Backend,
  NDArray,
  DType,
  AnyTypedArray,
  ArrayOrScalar,
  BinsParam,
  SortKind,
  DEFAULT_DTYPE,
  createTypedArray,
  createTypedArrayFrom,
} from './types.js';

class JsNDArray implements NDArray {
  data: AnyTypedArray;
  shape: number[];
  dtype: DType;

  constructor(data: AnyTypedArray | number[], shape: number[], dtype: DType = 'float64') {
    this.dtype = dtype;
    if (Array.isArray(data)) {
      this.data = createTypedArrayFrom(dtype, data);
    } else {
      this.data = data;
    }
    this.shape = shape;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  get ndim(): number {
    return this.shape.length;
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get T(): NDArray {
    const ndim = this.shape.length;
    if (ndim <= 1) return this;
    // Reverse axes for transpose
    const perm = [...Array(ndim).keys()].reverse();
    const newShape = perm.map(i => this.shape[i]);
    const size = this.data.length;
    const data = new Float64Array(size);

    const oldStrides = new Array(ndim);
    oldStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1] * this.shape[i + 1];
    }

    const newStrides = new Array(ndim);
    newStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    for (let newFlat = 0; newFlat < size; newFlat++) {
      let remaining = newFlat;
      let oldFlat = 0;
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(remaining / newStrides[d]);
        remaining -= coord * newStrides[d];
        oldFlat += coord * oldStrides[perm[d]];
      }
      data[newFlat] = this.data[oldFlat];
    }

    return new JsNDArray(data, newShape, this.dtype);
  }

  item(): number {
    if (this.data.length !== 1) {
      throw new Error('can only convert an array of size 1 to a scalar');
    }
    return this.data[0];
  }
}

function flattenNestedArray(data: any): { flat: number[]; shape: number[] } {
  if (!Array.isArray(data)) return { flat: [data], shape: [] };
  if (data.length === 0) return { flat: [], shape: [0] };
  if (!Array.isArray(data[0])) return { flat: data, shape: [data.length] };

  const shape: number[] = [];
  let current: any = data;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  function flatten(arr: any): number[] {
    if (!Array.isArray(arr)) return [arr];
    return arr.reduce((acc: number[], item: any) => acc.concat(flatten(item)), []);
  }

  return { flat: flatten(data), shape };
}

export class JsBackend implements Backend {
  name = 'js';

  // ============ Constants ============
  pi = Math.PI;
  e = Math.E;
  inf = Infinity;
  nan = NaN;
  newaxis = null as any;

  // ============ Creation ============

  zeros(shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JsNDArray(createTypedArray(dtype, size), shape, dtype);
  }

  ones(shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    for (let i = 0; i < size; i++) data[i] = 1;
    return new JsNDArray(data, shape, dtype);
  }

  full(shape: number[], value: number, dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    for (let i = 0; i < size; i++) data[i] = value;
    return new JsNDArray(data, shape, dtype);
  }

  arange(startOrStop: number, stop?: number, step?: number, dtype: DType = 'float64'): NDArray {
    let start: number;
    let actualStop: number;
    let actualStep: number;
    if (stop === undefined) {
      start = 0;
      actualStop = startOrStop;
      actualStep = 1;
    } else if (step === undefined) {
      start = startOrStop;
      actualStop = stop;
      actualStep = 1;
    } else {
      start = startOrStop;
      actualStop = stop;
      actualStep = step;
    }
    if (actualStep === 0) {
      throw new Error('step cannot be zero');
    }
    const data: number[] = [];
    if (actualStep > 0) {
      for (let x = start; x < actualStop; x += actualStep) {
        data.push(x);
      }
    } else {
      for (let x = start; x > actualStop; x += actualStep) {
        data.push(x);
      }
    }
    return new JsNDArray(createTypedArrayFrom(dtype, data), [data.length], dtype);
  }

  linspace(
    start: number,
    stop: number,
    num: number,
    endpoint?: boolean | DType,
    dtype: DType = 'float64'
  ): NDArray {
    // Backward compat: linspace(0, 1, 5, 'float32') where 4th arg is dtype
    if (typeof endpoint === 'string') {
      dtype = endpoint as DType;
      endpoint = true;
    }
    if (endpoint === undefined) endpoint = true;
    if (num === 0) return new JsNDArray(createTypedArray(dtype, 0), [0], dtype);
    if (num === 1) return new JsNDArray(createTypedArrayFrom(dtype, [start]), [1], dtype);
    const step = endpoint ? (stop - start) / (num - 1) : (stop - start) / num;
    const data: number[] = [];
    for (let i = 0; i < num; i++) {
      data.push(start + i * step);
    }
    return new JsNDArray(createTypedArrayFrom(dtype, data), [num], dtype);
  }

  eye(n: number, M?: number | DType, k: number = 0, dtype: DType = 'float64'): NDArray {
    // Handle backward compat: eye(3, 'float32') where 2nd arg is dtype
    let m: number;
    if (typeof M === 'string') {
      dtype = M;
      m = n;
    } else {
      m = M ?? n;
    }
    const data = createTypedArray(dtype, n * m);
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    return new JsNDArray(data, [n, m], dtype);
  }

  diag(arr: NDArray, k: number = 0): NDArray {
    if (arr.shape.length === 1) {
      // Create diagonal matrix from vector
      const n = arr.shape[0] + Math.abs(k);
      const data = new Float64Array(n * n);
      for (let i = 0; i < arr.shape[0]; i++) {
        const row = k >= 0 ? i : i - k;
        const col = k >= 0 ? i + k : i;
        data[row * n + col] = arr.data[i];
      }
      return new JsNDArray(data, [n, n]);
    } else if (arr.shape.length === 2) {
      // Extract diagonal from matrix
      const [rows, cols] = arr.shape;
      const startRow = k >= 0 ? 0 : -k;
      const startCol = k >= 0 ? k : 0;
      const diagLen = Math.min(rows - startRow, cols - startCol);
      const data: number[] = [];
      for (let i = 0; i < diagLen; i++) {
        data.push(arr.data[(startRow + i) * cols + (startCol + i)]);
      }
      return new JsNDArray(data, [data.length]);
    }
    throw new Error('diag requires 1D or 2D array');
  }

  array(data: number[] | number[][] | any[], shape?: number[], dtype: DType = 'float64'): NDArray {
    let flatData: number[];
    let inferredShape: number[];

    if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
      const result = flattenNestedArray(data);
      flatData = result.flat;
      inferredShape = shape || result.shape;
    } else {
      flatData = data as number[];
      inferredShape = shape || [flatData.length];
    }
    return new JsNDArray(createTypedArrayFrom(dtype, flatData), inferredShape, dtype);
  }

  asarray(a: NDArray | number[] | number, dtype?: DType): NDArray {
    if (typeof a === 'number') {
      const d = dtype || DEFAULT_DTYPE;
      return new JsNDArray(createTypedArrayFrom(d, [a]), [1], d);
    }
    if (Array.isArray(a)) {
      const d = dtype || DEFAULT_DTYPE;
      return new JsNDArray(createTypedArrayFrom(d, a), [a.length], d);
    }
    // Already an NDArray
    if (dtype && a.dtype !== dtype) {
      return this.astype(a, dtype);
    }
    return a;
  }

  // ============ Creation (Additional) ============

  fromfunction(
    fn: (...coords: number[]) => number,
    shape: number[],
    dtype: DType = 'float64'
  ): NDArray {
    const total = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, total);
    const strides = new Array(shape.length);
    strides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
    for (let flatIdx = 0; flatIdx < total; flatIdx++) {
      const coords: number[] = [];
      let rem = flatIdx;
      for (let d = 0; d < shape.length; d++) {
        coords.push(Math.floor(rem / strides[d]));
        rem = rem % strides[d];
      }
      data[flatIdx] = fn(...coords);
    }
    return new JsNDArray(data, [...shape], dtype);
  }

  fromiter(iter: Iterable<number>, count?: number, dtype: DType = 'float64'): NDArray {
    const values: number[] = [];
    let n = 0;
    for (const v of iter) {
      values.push(v);
      n++;
      if (count !== undefined && n >= count) break;
    }
    return new JsNDArray(createTypedArrayFrom(dtype, values), [values.length], dtype);
  }

  // ============ Complex Helpers ============
  // For real-valued arrays (Float64Array), these are trivial.
  // real() returns a copy, imag() returns zeros, conj() returns a copy.

  real(arr: NDArray): NDArray {
    return new JsNDArray(new Float64Array(arr.data), [...arr.shape]);
  }

  imag(arr: NDArray): NDArray {
    return new JsNDArray(new Float64Array(arr.data.length), [...arr.shape]);
  }

  conj(arr: NDArray): NDArray {
    // For real arrays, conjugate is identity
    return new JsNDArray(new Float64Array(arr.data), [...arr.shape]);
  }

  // ============ Type Casting ============

  astype(arr: NDArray, dtype: DType): NDArray {
    const data = createTypedArrayFrom(dtype, Array.from(arr.data));
    return new JsNDArray(data, [...arr.shape], dtype);
  }

  // ============ Math - Unary ============

  sin(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sin), arr.shape);
  }

  cos(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cos), arr.shape);
  }

  tan(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.tan), arr.shape);
  }

  arcsin(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.asin), arr.shape);
  }

  arccos(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.acos), arr.shape);
  }

  arctan(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.atan), arr.shape);
  }

  sinh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sinh), arr.shape);
  }

  cosh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cosh), arr.shape);
  }

  tanh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.tanh), arr.shape);
  }

  exp(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.exp), arr.shape);
  }

  log(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log), arr.shape);
  }

  log2(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log2), arr.shape);
  }

  log10(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log10), arr.shape);
  }

  sqrt(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sqrt), arr.shape);
  }

  cbrt(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cbrt), arr.shape);
  }

  abs(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.abs), arr.shape);
  }

  sign(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sign), arr.shape);
  }

  floor(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.floor), arr.shape);
  }

  ceil(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.ceil), arr.shape);
  }

  round(arr: NDArray, decimals: number = 0): NDArray {
    if (decimals === 0) {
      return new JsNDArray(arr.data.map(Math.round), arr.shape);
    }
    const factor = Math.pow(10, decimals);
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.round(arr.data[i] * factor) / factor;
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  negative(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => -x),
      arr.shape
    );
  }

  /** @deprecated Use negative() instead */
  neg(arr: NDArray): NDArray {
    return this.negative(arr);
  }

  reciprocal(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => 1 / x),
      arr.shape
    );
  }

  square(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => x * x),
      arr.shape
    );
  }

  // ============ Math - Unary (Extended) ============

  arcsinh(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.asinh(x)),
      arr.shape
    );
  }

  arccosh(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.acosh(x)),
      arr.shape
    );
  }

  arctanh(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.atanh(x)),
      arr.shape
    );
  }

  expm1(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.expm1(x)),
      arr.shape
    );
  }

  log1p(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.log1p(x)),
      arr.shape
    );
  }

  trunc(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.trunc(x)),
      arr.shape
    );
  }

  sinc(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => {
        if (x === 0) return 1;
        const px = Math.PI * x;
        return Math.sin(px) / px;
      }),
      arr.shape
    );
  }

  deg2rad(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => (x * Math.PI) / 180),
      arr.shape
    );
  }

  rad2deg(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => (x * 180) / Math.PI),
      arr.shape
    );
  }

  heaviside(arr: NDArray, h0: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => {
        if (x < 0) return 0;
        if (x === 0) return h0;
        return 1;
      }),
      arr.shape
    );
  }

  fix(arr: NDArray): NDArray {
    // Same as trunc - round toward zero
    return this.trunc(arr);
  }

  signbit(arr: NDArray): NDArray {
    // Returns 1.0 if sign bit is set (negative or -0), 0.0 otherwise
    // NumPy: signbit(NaN) = False, signbit(-0) = True, signbit(-inf) = True
    return new JsNDArray(
      arr.data.map(x => {
        if (Number.isNaN(x)) return 0; // NumPy returns False for NaN
        if (Object.is(x, -0)) return 1; // -0 has sign bit set
        return x < 0 ? 1 : 0;
      }),
      arr.shape
    );
  }

  // ============ Math - Decomposition ============

  modf(arr: NDArray): { frac: NDArray; integ: NDArray } {
    // Returns fractional and integral parts, both with same sign as input
    const frac = new Float64Array(arr.data.length);
    const integ = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      const x = arr.data[i];
      integ[i] = Math.trunc(x);
      frac[i] = x - integ[i];
    }
    return {
      frac: new JsNDArray(frac, arr.shape),
      integ: new JsNDArray(integ, arr.shape),
    };
  }

  frexp(arr: NDArray): { mantissa: NDArray; exponent: NDArray } {
    // Decompose into mantissa * 2^exponent where 0.5 <= |mantissa| < 1
    const mantissa = new Float64Array(arr.data.length);
    const exponent = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      const x = arr.data[i];
      if (x === 0 || !Number.isFinite(x) || Number.isNaN(x)) {
        mantissa[i] = x;
        exponent[i] = 0;
      } else {
        const exp = Math.floor(Math.log2(Math.abs(x))) + 1;
        mantissa[i] = x / Math.pow(2, exp);
        exponent[i] = exp;
      }
    }
    return {
      mantissa: new JsNDArray(mantissa, arr.shape),
      exponent: new JsNDArray(exponent, arr.shape),
    };
  }

  ldexp(arr: NDArray, exp: NDArray): NDArray {
    // Compute arr * 2^exp
    this._checkSameShape(arr, exp);
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      data[i] = arr.data[i] * Math.pow(2, exp.data[i]);
    }
    return new JsNDArray(data, arr.shape);
  }

  divmod(a: ArrayOrScalar, b: ArrayOrScalar): { quotient: NDArray; remainder: NDArray } {
    const arrA = this._toNDArray(a);
    const arrB = this._toNDArray(b);
    // Determine output shape via scalar broadcasting
    let len: number;
    let shape: number[];
    if (arrA.data.length === 1 && arrB.data.length > 1) {
      len = arrB.data.length;
      shape = [...arrB.shape];
    } else if (arrB.data.length === 1 && arrA.data.length > 1) {
      len = arrA.data.length;
      shape = [...arrA.shape];
    } else {
      this._checkSameShape(arrA, arrB);
      len = arrA.data.length;
      shape = [...arrA.shape];
    }
    const quotient = new Float64Array(len);
    const remainder = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      const av = arrA.data.length === 1 ? arrA.data[0] : arrA.data[i];
      const bv = arrB.data.length === 1 ? arrB.data[0] : arrB.data[i];
      quotient[i] = Math.floor(av / bv);
      const r = av % bv;
      remainder[i] = r !== 0 && Math.sign(r) !== Math.sign(bv) ? r + bv : r;
    }
    return {
      quotient: new JsNDArray(quotient, shape),
      remainder: new JsNDArray(remainder, shape),
    };
  }

  // ============ Math - Binary (Extended) ============

  mod(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      const r = x % y;
      return r !== 0 && Math.sign(r) !== Math.sign(y) ? r + y : r;
    });
  }

  fmod(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x % y);
  }

  remainder(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this.mod(a, b); // Same as mod in NumPy
  }

  copysign(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      const magnitude = Math.abs(x);
      const bNegative = y < 0 || Object.is(y, -0);
      return bNegative ? -magnitude : magnitude;
    });
  }

  hypot(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.hypot(x, y));
  }

  arctan2(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.atan2(x, y));
  }

  logaddexp(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      const mx = Math.max(x, y);
      if (mx === -Infinity) return -Infinity;
      return mx + Math.log(Math.exp(x - mx) + Math.exp(y - mx));
    });
  }

  logaddexp2(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    const log2 = Math.log(2);
    return this._binaryOp(a, b, (x, y) => {
      const mx = Math.max(x, y);
      if (mx === -Infinity) return -Infinity;
      return mx + Math.log(Math.pow(2, x - mx) + Math.pow(2, y - mx)) / log2;
    });
  }

  fmax(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      if (Number.isNaN(x)) return y;
      if (Number.isNaN(y)) return x;
      return Math.max(x, y);
    });
  }

  fmin(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      if (Number.isNaN(x)) return y;
      if (Number.isNaN(y)) return x;
      return Math.min(x, y);
    });
  }

  // ============ Comparison ============

  equal(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x === y ? 1 : 0));
  }

  notEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x !== y ? 1 : 0));
  }

  less(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x < y ? 1 : 0));
  }

  lessEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x <= y ? 1 : 0));
  }

  greater(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x > y ? 1 : 0));
  }

  greaterEqual(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x >= y ? 1 : 0));
  }

  isnan(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => (Number.isNaN(x) ? 1 : 0)),
      arr.shape
    );
  }

  isinf(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => (!Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0)),
      arr.shape
    );
  }

  isfinite(arr: NDArray): NDArray {
    return new JsNDArray(
      arr.data.map(x => (Number.isFinite(x) ? 1 : 0)),
      arr.shape
    );
  }

  // ============ Set Operations ============

  setdiff1d(a: NDArray, b: NDArray): NDArray {
    const setB = new Set(b.data);
    const result = Array.from(a.data).filter(x => !setB.has(x));
    const unique = [...new Set(result)].sort((x, y) => x - y);
    return new JsNDArray(new Float64Array(unique), [unique.length]);
  }

  union1d(a: NDArray, b: NDArray): NDArray {
    const combined = new Set([...a.data, ...b.data]);
    const result = [...combined].sort((x, y) => x - y);
    return new JsNDArray(new Float64Array(result), [result.length]);
  }

  intersect1d(a: NDArray, b: NDArray): NDArray {
    const setB = new Set(b.data);
    const result = [...new Set(Array.from(a.data).filter(x => setB.has(x)))].sort((x, y) => x - y);
    return new JsNDArray(new Float64Array(result), [result.length]);
  }

  isin(element: NDArray, testElements: NDArray): NDArray {
    const testSet = new Set(testElements.data);
    return new JsNDArray(
      element.data.map(x => (testSet.has(x) ? 1 : 0)),
      element.shape
    );
  }

  // ============ Array Manipulation (Extended) ============

  insert(arr: NDArray, index: number, values: NDArray | number, axis?: number): NDArray {
    if (axis === undefined) {
      // Insert into flattened array
      const flat = Array.from(this.flatten(arr).data);
      const toInsert = typeof values === 'number' ? [values] : Array.from(values.data);
      if (index < 0) index = flat.length + index + 1;
      flat.splice(index, 0, ...toInsert);
      return new JsNDArray(new Float64Array(flat), [flat.length]);
    }
    throw new Error('insert with axis not yet implemented');
  }

  deleteArr(arr: NDArray, index: number | number[], axis?: number): NDArray {
    if (axis === undefined) {
      const flat = Array.from(this.flatten(arr).data);
      const indices = Array.isArray(index) ? index : [index];
      const normalized = indices.map(i => (i < 0 ? flat.length + i : i)).sort((a, b) => b - a);
      for (const i of normalized) {
        flat.splice(i, 1);
      }
      return new JsNDArray(new Float64Array(flat), [flat.length]);
    }
    throw new Error('delete with axis not yet implemented');
  }

  append(arr: NDArray, values: NDArray, axis?: number): NDArray {
    if (axis === undefined) {
      const flat1 = this.flatten(arr);
      const flat2 = this.flatten(values);
      const result = new Float64Array(flat1.data.length + flat2.data.length);
      result.set(flat1.data);
      result.set(flat2.data, flat1.data.length);
      return new JsNDArray(result, [result.length]);
    }
    return this.concatenate([arr, values], axis);
  }

  atleast1d(arr: NDArray): NDArray {
    if (arr.shape.length === 0) {
      return new JsNDArray(arr.data, [1]);
    }
    return arr;
  }

  atleast2d(arr: NDArray): NDArray {
    if (arr.shape.length === 0) {
      return new JsNDArray(arr.data, [1, 1]);
    }
    if (arr.shape.length === 1) {
      return new JsNDArray(arr.data, [1, arr.shape[0]]);
    }
    return arr;
  }

  atleast3d(arr: NDArray): NDArray {
    if (arr.shape.length === 0) {
      return new JsNDArray(arr.data, [1, 1, 1]);
    }
    if (arr.shape.length === 1) {
      return new JsNDArray(arr.data, [1, arr.shape[0], 1]);
    }
    if (arr.shape.length === 2) {
      return new JsNDArray(arr.data, [arr.shape[0], arr.shape[1], 1]);
    }
    return arr;
  }

  countNonzero(arr: NDArray, axis?: number, keepdims?: boolean): NDArray | number {
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

    const out = new JsNDArray(result, outShape);
    if (keepdims) {
      const newShape = [...arr.shape];
      newShape[normalizedAxis] = 1;
      return this.reshape(out, newShape);
    }
    return out;
  }

  // ============ Advanced Linalg ============

  matrixPower(arr: NDArray, n: number): NDArray {
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

  kron(a: NDArray, b: NDArray): NDArray {
    const aFlat = a.shape.length === 1 ? this.reshape(a, [a.shape[0], 1]) : a;
    const bFlat = b.shape.length === 1 ? this.reshape(b, [b.shape[0], 1]) : b;

    if (aFlat.shape.length !== 2 || bFlat.shape.length !== 2) {
      throw new Error('kron requires 1D or 2D arrays');
    }

    const [am, an] = aFlat.shape;
    const [bm, bn] = bFlat.shape;
    const outShape = [am * bm, an * bn];
    const result = new Float64Array(outShape[0] * outShape[1]);

    for (let i = 0; i < am; i++) {
      for (let j = 0; j < an; j++) {
        const aVal = aFlat.data[i * an + j];
        for (let k = 0; k < bm; k++) {
          for (let l = 0; l < bn; l++) {
            const outRow = i * bm + k;
            const outCol = j * bn + l;
            result[outRow * outShape[1] + outCol] = aVal * bFlat.data[k * bn + l];
          }
        }
      }
    }

    return new JsNDArray(result, outShape);
  }

  cond(arr: NDArray, _p: number | 'fro' = 2): number {
    if (arr.shape.length !== 2) {
      throw new Error('cond requires a 2D matrix');
    }
    // Compute condition number using SVD
    const { s } = this.svd(arr);
    const sData = s.data;
    const sMax = Math.max(...sData);
    const sMin = Math.min(...sData.filter(v => v > 0)); // Exclude zeros
    if (sMin === 0 || sData.length === 0) {
      return Infinity;
    }
    return sMax / sMin;
  }

  slogdet(arr: NDArray): { sign: number; logabsdet: number } {
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

  multiDot(arrays: NDArray[]): NDArray {
    if (arrays.length === 0) {
      throw new Error('multiDot requires at least one array');
    }
    if (arrays.length === 1) {
      return new JsNDArray(arrays[0].data.slice(), arrays[0].shape);
    }
    // Simple left-to-right multiplication (could optimize with dynamic programming)
    let result = arrays[0];
    for (let i = 1; i < arrays.length; i++) {
      result = this.matmul(result, arrays[i]);
    }
    return result;
  }

  // ============ Polynomial ============

  polyval(p: NDArray, x: NDArray): NDArray {
    const coeffs = this.flatten(p).data;
    return new JsNDArray(
      x.data.map(xi => {
        let result = 0;
        for (let i = 0; i < coeffs.length; i++) {
          result = result * xi + coeffs[i];
        }
        return result;
      }),
      x.shape
    );
  }

  polyadd(a: NDArray, b: NDArray): NDArray {
    const aCoeffs = Array.from(this.flatten(a).data);
    const bCoeffs = Array.from(this.flatten(b).data);
    const maxLen = Math.max(aCoeffs.length, bCoeffs.length);
    const result = new Float64Array(maxLen);
    // Pad shorter array with zeros at the front
    const aPadded = new Array(maxLen - aCoeffs.length).fill(0).concat(aCoeffs);
    const bPadded = new Array(maxLen - bCoeffs.length).fill(0).concat(bCoeffs);
    for (let i = 0; i < maxLen; i++) {
      result[i] = aPadded[i] + bPadded[i];
    }
    return new JsNDArray(result, [maxLen]);
  }

  polymul(a: NDArray, b: NDArray): NDArray {
    const aCoeffs = Array.from(this.flatten(a).data);
    const bCoeffs = Array.from(this.flatten(b).data);
    const resultLen = aCoeffs.length + bCoeffs.length - 1;
    const result = new Float64Array(resultLen);
    for (let i = 0; i < aCoeffs.length; i++) {
      for (let j = 0; j < bCoeffs.length; j++) {
        result[i + j] += aCoeffs[i] * bCoeffs[j];
      }
    }
    return new JsNDArray(result, [resultLen]);
  }

  polyfit(x: NDArray, y: NDArray, deg: number): NDArray {
    // Least squares polynomial fit
    // Build Vandermonde matrix for x: [[x_i^deg, x_i^(deg-1), ..., x_i^0], ...]
    const xData = this.flatten(x).data;
    const yData = this.flatten(y).data;
    const n = xData.length;
    const m = deg + 1;

    // Vandermonde matrix V[i,j] = x[i]^(deg-j)
    const V = new Float64Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        V[i * m + j] = Math.pow(xData[i], deg - j);
      }
    }

    // Solve least squares using normal equations: (V^T V) c = V^T y
    // V^T V is (m x m), V^T y is (m x 1)
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

    // Solve VtV * c = Vty using Gaussian elimination
    const A = Array.from(VtV);
    const b = Array.from(Vty);

    // Forward elimination with partial pivoting
    for (let k = 0; k < m; k++) {
      // Find pivot
      let maxVal = Math.abs(A[k * m + k]);
      let maxRow = k;
      for (let i = k + 1; i < m; i++) {
        if (Math.abs(A[i * m + k]) > maxVal) {
          maxVal = Math.abs(A[i * m + k]);
          maxRow = i;
        }
      }
      // Swap rows
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
      // Eliminate
      for (let i = k + 1; i < m; i++) {
        const factor = A[i * m + k] / A[k * m + k];
        for (let j = k; j < m; j++) {
          A[i * m + j] -= factor * A[k * m + j];
        }
        b[i] -= factor * b[k];
      }
    }

    // Back substitution
    const c = new Float64Array(m);
    for (let i = m - 1; i >= 0; i--) {
      let sum = b[i];
      for (let j = i + 1; j < m; j++) {
        sum -= A[i * m + j] * c[j];
      }
      c[i] = sum / A[i * m + i];
    }

    return new JsNDArray(c, [m]);
  }

  roots(p: NDArray): NDArray {
    // Find roots of polynomial using companion matrix eigenvalues
    const coeffs = Array.from(this.flatten(p).data);

    // Remove leading zeros
    while (coeffs.length > 0 && coeffs[0] === 0) {
      coeffs.shift();
    }

    if (coeffs.length === 0) {
      return new JsNDArray(new Float64Array(0), [0]);
    }
    if (coeffs.length === 1) {
      return new JsNDArray(new Float64Array(0), [0]);
    }

    const n = coeffs.length - 1;

    // Normalize by leading coefficient
    const a0 = coeffs[0];
    for (let i = 0; i < coeffs.length; i++) {
      coeffs[i] /= a0;
    }

    // Build companion matrix
    // [0, 0, ..., 0, -c_n]
    // [1, 0, ..., 0, -c_{n-1}]
    // [0, 1, ..., 0, -c_{n-2}]
    // ...
    // [0, 0, ..., 1, -c_1]
    const C = new Float64Array(n * n);
    for (let i = 1; i < n; i++) {
      C[i * n + (i - 1)] = 1;
    }
    for (let i = 0; i < n; i++) {
      C[i * n + (n - 1)] = -coeffs[n - i];
    }

    // Use QR iteration to find eigenvalues (simplified for real roots)
    // This is a basic implementation - for complex roots, would need complex arithmetic
    const maxIter = 100;
    const tol = 1e-10;
    const roots: number[] = [];

    // Create a working copy
    const H = Array.from(C);

    // Use deflation and QR iteration
    let size = n;
    for (let deflation = 0; deflation < n; deflation++) {
      if (size === 0) break;

      if (size === 1) {
        roots.push(H[0]);
        break;
      }

      // Simple QR iteration with shifts
      for (let iter = 0; iter < maxIter; iter++) {
        // Check for convergence (bottom-left element near zero)
        const bottomLeft = Math.abs(H[(size - 1) * n + (size - 2)]);
        const diag1 = Math.abs(H[(size - 2) * n + (size - 2)]);
        const diag2 = Math.abs(H[(size - 1) * n + (size - 1)]);

        if (bottomLeft < tol * (diag1 + diag2 + tol)) {
          // Eigenvalue found
          roots.push(H[(size - 1) * n + (size - 1)]);
          size--;
          break;
        }

        // Wilkinson shift
        const d = (H[(size - 2) * n + (size - 2)] - H[(size - 1) * n + (size - 1)]) / 2;
        const bElem = H[(size - 1) * n + (size - 2)];
        const shift =
          H[(size - 1) * n + (size - 1)] -
          (bElem * bElem) / (d + Math.sign(d || 1) * Math.sqrt(d * d + bElem * bElem));

        // Apply shift
        for (let i = 0; i < size; i++) {
          H[i * n + i] -= shift;
        }

        // QR step via Givens rotations
        for (let i = 0; i < size - 1; i++) {
          const a = H[i * n + i];
          const b = H[(i + 1) * n + i];
          const r = Math.sqrt(a * a + b * b);
          if (r < tol) continue;
          const c = a / r;
          const s = b / r;

          // Apply rotation to H from left
          for (let j = 0; j < size; j++) {
            const t1 = H[i * n + j];
            const t2 = H[(i + 1) * n + j];
            H[i * n + j] = c * t1 + s * t2;
            H[(i + 1) * n + j] = -s * t1 + c * t2;
          }

          // Apply rotation to H from right
          for (let j = 0; j < size; j++) {
            const t1 = H[j * n + i];
            const t2 = H[j * n + (i + 1)];
            H[j * n + i] = c * t1 + s * t2;
            H[j * n + (i + 1)] = -s * t1 + c * t2;
          }
        }

        // Remove shift
        for (let i = 0; i < size; i++) {
          H[i * n + i] += shift;
        }
      }

      if (roots.length === deflation) {
        // Did not converge - use diagonal element as approximation
        roots.push(H[(size - 1) * n + (size - 1)]);
        size--;
      }
    }

    // Sort roots
    roots.sort((a, b) => b - a);

    return new JsNDArray(new Float64Array(roots), [roots.length]);
  }

  // ============ Interpolation ============

  interp(x: NDArray, xp: NDArray, fp: NDArray): NDArray {
    const xpData = this.flatten(xp).data;
    const fpData = this.flatten(fp).data;

    return new JsNDArray(
      x.data.map(xi => {
        // Binary search for interval
        if (xi <= xpData[0]) return fpData[0];
        if (xi >= xpData[xpData.length - 1]) return fpData[fpData.length - 1];

        let lo = 0,
          hi = xpData.length - 1;
        while (hi - lo > 1) {
          const mid = Math.floor((lo + hi) / 2);
          if (xpData[mid] <= xi) lo = mid;
          else hi = mid;
        }

        // Linear interpolation
        const t = (xi - xpData[lo]) / (xpData[hi] - xpData[lo]);
        return fpData[lo] + t * (fpData[hi] - fpData[lo]);
      }),
      x.shape
    );
  }

  // ============ Histogram ============

  bincount(x: NDArray, weights?: NDArray, minlength?: number): NDArray {
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

    for (let i = 0; i < xFlat.length; i++) {
      const bin = Math.floor(xFlat[i]);
      result[bin] += wFlat ? wFlat[i] : 1;
    }

    return new JsNDArray(result, [outLen]);
  }

  // ============ Advanced Indexing ============

  partition(arr: NDArray, kth: number, axis: number = -1): NDArray {
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

      // Extract slice along axis
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

      // Quickselect-style partition
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

      // Write back partitioned values
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

    return new JsNDArray(result, arr.shape);
  }

  argpartition(arr: NDArray, kth: number, axis: number = -1): NDArray {
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

      // Extract slice indices along axis
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

      // Quickselect-style partition
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

      // Write back indices
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

    return new JsNDArray(result, arr.shape);
  }

  lexsort(keys: NDArray[]): NDArray {
    // Sort by multiple keys - last key is primary sort key
    if (keys.length === 0) {
      return new JsNDArray(new Float64Array(0), [0]);
    }

    const n = keys[0].data.length;
    for (const key of keys) {
      if (key.data.length !== n) {
        throw new Error('all keys must have the same length');
      }
    }

    // Create indices array
    const indices = Array.from({ length: n }, (_, i) => i);

    // Sort indices using lexicographic comparison (last key is primary)
    indices.sort((a, b) => {
      for (let k = keys.length - 1; k >= 0; k--) {
        const va = keys[k].data[a];
        const vb = keys[k].data[b];
        if (va < vb) return -1;
        if (va > vb) return 1;
      }
      return 0;
    });

    return new JsNDArray(new Float64Array(indices), [n]);
  }

  compress(condition: NDArray, arr: NDArray, axis?: number): NDArray {
    const condFlat = this.flatten(condition).data;

    if (axis === undefined) {
      // Compress flattened array
      const arrFlat = this.flatten(arr).data;
      const result: number[] = [];
      const len = Math.min(condFlat.length, arrFlat.length);
      for (let i = 0; i < len; i++) {
        if (condFlat[i] !== 0) {
          result.push(arrFlat[i]);
        }
      }
      return new JsNDArray(new Float64Array(result), [result.length]);
    }

    const ndim = arr.shape.length;
    axis = axis < 0 ? axis + ndim : axis;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    // Count true conditions
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

    // Build mapping from compressed index to original index
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

      // Map compressed axis coordinate to original
      coords[axis] = mapping[coords[axis]];

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return new JsNDArray(result, outShape);
  }

  extract(condition: NDArray, arr: NDArray): NDArray {
    const condFlat = this.flatten(condition).data;
    const arrFlat = this.flatten(arr).data;
    const result: number[] = [];

    const len = Math.min(condFlat.length, arrFlat.length);
    for (let i = 0; i < len; i++) {
      if (condFlat[i] !== 0) {
        result.push(arrFlat[i]);
      }
    }

    return new JsNDArray(new Float64Array(result), [result.length]);
  }

  place(arr: NDArray, mask: NDArray, vals: NDArray): void {
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

  select(condlist: NDArray[], choicelist: NDArray[], defaultVal: number = 0): NDArray {
    if (condlist.length !== choicelist.length) {
      throw new Error('condlist and choicelist must have same length');
    }
    if (condlist.length === 0) {
      throw new Error('condlist and choicelist must not be empty');
    }

    // All arrays must be broadcastable to same shape
    const allArrays = [...condlist, ...choicelist];
    const broadcasted = this.broadcastArrays(...allArrays);
    const shape = broadcasted[0].shape;
    const size = broadcasted[0].data.length;

    const conditions = broadcasted.slice(0, condlist.length);
    const choices = broadcasted.slice(condlist.length);

    const result = new Float64Array(size).fill(defaultVal);

    // Process conditions in reverse order (last condition has highest priority)
    for (let c = condlist.length - 1; c >= 0; c--) {
      for (let i = 0; i < size; i++) {
        if (conditions[c].data[i] !== 0) {
          result[i] = choices[c].data[i];
        }
      }
    }

    // Now process in forward order so first true condition wins
    const selected = new Uint8Array(size);
    for (let c = 0; c < condlist.length; c++) {
      for (let i = 0; i < size; i++) {
        if (!selected[i] && conditions[c].data[i] !== 0) {
          result[i] = choices[c].data[i];
          selected[i] = 1;
        }
      }
    }

    return new JsNDArray(result, shape);
  }

  // ============ Math - Binary ============

  add(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x + y);
  }

  subtract(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x - y);
  }

  /** @deprecated Use subtract() instead */
  sub(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this.subtract(a, b);
  }

  multiply(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x * y);
  }

  /** @deprecated Use multiply() instead */
  mul(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this.multiply(a, b);
  }

  divide(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x / y);
  }

  /** @deprecated Use divide() instead */
  div(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this.divide(a, b);
  }

  power(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.pow(x, y));
  }

  /** @deprecated Use power() instead */
  pow(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this.power(a, b);
  }

  floorDivide(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.floor(x / y));
  }

  maximum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.max(x, y));
  }

  minimum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => Math.min(x, y));
  }

  // ============ Math - Scalar ============

  addScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => x + scalar),
      arr.shape
    );
  }

  subScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => x - scalar),
      arr.shape
    );
  }

  mulScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => x * scalar),
      arr.shape
    );
  }

  divScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => x / scalar),
      arr.shape
    );
  }

  powScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(
      arr.data.map(x => Math.pow(x, scalar)),
      arr.shape
    );
  }

  clip(arr: NDArray, min: number | null, max: number | null): NDArray {
    return new JsNDArray(
      arr.data.map(x => {
        let v = x;
        if (min !== null) v = Math.max(v, min);
        if (max !== null) v = Math.min(v, max);
        return v;
      }),
      arr.shape
    );
  }

  // ============ Stats ============

  sum(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.sumAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const d = arr.data;
    let s = 0;
    for (let i = 0; i < d.length; i++) s += d[i];
    return s;
  }

  prod(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.prodAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const d = arr.data;
    let p = 1;
    for (let i = 0; i < d.length; i++) p *= d[i];
    return p;
  }

  mean(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.meanAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) return NaN;
    return (this.sum(arr) as number) / arr.data.length;
  }

  var(arr: NDArray, axis?: number | null, ddof: number = 0, keepdims?: boolean): number | NDArray {
    if (axis !== undefined && axis !== null) {
      const result = this.varAxis(arr, axis, ddof);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) return NaN;
    const m = this.mean(arr) as number;
    const d = arr.data;
    let sumSq = 0;
    for (let i = 0; i < d.length; i++) sumSq += (d[i] - m) ** 2;
    return sumSq / (d.length - ddof);
  }

  std(arr: NDArray, axis?: number | null, ddof: number = 0, keepdims?: boolean): number | NDArray {
    if (axis !== undefined && axis !== null) {
      const result = this.stdAxis(arr, axis, ddof);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    return Math.sqrt(this.var(arr, null, ddof) as number);
  }

  min(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.minAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.min(...arr.data);
  }

  max(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.maxAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.max(...arr.data);
  }

  argmin(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.argminAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    let minIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < arr.data[minIdx]) minIdx = i;
    }
    return minIdx;
  }

  argmax(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this.argmaxAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    let maxIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > arr.data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  cumsum(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) return this.cumsumAxis(arr, axis);
    const data = new Float64Array(arr.data.length);
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      sum += arr.data[i];
      data[i] = sum;
    }
    return new JsNDArray(data, arr.shape);
  }

  cumprod(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) return this.cumprodAxis(arr, axis);
    const data = new Float64Array(arr.data.length);
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      prod *= arr.data[i];
      data[i] = prod;
    }
    return new JsNDArray(data, arr.shape);
  }

  all(arr: NDArray, axis?: number, keepdims?: boolean): boolean | NDArray {
    if (axis !== undefined) {
      const result = this.allAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    return arr.data.every(x => x !== 0);
  }

  any(arr: NDArray, axis?: number, keepdims?: boolean): boolean | NDArray {
    if (axis !== undefined) {
      const result = this.anyAxis(arr, axis);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    return arr.data.some(x => x !== 0);
  }

  sumAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('sumAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      // Sum along rows, result is [cols]
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          data[j] += arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      // Sum along cols, result is [rows]
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i] += arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [rows]);
    }
  }

  meanAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('meanAxis only supports 2D');
    const sumResult = this.sumAxis(arr, axis);
    const divisor = arr.shape[axis];
    return new JsNDArray(
      sumResult.data.map(x => x / divisor),
      sumResult.shape
    );
  }

  minAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('minAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        data[j] = arr.data[j];
        for (let i = 1; i < rows; i++) {
          data[j] = Math.min(data[j], arr.data[i * cols + j]);
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        data[i] = arr.data[i * cols];
        for (let j = 1; j < cols; j++) {
          data[i] = Math.min(data[i], arr.data[i * cols + j]);
        }
      }
      return new JsNDArray(data, [rows]);
    }
  }

  maxAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('maxAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        data[j] = arr.data[j];
        for (let i = 1; i < rows; i++) {
          data[j] = Math.max(data[j], arr.data[i * cols + j]);
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        data[i] = arr.data[i * cols];
        for (let j = 1; j < cols; j++) {
          data[i] = Math.max(data[i], arr.data[i * cols + j]);
        }
      }
      return new JsNDArray(data, [rows]);
    }
  }

  argminAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('argminAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        let minIdx = 0;
        for (let i = 1; i < rows; i++) {
          if (arr.data[i * cols + j] < arr.data[minIdx * cols + j]) minIdx = i;
        }
        data[j] = minIdx;
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let minIdx = 0;
        for (let j = 1; j < cols; j++) {
          if (arr.data[i * cols + j] < arr.data[i * cols + minIdx]) minIdx = j;
        }
        data[i] = minIdx;
      }
      return new JsNDArray(data, [rows]);
    }
  }

  argmaxAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('argmaxAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        let maxIdx = 0;
        for (let i = 1; i < rows; i++) {
          if (arr.data[i * cols + j] > arr.data[maxIdx * cols + j]) maxIdx = i;
        }
        data[j] = maxIdx;
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        let maxIdx = 0;
        for (let j = 1; j < cols; j++) {
          if (arr.data[i * cols + j] > arr.data[i * cols + maxIdx]) maxIdx = j;
        }
        data[i] = maxIdx;
      }
      return new JsNDArray(data, [rows]);
    }
  }

  varAxis(arr: NDArray, axis: number, ddof: number = 0): NDArray {
    if (arr.shape.length !== 2) throw new Error('varAxis only supports 2D');
    const mean = this.meanAxis(arr, axis);
    const [rows, cols] = arr.shape;
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
      return new JsNDArray(data, [cols]);
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
      return new JsNDArray(data, [rows]);
    }
  }

  stdAxis(arr: NDArray, axis: number, ddof: number = 0): NDArray {
    const variance = this.varAxis(arr, axis, ddof);
    return new JsNDArray(variance.data.map(Math.sqrt), variance.shape);
  }

  prodAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('prodAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      const data = new Float64Array(cols).fill(1);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          data[j] *= arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(1);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i] *= arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [rows]);
    }
  }

  allAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('allAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
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
      return new JsNDArray(data, [cols]);
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
      return new JsNDArray(data, [rows]);
    }
  }

  anyAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('anyAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
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
      return new JsNDArray(data, [cols]);
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
      return new JsNDArray(data, [rows]);
    }
  }

  cumsumAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('cumsumAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    const data = new Float64Array(arr.data);
    if (axis === 0) {
      for (let j = 0; j < cols; j++) {
        for (let i = 1; i < rows; i++) {
          data[i * cols + j] += data[(i - 1) * cols + j];
        }
      }
    } else {
      for (let i = 0; i < rows; i++) {
        for (let j = 1; j < cols; j++) {
          data[i * cols + j] += data[i * cols + j - 1];
        }
      }
    }
    return new JsNDArray(data, [rows, cols]);
  }

  cumprodAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('cumprodAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    const data = new Float64Array(arr.data);
    if (axis === 0) {
      for (let j = 0; j < cols; j++) {
        for (let i = 1; i < rows; i++) {
          data[i * cols + j] *= data[(i - 1) * cols + j];
        }
      }
    } else {
      for (let i = 0; i < rows; i++) {
        for (let j = 1; j < cols; j++) {
          data[i * cols + j] *= data[i * cols + j - 1];
        }
      }
    }
    return new JsNDArray(data, [rows, cols]);
  }

  // ============ Linalg ============

  matmul(a: NDArray, b: NDArray): NDArray {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('matmul requires 2D arrays');
    }
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error('matmul dimension mismatch');

    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          sum += a.data[i * k1 + k] * b.data[k * n + j];
        }
        data[i * n + j] = sum;
      }
    }
    return new JsNDArray(data, [m, n]);
  }

  dot(a: NDArray, b: NDArray): NDArray {
    if (a.shape.length === 1 && b.shape.length === 1) {
      // Vector dot product
      if (a.shape[0] !== b.shape[0]) throw new Error('dot dimension mismatch');
      let sum = 0;
      for (let i = 0; i < a.data.length; i++) {
        sum += a.data[i] * b.data[i];
      }
      return new JsNDArray([sum], [1]);
    }
    // For 2D, same as matmul
    return this.matmul(a, b);
  }

  inner(a: NDArray, b: NDArray): number | NDArray {
    if (a.shape[0] !== b.shape[0]) throw new Error('inner dimension mismatch');
    let sum = 0;
    for (let i = 0; i < a.data.length; i++) {
      sum += a.data[i] * b.data[i];
    }
    return sum;
  }

  outer(a: NDArray, b: NDArray): NDArray {
    const m = a.data.length;
    const n = b.data.length;
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        data[i * n + j] = a.data[i] * b.data[j];
      }
    }
    return new JsNDArray(data, [m, n]);
  }

  transpose(arr: NDArray, axes?: number[]): NDArray {
    const ndim = arr.shape.length;
    if (ndim === 1) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }

    // Determine permutation: default is reversed dimensions
    const perm = axes || [...Array(ndim).keys()].reverse();

    if (perm.length !== ndim) {
      throw new Error(`axes don't match array: expected ${ndim} axes, got ${perm.length}`);
    }

    const newShape = perm.map(i => arr.shape[i]);
    const size = arr.data.length;
    const data = new Float64Array(size);

    // Compute strides for the original array
    const oldStrides = new Array(ndim);
    oldStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1] * arr.shape[i + 1];
    }

    // Compute strides for the new array
    const newStrides = new Array(ndim);
    newStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    // For each position in the new array, find the corresponding old array position
    for (let newFlat = 0; newFlat < size; newFlat++) {
      let remaining = newFlat;
      let oldFlat = 0;
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(remaining / newStrides[d]);
        remaining -= coord * newStrides[d];
        // new dim d has coord, which maps to old dim perm[d]
        oldFlat += coord * oldStrides[perm[d]];
      }
      data[newFlat] = arr.data[oldFlat];
    }

    return new JsNDArray(data, newShape);
  }

  trace(arr: NDArray): number {
    if (arr.shape.length !== 2) throw new Error('trace requires 2D');
    const [rows, cols] = arr.shape;
    const n = Math.min(rows, cols);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += arr.data[i * cols + i];
    }
    return sum;
  }

  det(arr: NDArray): number {
    if (arr.shape.length !== 2) throw new Error('det requires 2D');
    const [rows, cols] = arr.shape;
    if (rows !== cols) throw new Error('det requires square matrix');

    // Simple 2x2 and 3x3 determinants
    if (rows === 2) {
      return arr.data[0] * arr.data[3] - arr.data[1] * arr.data[2];
    }
    if (rows === 3) {
      const [a, b, c, d, e, f, g, h, i] = arr.data;
      return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    // LU decomposition for larger matrices
    const lu = this._luDecompose(arr);
    let det = lu.sign;
    for (let i = 0; i < rows; i++) {
      det *= lu.u.data[i * cols + i];
    }
    return det;
  }

  inv(arr: NDArray): NDArray {
    if (arr.shape.length !== 2) throw new Error('inv requires 2D');
    const [rows, cols] = arr.shape;
    if (rows !== cols) throw new Error('inv requires square matrix');

    const n = rows;
    // Gauss-Jordan elimination
    const aug = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aug[i * 2 * n + j] = arr.data[i * n + j];
      }
      aug[i * 2 * n + n + i] = 1;
    }

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k * 2 * n + i]) > Math.abs(aug[maxRow * 2 * n + i])) {
          maxRow = k;
        }
      }
      // Swap rows
      for (let k = 0; k < 2 * n; k++) {
        const tmp = aug[i * 2 * n + k];
        aug[i * 2 * n + k] = aug[maxRow * 2 * n + k];
        aug[maxRow * 2 * n + k] = tmp;
      }

      const pivot = aug[i * 2 * n + i];
      if (Math.abs(pivot) < 1e-10) throw new Error('singular matrix');

      // Scale row
      for (let k = 0; k < 2 * n; k++) {
        aug[i * 2 * n + k] /= pivot;
      }

      // Eliminate
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k * 2 * n + i];
          for (let j = 0; j < 2 * n; j++) {
            aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
          }
        }
      }
    }

    // Extract inverse
    const result = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aug[i * 2 * n + n + j];
      }
    }
    return new JsNDArray(result, [n, n]);
  }

  solve(a: NDArray, b: NDArray): NDArray {
    // Solve Ax = b using LU decomposition
    const aInv = this.inv(a);
    return this.matmul(aInv, b);
  }

  norm(arr: NDArray, ord: number | 'fro' | 'nuc' = 2, axis?: number): number | NDArray {
    // Handle 'fro' (Frobenius norm) — sqrt of sum of squares
    if (ord === 'fro') {
      const d = arr.data;
      let s = 0;
      for (let i = 0; i < d.length; i++) s += d[i] * d[i];
      return Math.sqrt(s);
    }
    if (ord === 'nuc') {
      // Nuclear norm: sum of singular values
      const { s } = this.svd(arr);
      let sum = 0;
      for (let i = 0; i < s.data.length; i++) sum += s.data[i];
      return sum;
    }
    if (axis !== undefined) {
      // Compute norm along the given axis (2D arrays only for now)
      if (arr.shape.length !== 2) throw new Error('norm with axis only supports 2D arrays');
      const [rows, cols] = arr.shape;
      if (axis < 0 || axis >= 2) throw new Error('invalid axis for 2D array');
      if (axis === 0) {
        // Norm along rows -> result shape [cols]
        const result = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let s = 0;
          for (let i = 0; i < rows; i++) {
            const v = arr.data[i * cols + j];
            if (ord === 1) s += Math.abs(v);
            else if (ord === Infinity) s = Math.max(s, Math.abs(v));
            else if (ord === -Infinity) {
              s = i === 0 ? Math.abs(v) : Math.min(s, Math.abs(v));
            } else s += v * v; // L2
          }
          result[j] =
            ord === 2
              ? Math.sqrt(s)
              : ord === 1 || ord === Infinity || ord === -Infinity
                ? s
                : Math.pow(s, 1 / ord);
        }
        return new JsNDArray(result, [cols]);
      } else {
        // Norm along cols -> result shape [rows]
        const result = new Float64Array(rows);
        for (let i = 0; i < rows; i++) {
          let s = 0;
          for (let j = 0; j < cols; j++) {
            const v = arr.data[i * cols + j];
            if (ord === 1) s += Math.abs(v);
            else if (ord === Infinity) s = Math.max(s, Math.abs(v));
            else if (ord === -Infinity) {
              s = j === 0 ? Math.abs(v) : Math.min(s, Math.abs(v));
            } else s += v * v; // L2
          }
          result[i] =
            ord === 2
              ? Math.sqrt(s)
              : ord === 1 || ord === Infinity || ord === -Infinity
                ? s
                : Math.pow(s, 1 / ord);
        }
        return new JsNDArray(result, [rows]);
      }
    }
    const d = arr.data;
    if (ord === 1) {
      let s = 0;
      for (let i = 0; i < d.length; i++) s += Math.abs(d[i]);
      return s;
    }
    if (ord === Infinity) {
      let mx = -Infinity;
      for (let i = 0; i < d.length; i++) {
        const v = Math.abs(d[i]);
        if (v > mx) mx = v;
      }
      return mx;
    }
    if (ord === -Infinity) {
      let mn = Infinity;
      for (let i = 0; i < d.length; i++) {
        const v = Math.abs(d[i]);
        if (v < mn) mn = v;
      }
      return mn;
    }
    // Default L2 norm
    let s = 0;
    for (let i = 0; i < d.length; i++) s += d[i] * d[i];
    return Math.sqrt(s);
  }

  qr(arr: NDArray, mode: 'reduced' | 'complete' = 'reduced'): { q: NDArray; r: NDArray } {
    if (arr.shape.length !== 2) throw new Error('qr requires 2D');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);

    if (mode === 'complete') {
      // Complete QR via Householder reflections: Q is m×m, R is m×n
      const qFull = new Float64Array(m * m);
      const rFull = new Float64Array(m * n);

      // Initialize R as a copy of A
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          rFull[i * n + j] = arr.data[i * n + j];
        }
      }

      // Initialize Q as identity
      for (let i = 0; i < m; i++) qFull[i * m + i] = 1;

      for (let j = 0; j < k; j++) {
        // Extract column j of R from row j downward
        const v = new Float64Array(m - j);
        for (let i = j; i < m; i++) v[i - j] = rFull[i * n + j];

        const vNorm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
        if (vNorm < 1e-15) continue;

        // Householder vector with sign choice to avoid cancellation
        const sign = v[0] >= 0 ? 1 : -1;
        v[0] += sign * vNorm;
        const v2Norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
        if (v2Norm < 1e-15) continue;
        for (let i = 0; i < v.length; i++) v[i] /= v2Norm;

        // Apply Householder to R: R[j:, :] -= 2 * v * (v^T * R[j:, :])
        for (let col = 0; col < n; col++) {
          let dot = 0;
          for (let i = 0; i < v.length; i++) {
            dot += v[i] * rFull[(j + i) * n + col];
          }
          for (let i = 0; i < v.length; i++) {
            rFull[(j + i) * n + col] -= 2 * v[i] * dot;
          }
        }

        // Apply Householder to Q: Q[:, j:] -= 2 * (Q[:, j:] * v) * v^T
        for (let row = 0; row < m; row++) {
          let dot = 0;
          for (let i = 0; i < v.length; i++) {
            dot += qFull[row * m + (j + i)] * v[i];
          }
          for (let i = 0; i < v.length; i++) {
            qFull[row * m + (j + i)] -= 2 * dot * v[i];
          }
        }
      }

      return {
        q: new JsNDArray(qFull, [m, m]),
        r: new JsNDArray(rFull, [m, n]),
      };
    }

    // Reduced QR (default) via Modified Gram-Schmidt: Q is m×n, R is n×n
    const q = new Float64Array(m * n);
    const r = new Float64Array(n * n);

    // Copy A to Q
    for (let i = 0; i < m * n; i++) q[i] = arr.data[i];

    for (let j = 0; j < n; j++) {
      // Compute norm of column j
      let norm = 0;
      for (let i = 0; i < m; i++) {
        norm += q[i * n + j] ** 2;
      }
      norm = Math.sqrt(norm);
      r[j * n + j] = norm;

      // Normalize column j
      for (let i = 0; i < m; i++) {
        q[i * n + j] /= norm;
      }

      // Orthogonalize remaining columns
      for (let kk = j + 1; kk < n; kk++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += q[i * n + j] * q[i * n + kk];
        }
        r[j * n + kk] = dot;
        for (let i = 0; i < m; i++) {
          q[i * n + kk] -= dot * q[i * n + j];
        }
      }
    }

    return {
      q: new JsNDArray(q, [m, n]),
      r: new JsNDArray(r, [n, n]),
    };
  }

  svd(arr: NDArray, fullMatrices: boolean = true): { u: NDArray; s: NDArray; vt: NDArray } {
    // SVD using power iteration with deflation
    // Computes A = U * Σ * V^T
    if (arr.shape.length !== 2) throw new Error('svd requires 2D');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);

    // Compute A^T A
    const at = this.transpose(arr);
    const ata = this.matmul(at, arr);
    const ataData = ata.data;

    // Working copy for deflation
    const ataWork = new Float64Array(ataData);
    const singularValues = new Float64Array(k);
    const vMatrix = new Float64Array(n * k);

    // Power iteration to find eigenvalues and eigenvectors of A^T A
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
      const CONV_TOL = 1e-10;
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

      // Deflate: ATA -= eigenvalue * v * v^T
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
    const vArr = new JsNDArray(sortedV, [n, k]);
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

    if (!fullMatrices || (m === k && n === k)) {
      // Reduced SVD or square matrix (full == reduced)
      return {
        u: new JsNDArray(uData, [m, k]),
        s: new JsNDArray(sortedS, [k]),
        vt: new JsNDArray(vtData, [k, n]),
      };
    }

    // Full SVD: extend U to m×m and Vt to n×n by completing orthonormal bases
    // Extend U (m×k -> m×m) using QR of current U columns
    let fullU: Float64Array;
    if (m > k) {
      // Build m×m orthogonal matrix whose first k columns are uData
      fullU = new Float64Array(m * m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          fullU[i * m + j] = uData[i * k + j];
        }
      }
      // Fill remaining columns with random vectors, then orthogonalize via Gram-Schmidt
      for (let j = k; j < m; j++) {
        // Start with a standard basis vector e_j (good for linear independence)
        for (let i = 0; i < m; i++) fullU[i * m + j] = i === j ? 1 : 0;
        // Orthogonalize against all previous columns
        for (let prev = 0; prev < j; prev++) {
          let dot = 0;
          for (let i = 0; i < m; i++) dot += fullU[i * m + j] * fullU[i * m + prev];
          for (let i = 0; i < m; i++) fullU[i * m + j] -= dot * fullU[i * m + prev];
        }
        // Normalize
        let norm = 0;
        for (let i = 0; i < m; i++) norm += fullU[i * m + j] ** 2;
        norm = Math.sqrt(norm);
        if (norm < 1e-12) {
          // Try another basis vector if this one was linearly dependent
          for (let i = 0; i < m; i++) fullU[i * m + j] = i === (j + 1) % m ? 1 : 0;
          for (let prev = 0; prev < j; prev++) {
            let dot = 0;
            for (let i = 0; i < m; i++) dot += fullU[i * m + j] * fullU[i * m + prev];
            for (let i = 0; i < m; i++) fullU[i * m + j] -= dot * fullU[i * m + prev];
          }
          norm = 0;
          for (let i = 0; i < m; i++) norm += fullU[i * m + j] ** 2;
          norm = Math.sqrt(norm);
        }
        if (norm > 1e-12) {
          for (let i = 0; i < m; i++) fullU[i * m + j] /= norm;
        }
      }
    } else {
      fullU = uData;
    }

    // Extend Vt (k×n -> n×n)
    let fullVt: Float64Array;
    if (n > k) {
      // sortedV is n×k (columns of V), vtData is k×n (rows of Vt)
      // Build n×n: first k rows are vtData, remaining rows complete the basis
      fullVt = new Float64Array(n * n);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < n; j++) {
          fullVt[i * n + j] = vtData[i * n + j];
        }
      }
      // Work in V-space (columns): V is n×k, extend to n×n then transpose back
      // Easier: work with rows of Vt. Each row is a basis vector of length n.
      for (let row = k; row < n; row++) {
        // Start with standard basis vector
        for (let j = 0; j < n; j++) fullVt[row * n + j] = j === row ? 1 : 0;
        // Orthogonalize against all previous rows
        for (let prev = 0; prev < row; prev++) {
          let dot = 0;
          for (let j = 0; j < n; j++) dot += fullVt[row * n + j] * fullVt[prev * n + j];
          for (let j = 0; j < n; j++) fullVt[row * n + j] -= dot * fullVt[prev * n + j];
        }
        let norm = 0;
        for (let j = 0; j < n; j++) norm += fullVt[row * n + j] ** 2;
        norm = Math.sqrt(norm);
        if (norm < 1e-12) {
          for (let j = 0; j < n; j++) fullVt[row * n + j] = j === (row + 1) % n ? 1 : 0;
          for (let prev = 0; prev < row; prev++) {
            let dot = 0;
            for (let j = 0; j < n; j++) dot += fullVt[row * n + j] * fullVt[prev * n + j];
            for (let j = 0; j < n; j++) fullVt[row * n + j] -= dot * fullVt[prev * n + j];
          }
          norm = 0;
          for (let j = 0; j < n; j++) norm += fullVt[row * n + j] ** 2;
          norm = Math.sqrt(norm);
        }
        if (norm > 1e-12) {
          for (let j = 0; j < n; j++) fullVt[row * n + j] /= norm;
        }
      }
    } else {
      fullVt = vtData;
    }

    return {
      u: new JsNDArray(fullU, [m, m > k ? m : k]),
      s: new JsNDArray(sortedS, [k]),
      vt: new JsNDArray(fullVt, [n > k ? n : k, n]),
    };
  }

  // ============ Helpers ============

  /** Convert ArrayOrScalar to NDArray */
  private _toNDArray(x: ArrayOrScalar): NDArray {
    if (typeof x === 'number') {
      return new JsNDArray(new Float64Array([x]), [1], 'float64');
    }
    return x;
  }

  /** Generic binary op with scalar broadcasting */
  private _binaryOp(
    a: ArrayOrScalar,
    b: ArrayOrScalar,
    fn: (x: number, y: number) => number
  ): NDArray {
    const arrA = this._toNDArray(a);
    const arrB = this._toNDArray(b);
    // Scalar broadcasting
    if (arrA.data.length === 1 && arrB.data.length > 1) {
      const data = new Float64Array(arrB.data.length);
      const scalar = arrA.data[0];
      for (let i = 0; i < arrB.data.length; i++) data[i] = fn(scalar, arrB.data[i]);
      return new JsNDArray(data, [...arrB.shape]);
    }
    if (arrB.data.length === 1 && arrA.data.length > 1) {
      const data = new Float64Array(arrA.data.length);
      const scalar = arrB.data[0];
      for (let i = 0; i < arrA.data.length; i++) data[i] = fn(arrA.data[i], scalar);
      return new JsNDArray(data, [...arrA.shape]);
    }
    this._checkSameShape(arrA, arrB);
    const data = new Float64Array(arrA.data.length);
    for (let i = 0; i < arrA.data.length; i++) data[i] = fn(arrA.data[i], arrB.data[i]);
    return new JsNDArray(data, [...arrA.shape]);
  }

  private _checkSameShape(a: NDArray, b: NDArray): void {
    if (a.shape.length !== b.shape.length) {
      throw new Error('shape mismatch');
    }
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) {
        throw new Error('shape mismatch');
      }
    }
  }

  private _luDecompose(arr: NDArray): { l: NDArray; u: NDArray; sign: number } {
    const n = arr.shape[0];
    const l = new Float64Array(n * n);
    const u = new Float64Array(n * n);
    let sign = 1;

    // Copy to U
    for (let i = 0; i < n * n; i++) u[i] = arr.data[i];

    // Initialize L as identity
    for (let i = 0; i < n; i++) l[i * n + i] = 1;

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(u[k * n + i]) > Math.abs(u[maxRow * n + i])) {
          maxRow = k;
        }
      }
      if (maxRow !== i) {
        sign *= -1;
        for (let k = 0; k < n; k++) {
          const tmp = u[i * n + k];
          u[i * n + k] = u[maxRow * n + k];
          u[maxRow * n + k] = tmp;
        }
      }

      for (let k = i + 1; k < n; k++) {
        const factor = u[k * n + i] / u[i * n + i];
        l[k * n + i] = factor;
        for (let j = i; j < n; j++) {
          u[k * n + j] -= factor * u[i * n + j];
        }
      }
    }

    return {
      l: new JsNDArray(l, [n, n]),
      u: new JsNDArray(u, [n, n]),
      sign,
    };
  }

  // ============ Creation - Like Functions ============

  zerosLike(arr: NDArray, dtype?: DType): NDArray {
    const dt = dtype ?? arr.dtype ?? 'float64';
    return this.zeros(arr.shape, dt);
  }

  onesLike(arr: NDArray, dtype?: DType): NDArray {
    const dt = dtype ?? arr.dtype ?? 'float64';
    return this.ones(arr.shape, dt);
  }

  emptyLike(arr: NDArray, dtype?: DType): NDArray {
    // In JS, we can't have uninitialized memory, so same as zeros
    const dt = dtype ?? arr.dtype ?? 'float64';
    return this.zeros(arr.shape, dt);
  }

  fullLike(arr: NDArray, value: number, dtype?: DType): NDArray {
    const dt = dtype ?? arr.dtype ?? 'float64';
    return this.full(arr.shape, value, dt);
  }

  // ============ Broadcasting ============

  broadcastTo(arr: NDArray, shape: number[]): NDArray {
    // Validate shapes are compatible
    const arrShape = arr.shape;
    if (arrShape.length > shape.length) {
      throw new Error('Cannot broadcast to smaller number of dimensions');
    }

    // Pad arr shape with 1s on the left
    const paddedShape = new Array(shape.length - arrShape.length).fill(1).concat(arrShape);

    // Check compatibility
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

    return new JsNDArray(result, shape);
  }

  broadcastArrays(...arrays: NDArray[]): NDArray[] {
    if (arrays.length === 0) return [];
    if (arrays.length === 1) return [new JsNDArray(arrays[0].data.slice(), arrays[0].shape)];

    // Compute broadcast shape
    const shapes = arrays.map(a => a.shape);
    const maxDims = Math.max(...shapes.map(s => s.length));

    // Pad all shapes with 1s on the left
    const paddedShapes = shapes.map(s => {
      const padded = new Array(maxDims - s.length).fill(1);
      return padded.concat(s);
    });

    // Compute output shape
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

    // Broadcast each array
    return arrays.map(arr => this.broadcastTo(arr, outShape));
  }

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
      // If dimension is 1, stride is 0 (broadcast)
      strides[i] = srcShape[i] === 1 ? 0 : srcStride;
      srcStride *= srcShape[i];
    }
    return strides;
  }

  // ============ Shape Manipulation ============

  private _normalizeAxis(axis: number, ndim: number): number {
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    return axis;
  }

  swapaxes(arr: NDArray, axis1: number, axis2: number): NDArray {
    const ndim = arr.shape.length;
    axis1 = this._normalizeAxis(axis1, ndim);
    axis2 = this._normalizeAxis(axis2, ndim);

    if (axis1 === axis2) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }

    // Create new shape with swapped axes
    const newShape = [...arr.shape];
    [newShape[axis1], newShape[axis2]] = [newShape[axis2], newShape[axis1]];

    // Create permutation array
    const perm = Array.from({ length: ndim }, (_, i) => i);
    [perm[axis1], perm[axis2]] = [perm[axis2], perm[axis1]];

    return this._transposeGeneral(arr, perm, newShape);
  }

  moveaxis(arr: NDArray, source: number | number[], destination: number | number[]): NDArray {
    const ndim = arr.shape.length;
    const srcArr = Array.isArray(source) ? source : [source];
    const dstArr = Array.isArray(destination) ? destination : [destination];

    if (srcArr.length !== dstArr.length) {
      throw new Error(`source and destination must have the same number of elements`);
    }

    const normSrc = srcArr.map(s => this._normalizeAxis(s, ndim));
    const normDst = dstArr.map(d => this._normalizeAxis(d, ndim));

    // Build permutation: start with axes not in source, in order
    const order: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (!normSrc.includes(i)) order.push(i);
    }
    // Insert source axes at destination positions (process in sorted dst order)
    const pairs = normDst.map((d, i) => ({ dst: d, src: normSrc[i] }));
    pairs.sort((a, b) => a.dst - b.dst);
    for (const { dst, src } of pairs) {
      order.splice(dst, 0, src);
    }

    const newShape = order.map(i => arr.shape[i]);
    return this._transposeGeneral(arr, order, newShape);
  }

  private _transposeGeneral(arr: NDArray, perm: number[], newShape: number[]): NDArray {
    const size = arr.data.length;
    const result = new Float64Array(size);

    const oldStrides = this._computeStrides(arr.shape);
    const newStrides = this._computeStrides(newShape);

    for (let i = 0; i < size; i++) {
      // Convert flat index to coordinates in new array
      const coords = new Array(newShape.length);
      let remaining = i;
      for (let d = 0; d < newShape.length; d++) {
        coords[d] = Math.floor(remaining / newStrides[d]);
        remaining = remaining % newStrides[d];
      }

      // Map to old coordinates using inverse permutation
      let oldIdx = 0;
      for (let d = 0; d < perm.length; d++) {
        oldIdx += coords[d] * oldStrides[perm[d]];
      }

      result[i] = arr.data[oldIdx];
    }

    return new JsNDArray(result, newShape);
  }

  squeeze(arr: NDArray, axis?: number | number[]): NDArray {
    if (axis !== undefined) {
      const axes = Array.isArray(axis) ? axis : [axis];
      const normalizedAxes = axes.map(a => this._normalizeAxis(a, arr.shape.length));
      for (const na of normalizedAxes) {
        if (arr.shape[na] !== 1) {
          throw new Error(`cannot squeeze axis ${na} with size ${arr.shape[na]}`);
        }
      }
      const newShape = arr.shape.filter((_, i) => !normalizedAxes.includes(i));
      return new JsNDArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
    }

    // Remove all dimensions of size 1
    const newShape = arr.shape.filter(d => d !== 1);
    return new JsNDArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
  }

  expandDims(arr: NDArray, axis: number): NDArray {
    const ndim = arr.shape.length + 1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    const newShape = [...arr.shape];
    newShape.splice(axis, 0, 1);
    return new JsNDArray(arr.data.slice(), newShape);
  }

  reshape(arr: NDArray, shape: number[]): NDArray {
    // Handle -1 in shape (infer dimension)
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

    return new JsNDArray(arr.data.slice(), newShape);
  }

  resize(arr: NDArray, newShape: number[]): NDArray {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(newSize);
    const srcLen = arr.data.length;
    if (srcLen === 0) {
      // If source is empty, fill with zeros (already done by Float64Array)
      return new JsNDArray(data, [...newShape]);
    }
    for (let i = 0; i < newSize; i++) {
      data[i] = arr.data[i % srcLen];
    }
    return new JsNDArray(data, [...newShape]);
  }

  flatten(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.slice(), [arr.data.length]);
  }

  concatenate(arrays: NDArray[], axis: number | null = 0): NDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');

    // axis=null: flatten all arrays then concatenate as 1D
    if (axis === null || axis === undefined) {
      const flattened = arrays.map(a => this.flatten(a));
      const totalLen = flattened.reduce((sum, a) => sum + a.data.length, 0);
      const result = new Float64Array(totalLen);
      let offset = 0;
      for (const a of flattened) {
        result.set(a.data, offset);
        offset += a.data.length;
      }
      return new JsNDArray(result, [totalLen]);
    }

    if (arrays.length === 1) return new JsNDArray(arrays[0].data.slice(), arrays[0].shape);

    const ndim = arrays[0].shape.length;
    axis = this._normalizeAxis(axis, ndim);

    // Verify shapes match except along concat axis
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

    // Compute output shape
    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    // For 1D, simple concatenation
    if (ndim === 1) {
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return new JsNDArray(result, outShape);
    }

    // For nD, use stride-based copy
    const outStrides = this._computeStrides(outShape);

    let axisOffset = 0;
    for (const arr of arrays) {
      const srcStrides = this._computeStrides(arr.shape);
      const srcSize = arr.data.length;

      for (let srcIdx = 0; srcIdx < srcSize; srcIdx++) {
        // Convert to coordinates
        const coords = new Array(ndim);
        let remaining = srcIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / srcStrides[d]);
          remaining = remaining % srcStrides[d];
        }

        // Add offset along concat axis
        coords[axis] += axisOffset;

        // Convert to dest index
        let dstIdx = 0;
        for (let d = 0; d < ndim; d++) {
          dstIdx += coords[d] * outStrides[d];
        }

        result[dstIdx] = arr.data[srcIdx];
      }

      axisOffset += arr.shape[axis];
    }

    return new JsNDArray(result, outShape);
  }

  stack(arrays: NDArray[], axis: number = 0): NDArray {
    if (arrays.length === 0) throw new Error('need at least one array to stack');

    // Verify all shapes are the same
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

    // Expand dims on each array, then concatenate
    const expanded = arrays.map(arr => this.expandDims(arr, axis));
    return this.concatenate(expanded, axis);
  }

  split(arr: NDArray, indices: number | number[], axis: number = 0): NDArray[] {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const axisSize = arr.shape[axis];

    let splitIndices: number[];
    if (typeof indices === 'number') {
      // Split into n equal parts
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

    const results: NDArray[] = [];
    let start = 0;

    const getSlice = (startIdx: number, endIdx: number): NDArray => {
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

      return new JsNDArray(sliceData, sliceShape);
    };

    for (const idx of splitIndices) {
      results.push(getSlice(start, idx));
      start = idx;
    }
    results.push(getSlice(start, axisSize));

    return results;
  }

  // ============ Conditional ============

  where(condition: NDArray, x?: NDArray, y?: NDArray): NDArray | NDArray[] {
    if (x === undefined || y === undefined) {
      return this.nonzero(condition);
    }
    // Broadcast all arrays to the same shape
    const [condBcast, xBcast, yBcast] = this.broadcastArrays(condition, x, y);
    const size = condBcast.data.length;
    const result = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      result[i] = condBcast.data[i] !== 0 ? xBcast.data[i] : yBcast.data[i];
    }

    return new JsNDArray(result, condBcast.shape);
  }

  // ============ Advanced Indexing ============

  take(arr: NDArray, indices: NDArray | number[], axis?: number): NDArray {
    const indexArray = Array.isArray(indices) ? indices : Array.from(indices.data);

    if (axis === undefined) {
      // Take from flattened array
      const result = new Float64Array(indexArray.length);
      for (let i = 0; i < indexArray.length; i++) {
        let idx = indexArray[i];
        if (idx < 0) idx += arr.data.length;
        result[i] = arr.data[idx];
      }
      return new JsNDArray(result, [indexArray.length]);
    }

    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    // Output shape: replace axis dimension with indices length
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

      // Map the axis coordinate through indices
      let srcAxisCoord = indexArray[coords[axis]];
      if (srcAxisCoord < 0) srcAxisCoord += arr.shape[axis];
      coords[axis] = srcAxisCoord;

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return new JsNDArray(result, outShape);
  }

  // ============ Batched Operations ============

  batchedMatmul(a: NDArray, b: NDArray): NDArray {
    // Supports shapes like (batch, M, K) @ (batch, K, N) -> (batch, M, N)
    // Or (batch, M, K) @ (K, N) -> (batch, M, N) with broadcasting
    if (a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('batchedMatmul requires at least 2D arrays');
    }

    // Get matrix dimensions (last two axes)
    const aM = a.shape[a.shape.length - 2];
    const aK = a.shape[a.shape.length - 1];
    const bK = b.shape[b.shape.length - 2];
    const bN = b.shape[b.shape.length - 1];

    if (aK !== bK) throw new Error('matmul inner dimensions must match');

    // Compute batch dimensions
    const aBatchShape = a.shape.slice(0, -2);
    const bBatchShape = b.shape.slice(0, -2);

    // Broadcast batch dimensions
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
    const batchSize = outBatchShape.reduce((a, b) => a * b, 1);
    const matSize = aM * bN;
    const result = new Float64Array(batchSize * matSize);

    // Compute strides for batch indexing
    const aBatchStrides = this._computeStrides(paddedABatch);
    const bBatchStrides = this._computeStrides(paddedBBatch);
    const outBatchStrides = this._computeStrides(outBatchShape);

    const aMatStride = aM * aK;
    const bMatStride = bK * bN;

    for (let batch = 0; batch < batchSize; batch++) {
      // Convert batch index to coordinates
      const coords = new Array(maxBatchDims);
      let remaining = batch;
      for (let d = 0; d < maxBatchDims; d++) {
        coords[d] = Math.floor(remaining / outBatchStrides[d]);
        remaining = remaining % outBatchStrides[d];
      }

      // Map to source batch indices with broadcasting
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

      // Perform matmul for this batch
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

    return new JsNDArray(result, outShape);
  }

  // ============ Einstein Summation ============

  einsum(subscripts: string, ...operands: NDArray[]): NDArray {
    // Parse einsum string
    const [inputStr, outputStr] = subscripts.split('->').map(s => s.trim());
    const inputSubscripts = inputStr.split(',').map(s => s.trim());

    if (inputSubscripts.length !== operands.length) {
      throw new Error(
        `einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`
      );
    }

    // Map each label to its dimension size
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

    // Determine output labels
    let outputLabels: string[];
    if (outputStr !== undefined) {
      outputLabels = outputStr.split('');
    } else {
      // Implicit mode: output labels are those that appear exactly once
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

    // Compute output shape
    const outputShape = outputLabels.map(l => labelSizes.get(l)!);
    const outputSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

    // Find contracted (summed) labels
    const outputSet = new Set(outputLabels);
    const allLabels = Array.from(labelSizes.keys());
    const contractedLabels = allLabels.filter(l => !outputSet.has(l));

    // Compute contracted dimensions size
    const contractedSizes = contractedLabels.map(l => labelSizes.get(l)!);
    const contractedTotal =
      contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

    const result = new Float64Array(outputSize);

    // Compute input strides
    const inputStrides = operands.map(op => this._computeStrides(op.shape));

    // For each output position
    const outputStrides = outputShape.length === 0 ? [] : this._computeStrides(outputShape);

    for (let outIdx = 0; outIdx < outputSize; outIdx++) {
      // Convert to output coordinates
      const outCoords: Map<string, number> = new Map();
      let remaining = outIdx;
      for (let d = 0; d < outputLabels.length; d++) {
        const coord = Math.floor(remaining / outputStrides[d]);
        remaining = remaining % outputStrides[d];
        outCoords.set(outputLabels[d], coord);
      }

      // Sum over contracted indices
      let sum = 0;
      for (let contrIdx = 0; contrIdx < contractedTotal; contrIdx++) {
        // Convert to contracted coordinates
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

        // Merge coordinates
        const allCoords = new Map([...outCoords, ...contrCoords]);

        // Compute product of all operands at these coordinates
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

    return new JsNDArray(result, outputShape.length === 0 ? [1] : outputShape);
  }

  // ============ Differences ============

  diff(
    arr: NDArray,
    n: number = 1,
    axis: number = -1,
    prepend?: NDArray | number,
    append?: NDArray | number
  ): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    // Handle prepend and append by concatenating before computing diff
    let input: NDArray = arr;
    if (prepend !== undefined || append !== undefined) {
      const parts: NDArray[] = [];
      if (prepend !== undefined) {
        if (typeof prepend === 'number') {
          // Create array with matching shape except axis dimension = 1
          const pShape = [...arr.shape];
          pShape[axis] = 1;
          const pSize = pShape.reduce((a, b) => a * b, 1);
          const pData = new Float64Array(pSize);
          pData.fill(prepend);
          parts.push(new JsNDArray(pData, pShape));
        } else {
          parts.push(prepend);
        }
      }
      parts.push(arr);
      if (append !== undefined) {
        if (typeof append === 'number') {
          const aShape = [...arr.shape];
          aShape[axis] = 1;
          const aSize = aShape.reduce((a, b) => a * b, 1);
          const aData = new Float64Array(aSize);
          aData.fill(append);
          parts.push(new JsNDArray(aData, aShape));
        } else {
          parts.push(append);
        }
      }
      input = this.concatenate(parts, axis);
    }

    let result: NDArray = input;
    for (let i = 0; i < n; i++) {
      result = this._diffOnce(result, axis);
    }
    return result;
  }

  private _diffOnce(arr: NDArray, axis: number): NDArray {
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

    return new JsNDArray(result, newShape);
  }

  gradient(arr: NDArray, axis: number = -1, edgeOrder: 1 | 2 = 1): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const shape = arr.shape;
    const axisLen = shape[axis];

    if (axisLen < 2) {
      throw new Error('gradient requires at least 2 elements along axis');
    }
    if (edgeOrder === 2 && axisLen < 3) {
      throw new Error('gradient with edge_order=2 requires at least 3 elements along axis');
    }

    const result = new Float64Array(arr.data.length);
    const strides = this._computeStrides(shape);

    // Helper to get flat index from coords
    const flatIdx = (coords: number[]) => {
      let idx = 0;
      for (let d = 0; d < ndim; d++) idx += coords[d] * strides[d];
      return idx;
    };

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
        if (edgeOrder === 2) {
          // Second-order accurate forward: (-3f0 + 4f1 - f2) / 2
          const c1 = [...coords];
          c1[axis] = 1;
          const c2 = [...coords];
          c2[axis] = 2;
          grad = (-3 * arr.data[i] + 4 * arr.data[flatIdx(c1)] - arr.data[flatIdx(c2)]) / 2;
        } else {
          // First-order forward difference
          const nextCoords = [...coords];
          nextCoords[axis] = 1;
          grad = arr.data[flatIdx(nextCoords)] - arr.data[i];
        }
      } else if (axisCoord === axisLen - 1) {
        if (edgeOrder === 2) {
          // Second-order accurate backward: (3fN - 4fN-1 + fN-2) / 2
          const c1 = [...coords];
          c1[axis] = axisLen - 2;
          const c2 = [...coords];
          c2[axis] = axisLen - 3;
          grad = (3 * arr.data[i] - 4 * arr.data[flatIdx(c1)] + arr.data[flatIdx(c2)]) / 2;
        } else {
          // First-order backward difference
          const prevCoords = [...coords];
          prevCoords[axis] = axisLen - 2;
          grad = arr.data[i] - arr.data[flatIdx(prevCoords)];
        }
      } else {
        // Central difference (same for both edge orders)
        const prevCoords = [...coords];
        const nextCoords = [...coords];
        prevCoords[axis] = axisCoord - 1;
        nextCoords[axis] = axisCoord + 1;
        grad = (arr.data[flatIdx(nextCoords)] - arr.data[flatIdx(prevCoords)]) / 2;
      }

      result[i] = grad;
    }

    return new JsNDArray(result, shape);
  }

  ediff1d(arr: NDArray): NDArray {
    const flat = this.flatten(arr);
    return this.diff(flat, 1, 0);
  }

  // ============ Cross Product ============

  cross(a: NDArray, b: NDArray): NDArray {
    // Only supports 3D vectors
    const aFlat = this.flatten(a);
    const bFlat = this.flatten(b);

    if (aFlat.data.length !== 3 || bFlat.data.length !== 3) {
      throw new Error('cross product only supports 3D vectors');
    }

    const [a1, a2, a3] = aFlat.data;
    const [b1, b2, b3] = bFlat.data;

    return new JsNDArray(
      new Float64Array([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]),
      [3]
    );
  }

  // ============ Statistics ============

  cov(
    x: NDArray,
    y?: NDArray,
    rowvar: boolean = true,
    bias: boolean = false,
    ddof?: number | null
  ): NDArray {
    // Determine normalization factor
    // ddof overrides bias: if ddof is given, use N - ddof
    // if bias=true, use N (equivalent to ddof=0)
    // if bias=false (default), use N-1 (equivalent to ddof=1)
    const getDivisor = (nObs: number) => {
      if (ddof != null) return nObs - ddof;
      return bias ? nObs : nObs - 1;
    };

    if (y === undefined) {
      // Compute covariance matrix for rows (or columns) of x
      let data2d: NDArray;
      if (x.shape.length === 1) {
        // 1D: treat as single variable (1 row of observations)
        data2d = this.reshape(x, [1, x.data.length]);
      } else if (x.shape.length === 2) {
        data2d = x;
      } else {
        throw new Error('cov requires 1D or 2D array when y is not provided');
      }

      // If rowvar=false, transpose so rows become variables
      if (!rowvar) {
        data2d = this.transpose(data2d);
      }

      const [nVars, nObs] = data2d.shape;
      const divisor = getDivisor(nObs);
      const result = new Float64Array(nVars * nVars);

      // Compute means for each row
      const means = new Float64Array(nVars);
      for (let i = 0; i < nVars; i++) {
        let sum = 0;
        for (let j = 0; j < nObs; j++) {
          sum += data2d.data[i * nObs + j];
        }
        means[i] = sum / nObs;
      }

      // Compute covariance matrix
      for (let i = 0; i < nVars; i++) {
        for (let j = 0; j < nVars; j++) {
          let cov = 0;
          for (let k = 0; k < nObs; k++) {
            cov += (data2d.data[i * nObs + k] - means[i]) * (data2d.data[j * nObs + k] - means[j]);
          }
          result[i * nVars + j] = divisor > 0 ? cov / divisor : 0;
        }
      }

      return new JsNDArray(result, [nVars, nVars]);
    } else {
      // Compute covariance between x and y (both 1D)
      const xFlat = this.flatten(x);
      const yFlat = this.flatten(y);

      if (xFlat.data.length !== yFlat.data.length) {
        throw new Error('x and y must have same length');
      }

      const n = xFlat.data.length;
      const divisor = getDivisor(n);
      const xMean = this.mean(xFlat) as number;
      const yMean = this.mean(yFlat) as number;

      let covXY = 0;
      let varX = 0;
      let varY = 0;
      for (let i = 0; i < n; i++) {
        const dx = xFlat.data[i] - xMean;
        const dy = yFlat.data[i] - yMean;
        covXY += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
      }

      const d = divisor > 0 ? divisor : 1;
      return new JsNDArray(new Float64Array([varX / d, covXY / d, covXY / d, varY / d]), [2, 2]);
    }
  }

  corrcoef(x: NDArray, y?: NDArray, rowvar: boolean = true): NDArray {
    const covMatrix = this.cov(x, y, rowvar);
    const n = covMatrix.shape[0];
    const result = new Float64Array(n * n);

    // Correlation = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const covIJ = covMatrix.data[i * n + j];
        const varI = covMatrix.data[i * n + i];
        const varJ = covMatrix.data[j * n + j];
        result[i * n + j] = covIJ / Math.sqrt(varI * varJ);
      }
    }

    return new JsNDArray(result, [n, n]);
  }

  // ============ Convolution ============

  convolve(a: NDArray, v: NDArray, mode: 'full' | 'same' | 'valid' = 'full'): NDArray {
    const aFlat = this.flatten(a);
    const vFlat = this.flatten(v);
    const aLen = aFlat.data.length;
    const vLen = vFlat.data.length;

    let outLen: number;
    let startIdx: number;

    if (mode === 'full') {
      outLen = aLen + vLen - 1;
      startIdx = 0;
    } else if (mode === 'same') {
      outLen = aLen;
      startIdx = Math.floor((vLen - 1) / 2);
    } else {
      // valid
      outLen = Math.max(aLen - vLen + 1, 0);
      startIdx = vLen - 1;
    }

    const result = new Float64Array(outLen);

    // Full convolution
    const fullLen = aLen + vLen - 1;
    const full = new Float64Array(fullLen);

    for (let i = 0; i < fullLen; i++) {
      let sum = 0;
      for (let j = 0; j < vLen; j++) {
        const aIdx = i - j;
        if (aIdx >= 0 && aIdx < aLen) {
          sum += aFlat.data[aIdx] * vFlat.data[j];
        }
      }
      full[i] = sum;
    }

    // Extract the relevant portion
    for (let i = 0; i < outLen; i++) {
      result[i] = full[startIdx + i];
    }

    return new JsNDArray(result, [outLen]);
  }

  correlate(a: NDArray, v: NDArray, mode: 'full' | 'same' | 'valid' = 'valid'): NDArray {
    // Correlation is convolution with reversed v
    const vFlat = this.flatten(v);
    const vReversed = new JsNDArray(new Float64Array([...vFlat.data].reverse()), vFlat.shape);
    return this.convolve(a, vReversed, mode);
  }

  // ============ Matrix Creation ============

  identity(n: number, dtype: DType = 'float64'): NDArray {
    return this.eye(n, dtype);
  }

  tril(arr: NDArray, k: number = 0): NDArray {
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

    return new JsNDArray(result, [rows, cols]);
  }

  triu(arr: NDArray, k: number = 0): NDArray {
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

    return new JsNDArray(result, [rows, cols]);
  }

  // ============ Grid Creation ============

  meshgrid(...args: (NDArray | 'xy' | 'ij')[]): NDArray[] {
    // Separate arrays from indexing option
    let indexing: 'xy' | 'ij' = 'xy';
    const arrays: NDArray[] = [];
    for (const arg of args) {
      if (arg === 'xy' || arg === 'ij') {
        indexing = arg;
      } else {
        arrays.push(arg as NDArray);
      }
    }

    if (arrays.length < 2) throw new Error('meshgrid requires at least 2 arrays');

    const sizes = arrays.map(a => this.flatten(a).data.length);
    const flatArrays = arrays.map(a => this.flatten(a));
    const n = arrays.length;

    // Output shape: 'ij' -> [n0, n1, ...], 'xy' -> [n1, n0, n2, ...]
    let outputShape: number[];
    if (indexing === 'ij') {
      outputShape = sizes;
    } else {
      outputShape = [sizes[1], sizes[0], ...sizes.slice(2)];
    }

    const totalSize = outputShape.reduce((a, b) => a * b, 1);

    // Compute strides for the output array
    const strides: number[] = new Array(n);
    let stride = 1;
    for (let d = n - 1; d >= 0; d--) {
      strides[d] = stride;
      stride *= outputShape[d];
    }

    const result: NDArray[] = [];
    for (let dim = 0; dim < n; dim++) {
      const data = new Float64Array(totalSize);
      const arrData = flatArrays[dim].data;

      // For 'ij' indexing, dim maps directly to output dimension dim
      // For 'xy' indexing, dims 0 and 1 are swapped in output
      let outDim = dim;
      if (indexing === 'xy' && dim < 2) {
        outDim = 1 - dim; // swap 0<->1
      }

      for (let i = 0; i < totalSize; i++) {
        const coord = Math.floor(i / strides[outDim]) % outputShape[outDim];
        data[i] = arrData[coord];
      }

      result.push(new JsNDArray(data, [...outputShape]));
    }

    return result;
  }

  logspace(
    start: number,
    stop: number,
    num: number,
    base: number = 10,
    endpoint: boolean = true,
    dtype: DType = 'float64'
  ): NDArray {
    const linear = this.linspace(start, stop, num, endpoint);
    const result = createTypedArray(dtype, num);
    for (let i = 0; i < num; i++) {
      result[i] = Math.pow(base, linear.data[i]);
    }
    return new JsNDArray(result, [num], dtype);
  }

  geomspace(
    start: number,
    stop: number,
    num: number,
    endpoint: boolean = true,
    dtype: DType = 'float64'
  ): NDArray {
    if (start === 0 || stop === 0) {
      throw new Error('geomspace: start and stop must be non-zero');
    }
    if (start < 0 !== stop < 0) {
      throw new Error('geomspace: start and stop must have same sign');
    }

    const logStart = Math.log(Math.abs(start));
    const logStop = Math.log(Math.abs(stop));
    const linear = this.linspace(logStart, logStop, num, endpoint);
    const result = createTypedArray(dtype, num);
    const sign = start < 0 ? -1 : 1;

    for (let i = 0; i < num; i++) {
      result[i] = sign * Math.exp(linear.data[i]);
    }

    return new JsNDArray(result, [num], dtype);
  }

  // ============ Stacking Shortcuts ============

  vstack(arrays: NDArray[]): NDArray {
    // Stack arrays vertically (row-wise)
    const processed = arrays.map(arr => {
      if (arr.shape.length === 1) {
        return this.reshape(arr, [1, arr.shape[0]]);
      }
      return arr;
    });
    return this.concatenate(processed, 0);
  }

  rowStack(arrays: NDArray[]): NDArray {
    return this.vstack(arrays);
  }

  hstack(arrays: NDArray[]): NDArray {
    // Stack arrays horizontally (column-wise)
    if (arrays[0].shape.length === 1) {
      return this.concatenate(arrays, 0);
    }
    return this.concatenate(arrays, 1);
  }

  dstack(arrays: NDArray[]): NDArray {
    // Stack arrays along third axis
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

  vsplit(arr: NDArray, indices: number | number[]): NDArray[] {
    if (arr.shape.length < 2) {
      throw new Error('vsplit requires array with at least 2 dimensions');
    }
    return this.split(arr, indices, 0);
  }

  hsplit(arr: NDArray, indices: number | number[]): NDArray[] {
    if (arr.shape.length === 1) {
      return this.split(arr, indices, 0);
    }
    return this.split(arr, indices, 1);
  }

  dsplit(arr: NDArray, indices: number | number[]): NDArray[] {
    if (arr.shape.length < 3) {
      throw new Error('dsplit requires array with at least 3 dimensions');
    }
    return this.split(arr, indices, 2);
  }

  // ============ Array Replication ============

  tile(arr: NDArray, reps: number | number[]): NDArray {
    const repsArray = Array.isArray(reps) ? reps : [reps];

    // Pad reps to match arr dimensions
    const ndim = Math.max(arr.shape.length, repsArray.length);
    const paddedReps = new Array(ndim - repsArray.length).fill(1).concat(repsArray);
    const paddedShape = new Array(ndim - arr.shape.length).fill(1).concat(arr.shape);

    // Compute output shape
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

      // Map to source coordinates (modulo original shape)
      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += (coords[d] % paddedShape[d]) * srcStrides[d];
      }

      result[i] = arr.data[srcIdx];
    }

    return new JsNDArray(result, outShape);
  }

  repeat(arr: NDArray, repeats: number, axis?: number): NDArray {
    if (axis === undefined) {
      // Repeat on flattened array
      const flat = this.flatten(arr);
      const result = new Float64Array(flat.data.length * repeats);
      for (let i = 0; i < flat.data.length; i++) {
        for (let j = 0; j < repeats; j++) {
          result[i * repeats + j] = flat.data[i];
        }
      }
      return new JsNDArray(result, [result.length]);
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

      // Map axis coordinate back to source
      coords[axis] = Math.floor(coords[axis] / repeats);

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return new JsNDArray(result, outShape);
  }

  // ============ Index Finding ============

  nonzero(arr: NDArray): NDArray[] {
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

    return indices.map(idx => new JsNDArray(new Float64Array(idx), [idx.length]));
  }

  argwhere(arr: NDArray): NDArray {
    const indices = this.nonzero(arr);
    if (indices.length === 0 || indices[0].data.length === 0) {
      return new JsNDArray(new Float64Array(0), [0, arr.shape.length]);
    }

    const nNonzero = indices[0].data.length;
    const ndim = arr.shape.length;
    const result = new Float64Array(nNonzero * ndim);

    for (let i = 0; i < nNonzero; i++) {
      for (let d = 0; d < ndim; d++) {
        result[i * ndim + d] = indices[d].data[i];
      }
    }

    return new JsNDArray(result, [nNonzero, ndim]);
  }

  flatnonzero(arr: NDArray): NDArray {
    const indices: number[] = [];
    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] !== 0) {
        indices.push(i);
      }
    }
    return new JsNDArray(new Float64Array(indices), [indices.length]);
  }

  // ============ Value Handling ============

  nanToNum(arr: NDArray, nan: number = 0, posInf?: number, negInf?: number): NDArray {
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

    return new JsNDArray(result, arr.shape);
  }

  // ============ Sorting ============

  sort(arr: NDArray, axis: number = -1, _kind?: SortKind): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const result = new Float64Array(arr.data);
    const shape = arr.shape;
    const strides = this._computeStrides(shape);
    const axisLen = shape[axis];

    // Compute the stride pattern excluding the sort axis
    const outerShape = shape.filter((_, i) => i !== axis);
    const outerStrides = outerShape.length > 0 ? this._computeStrides(outerShape) : [1];
    const outerSize = outerShape.reduce((a, b) => a * b, 1) || 1;

    for (let outerIdx = 0; outerIdx < outerSize; outerIdx++) {
      // Extract 1D slice
      const slice = new Float64Array(axisLen);
      const outerCoords = new Array(outerShape.length);
      let remaining = outerIdx;
      for (let d = 0; d < outerShape.length; d++) {
        outerCoords[d] = Math.floor(remaining / outerStrides[d]);
        remaining = remaining % outerStrides[d];
      }

      // Map outer coords to full coords
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

      // Sort (handling NaN - they go to end)
      const sorted = Array.from(slice).sort((a, b) => {
        if (Number.isNaN(a) && Number.isNaN(b)) return 0;
        if (Number.isNaN(a)) return 1;
        if (Number.isNaN(b)) return -1;
        return a - b;
      });

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

    return new JsNDArray(result, shape);
  }

  argsort(arr: NDArray, axis: number = -1, _kind?: SortKind): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const result = new Float64Array(arr.data.length);
    const shape = arr.shape;
    const strides = this._computeStrides(shape);
    const axisLen = shape[axis];

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

      // Create index array and sort by values
      const indices = Array.from({ length: axisLen }, (_, i) => i);
      const values = new Array(axisLen);

      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        values[i] = arr.data[idx];
      }

      indices.sort((a, b) => {
        const va = values[a],
          vb = values[b];
        if (Number.isNaN(va) && Number.isNaN(vb)) return 0;
        if (Number.isNaN(va)) return 1;
        if (Number.isNaN(vb)) return -1;
        return va - vb;
      });

      // Write indices back
      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        result[idx] = indices[i];
      }
    }

    return new JsNDArray(result, shape);
  }

  searchsorted(
    arr: NDArray,
    v: number | NDArray,
    side: 'left' | 'right' = 'left',
    sorter?: NDArray
  ): NDArray | number {
    const flat = this.flatten(arr);
    // If sorter is provided, reindex the array through the sorter permutation
    let data: AnyTypedArray;
    if (sorter) {
      const sortedData = new Float64Array(flat.data.length);
      for (let i = 0; i < flat.data.length; i++) {
        sortedData[i] = flat.data[sorter.data[i]];
      }
      data = sortedData;
    } else {
      data = flat.data;
    }

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
      return new JsNDArray(result, vFlat.shape);
    }
  }

  unique(
    arr: NDArray,
    returnIndex?: boolean,
    returnInverse?: boolean,
    returnCounts?: boolean
  ): NDArray | { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray } {
    const flat = this.flatten(arr);
    const data = Array.from(flat.data);
    const seen = new Set<number>();
    const result: number[] = [];

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!seen.has(v)) {
        seen.add(v);
        result.push(v);
      }
    }

    // Sort the result
    result.sort((a, b) => {
      if (Number.isNaN(a) && Number.isNaN(b)) return 0;
      if (Number.isNaN(a)) return 1;
      if (Number.isNaN(b)) return -1;
      return a - b;
    });

    const values = new JsNDArray(new Float64Array(result), [result.length]);

    if (!returnIndex && !returnInverse && !returnCounts) {
      return values;
    }

    const out: { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray } = {
      values,
    };

    if (returnIndex) {
      const indices = result.map(v => data.indexOf(v));
      out.indices = new JsNDArray(new Float64Array(indices), [indices.length]);
    }

    if (returnInverse) {
      const valueMap = new Map(result.map((v, i) => [v, i]));
      const inverse = data.map(v => valueMap.get(v)!);
      out.inverse = new JsNDArray(new Float64Array(inverse), [inverse.length]);
    }

    if (returnCounts) {
      const countMap = new Map<number, number>();
      for (const v of data) countMap.set(v, (countMap.get(v) || 0) + 1);
      const counts = result.map(v => countMap.get(v)!);
      out.counts = new JsNDArray(new Float64Array(counts), [counts.length]);
    }

    return out;
  }

  // No-op for JS backend (all data is already on CPU)
  async materializeAll(): Promise<void> {
    // Nothing to do
  }

  // ============ NaN-aware Stats ============

  /**
   * Generic axis reduction helper: for each slice along the given axis,
   * gathers values into a flat array and calls the scalar reducer.
   */
  private _reduceAlongAxis(
    arr: NDArray,
    axis: number,
    reducer: (vals: Float64Array) => number
  ): NDArray {
    const shape = arr.shape;
    if (axis < 0 || axis >= shape.length) throw new Error(`Invalid axis ${axis}`);
    const axisLen = shape[axis];
    const resultShape = shape.filter((_, i) => i !== axis);
    if (resultShape.length === 0) resultShape.push(1);
    const resultSize = resultShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(resultSize);

    // Compute strides for the source array
    const strides: number[] = new Array(shape.length);
    strides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];

    // Compute strides for the result array
    const resultStrides: number[] = new Array(resultShape.length);
    if (resultShape.length > 0) {
      resultStrides[resultShape.length - 1] = 1;
      for (let i = resultShape.length - 2; i >= 0; i--)
        resultStrides[i] = resultStrides[i + 1] * resultShape[i + 1];
    }

    const sliceVals = new Float64Array(axisLen);
    for (let ri = 0; ri < resultSize; ri++) {
      // Convert flat result index to multi-dim coords in result space
      let tmp = ri;
      const srcCoords: number[] = new Array(shape.length);
      let rDim = 0;
      for (let d = 0; d < shape.length; d++) {
        if (d === axis) {
          srcCoords[d] = 0; // will iterate
        } else {
          srcCoords[d] = Math.floor(tmp / resultStrides[rDim]);
          tmp %= resultStrides[rDim];
          rDim++;
        }
      }
      // Gather values along the axis
      for (let k = 0; k < axisLen; k++) {
        srcCoords[axis] = k;
        let flatIdx = 0;
        for (let d = 0; d < shape.length; d++) flatIdx += srcCoords[d] * strides[d];
        sliceVals[k] = arr.data[flatIdx];
      }
      result[ri] = reducer(sliceVals);
    }
    return new JsNDArray(result, resultShape);
  }

  nansum(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let sum = 0;
        for (let i = 0; i < vals.length; i++) if (!Number.isNaN(vals[i])) sum += vals[i];
        return sum;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) sum += arr.data[i];
    }
    return sum;
  }

  nanmean(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let sum = 0,
          count = 0;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i])) {
            sum += vals[i];
            count++;
          }
        return count > 0 ? sum / count : NaN;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
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

  nanvar(
    arr: NDArray,
    axis?: number | null,
    ddof: number = 0,
    keepdims?: boolean
  ): number | NDArray {
    if (axis !== undefined && axis !== null) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let sum = 0,
          count = 0;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i])) {
            sum += vals[i];
            count++;
          }
        if (count === 0) return NaN;
        const mean = sum / count;
        let sumSq = 0;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i])) {
            const d = vals[i] - mean;
            sumSq += d * d;
          }
        return count > ddof ? sumSq / (count - ddof) : NaN;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const mean = this.nanmean(arr) as number;
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

  nanstd(
    arr: NDArray,
    axis?: number | null,
    ddof: number = 0,
    keepdims?: boolean
  ): number | NDArray {
    if (axis !== undefined && axis !== null) {
      const variance = this.nanvar(arr, axis, ddof) as NDArray;
      const result = this.sqrt(variance);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    return Math.sqrt(this.nanvar(arr, null, ddof) as number);
  }

  nanmin(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let min = Infinity;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i]) && vals[i] < min) min = vals[i];
        return min === Infinity ? NaN : min;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    let min = Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] < min) min = arr.data[i];
    }
    return min === Infinity ? NaN : min;
  }

  nanmax(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let max = -Infinity;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i]) && vals[i] > max) max = vals[i];
        return max === -Infinity ? NaN : max;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    let max = -Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] > max) max = arr.data[i];
    }
    return max === -Infinity ? NaN : max;
  }

  nanargmin(arr: NDArray, axis?: number): number | NDArray {
    if (axis !== undefined) {
      return this._reduceAlongAxis(arr, axis, vals => {
        let minIdx = -1,
          minVal = Infinity;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i]) && vals[i] < minVal) {
            minVal = vals[i];
            minIdx = i;
          }
        return minIdx;
      });
    }
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

  nanargmax(arr: NDArray, axis?: number): number | NDArray {
    if (axis !== undefined) {
      return this._reduceAlongAxis(arr, axis, vals => {
        let maxIdx = -1,
          maxVal = -Infinity;
        for (let i = 0; i < vals.length; i++)
          if (!Number.isNaN(vals[i]) && vals[i] > maxVal) {
            maxVal = vals[i];
            maxIdx = i;
          }
        return maxIdx;
      });
    }
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

  nanprod(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        let prod = 1;
        for (let i = 0; i < vals.length; i++) if (!Number.isNaN(vals[i])) prod *= vals[i];
        return prod;
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) prod *= arr.data[i];
    }
    return prod;
  }

  // ============ Order Statistics ============

  private _sortedData(arr: NDArray): number[] {
    return Array.from(arr.data).sort((a, b) => a - b);
  }

  private _sortedNonNaN(arr: NDArray): number[] {
    return Array.from(arr.data)
      .filter(x => !Number.isNaN(x))
      .sort((a, b) => a - b);
  }

  private static _medianOfSorted(sorted: number[]): number {
    if (sorted.length === 0) return NaN;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  private static _quantileOfSorted(sorted: number[], q: number): number {
    if (sorted.length === 0) return NaN;
    if (sorted.length === 1) return sorted[0];
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    const frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  median(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        const sorted = Array.from(vals).sort((a, b) => a - b);
        return JsBackend._medianOfSorted(sorted);
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const sorted = this._sortedData(arr);
    return JsBackend._medianOfSorted(sorted);
  }

  percentile(arr: NDArray, q: number, axis?: number, keepdims?: boolean): number | NDArray {
    if (q < 0 || q > 100) throw new Error('percentile must be 0-100');
    const result = this.quantile(arr, q / 100, axis);
    if (keepdims && axis !== undefined && typeof result !== 'number') {
      const newShape = [...arr.shape];
      newShape[axis] = 1;
      return this.reshape(result, newShape);
    }
    return result;
  }

  quantile(arr: NDArray, q: number, axis?: number): number | NDArray {
    if (q < 0 || q > 1) throw new Error('quantile must be 0-1');
    if (axis !== undefined) {
      return this._reduceAlongAxis(arr, axis, vals => {
        const sorted = Array.from(vals).sort((a, b) => a - b);
        return JsBackend._quantileOfSorted(sorted, q);
      });
    }
    const sorted = this._sortedData(arr);
    return JsBackend._quantileOfSorted(sorted, q);
  }

  nanmedian(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        const sorted = Array.from(vals)
          .filter(x => !Number.isNaN(x))
          .sort((a, b) => a - b);
        return JsBackend._medianOfSorted(sorted);
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const sorted = this._sortedNonNaN(arr);
    return JsBackend._medianOfSorted(sorted);
  }

  nanpercentile(arr: NDArray, q: number, axis?: number, keepdims?: boolean): number | NDArray {
    if (q < 0 || q > 100) throw new Error('percentile must be 0-100');
    if (axis !== undefined) {
      const result = this._reduceAlongAxis(arr, axis, vals => {
        const sorted = Array.from(vals)
          .filter(x => !Number.isNaN(x))
          .sort((a, b) => a - b);
        return JsBackend._quantileOfSorted(sorted, q / 100);
      });
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    const sorted = this._sortedNonNaN(arr);
    if (sorted.length === 0) return NaN;
    if (sorted.length === 1) return sorted[0];
    const pos = (q / 100) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    const frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  // ============ Histogram ============

  private _computeOptimalBins(data: AnyTypedArray, strategy: string): number {
    // Filter out NaN values
    const valid: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (!Number.isNaN(data[i])) valid.push(data[i]);
    }
    const n = valid.length;
    if (n === 0) return 10;

    valid.sort((a, b) => a - b);
    const min = valid[0];
    const max = valid[n - 1];
    const range = max - min || 1;

    // Helper: IQR
    const q1Idx = Math.floor(n * 0.25);
    const q3Idx = Math.floor(n * 0.75);
    const iqr = valid[q3Idx] - valid[q1Idx];

    // Helper: std
    let sum = 0;
    for (let i = 0; i < n; i++) sum += valid[i];
    const mean = sum / n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (valid[i] - mean) ** 2;
    const std = Math.sqrt(variance / n);

    switch (strategy) {
      case 'sqrt':
        return Math.max(1, Math.ceil(Math.sqrt(n)));
      case 'sturges':
        return Math.max(1, Math.ceil(Math.log2(n) + 1));
      case 'rice':
        return Math.max(1, Math.ceil(2 * Math.cbrt(n)));
      case 'scott': {
        const h = 3.5 * std * Math.pow(n, -1 / 3);
        return Math.max(1, Math.ceil(range / (h || 1)));
      }
      case 'fd': {
        const h = 2 * iqr * Math.pow(n, -1 / 3);
        return Math.max(1, Math.ceil(range / (h || 1)));
      }
      case 'doane': {
        if (n < 3) return 1;
        // Skewness
        let g1 = 0;
        for (let i = 0; i < n; i++) g1 += ((valid[i] - mean) / (std || 1)) ** 3;
        g1 /= n;
        const sigmaG1 = Math.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)));
        return Math.max(
          1,
          Math.ceil(1 + Math.log2(n) + Math.log2(1 + Math.abs(g1) / (sigmaG1 || 1)))
        );
      }
      case 'auto': {
        // auto uses max of sturges and fd
        const sturges = Math.max(1, Math.ceil(Math.log2(n) + 1));
        const hFd = 2 * iqr * Math.pow(n, -1 / 3);
        const fd = hFd > 0 ? Math.max(1, Math.ceil(range / hFd)) : sturges;
        return Math.max(sturges, fd);
      }
      default:
        return 10;
    }
  }

  histogram(
    arr: NDArray,
    bins: BinsParam = 10,
    range?: [number, number] | null,
    density?: boolean,
    weights?: NDArray
  ): { hist: NDArray; binEdges: NDArray } {
    const data = arr.data;

    // If bins is an NDArray, use those as bin edges directly
    if (typeof bins === 'object' && bins !== null && 'data' in bins) {
      const edges = bins as NDArray;
      const numBins = edges.data.length - 1;
      const hist = new Float64Array(numBins);
      for (let i = 0; i < data.length; i++) {
        if (Number.isNaN(data[i])) continue;
        const val = data[i];
        // Find which bin this value falls into
        let binIdx = -1;
        for (let b = 0; b < numBins; b++) {
          if (b === numBins - 1) {
            // Last bin is inclusive on the right
            if (val >= edges.data[b] && val <= edges.data[b + 1]) {
              binIdx = b;
              break;
            }
          } else {
            if (val >= edges.data[b] && val < edges.data[b + 1]) {
              binIdx = b;
              break;
            }
          }
        }
        if (binIdx >= 0) {
          const w = weights ? weights.data[i] : 1;
          hist[binIdx] += w;
        }
      }

      if (density) {
        let totalWeight = 0;
        for (let i = 0; i < numBins; i++) totalWeight += hist[i];
        if (totalWeight > 0) {
          for (let i = 0; i < numBins; i++) {
            const binWidth = edges.data[i + 1] - edges.data[i];
            hist[i] = hist[i] / (totalWeight * (binWidth || 1));
          }
        }
      }

      return {
        hist: new JsNDArray(hist, [numBins]),
        binEdges: new JsNDArray(new Float64Array(edges.data), [...edges.shape]),
      };
    }

    // If bins is a string, compute optimal bin count
    let numBins: number;
    if (typeof bins === 'string') {
      numBins = this._computeOptimalBins(data, bins);
    } else {
      numBins = bins;
    }

    let min: number, max: number;

    if (range != null) {
      [min, max] = range;
    } else {
      min = Infinity;
      max = -Infinity;
      for (let i = 0; i < data.length; i++) {
        if (!Number.isNaN(data[i])) {
          if (data[i] < min) min = data[i];
          if (data[i] > max) max = data[i];
        }
      }
    }

    if (min === Infinity || (min === max && range == null)) {
      if (min === Infinity) {
        // All NaN
        return {
          hist: new JsNDArray(new Float64Array(numBins), [numBins]),
          binEdges: new JsNDArray(new Float64Array(numBins + 1), [numBins + 1]),
        };
      }
    }

    const rangeSpan = max - min;
    const binWidth = rangeSpan / numBins || 1;
    const edges = new Float64Array(numBins + 1);
    for (let i = 0; i <= numBins; i++) edges[i] = min + i * binWidth;

    const hist = new Float64Array(numBins);
    for (let i = 0; i < data.length; i++) {
      if (Number.isNaN(data[i])) continue;
      // Skip values outside range when range is specified
      if (range != null && (data[i] < min || data[i] > max)) continue;
      let binIdx = Math.floor((data[i] - min) / binWidth);
      if (binIdx >= numBins) binIdx = numBins - 1;
      if (binIdx < 0) binIdx = 0;
      const w = weights ? weights.data[i] : 1;
      hist[binIdx] += w;
    }

    // Apply density normalization: normalize so integral over bins equals 1
    if (density) {
      let totalWeight = 0;
      for (let i = 0; i < numBins; i++) totalWeight += hist[i];
      if (totalWeight > 0) {
        for (let i = 0; i < numBins; i++) {
          hist[i] = hist[i] / (totalWeight * binWidth);
        }
      }
    }

    return {
      hist: new JsNDArray(hist, [numBins]),
      binEdges: new JsNDArray(edges, [numBins + 1]),
    };
  }

  histogramBinEdges(arr: NDArray, bins: BinsParam = 10): NDArray {
    const { binEdges } = this.histogram(arr, bins);
    return binEdges;
  }

  // ============ Random ============
  private _rngState: number = Date.now();

  private _xorshift(): number {
    // Simple xorshift32 PRNG for reproducibility
    let x = this._rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this._rngState = x >>> 0;
    return (this._rngState >>> 0) / 0xffffffff;
  }

  seed(s: number): void {
    this._rngState = s >>> 0;
    if (this._rngState === 0) this._rngState = 1; // Avoid zero state
  }

  rand(shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift();
    }
    return new JsNDArray(data, shape, dtype);
  }

  randn(shape: number[], dtype: DType = 'float64'): NDArray {
    // Box-Muller transform for normal distribution
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    for (let i = 0; i < size; i += 2) {
      const u1 = this._xorshift();
      const u2 = this._xorshift();
      const r = Math.sqrt(-2.0 * Math.log(u1 || 1e-10));
      const theta = 2.0 * Math.PI * u2;
      data[i] = r * Math.cos(theta);
      if (i + 1 < size) data[i + 1] = r * Math.sin(theta);
    }
    return new JsNDArray(data, shape, dtype);
  }

  randint(low: number, high: number, shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(this._xorshift() * range) + low;
    }
    return new JsNDArray(data, shape, dtype);
  }

  uniform(low: number, high: number, shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = createTypedArray(dtype, size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift() * range + low;
    }
    return new JsNDArray(data, shape, dtype);
  }

  normal(loc: number, scale: number, shape: number[], dtype: DType = 'float64'): NDArray {
    const arr = this.randn(shape, dtype);
    const data = arr.data;
    for (let i = 0; i < data.length; i++) {
      data[i] = data[i] * scale + loc;
    }
    return new JsNDArray(data, shape, dtype);
  }

  shuffle(arr: NDArray): void {
    // Fisher-Yates shuffle on first axis, in-place
    const data = arr.data;
    const shape = arr.shape;
    if (shape.length === 1) {
      // 1D shuffle
      for (let i = data.length - 1; i > 0; i--) {
        const j = Math.floor(this._xorshift() * (i + 1));
        const tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
      }
    } else {
      // Shuffle along first axis
      const stride = shape.slice(1).reduce((a, b) => a * b, 1);
      const n = shape[0];
      const temp = new Float64Array(stride);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this._xorshift() * (i + 1));
        // Swap row i with row j
        temp.set(data.subarray(i * stride, (i + 1) * stride));
        data.copyWithin(i * stride, j * stride, (j + 1) * stride);
        data.set(temp, j * stride);
      }
    }
  }

  choice(arr: NDArray, size: number, replace: boolean = true, p?: NDArray | number[]): NDArray {
    const n = arr.data.length;
    const data = new Float64Array(size);

    if (p !== undefined) {
      // Weighted sampling
      const weights = Array.isArray(p) ? p : Array.from(p.data);
      if (weights.length !== n) throw new Error('p must have same length as arr');

      // Build cumulative distribution function
      const cdf = new Float64Array(n);
      cdf[0] = weights[0];
      for (let i = 1; i < n; i++) {
        cdf[i] = cdf[i - 1] + weights[i];
      }

      if (replace) {
        for (let i = 0; i < size; i++) {
          const r = this._xorshift();
          // Binary search in CDF
          let lo = 0,
            hi = n - 1;
          while (lo < hi) {
            const mid = (lo + hi) >>> 1;
            if (cdf[mid] <= r) lo = mid + 1;
            else hi = mid;
          }
          data[i] = arr.data[lo];
        }
      } else {
        if (size > n) throw new Error('Cannot sample more than array size without replacement');
        // Weighted sampling without replacement: remove selected items
        const remaining = Array.from({ length: n }, (_, i) => i);
        const remainingWeights = [...weights];
        for (let i = 0; i < size; i++) {
          // Rebuild CDF for remaining items
          let totalWeight = 0;
          for (let j = 0; j < remaining.length; j++) totalWeight += remainingWeights[j];

          const r = this._xorshift() * totalWeight;
          let cumulative = 0;
          let chosen = 0;
          for (let j = 0; j < remaining.length; j++) {
            cumulative += remainingWeights[j];
            if (cumulative > r) {
              chosen = j;
              break;
            }
          }
          data[i] = arr.data[remaining[chosen]];
          remaining.splice(chosen, 1);
          remainingWeights.splice(chosen, 1);
        }
      }
    } else if (replace) {
      for (let i = 0; i < size; i++) {
        const idx = Math.floor(this._xorshift() * n);
        data[i] = arr.data[idx];
      }
    } else {
      if (size > n) throw new Error('Cannot sample more than array size without replacement');
      // Fisher-Yates partial shuffle
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < size; i++) {
        const j = i + Math.floor(this._xorshift() * (n - i));
        [indices[i], indices[j]] = [indices[j], indices[i]];
        data[i] = arr.data[indices[i]];
      }
    }
    return new JsNDArray(data, [size]);
  }

  permutation(n: number | NDArray): NDArray {
    let arr: NDArray;
    if (typeof n === 'number') {
      arr = this.arange(0, n, 1);
    } else {
      arr = new JsNDArray(new Float64Array(n.data), [...n.shape]);
    }
    this.shuffle(arr);
    return arr;
  }

  // ============ Logic ============

  logicalAnd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x !== 0 && y !== 0 ? 1 : 0));
  }

  logicalOr(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x !== 0 || y !== 0 ? 1 : 0));
  }

  logicalNot(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = arr.data[i] === 0 ? 1 : 0;
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  logicalXor(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => ((x !== 0) !== (y !== 0) ? 1 : 0));
  }

  private _iscloseScalar(x: number, y: number, rtol: number, atol: number): boolean {
    // NumPy isclose: handles inf, -inf, NaN specially
    if (x === y) return true; // handles inf==inf, -inf==-inf
    if (Number.isNaN(x) || Number.isNaN(y)) return false;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    return Math.abs(x - y) <= atol + rtol * Math.abs(y);
  }

  isclose(a: NDArray, b: NDArray, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = this._iscloseScalar(a.data[i], b.data[i], rtol, atol) ? 1 : 0;
    }
    return new JsNDArray(data, [...a.shape]);
  }

  allclose(a: NDArray, b: NDArray, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    this._checkSameShape(a, b);
    for (let i = 0; i < a.data.length; i++) {
      if (!this._iscloseScalar(a.data[i], b.data[i], rtol, atol)) {
        return false;
      }
    }
    return true;
  }

  arrayEqual(a: NDArray, b: NDArray): boolean {
    if (a.shape.length !== b.shape.length) return false;
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) return false;
    }
    for (let i = 0; i < a.data.length; i++) {
      if (a.data[i] !== b.data[i]) return false;
    }
    return true;
  }

  // ============ Bitwise ============

  bitwiseAnd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x | 0) & (y | 0));
  }

  bitwiseOr(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => x | 0 | (y | 0));
  }

  bitwiseXor(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x | 0) ^ (y | 0));
  }

  bitwiseNot(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = ~(arr.data[i] | 0);
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  leftShift(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x | 0) << (y | 0));
  }

  rightShift(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => (x | 0) >> (y | 0));
  }

  // ============ Array Manipulation (Additional) ============

  copy(arr: NDArray, dtype?: DType): NDArray {
    const dt = dtype ?? arr.dtype ?? 'float64';
    return new JsNDArray(createTypedArrayFrom(dt, Array.from(arr.data)), [...arr.shape], dt);
  }

  empty(shape: number[], dtype: DType = 'float64'): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JsNDArray(createTypedArray(dtype, size), shape, dtype);
  }

  flip(arr: NDArray, axis?: number): NDArray {
    if (axis === undefined) {
      // Flip all elements
      const data = new Float64Array(arr.data.length);
      for (let i = 0; i < data.length; i++) {
        data[i] = arr.data[data.length - 1 - i];
      }
      return new JsNDArray(data, [...arr.shape]);
    }
    const ndim = arr.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const data = new Float64Array(arr.data.length);
    const strides = this._computeStrides(arr.shape);
    const totalSize = arr.data.length;

    for (let idx = 0; idx < totalSize; idx++) {
      // Compute multi-index
      let rem = idx;
      const multiIdx: number[] = [];
      for (let d = 0; d < ndim; d++) {
        multiIdx.push(Math.floor(rem / strides[d]));
        rem = rem % strides[d];
      }
      // Flip the axis
      multiIdx[ax] = arr.shape[ax] - 1 - multiIdx[ax];
      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += multiIdx[d] * strides[d];
      }
      data[idx] = arr.data[srcIdx];
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  fliplr(arr: NDArray): NDArray {
    if (arr.shape.length < 2) throw new Error('fliplr requires at least 2D');
    return this.flip(arr, 1);
  }

  flipud(arr: NDArray): NDArray {
    if (arr.shape.length < 1) throw new Error('flipud requires at least 1D');
    return this.flip(arr, 0);
  }

  roll(arr: NDArray, shift: number | number[], axis?: number | number[]): NDArray {
    if (axis === undefined) {
      // Roll flat — if shift is array, sum the shifts
      const totalShift = Array.isArray(shift) ? shift.reduce((a, b) => a + b, 0) : shift;
      const n = arr.data.length;
      const s = ((totalShift % n) + n) % n;
      const data = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        data[(i + s) % n] = arr.data[i];
      }
      return new JsNDArray(data, [...arr.shape]);
    }

    // Normalize to arrays
    const shifts = Array.isArray(shift) ? shift : [shift];
    const axes = Array.isArray(axis) ? axis : [axis];
    if (shifts.length !== axes.length) {
      throw new Error('shift and axis must have the same length');
    }

    // Apply rolls sequentially for each axis
    let result = arr;
    for (let k = 0; k < shifts.length; k++) {
      const ndim = result.shape.length;
      const ax = axes[k] < 0 ? axes[k] + ndim : axes[k];
      const axLen = result.shape[ax];
      const s = ((shifts[k] % axLen) + axLen) % axLen;
      const data = new Float64Array(result.data.length);
      const strides = this._computeStrides(result.shape);
      const totalSize = result.data.length;

      for (let idx = 0; idx < totalSize; idx++) {
        let rem = idx;
        const multiIdx: number[] = [];
        for (let d = 0; d < ndim; d++) {
          multiIdx.push(Math.floor(rem / strides[d]));
          rem = rem % strides[d];
        }
        const srcMultiIdx = [...multiIdx];
        srcMultiIdx[ax] = (multiIdx[ax] - s + axLen) % axLen;
        let srcIdx = 0;
        for (let d = 0; d < ndim; d++) {
          srcIdx += srcMultiIdx[d] * strides[d];
        }
        data[idx] = result.data[srcIdx];
      }
      result = new JsNDArray(data, [...result.shape]);
    }
    return result;
  }

  rot90(arr: NDArray, k: number = 1, axes?: [number, number]): NDArray {
    if (arr.shape.length < 2) throw new Error('rot90 requires at least 2D');
    const [ax0, ax1] = axes
      ? [
          this._normalizeAxis(axes[0], arr.shape.length),
          this._normalizeAxis(axes[1], arr.shape.length),
        ]
      : [0, 1];
    if (ax0 === ax1) throw new Error('axes must be different');

    const nk = ((k % 4) + 4) % 4;
    if (nk === 0) return this.copy(arr);

    let result = arr;
    for (let i = 0; i < nk; i++) {
      // Rotate 90 degrees counterclockwise in the plane (ax0, ax1):
      // 1. Swap ax0 and ax1 via transpose
      const perm = Array.from({ length: result.shape.length }, (_, j) => j);
      perm[ax0] = ax1;
      perm[ax1] = ax0;
      result = this.transpose(result, perm);
      // 2. Flip along ax0
      result = this.flip(result, ax0);
    }
    return result;
  }

  ravel(arr: NDArray): NDArray {
    return new JsNDArray(new Float64Array(arr.data), [arr.data.length]);
  }

  private _pad1dIndex(i: number, padBefore: number, n: number, mode: string): number {
    if (mode === 'edge') return i < padBefore ? 0 : n - 1;
    if (mode === 'reflect') {
      // reflect does NOT include edge: [3,2, | 1,2,3, | 2,1]
      return i < padBefore ? padBefore - i : n - 2 - (i - padBefore - n);
    }
    if (mode === 'symmetric') {
      // symmetric INCLUDES edge: [2,1, | 1,2,3, | 3,2]
      return i < padBefore ? padBefore - 1 - i : n - 1 - (i - padBefore - n);
    }
    if (mode === 'wrap') {
      return i < padBefore ? (((n - padBefore + i) % n) + n) % n : (i - padBefore - n) % n;
    }
    return -1; // should not happen
  }

  pad(
    arr: NDArray,
    padWidth: number | [number, number],
    mode:
      | 'constant'
      | 'edge'
      | 'reflect'
      | 'wrap'
      | 'symmetric'
      | 'linear_ramp'
      | 'mean'
      | 'minimum'
      | 'maximum' = 'constant',
    constantValue: number = 0
  ): NDArray {
    const [padBefore, padAfter] = typeof padWidth === 'number' ? [padWidth, padWidth] : padWidth;

    if (arr.shape.length === 1) {
      const n = arr.shape[0];
      const newLen = n + padBefore + padAfter;
      const data = new Float64Array(newLen);

      // Fill center
      for (let i = 0; i < n; i++) {
        data[padBefore + i] = arr.data[i];
      }

      // Compute stat values for stat-based modes
      let statVal = 0;
      if (mode === 'mean' || mode === 'minimum' || mode === 'maximum') {
        if (mode === 'mean') {
          let sum = 0;
          for (let i = 0; i < n; i++) sum += arr.data[i];
          statVal = sum / n;
        } else if (mode === 'minimum') {
          statVal = arr.data[0];
          for (let i = 1; i < n; i++) if (arr.data[i] < statVal) statVal = arr.data[i];
        } else {
          statVal = arr.data[0];
          for (let i = 1; i < n; i++) if (arr.data[i] > statVal) statVal = arr.data[i];
        }
      }

      // Fill padding before
      for (let i = 0; i < padBefore; i++) {
        if (mode === 'constant') {
          data[i] = constantValue;
        } else if (mode === 'mean' || mode === 'minimum' || mode === 'maximum') {
          data[i] = statVal;
        } else if (mode === 'linear_ramp') {
          // Linearly ramp from constantValue (at outermost) to edge value
          const edgeVal = arr.data[0];
          const t = padBefore > 1 ? (padBefore - 1 - i) / padBefore : 0;
          data[i] = edgeVal + t * (constantValue - edgeVal);
        } else {
          const srcIdx = this._pad1dIndex(i, padBefore, n, mode);
          data[i] = arr.data[Math.max(0, Math.min(n - 1, srcIdx))];
        }
      }
      // Fill padding after
      for (let i = 0; i < padAfter; i++) {
        const outIdx = padBefore + n + i;
        if (mode === 'constant') {
          data[outIdx] = constantValue;
        } else if (mode === 'mean' || mode === 'minimum' || mode === 'maximum') {
          data[outIdx] = statVal;
        } else if (mode === 'linear_ramp') {
          const edgeVal = arr.data[n - 1];
          // fix: at i=0 we are at edge, ramp toward constantValue
          data[outIdx] = edgeVal + ((i + 1) / padAfter) * (constantValue - edgeVal);
        } else {
          const srcIdx = this._pad1dIndex(padBefore + n + i, padBefore, n, mode);
          data[outIdx] = arr.data[Math.max(0, Math.min(n - 1, srcIdx))];
        }
      }

      return new JsNDArray(data, [newLen]);
    }

    // 2D case: pad each axis
    if (arr.shape.length === 2) {
      const [rows, cols] = arr.shape;
      const newRows = rows + padBefore + padAfter;
      const newCols = cols + padBefore + padAfter;
      const data = new Float64Array(newRows * newCols);

      // For stat-based modes, compute the stat value
      let statVal = 0;
      if (mode === 'mean' || mode === 'minimum' || mode === 'maximum') {
        if (mode === 'mean') {
          let sum = 0;
          for (let i = 0; i < arr.data.length; i++) sum += arr.data[i];
          statVal = sum / arr.data.length;
        } else if (mode === 'minimum') {
          statVal = arr.data[0];
          for (let i = 1; i < arr.data.length; i++)
            if (arr.data[i] < statVal) statVal = arr.data[i];
        } else {
          statVal = arr.data[0];
          for (let i = 1; i < arr.data.length; i++)
            if (arr.data[i] > statVal) statVal = arr.data[i];
        }
      }

      if (mode === 'constant') {
        data.fill(constantValue);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            data[(i + padBefore) * newCols + (j + padBefore)] = arr.data[i * cols + j];
          }
        }
      } else if (mode === 'mean' || mode === 'minimum' || mode === 'maximum') {
        data.fill(statVal);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            data[(i + padBefore) * newCols + (j + padBefore)] = arr.data[i * cols + j];
          }
        }
      } else {
        // For reflect, symmetric, edge, wrap, linear_ramp on 2D
        for (let i = 0; i < newRows; i++) {
          for (let j = 0; j < newCols; j++) {
            let srcRow: number, srcCol: number;

            if (mode === 'linear_ramp') {
              // For 2D linear_ramp, compute per-axis ramp factors and multiply
              let rowFactor = 1.0;
              let colFactor = 1.0;
              if (i < padBefore) {
                rowFactor = (i + 1) / (padBefore + 1);
              } else if (i >= padBefore + rows) {
                rowFactor = (newRows - i) / (padAfter + 1);
              }
              if (j < padBefore) {
                colFactor = (j + 1) / (padBefore + 1);
              } else if (j >= padBefore + cols) {
                colFactor = (newCols - j) / (padAfter + 1);
              }
              srcRow = Math.max(0, Math.min(rows - 1, i - padBefore));
              srcCol = Math.max(0, Math.min(cols - 1, j - padBefore));
              const val = arr.data[srcRow * cols + srcCol];
              data[i * newCols + j] = constantValue + (val - constantValue) * rowFactor * colFactor;
              continue;
            }

            // Compute source row index
            if (i >= padBefore && i < padBefore + rows) {
              srcRow = i - padBefore;
            } else {
              srcRow = this._pad1dIndex(i, padBefore, rows, mode);
            }

            // Compute source col index
            if (j >= padBefore && j < padBefore + cols) {
              srcCol = j - padBefore;
            } else {
              srcCol = this._pad1dIndex(j, padBefore, cols, mode);
            }

            srcRow = Math.max(0, Math.min(rows - 1, srcRow));
            srcCol = Math.max(0, Math.min(cols - 1, srcCol));
            data[i * newCols + j] = arr.data[srcRow * cols + srcCol];
          }
        }
      }
      return new JsNDArray(data, [newRows, newCols]);
    }

    throw new Error('pad only supports 1D and 2D arrays');
  }

  columnStack(arrays: NDArray[]): NDArray {
    // For 1D arrays, treat each as a column and stack
    if (arrays[0].shape.length === 1) {
      const rows = arrays[0].data.length;
      const cols = arrays.length;
      const data = new Float64Array(rows * cols);
      for (let j = 0; j < cols; j++) {
        if (arrays[j].data.length !== rows) throw new Error('All arrays must have same length');
        for (let i = 0; i < rows; i++) {
          data[i * cols + j] = arrays[j].data[i];
        }
      }
      return new JsNDArray(data, [rows, cols]);
    }
    // For 2D arrays, hstack
    return this.hstack(arrays);
  }

  arraySplit(arr: NDArray, indices: number | number[], axis: number = 0): NDArray[] {
    return this.split(arr, indices, axis);
  }

  putAlongAxis(arr: NDArray, indices: NDArray, values: NDArray, axis: number): NDArray {
    const ndim = arr.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const result = new Float64Array(arr.data);
    const strides = this._computeStrides(arr.shape);
    const shape = arr.shape;

    // Iterate over all elements in the indices/values arrays
    const idxStrides = this._computeStrides(indices.shape);
    const totalIdxSize = indices.data.length;

    for (let flatIdx = 0; flatIdx < totalIdxSize; flatIdx++) {
      // Compute multi-index of the indices array
      let rem = flatIdx;
      const multiIdx: number[] = [];
      for (let d = 0; d < indices.shape.length; d++) {
        multiIdx.push(Math.floor(rem / idxStrides[d]));
        rem = rem % idxStrides[d];
      }

      // Build target multi-index in arr
      const targetMultiIdx = [...multiIdx];
      targetMultiIdx[ax] = indices.data[flatIdx];

      let targetFlat = 0;
      for (let d = 0; d < ndim; d++) {
        targetFlat += targetMultiIdx[d] * strides[d];
      }
      result[targetFlat] = values.data[flatIdx];
    }
    return new JsNDArray(result, [...shape]);
  }

  takeAlongAxis(arr: NDArray, indices: NDArray, axis: number): NDArray {
    const ndim = arr.shape.length;
    const ax = axis < 0 ? axis + ndim : axis;
    const strides = this._computeStrides(arr.shape);
    const idxStrides = this._computeStrides(indices.shape);
    const totalIdxSize = indices.data.length;
    const result = new Float64Array(totalIdxSize);

    for (let flatIdx = 0; flatIdx < totalIdxSize; flatIdx++) {
      let rem = flatIdx;
      const multiIdx: number[] = [];
      for (let d = 0; d < indices.shape.length; d++) {
        multiIdx.push(Math.floor(rem / idxStrides[d]));
        rem = rem % idxStrides[d];
      }

      const srcMultiIdx = [...multiIdx];
      srcMultiIdx[ax] = indices.data[flatIdx];

      let srcFlat = 0;
      for (let d = 0; d < ndim; d++) {
        srcFlat += srcMultiIdx[d] * strides[d];
      }
      result[flatIdx] = arr.data[srcFlat];
    }
    return new JsNDArray(result, [...indices.shape]);
  }

  // ============ Additional Linalg ============

  eig(arr: NDArray): { values: NDArray; vectors: NDArray } {
    if (arr.shape.length !== 2) throw new Error('eig requires 2D');
    const [n, m] = arr.shape;
    if (n !== m) throw new Error('eig requires square matrix');

    // QR algorithm with shifts for eigenvalue decomposition
    const MAX_ITER = 200;
    const TOL = 1e-10;

    // Work on a copy (Hessenberg reduction would improve, but keep simple)
    let A = new Float64Array(arr.data);
    // Accumulate eigenvectors
    let V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;

    for (let iter = 0; iter < MAX_ITER; iter++) {
      // Wilkinson shift using trailing 2x2 submatrix
      let shift: number;
      if (n >= 2) {
        const a11 = A[(n - 2) * n + (n - 2)];
        const a12 = A[(n - 2) * n + (n - 1)];
        const a21 = A[(n - 1) * n + (n - 2)];
        const a22 = A[(n - 1) * n + (n - 1)];
        const delta = (a11 - a22) / 2;
        const sign = delta >= 0 ? 1 : -1;
        shift = a22 - (sign * a21 * a12) / (Math.abs(delta) + Math.sqrt(delta * delta + a21 * a12));
      } else {
        shift = A[0];
      }

      // Shift: A - shift * I
      for (let i = 0; i < n; i++) A[i * n + i] -= shift;

      // QR decomposition
      const qrResult = this.qr(new JsNDArray(new Float64Array(A), [n, n]));

      // A = R * Q + shift * I
      const RQ = this.matmul(qrResult.r, qrResult.q);
      A = new Float64Array(RQ.data);
      for (let i = 0; i < n; i++) A[i * n + i] += shift;

      // Accumulate eigenvectors: V = V * Q
      const newV = this.matmul(new JsNDArray(V, [n, n]), qrResult.q);
      V = new Float64Array(newV.data);

      // Check convergence: sub-diagonal elements near zero
      let maxOffDiag = 0;
      for (let i = 1; i < n; i++) {
        maxOffDiag = Math.max(maxOffDiag, Math.abs(A[i * n + (i - 1)]));
      }
      if (maxOffDiag < TOL) break;
    }

    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = A[i * n + i];

    return {
      values: new JsNDArray(values, [n]),
      vectors: new JsNDArray(V, [n, n]),
    };
  }

  eigh(arr: NDArray): { values: NDArray; vectors: NDArray } {
    // For symmetric matrices, eig works the same but guarantees real eigenvalues
    return this.eig(arr);
  }

  eigvals(arr: NDArray): NDArray {
    return this.eig(arr).values;
  }

  cholesky(arr: NDArray): NDArray {
    if (arr.shape.length !== 2) throw new Error('cholesky requires 2D');
    const [n, m] = arr.shape;
    if (n !== m) throw new Error('cholesky requires square matrix');

    const L = new Float64Array(n * n);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i * n + k] * L[j * n + k];
        }
        if (i === j) {
          const val = arr.data[i * n + i] - sum;
          if (val < 0) throw new Error('Matrix is not positive definite');
          L[i * n + j] = Math.sqrt(val);
        } else {
          L[i * n + j] = (arr.data[i * n + j] - sum) / L[j * n + j];
        }
      }
    }
    return new JsNDArray(L, [n, n]);
  }

  lstsq(
    a: NDArray,
    b: NDArray,
    rcond?: number | null
  ): { x: NDArray; residuals: NDArray; rank: number; singularValues: NDArray } {
    // Solve least squares via normal equations: x = (A^T A)^{-1} A^T b
    // Handle 1D b by reshaping to column vector
    let bMat = b;
    const was1D = b.shape.length === 1;
    if (was1D) {
      bMat = new JsNDArray(new Float64Array(b.data), [b.data.length, 1]);
    }
    const at = this.transpose(a);
    const ata = this.matmul(at, a);
    const atb = this.matmul(at, bMat);
    const xMat = this.solve(ata, atb);
    const x = was1D ? this.flatten(xMat) : xMat;

    // Compute residuals
    const ax = this.matmul(a, was1D ? xMat : x);
    const residArr = this.subtract(bMat, ax);
    let residSum = 0;
    for (let i = 0; i < residArr.data.length; i++) {
      residSum += residArr.data[i] * residArr.data[i];
    }
    const residuals = new JsNDArray(new Float64Array([residSum]), [1]);

    // SVD for rank and singular values
    const svdResult = this.svd(a, false);
    const singularValues = svdResult.s;

    // Determine rcond cutoff for rank calculation
    // rcond=null or rcond=-1 means use machine epsilon * max(M, N)
    const [M, N] = a.shape;
    const eps = 2.220446049250313e-16; // float64 machine epsilon
    let cutoff: number;
    if (rcond === null || rcond === -1) {
      cutoff = eps * Math.max(M, N);
    } else if (rcond !== undefined) {
      cutoff = rcond;
    } else {
      // Default (no rcond provided): use small tolerance * max singular value
      cutoff = 1e-10 * Math.max(...Array.from(singularValues.data));
    }

    // If rcond is relative (null/-1/undefined with max sv), apply to max singular value
    const maxSV = singularValues.data.length > 0 ? Math.max(...Array.from(singularValues.data)) : 0;
    const threshold = rcond === null || rcond === -1 ? cutoff * maxSV : cutoff;

    let rank = 0;
    for (let i = 0; i < singularValues.data.length; i++) {
      if (singularValues.data[i] > threshold) rank++;
    }

    return { x, residuals, rank, singularValues };
  }

  pinv(arr: NDArray): NDArray {
    // Moore-Penrose pseudoinverse via SVD: A+ = V * S+ * U^T
    const { u, s, vt } = this.svd(arr, false);
    const [m, n] = arr.shape;
    const k = s.data.length;
    const tol = 1e-10 * Math.max(...Array.from(s.data));

    // S+ is diagonal with 1/s_i for nonzero singular values
    const sInv = new Float64Array(n * m);
    for (let i = 0; i < k; i++) {
      if (s.data[i] > tol) {
        sInv[i * m + i] = 1.0 / s.data[i];
      }
    }

    // A+ = V^T^T * S+ * U^T = V * S+ * U^T
    const v = this.transpose(vt); // n x k
    const ut = this.transpose(u); // k x m
    // pinv = V * S+ * U^T  but dimensions: v is n x k, sInv is n x m is wrong
    // Correct: pinv = V * diag(1/s) * U^T
    // V is n x k, diag(1/s) is k x k, U^T is k x m => result is n x m
    const sInvDiag = new Float64Array(k * k);
    for (let i = 0; i < k; i++) {
      if (s.data[i] > tol) {
        sInvDiag[i * k + i] = 1.0 / s.data[i];
      }
    }
    const step1 = this.matmul(v, new JsNDArray(sInvDiag, [k, k]));
    const result = this.matmul(step1, ut);
    return result;
  }

  matrixRank(arr: NDArray, tol?: number): number {
    const { s } = this.svd(arr);
    const threshold = tol ?? 1e-10 * Math.max(...Array.from(s.data));
    let rank = 0;
    for (let i = 0; i < s.data.length; i++) {
      if (s.data[i] > threshold) rank++;
    }
    return rank;
  }

  tensordot(a: NDArray, b: NDArray, axes: number | [number[], number[]] = 2): NDArray {
    let axesA: number[];
    let axesB: number[];

    if (typeof axes === 'number') {
      // Last `axes` dims of a, first `axes` dims of b
      axesA = [];
      axesB = [];
      for (let i = 0; i < axes; i++) {
        axesA.push(a.shape.length - axes + i);
        axesB.push(i);
      }
    } else {
      axesA = axes[0];
      axesB = axes[1];
    }

    // Compute contracted size
    let contractedSize = 1;
    for (const ax of axesA) contractedSize *= a.shape[ax];

    // Free axes
    const freeAxesA = a.shape.map((_, i) => i).filter(i => !axesA.includes(i));
    const freeAxesB = b.shape.map((_, i) => i).filter(i => !axesB.includes(i));

    const freeShapeA = freeAxesA.map(i => a.shape[i]);
    const freeShapeB = freeAxesB.map(i => b.shape[i]);
    const outShape = [...freeShapeA, ...freeShapeB];
    // Reshape a to (freeA_product, contractedSize) and b to (contractedSize, freeB_product)
    const freeASize = freeShapeA.length === 0 ? 1 : freeShapeA.reduce((x, y) => x * y, 1);
    const freeBSize = freeShapeB.length === 0 ? 1 : freeShapeB.reduce((x, y) => x * y, 1);

    // Permute a: free axes first, then contracted
    const permA = [...freeAxesA, ...axesA];
    const permShapeA = permA.map(i => a.shape[i]);
    const aT = this._transposeGeneral(a, permA, permShapeA);

    // Permute b: contracted first, then free
    const permB = [...axesB, ...freeAxesB];
    const permShapeB = permB.map(i => b.shape[i]);
    const bT = this._transposeGeneral(b, permB, permShapeB);

    // Now matmul: (freeASize x contractedSize) @ (contractedSize x freeBSize)
    const aReshaped = new JsNDArray(aT.data, [freeASize, contractedSize]);
    const bReshaped = new JsNDArray(bT.data, [contractedSize, freeBSize]);
    const result = this.matmul(aReshaped, bReshaped);

    if (outShape.length === 0) {
      return new JsNDArray(new Float64Array([result.data[0]]), []);
    }
    return new JsNDArray(result.data, outShape);
  }

  vdot(a: NDArray, b: NDArray): number {
    const aFlat = a.data;
    const bFlat = b.data;
    if (aFlat.length !== bFlat.length) throw new Error('vdot requires same number of elements');
    let sum = 0;
    for (let i = 0; i < aFlat.length; i++) {
      sum += aFlat[i] * bFlat[i];
    }
    return sum;
  }

  // ============ FFT ============

  private _fftCore(
    realIn: Float64Array,
    imagIn: Float64Array,
    inverse: boolean
  ): { real: Float64Array; imag: Float64Array } {
    const n = realIn.length;
    if (n === 0) return { real: new Float64Array(0), imag: new Float64Array(0) };
    if (n === 1)
      return { real: new Float64Array([realIn[0]]), imag: new Float64Array([imagIn[0]]) };

    // Check if power of 2
    if ((n & (n - 1)) === 0) {
      return this._fftRadix2(realIn, imagIn, inverse);
    } else {
      // Bluestein's algorithm for non-power-of-2
      return this._fftBluestein(realIn, imagIn, inverse);
    }
  }

  private _fftRadix2(
    realIn: Float64Array,
    imagIn: Float64Array,
    inverse: boolean
  ): { real: Float64Array; imag: Float64Array } {
    const n = realIn.length;
    const real = new Float64Array(realIn);
    const imag = new Float64Array(imagIn);

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n; i++) {
      if (j > i) {
        let tmp = real[i];
        real[i] = real[j];
        real[j] = tmp;
        tmp = imag[i];
        imag[i] = imag[j];
        imag[j] = tmp;
      }
      let m = n >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }

    // Cooley-Tukey
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size *= 2) {
      const halfSize = size / 2;
      const angle = (sign * 2 * Math.PI) / size;
      const wReal = Math.cos(angle);
      const wImag = Math.sin(angle);

      for (let i = 0; i < n; i += size) {
        let curReal = 1,
          curImag = 0;
        for (let k = 0; k < halfSize; k++) {
          const evenIdx = i + k;
          const oddIdx = i + k + halfSize;

          const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
          const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] += tReal;
          imag[evenIdx] += tImag;

          const newCurReal = curReal * wReal - curImag * wImag;
          curImag = curReal * wImag + curImag * wReal;
          curReal = newCurReal;
        }
      }
    }

    if (inverse) {
      for (let i = 0; i < n; i++) {
        real[i] /= n;
        imag[i] /= n;
      }
    }

    return { real, imag };
  }

  private _fftBluestein(
    realIn: Float64Array,
    imagIn: Float64Array,
    inverse: boolean
  ): { real: Float64Array; imag: Float64Array } {
    const n = realIn.length;
    // Find next power of 2 >= 2*n - 1
    let m = 1;
    while (m < 2 * n - 1) m *= 2;

    const sign = inverse ? 1 : -1;

    // Chirp sequence
    const chirpReal = new Float64Array(n);
    const chirpImag = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const angle = (sign * Math.PI * k * k) / n;
      chirpReal[k] = Math.cos(angle);
      chirpImag[k] = Math.sin(angle);
    }

    // a_k = x_k * conj(chirp_k) padded to m
    const aReal = new Float64Array(m);
    const aImag = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      aReal[k] = realIn[k] * chirpReal[k] + imagIn[k] * chirpImag[k];
      aImag[k] = imagIn[k] * chirpReal[k] - realIn[k] * chirpImag[k];
    }

    // b_k = chirp padded to m (wrap around)
    const bReal = new Float64Array(m);
    const bImag = new Float64Array(m);
    bReal[0] = chirpReal[0];
    bImag[0] = chirpImag[0];
    for (let k = 1; k < n; k++) {
      bReal[k] = chirpReal[k];
      bImag[k] = chirpImag[k];
      bReal[m - k] = chirpReal[k];
      bImag[m - k] = chirpImag[k];
    }

    // Convolution via FFT
    const aFFT = this._fftRadix2(aReal, aImag, false);
    const bFFT = this._fftRadix2(bReal, bImag, false);

    // Pointwise multiply
    const cReal = new Float64Array(m);
    const cImag = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      cReal[k] = aFFT.real[k] * bFFT.real[k] - aFFT.imag[k] * bFFT.imag[k];
      cImag[k] = aFFT.real[k] * bFFT.imag[k] + aFFT.imag[k] * bFFT.real[k];
    }

    // IFFT
    const conv = this._fftRadix2(cReal, cImag, true);

    // Extract and multiply by conj(chirp)
    const outReal = new Float64Array(n);
    const outImag = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      outReal[k] = conv.real[k] * chirpReal[k] + conv.imag[k] * chirpImag[k];
      outImag[k] = conv.imag[k] * chirpReal[k] - conv.real[k] * chirpImag[k];
    }

    if (inverse) {
      for (let k = 0; k < n; k++) {
        outReal[k] /= n;
        outImag[k] /= n;
      }
    }

    return { real: outReal, imag: outImag };
  }

  fft(arr: NDArray): { real: NDArray; imag: NDArray } {
    const n = arr.data.length;
    const result = this._fftCore(new Float64Array(arr.data), new Float64Array(n), false);
    return {
      real: new JsNDArray(result.real, [n]),
      imag: new JsNDArray(result.imag, [n]),
    };
  }

  ifft(real: NDArray, imag: NDArray): { real: NDArray; imag: NDArray } {
    const n = real.data.length;
    const result = this._fftCore(new Float64Array(real.data), new Float64Array(imag.data), true);
    return {
      real: new JsNDArray(result.real, [n]),
      imag: new JsNDArray(result.imag, [n]),
    };
  }

  fft2(arr: NDArray): { real: NDArray; imag: NDArray } {
    if (arr.shape.length !== 2) throw new Error('fft2 requires 2D');
    const [rows, cols] = arr.shape;

    // FFT along rows
    const realData = new Float64Array(arr.data);
    const imagData = new Float64Array(rows * cols);

    for (let i = 0; i < rows; i++) {
      const rowReal = realData.slice(i * cols, (i + 1) * cols);
      const rowImag = imagData.slice(i * cols, (i + 1) * cols);
      const result = this._fftCore(rowReal, rowImag, false);
      realData.set(result.real, i * cols);
      imagData.set(result.imag, i * cols);
    }

    // FFT along columns
    for (let j = 0; j < cols; j++) {
      const colReal = new Float64Array(rows);
      const colImag = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        colReal[i] = realData[i * cols + j];
        colImag[i] = imagData[i * cols + j];
      }
      const result = this._fftCore(colReal, colImag, false);
      for (let i = 0; i < rows; i++) {
        realData[i * cols + j] = result.real[i];
        imagData[i * cols + j] = result.imag[i];
      }
    }

    return {
      real: new JsNDArray(realData, [rows, cols]),
      imag: new JsNDArray(imagData, [rows, cols]),
    };
  }

  ifft2(real: NDArray, imag: NDArray): { real: NDArray; imag: NDArray } {
    if (real.shape.length !== 2) throw new Error('ifft2 requires 2D');
    const [rows, cols] = real.shape;

    const realData = new Float64Array(real.data);
    const imagData = new Float64Array(imag.data);

    // IFFT along rows
    for (let i = 0; i < rows; i++) {
      const rowReal = realData.slice(i * cols, (i + 1) * cols);
      const rowImag = imagData.slice(i * cols, (i + 1) * cols);
      const result = this._fftCore(rowReal, rowImag, true);
      realData.set(result.real, i * cols);
      imagData.set(result.imag, i * cols);
    }

    // IFFT along columns
    for (let j = 0; j < cols; j++) {
      const colReal = new Float64Array(rows);
      const colImag = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        colReal[i] = realData[i * cols + j];
        colImag[i] = imagData[i * cols + j];
      }
      const result = this._fftCore(colReal, colImag, true);
      for (let i = 0; i < rows; i++) {
        realData[i * cols + j] = result.real[i];
        imagData[i * cols + j] = result.imag[i];
      }
    }

    return {
      real: new JsNDArray(realData, [rows, cols]),
      imag: new JsNDArray(imagData, [rows, cols]),
    };
  }

  rfft(arr: NDArray): { real: NDArray; imag: NDArray } {
    const n = arr.data.length;
    const full = this.fft(arr);
    const outLen = Math.floor(n / 2) + 1;
    return {
      real: new JsNDArray(full.real.data.slice(0, outLen), [outLen]),
      imag: new JsNDArray(full.imag.data.slice(0, outLen), [outLen]),
    };
  }

  irfft(real: NDArray, imag: NDArray, n?: number): NDArray {
    const outLen = n ?? (real.data.length - 1) * 2;
    // Reconstruct full spectrum using Hermitian symmetry
    const fullReal = new Float64Array(outLen);
    const fullImag = new Float64Array(outLen);

    for (let i = 0; i < real.data.length; i++) {
      fullReal[i] = real.data[i];
      fullImag[i] = imag.data[i];
    }
    for (let i = real.data.length; i < outLen; i++) {
      const mirror = outLen - i;
      fullReal[i] = fullReal[mirror];
      fullImag[i] = -fullImag[mirror];
    }

    const result = this._fftCore(fullReal, fullImag, true);
    return new JsNDArray(result.real, [outLen]);
  }

  fftfreq(n: number, d: number = 1.0): NDArray {
    const data = new Float64Array(n);
    const N = Math.floor((n - 1) / 2) + 1;
    for (let i = 0; i < N; i++) {
      data[i] = i / (n * d);
    }
    for (let i = N; i < n; i++) {
      data[i] = (i - n) / (n * d);
    }
    return new JsNDArray(data, [n]);
  }

  rfftfreq(n: number, d: number = 1.0): NDArray {
    const outLen = Math.floor(n / 2) + 1;
    const data = new Float64Array(outLen);
    for (let i = 0; i < outLen; i++) {
      data[i] = i / (n * d);
    }
    return new JsNDArray(data, [outLen]);
  }

  fftshift(arr: NDArray): NDArray {
    const n = arr.data.length;
    const shift = Math.floor(n / 2);
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = arr.data[(i - shift + n) % n];
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  ifftshift(arr: NDArray): NDArray {
    const n = arr.data.length;
    const shift = Math.ceil(n / 2);
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = arr.data[(i - shift + n) % n];
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  // ============ Additional Random Distributions ============

  private _boxMullerSingle(): number {
    const u1 = this._xorshift() || 1e-10;
    const u2 = this._xorshift();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  private _gammaRandom(shape_param: number): number {
    // Marsaglia and Tsang's method for shape >= 1
    if (shape_param < 1) {
      // For shape < 1, use shape+1 then scale
      const g = this._gammaRandom(shape_param + 1);
      return g * Math.pow(this._xorshift() || 1e-10, 1.0 / shape_param);
    }

    const d = shape_param - 1.0 / 3.0;
    const c = 1.0 / Math.sqrt(9.0 * d);

    while (true) {
      let x: number, v: number;
      do {
        x = this._boxMullerSingle();
        v = 1.0 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = this._xorshift();

      if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) return d * v;
    }
  }

  exponential(scale: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = -scale * Math.log(this._xorshift() || 1e-10);
    }
    return new JsNDArray(data, shape);
  }

  poisson(lam: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    // Knuth's algorithm for small lambda
    const L = Math.exp(-lam);
    for (let i = 0; i < size; i++) {
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this._xorshift();
      } while (p > L);
      data[i] = k - 1;
    }
    return new JsNDArray(data, shape);
  }

  binomial(n: number, p: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      let successes = 0;
      for (let j = 0; j < n; j++) {
        if (this._xorshift() < p) successes++;
      }
      data[i] = successes;
    }
    return new JsNDArray(data, shape);
  }

  beta(a: number, b: number, shape: number[]): NDArray {
    const size = shape.reduce((acc, x) => acc * x, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const x = this._gammaRandom(a);
      const y = this._gammaRandom(b);
      data[i] = x / (x + y);
    }
    return new JsNDArray(data, shape);
  }

  gamma(shape_param: number, scale: number, size: number[]): NDArray {
    const totalSize = size.reduce((a, b) => a * b, 1);
    const data = new Float64Array(totalSize);
    for (let i = 0; i < totalSize; i++) {
      data[i] = this._gammaRandom(shape_param) * scale;
    }
    return new JsNDArray(data, size);
  }

  lognormal(mean: number, sigma: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.exp(mean + sigma * this._boxMullerSingle());
    }
    return new JsNDArray(data, shape);
  }

  chisquare(df: number, shape: number[]): NDArray {
    // Chi-square is Gamma(df/2, 2)
    return this.gamma(df / 2, 2, shape);
  }

  standardT(df: number, shape: number[]): NDArray {
    // t = Z / sqrt(V/df) where Z ~ N(0,1), V ~ chi-square(df)
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const z = this._boxMullerSingle();
      const v = this._gammaRandom(df / 2) * 2; // chi-square
      data[i] = z / Math.sqrt(v / df);
    }
    return new JsNDArray(data, shape);
  }

  multivariateNormal(mean: NDArray, cov: NDArray, size: number = 1): NDArray {
    const n = mean.data.length;
    // Cholesky decomposition of covariance
    const L = this.cholesky(cov);

    const data = new Float64Array(size * n);
    for (let s = 0; s < size; s++) {
      // Generate n standard normals
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        z[i] = this._boxMullerSingle();
      }

      // x = mean + L * z
      for (let i = 0; i < n; i++) {
        let val = mean.data[i];
        for (let j = 0; j <= i; j++) {
          val += L.data[i * n + j] * z[j];
        }
        data[s * n + i] = val;
      }
    }
    return new JsNDArray(data, size === 1 ? [n] : [size, n]);
  }

  geometric(p: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const logP = Math.log(1 - p);
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(Math.log(this._xorshift() || 1e-10) / logP) + 1;
    }
    return new JsNDArray(data, shape);
  }

  weibull(a: number, shape: number[]): NDArray {
    const size = shape.reduce((acc, x) => acc * x, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.pow(-Math.log(this._xorshift() || 1e-10), 1.0 / a);
    }
    return new JsNDArray(data, shape);
  }

  standardNormal(shape: number[]): NDArray {
    return this.randn(shape);
  }

  standardCauchy(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.tan(Math.PI * (this._xorshift() - 0.5));
    }
    return new JsNDArray(data, shape);
  }

  multinomial(n: number, pvals: number[], size: number = 1): NDArray {
    const k = pvals.length;
    const data = new Float64Array(size * k);
    for (let s = 0; s < size; s++) {
      let remaining = n;
      let pRemaining = 1.0;
      for (let j = 0; j < k - 1; j++) {
        // Each category is binomial(remaining, pvals[j] / pRemaining)
        const p = pRemaining > 0 ? pvals[j] / pRemaining : 0;
        let successes = 0;
        for (let t = 0; t < remaining; t++) {
          if (this._xorshift() < p) successes++;
        }
        data[s * k + j] = successes;
        remaining -= successes;
        pRemaining -= pvals[j];
      }
      data[s * k + (k - 1)] = remaining;
    }
    return new JsNDArray(data, size === 1 ? [k] : [size, k]);
  }

  dirichlet(alpha: number[], size: number = 1): NDArray {
    const k = alpha.length;
    const data = new Float64Array(size * k);
    for (let s = 0; s < size; s++) {
      let sum = 0;
      for (let j = 0; j < k; j++) {
        const g = this._gammaRandom(alpha[j]);
        data[s * k + j] = g;
        sum += g;
      }
      // Normalize
      for (let j = 0; j < k; j++) {
        data[s * k + j] /= sum;
      }
    }
    return new JsNDArray(data, size === 1 ? [k] : [size, k]);
  }

  random(shape: number[]): NDArray {
    return this.rand(shape);
  }

  f(dfnum: number, dfden: number, shape: number[]): NDArray {
    // F = (X1/d1) / (X2/d2) where X1 ~ chi-square(d1), X2 ~ chi-square(d2)
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const x1 = this._gammaRandom(dfnum / 2) * 2;
      const x2 = this._gammaRandom(dfden / 2) * 2;
      data[i] = x1 / dfnum / (x2 / dfden);
    }
    return new JsNDArray(data, shape);
  }

  hypergeometric(ngood: number, nbad: number, nsample: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      // Draw nsample from population of ngood + nbad without replacement
      let good = ngood,
        bad = nbad,
        drawn = 0;
      for (let j = 0; j < nsample; j++) {
        const total = good + bad;
        if (this._xorshift() < good / total) {
          drawn++;
          good--;
        } else {
          bad--;
        }
      }
      data[i] = drawn;
    }
    return new JsNDArray(data, shape);
  }

  negativeBinomial(n: number, p: number, shape: number[]): NDArray {
    // NB via Poisson-Gamma mixture: X ~ Poisson(Y) where Y ~ Gamma(n, (1-p)/p)
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const y = (this._gammaRandom(n) * (1 - p)) / p;
      // Poisson(y)
      const L = Math.exp(-y);
      let k = 0,
        prob = 1;
      do {
        k++;
        prob *= this._xorshift();
      } while (prob > L && y > 0);
      data[i] = y > 0 ? k - 1 : 0;
    }
    return new JsNDArray(data, shape);
  }

  pareto(a: number, shape: number[]): NDArray {
    const size = shape.reduce((acc, x) => acc * x, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.pow(1 - this._xorshift() || 1e-10, -1 / a) - 1;
    }
    return new JsNDArray(data, shape);
  }

  rayleigh(scale: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = scale * Math.sqrt(-2 * Math.log(this._xorshift() || 1e-10));
    }
    return new JsNDArray(data, shape);
  }

  triangular(left: number, mode: number, right: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const fc = (mode - left) / (right - left);
    for (let i = 0; i < size; i++) {
      const u = this._xorshift();
      if (u < fc) {
        data[i] = left + Math.sqrt(u * (right - left) * (mode - left));
      } else {
        data[i] = right - Math.sqrt((1 - u) * (right - left) * (right - mode));
      }
    }
    return new JsNDArray(data, shape);
  }

  vonmises(mu: number, kappa: number, shape: number[]): NDArray {
    // Best-Fisher algorithm for von Mises distribution
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const tau = 1 + Math.sqrt(1 + 4 * kappa * kappa);
    const rho = (tau - Math.sqrt(2 * tau)) / (2 * kappa);
    const r = (1 + rho * rho) / (2 * rho);
    for (let i = 0; i < size; i++) {
      let f: number;
      while (true) {
        const u1 = this._xorshift();
        const z = Math.cos(Math.PI * u1);
        f = (1 + r * z) / (r + z);
        const c = kappa * (r - f);
        const u2 = this._xorshift();
        if (c * (2 - c) > u2 || Math.log(c / u2) + 1 >= c) break;
      }
      const u3 = this._xorshift();
      data[i] = mu + (u3 > 0.5 ? 1 : -1) * Math.acos(f);
    }
    return new JsNDArray(data, shape);
  }

  wald(mean: number, scale: number, shape: number[]): NDArray {
    // Inverse Gaussian distribution (Michael/Schucany/Haas algorithm)
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const v = this._boxMullerSingle();
      const y = v * v;
      const x =
        mean +
        (mean * mean * y) / (2 * scale) -
        (mean / (2 * scale)) * Math.sqrt(4 * mean * scale * y + mean * mean * y * y);
      const u = this._xorshift();
      data[i] = u <= mean / (mean + x) ? x : (mean * mean) / x;
    }
    return new JsNDArray(data, shape);
  }

  zipf(a: number, shape: number[]): NDArray {
    // Rejection method for Zipf distribution
    const size = shape.reduce((acc, x) => acc * x, 1);
    const data = new Float64Array(size);
    const am1 = a - 1;
    const b = Math.pow(2, am1);
    for (let i = 0; i < size; i++) {
      let x: number;
      while (true) {
        const u = 1 - this._xorshift();
        const v = this._xorshift();
        x = Math.floor(Math.pow(u, -1 / am1));
        if (x < 1) continue;
        const t = Math.pow(1 + 1 / x, am1);
        if ((v * x * (t - 1)) / (b - 1) <= t / b) break;
      }
      data[i] = x;
    }
    return new JsNDArray(data, shape);
  }

  // ============ Additional Stats ============

  average(arr: NDArray, weights?: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      let result: NDArray;
      if (!weights) {
        result = this.mean(arr, axis) as NDArray;
      } else {
        const weighted = this.multiply(arr, weights);
        const sumWX = this.sum(weighted, axis) as NDArray;
        const sumW = this.sum(weights, axis) as NDArray;
        result = this.divide(sumWX, sumW);
      }
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        return this.reshape(result, newShape);
      }
      return result;
    }
    if (!weights) return this.mean(arr) as number;
    if (arr.data.length !== weights.data.length)
      throw new Error('arr and weights must have same length');
    let sumW = 0,
      sumWX = 0;
    for (let i = 0; i < arr.data.length; i++) {
      sumW += weights.data[i];
      sumWX += arr.data[i] * weights.data[i];
    }
    return sumWX / sumW;
  }

  ptp(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis === undefined) {
      return (this.max(arr) as number) - (this.min(arr) as number);
    }
    const maxArr = this.maxAxis(arr, axis);
    const minArr = this.minAxis(arr, axis);
    const result = this.subtract(maxArr, minArr);
    if (keepdims) {
      const newShape = [...arr.shape];
      newShape[axis] = 1;
      return this.reshape(result, newShape);
    }
    return result;
  }

  digitize(x: NDArray, bins: NDArray, right: boolean = false): NDArray {
    const data = new Float64Array(x.data.length);
    for (let i = 0; i < x.data.length; i++) {
      const val = x.data[i];
      // Binary search
      let lo = 0,
        hi = bins.data.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (right ? bins.data[mid] < val : bins.data[mid] <= val) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      data[i] = lo;
    }
    return new JsNDArray(data, [...x.shape]);
  }

  nanquantile(arr: NDArray, q: number, axis?: number): number | NDArray {
    if (axis !== undefined) {
      return this._reduceAlongAxis(arr, axis, vals => {
        const nonNaN = Array.from(vals).filter(x => !isNaN(x));
        if (nonNaN.length === 0) return NaN;
        const sorted = nonNaN.sort((a, b) => a - b);
        const pos = q * (sorted.length - 1);
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        if (lo === hi) return sorted[lo];
        return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
      });
    }
    const sorted = this._sortedNonNaN(arr);
    if (sorted.length === 0) return NaN;
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
  }

  nancumsum(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) {
      const cleaned = this.nanToNum(arr, 0, 0, 0);
      return this.cumsum(cleaned, axis);
    }
    const data = new Float64Array(arr.data.length);
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!isNaN(arr.data[i])) sum += arr.data[i];
      data[i] = sum;
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  nancumprod(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) {
      const cleaned = this.nanToNum(arr, 1, 1, 1);
      return this.cumprod(cleaned, axis);
    }
    const data = new Float64Array(arr.data.length);
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      if (!isNaN(arr.data[i])) prod *= arr.data[i];
      data[i] = prod;
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  uniqueCounts(arr: NDArray): { values: NDArray; counts: NDArray } {
    const sorted = Array.from(arr.data).sort((a, b) => a - b);
    const values: number[] = [];
    const counts: number[] = [];
    let i = 0;
    while (i < sorted.length) {
      const val = sorted[i];
      let count = 0;
      while (i < sorted.length && sorted[i] === val) {
        count++;
        i++;
      }
      values.push(val);
      counts.push(count);
    }
    return {
      values: new JsNDArray(new Float64Array(values), [values.length]),
      counts: new JsNDArray(new Float64Array(counts), [counts.length]),
    };
  }

  uniqueInverse(arr: NDArray): { values: NDArray; inverse: NDArray } {
    const uniqueVals = Array.from(new Set(Array.from(arr.data))).sort((a, b) => a - b);
    const valToIdx = new Map<number, number>();
    uniqueVals.forEach((v, i) => valToIdx.set(v, i));
    const inverse = new Float64Array(arr.data.length);
    for (let i = 0; i < arr.data.length; i++) {
      inverse[i] = valToIdx.get(arr.data[i])!;
    }
    return {
      values: new JsNDArray(new Float64Array(uniqueVals), [uniqueVals.length]),
      inverse: new JsNDArray(inverse, [...arr.shape]),
    };
  }

  histogram2d(
    x: NDArray,
    y: NDArray,
    bins: number = 10,
    range?: [[number, number], [number, number]] | null,
    density?: boolean,
    weights?: NDArray
  ): { hist: NDArray; xEdges: NDArray; yEdges: NDArray } {
    const n = x.data.length;
    let xMin: number, xMax: number, yMin: number, yMax: number;

    if (range != null) {
      [[xMin, xMax], [yMin, yMax]] = range;
    } else {
      xMin = this.min(x) as number;
      xMax = this.max(x) as number;
      yMin = this.min(y) as number;
      yMax = this.max(y) as number;
    }

    const xEdges = new Float64Array(bins + 1);
    const yEdges = new Float64Array(bins + 1);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const xBinWidth = xRange / bins;
    const yBinWidth = yRange / bins;
    for (let i = 0; i <= bins; i++) {
      xEdges[i] = xMin + i * xBinWidth;
      yEdges[i] = yMin + i * yBinWidth;
    }

    const hist = new Float64Array(bins * bins);
    for (let i = 0; i < n; i++) {
      const xv = x.data[i];
      const yv = y.data[i];
      if (Number.isNaN(xv) || Number.isNaN(yv)) continue;
      // Skip values outside range when range is specified
      if (range != null && (xv < xMin || xv > xMax || yv < yMin || yv > yMax)) continue;
      let xi = Math.floor((xv - xMin) / xBinWidth);
      let yi = Math.floor((yv - yMin) / yBinWidth);
      xi = Math.min(xi, bins - 1);
      yi = Math.min(yi, bins - 1);
      xi = Math.max(xi, 0);
      yi = Math.max(yi, 0);
      const w = weights ? weights.data[i] : 1;
      hist[xi * bins + yi] += w;
    }

    if (density) {
      let totalWeight = 0;
      for (let i = 0; i < bins * bins; i++) totalWeight += hist[i];
      if (totalWeight > 0) {
        const area = xBinWidth * yBinWidth;
        for (let i = 0; i < bins * bins; i++) {
          hist[i] = hist[i] / (totalWeight * area);
        }
      }
    }

    return {
      hist: new JsNDArray(hist, [bins, bins]),
      xEdges: new JsNDArray(xEdges, [bins + 1]),
      yEdges: new JsNDArray(yEdges, [bins + 1]),
    };
  }

  // ============ Additional Comparison ============

  rint(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.round(arr.data[i]);
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  around(arr: NDArray, decimals: number = 0): NDArray {
    const factor = Math.pow(10, decimals);
    const data = new Float64Array(arr.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.round(arr.data[i] * factor) / factor;
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  // ============ Additional Polynomial ============

  polyder(p: NDArray, m: number = 1): NDArray {
    let coeffs = Array.from(p.data);
    for (let iter = 0; iter < m; iter++) {
      const n = coeffs.length - 1;
      if (n < 1) return new JsNDArray(new Float64Array([0]), [1]);
      const newCoeffs: number[] = [];
      for (let i = 0; i < n; i++) {
        newCoeffs.push(coeffs[i] * (n - i));
      }
      coeffs = newCoeffs;
    }
    return new JsNDArray(new Float64Array(coeffs), [coeffs.length]);
  }

  polyint(p: NDArray, m: number = 1, k: number = 0): NDArray {
    let coeffs = Array.from(p.data);
    for (let iter = 0; iter < m; iter++) {
      const n = coeffs.length;
      const newCoeffs: number[] = [];
      for (let i = 0; i < n; i++) {
        newCoeffs.push(coeffs[i] / (n - i));
      }
      newCoeffs.push(k);
      coeffs = newCoeffs;
    }
    return new JsNDArray(new Float64Array(coeffs), [coeffs.length]);
  }

  polydiv(u: NDArray, v: NDArray): { q: NDArray; r: NDArray } {
    const num = Array.from(u.data);
    const den = Array.from(v.data);
    const nq = num.length - den.length + 1;

    if (nq <= 0) {
      return {
        q: new JsNDArray(new Float64Array([0]), [1]),
        r: new JsNDArray(new Float64Array(num), [num.length]),
      };
    }

    const q: number[] = [];
    for (let i = 0; i < nq; i++) {
      const coeff = num[i] / den[0];
      q.push(coeff);
      for (let j = 0; j < den.length; j++) {
        num[i + j] -= coeff * den[j];
      }
    }

    // Remainder is the remaining part
    const r = num.slice(nq);
    // Remove leading near-zeros from remainder
    while (r.length > 1 && Math.abs(r[0]) < 1e-15) {
      r.shift();
    }

    return {
      q: new JsNDArray(new Float64Array(q), [q.length]),
      r: new JsNDArray(new Float64Array(r), [r.length]),
    };
  }

  polysub(a: NDArray, b: NDArray): NDArray {
    const maxLen = Math.max(a.data.length, b.data.length);
    const result = new Float64Array(maxLen);
    const aOffset = maxLen - a.data.length;
    const bOffset = maxLen - b.data.length;
    for (let i = 0; i < a.data.length; i++) {
      result[aOffset + i] += a.data[i];
    }
    for (let i = 0; i < b.data.length; i++) {
      result[bOffset + i] -= b.data[i];
    }
    return new JsNDArray(result, [maxLen]);
  }

  // ============ Integration ============

  trapezoid(y: NDArray, x?: NDArray): number {
    const n = y.data.length;
    if (n < 2) return 0;
    let sum = 0;
    if (x) {
      for (let i = 0; i < n - 1; i++) {
        sum += 0.5 * (y.data[i] + y.data[i + 1]) * (x.data[i + 1] - x.data[i]);
      }
    } else {
      for (let i = 0; i < n - 1; i++) {
        sum += 0.5 * (y.data[i] + y.data[i + 1]);
      }
    }
    return sum;
  }

  // ============ Index Utilities ============

  unravelIndex(indices: NDArray | number, shape: number[]): NDArray[] {
    const flatIndices = typeof indices === 'number' ? [indices] : Array.from(indices.data);
    const ndim = shape.length;
    const result: NDArray[] = [];
    for (let d = 0; d < ndim; d++) {
      result.push(new JsNDArray(new Float64Array(flatIndices.length), [flatIndices.length]));
    }
    for (let i = 0; i < flatIndices.length; i++) {
      let rem = flatIndices[i];
      for (let d = ndim - 1; d >= 0; d--) {
        result[d].data[i] = rem % shape[d];
        rem = Math.floor(rem / shape[d]);
      }
    }
    return result;
  }

  ravelMultiIndex(multiIndex: NDArray[], shape: number[]): NDArray {
    const n = multiIndex[0].data.length;
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let flat = 0;
      let multiplier = 1;
      for (let d = shape.length - 1; d >= 0; d--) {
        flat += multiIndex[d].data[i] * multiplier;
        multiplier *= shape[d];
      }
      data[i] = flat;
    }
    return new JsNDArray(data, [n]);
  }

  // ============ Integer Math ============

  gcd(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      x = Math.abs(Math.round(x));
      y = Math.abs(Math.round(y));
      while (y) {
        const t = y;
        y = x % y;
        x = t;
      }
      return x;
    });
  }

  lcm(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._binaryOp(a, b, (x, y) => {
      x = Math.abs(Math.round(x));
      y = Math.abs(Math.round(y));
      if (x === 0 || y === 0) return 0;
      let a = x,
        b = y;
      while (b) {
        const t = b;
        b = a % b;
        a = t;
      }
      return (x / a) * y;
    });
  }

  // ============ Matrix Utilities ============

  tri(n: number, m?: number, k: number = 0, dtype: DType = 'float64'): NDArray {
    const cols = m ?? n;
    const data = createTypedArray(dtype, n * cols);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i + k && j < cols; j++) {
        if (j >= 0) data[i * cols + j] = 1;
      }
    }
    return new JsNDArray(data, [n, cols], dtype);
  }

  diagflat(v: NDArray, k: number = 0): NDArray {
    // Flatten input first, then create diagonal matrix
    const flat = this.flatten(v);
    return this.diag(flat, k);
  }

  block(arrays: (NDArray | NDArray[])[]): NDArray {
    // Check if all elements are NDArrays (1D case = simple concatenate)
    const allScalar = arrays.every(a => !Array.isArray(a));
    if (allScalar) {
      return this.concatenate(arrays as NDArray[]);
    }
    // 2D case: each element is a row of arrays to hstack, then vstack the rows
    const rows: NDArray[] = [];
    for (const row of arrays) {
      if (Array.isArray(row)) {
        rows.push(this.hstack(row));
      } else {
        rows.push(row);
      }
    }
    return this.vstack(rows);
  }

  fillDiagonal(arr: NDArray, val: number, wrap: boolean = false): NDArray {
    if (arr.shape.length < 2) {
      throw new Error('fillDiagonal requires at least a 2-d array');
    }
    const data = new Float64Array(arr.data);
    const rows = arr.shape[0];
    const cols = arr.shape[1];
    const step = cols + 1; // stride along diagonal in row-major
    if (wrap) {
      // Fill diagonal wrapping around for tall matrices
      for (let i = 0; i < rows; i++) {
        const col = i % cols;
        const flatIdx = i * cols + col;
        if (flatIdx < data.length) {
          data[flatIdx] = val;
        }
      }
    } else {
      const diagLen = Math.min(rows, cols);
      for (let i = 0; i < diagLen; i++) {
        data[i * step] = val;
      }
    }
    return new JsNDArray(data, [...arr.shape]);
  }

  // ============ Index Arrays ============

  indices(dimensions: number[], dtype: DType = 'float64'): NDArray[] {
    const ndim = dimensions.length;
    const total = dimensions.reduce((a, b) => a * b, 1);
    const result: NDArray[] = [];

    // Compute strides
    const strides = new Array(ndim);
    strides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) strides[i] = strides[i + 1] * dimensions[i + 1];

    for (let d = 0; d < ndim; d++) {
      const data = createTypedArray(dtype, total);
      for (let flatIdx = 0; flatIdx < total; flatIdx++) {
        data[flatIdx] = Math.floor(flatIdx / strides[d]) % dimensions[d];
      }
      result.push(new JsNDArray(data, [...dimensions], dtype));
    }
    return result;
  }

  diagIndices(n: number, ndim: number = 2): NDArray[] {
    const idx = new Float64Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const result: NDArray[] = [];
    for (let d = 0; d < ndim; d++) {
      result.push(new JsNDArray(new Float64Array(idx), [n]));
    }
    return result;
  }

  trilIndices(n: number, k: number = 0, m?: number): [NDArray, NDArray] {
    const cols = m ?? n;
    const rowList: number[] = [];
    const colList: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < cols; j++) {
        if (j <= i + k) {
          rowList.push(i);
          colList.push(j);
        }
      }
    }
    return [
      new JsNDArray(new Float64Array(rowList), [rowList.length]),
      new JsNDArray(new Float64Array(colList), [colList.length]),
    ];
  }

  triuIndices(n: number, k: number = 0, m?: number): [NDArray, NDArray] {
    const cols = m ?? n;
    const rowList: number[] = [];
    const colList: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < cols; j++) {
        if (j >= i + k) {
          rowList.push(i);
          colList.push(j);
        }
      }
    }
    return [
      new JsNDArray(new Float64Array(rowList), [rowList.length]),
      new JsNDArray(new Float64Array(colList), [colList.length]),
    ];
  }

  // ============ Window Functions ============

  bartlett(M: number): NDArray {
    if (M <= 0) return new JsNDArray(new Float64Array(0), [0]);
    if (M === 1) return new JsNDArray(new Float64Array([1]), [1]);
    const data = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      data[i] = 1 - Math.abs((2 * i) / (M - 1) - 1);
    }
    return new JsNDArray(data, [M]);
  }

  blackman(M: number): NDArray {
    if (M <= 0) return new JsNDArray(new Float64Array(0), [0]);
    if (M === 1) return new JsNDArray(new Float64Array([1]), [1]);
    const data = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      data[i] =
        0.42 -
        0.5 * Math.cos((2 * Math.PI * i) / (M - 1)) +
        0.08 * Math.cos((4 * Math.PI * i) / (M - 1));
    }
    return new JsNDArray(data, [M]);
  }

  hamming(M: number): NDArray {
    if (M <= 0) return new JsNDArray(new Float64Array(0), [0]);
    if (M === 1) return new JsNDArray(new Float64Array([1]), [1]);
    const data = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      data[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (M - 1));
    }
    return new JsNDArray(data, [M]);
  }

  hanning(M: number): NDArray {
    if (M <= 0) return new JsNDArray(new Float64Array(0), [0]);
    if (M === 1) return new JsNDArray(new Float64Array([1]), [1]);
    const data = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      data[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (M - 1)));
    }
    return new JsNDArray(data, [M]);
  }

  kaiser(M: number, beta: number): NDArray {
    if (M <= 0) return new JsNDArray(new Float64Array(0), [0]);
    if (M === 1) return new JsNDArray(new Float64Array([1]), [1]);
    const data = new Float64Array(M);
    const i0Beta = this._besselI0(beta);
    for (let i = 0; i < M; i++) {
      const x = (2 * i) / (M - 1) - 1;
      data[i] = this._besselI0(beta * Math.sqrt(1 - x * x)) / i0Beta;
    }
    return new JsNDArray(data, [M]);
  }

  /** Modified Bessel function of the first kind, order 0 */
  private _besselI0(x: number): number {
    // Series expansion: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
    let sum = 1;
    let term = 1;
    const halfX = x / 2;
    for (let k = 1; k <= 50; k++) {
      term *= halfX / k;
      const t2 = term * term;
      sum += t2;
      if (t2 < sum * 1e-16) break;
    }
    return sum;
  }

  // ============ Bit Manipulation ============

  packbits(arr: NDArray, axis?: number, bitorder: 'big' | 'little' = 'big'): NDArray {
    // Flatten and pack along last axis (simple 1D case)
    const input = axis !== undefined ? arr : this.flatten(arr);
    const n = input.data.length;
    const numBytes = Math.ceil(n / 8);
    const data = new Uint8Array(numBytes);
    for (let i = 0; i < n; i++) {
      const byteIdx = Math.floor(i / 8);
      const bitIdx = i % 8;
      if (input.data[i]) {
        if (bitorder === 'big') {
          data[byteIdx] |= 1 << (7 - bitIdx);
        } else {
          data[byteIdx] |= 1 << bitIdx;
        }
      }
    }
    return new JsNDArray(data, [numBytes], 'uint8');
  }

  unpackbits(
    arr: NDArray,
    axis?: number,
    count?: number,
    bitorder: 'big' | 'little' = 'big'
  ): NDArray {
    const n = arr.data.length;
    const totalBits = count ?? n * 8;
    const data = new Uint8Array(totalBits);
    let outIdx = 0;
    for (let i = 0; i < n && outIdx < totalBits; i++) {
      for (let bit = 0; bit < 8 && outIdx < totalBits; bit++) {
        if (bitorder === 'big') {
          data[outIdx] = (arr.data[i] >> (7 - bit)) & 1;
        } else {
          data[outIdx] = (arr.data[i] >> bit) & 1;
        }
        outIdx++;
      }
    }
    return new JsNDArray(data, [totalBits], 'uint8');
  }

  // ============ Additional Linalg ============

  eigvalsh(arr: NDArray): NDArray {
    return this.eigh(arr).values;
  }

  // ============ N-dimensional FFT ============

  fftn(arr: NDArray, shape?: number[]): { real: NDArray; imag: NDArray } {
    const ndim = arr.shape.length;
    if (ndim === 1) return this.fft(arr);
    if (ndim === 2) return this.fft2(arr);

    // For higher dimensions, apply FFT along each axis sequentially
    const totalSize = arr.data.length;
    const realData = new Float64Array(arr.data);
    const imagData = new Float64Array(totalSize);
    const currentShape = shape ? [...shape] : [...arr.shape];

    // Compute strides
    const strides = new Array(ndim);
    strides[ndim - 1] = 1;
    for (let d = ndim - 2; d >= 0; d--) strides[d] = strides[d + 1] * currentShape[d + 1];

    // Apply FFT along each axis
    for (let axis = ndim - 1; axis >= 0; axis--) {
      const axisLen = currentShape[axis];

      // For each 1D slice along this axis
      const stride = strides[axis];
      const outerStride = stride * axisLen;

      for (let outer = 0; outer < totalSize; outer += outerStride) {
        for (let inner = 0; inner < stride; inner++) {
          const sliceReal = new Float64Array(axisLen);
          const sliceImag = new Float64Array(axisLen);
          for (let k = 0; k < axisLen; k++) {
            const idx = outer + k * stride + inner;
            sliceReal[k] = realData[idx];
            sliceImag[k] = imagData[idx];
          }
          const result = this._fftCore(sliceReal, sliceImag, false);
          for (let k = 0; k < axisLen; k++) {
            const idx = outer + k * stride + inner;
            realData[idx] = result.real[k];
            imagData[idx] = result.imag[k];
          }
        }
      }
    }

    return {
      real: new JsNDArray(realData, [...currentShape]),
      imag: new JsNDArray(imagData, [...currentShape]),
    };
  }

  ifftn(real: NDArray, imag: NDArray, shape?: number[]): { real: NDArray; imag: NDArray } {
    const ndim = real.shape.length;
    if (ndim === 1) return this.ifft(real, imag);
    if (ndim === 2) return this.ifft2(real, imag);

    const totalSize = real.data.length;
    const realData = new Float64Array(real.data);
    const imagData = new Float64Array(imag.data);
    const currentShape = shape ? [...shape] : [...real.shape];

    const strides = new Array(ndim);
    strides[ndim - 1] = 1;
    for (let d = ndim - 2; d >= 0; d--) strides[d] = strides[d + 1] * currentShape[d + 1];

    for (let axis = ndim - 1; axis >= 0; axis--) {
      const axisLen = currentShape[axis];
      const stride = strides[axis];
      const outerStride = stride * axisLen;

      for (let outer = 0; outer < totalSize; outer += outerStride) {
        for (let inner = 0; inner < stride; inner++) {
          const sliceReal = new Float64Array(axisLen);
          const sliceImag = new Float64Array(axisLen);
          for (let k = 0; k < axisLen; k++) {
            const idx = outer + k * stride + inner;
            sliceReal[k] = realData[idx];
            sliceImag[k] = imagData[idx];
          }
          const result = this._fftCore(sliceReal, sliceImag, true);
          for (let k = 0; k < axisLen; k++) {
            const idx = outer + k * stride + inner;
            realData[idx] = result.real[k];
            imagData[idx] = result.imag[k];
          }
        }
      }
    }

    return {
      real: new JsNDArray(realData, [...currentShape]),
      imag: new JsNDArray(imagData, [...currentShape]),
    };
  }
}

export function createJsBackend(): Backend {
  return new JsBackend();
}
