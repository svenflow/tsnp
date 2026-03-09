/**
 * Pure JavaScript reference backend for testing
 *
 * This implements all Backend operations in pure JS, serving as:
 * 1. A reference implementation for testing
 * 2. A fallback when WASM is not available
 * 3. A baseline for performance comparisons
 */

import { Backend, NDArray } from './test-utils';

class JsNDArray implements NDArray {
  data: Float64Array;
  shape: number[];

  constructor(data: Float64Array | number[], shape: number[]) {
    this.data = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = shape;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }
}

export class JsBackend implements Backend {
  name = 'js';

  // ============ Creation ============

  zeros(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JsNDArray(new Float64Array(size), shape);
  }

  ones(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(1.0);
    return new JsNDArray(data, shape);
  }

  full(shape: number[], value: number): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(value);
    return new JsNDArray(data, shape);
  }

  arange(start: number, stop: number, step: number): NDArray {
    if (step === 0) {
      throw new Error('step cannot be zero');
    }
    const data: number[] = [];
    if (step > 0) {
      for (let x = start; x < stop; x += step) {
        data.push(x);
      }
    } else {
      for (let x = start; x > stop; x += step) {
        data.push(x);
      }
    }
    return new JsNDArray(data, [data.length]);
  }

  linspace(start: number, stop: number, num: number): NDArray {
    if (num === 0) return new JsNDArray([], [0]);
    if (num === 1) return new JsNDArray([start], [1]);
    const step = (stop - start) / (num - 1);
    const data: number[] = [];
    for (let i = 0; i < num; i++) {
      data.push(start + i * step);
    }
    return new JsNDArray(data, [num]);
  }

  eye(n: number): NDArray {
    const data = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1.0;
    }
    return new JsNDArray(data, [n, n]);
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

  array(data: number[], shape?: number[]): NDArray {
    const s = shape || [data.length];
    return new JsNDArray(data, s);
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

  round(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.round), arr.shape);
  }

  neg(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => -x), arr.shape);
  }

  reciprocal(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => 1 / x), arr.shape);
  }

  square(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => x * x), arr.shape);
  }

  // ============ Math - Unary (Extended) ============

  arcsinh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.asinh(x)), arr.shape);
  }

  arccosh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.acosh(x)), arr.shape);
  }

  arctanh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.atanh(x)), arr.shape);
  }

  expm1(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.expm1(x)), arr.shape);
  }

  log1p(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.log1p(x)), arr.shape);
  }

  trunc(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.trunc(x)), arr.shape);
  }

  sinc(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => {
      if (x === 0) return 1;
      const px = Math.PI * x;
      return Math.sin(px) / px;
    }), arr.shape);
  }

  deg2rad(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => x * Math.PI / 180), arr.shape);
  }

  rad2deg(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => x * 180 / Math.PI), arr.shape);
  }

  heaviside(arr: NDArray, h0: number): NDArray {
    return new JsNDArray(arr.data.map((x) => {
      if (x < 0) return 0;
      if (x === 0) return h0;
      return 1;
    }), arr.shape);
  }

  fix(arr: NDArray): NDArray {
    // Same as trunc - round toward zero
    return this.trunc(arr);
  }

  signbit(arr: NDArray): NDArray {
    // Returns 1.0 if sign bit is set (negative or -0), 0.0 otherwise
    // NumPy: signbit(NaN) = False, signbit(-0) = True, signbit(-inf) = True
    return new JsNDArray(arr.data.map((x) => {
      if (Number.isNaN(x)) return 0;  // NumPy returns False for NaN
      if (Object.is(x, -0)) return 1;  // -0 has sign bit set
      return x < 0 ? 1 : 0;
    }), arr.shape);
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

  divmod(a: NDArray, b: NDArray): { quotient: NDArray; remainder: NDArray } {
    // Floor division and modulo
    this._checkSameShape(a, b);
    const quotient = new Float64Array(a.data.length);
    const remainder = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      quotient[i] = Math.floor(a.data[i] / b.data[i]);
      // Python-style modulo
      const r = a.data[i] % b.data[i];
      remainder[i] = r !== 0 && Math.sign(r) !== Math.sign(b.data[i]) ? r + b.data[i] : r;
    }
    return {
      quotient: new JsNDArray(quotient, a.shape),
      remainder: new JsNDArray(remainder, a.shape),
    };
  }

  // ============ Math - Binary (Extended) ============

  mod(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      // Python-style modulo: result has same sign as divisor
      const r = a.data[i] % b.data[i];
      data[i] = r !== 0 && Math.sign(r) !== Math.sign(b.data[i]) ? r + b.data[i] : r;
    }
    return new JsNDArray(data, a.shape);
  }

  fmod(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      // C-style modulo: result has same sign as dividend
      data[i] = a.data[i] % b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  remainder(a: NDArray, b: NDArray): NDArray {
    return this.mod(a, b); // Same as mod in NumPy
  }

  copysign(a: NDArray, b: NDArray): NDArray {
    // Copy sign of b to magnitude of a
    // NumPy: copysign(5, -0) = -5, copysign(5, +0) = 5
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      const magnitude = Math.abs(a.data[i]);
      // Use Object.is to detect -0, since Math.sign(0) = 0 and Math.sign(-0) = -0
      const bNegative = b.data[i] < 0 || Object.is(b.data[i], -0);
      data[i] = bNegative ? -magnitude : magnitude;
    }
    return new JsNDArray(data, a.shape);
  }

  hypot(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.hypot(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  arctan2(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.atan2(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  logaddexp(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      // log(exp(a) + exp(b)) - numerically stable
      const mx = Math.max(a.data[i], b.data[i]);
      if (mx === -Infinity) {
        data[i] = -Infinity;
      } else {
        data[i] = mx + Math.log(Math.exp(a.data[i] - mx) + Math.exp(b.data[i] - mx));
      }
    }
    return new JsNDArray(data, a.shape);
  }

  logaddexp2(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    const log2 = Math.log(2);
    for (let i = 0; i < a.data.length; i++) {
      // log2(2^a + 2^b)
      const mx = Math.max(a.data[i], b.data[i]);
      if (mx === -Infinity) {
        data[i] = -Infinity;
      } else {
        data[i] = mx + Math.log(Math.pow(2, a.data[i] - mx) + Math.pow(2, b.data[i] - mx)) / log2;
      }
    }
    return new JsNDArray(data, a.shape);
  }

  fmax(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      // Ignore NaN
      if (Number.isNaN(a.data[i])) data[i] = b.data[i];
      else if (Number.isNaN(b.data[i])) data[i] = a.data[i];
      else data[i] = Math.max(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  fmin(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      // Ignore NaN
      if (Number.isNaN(a.data[i])) data[i] = b.data[i];
      else if (Number.isNaN(b.data[i])) data[i] = a.data[i];
      else data[i] = Math.min(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  // ============ Comparison ============

  equal(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] === b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  notEqual(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] !== b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  less(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] < b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  lessEqual(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] <= b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  greater(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] > b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  greaterEqual(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] >= b.data[i] ? 1 : 0;
    }
    return new JsNDArray(data, a.shape);
  }

  isnan(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Number.isNaN(x) ? 1 : 0), arr.shape);
  }

  isinf(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => !Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0), arr.shape);
  }

  isfinite(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => Number.isFinite(x) ? 1 : 0), arr.shape);
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
    return new JsNDArray(element.data.map(x => testSet.has(x) ? 1 : 0), element.shape);
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
      const normalized = indices.map(i => i < 0 ? flat.length + i : i).sort((a, b) => b - a);
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

  countNonzero(arr: NDArray, axis?: number): NDArray | number {
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

    return new JsNDArray(result, outShape);
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

  cond(arr: NDArray, p: number | 'fro' = 2): number {
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
    return new JsNDArray(x.data.map(xi => {
      let result = 0;
      for (let i = 0; i < coeffs.length; i++) {
        result = result * xi + coeffs[i];
      }
      return result;
    }), x.shape);
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
        const shift = H[(size - 1) * n + (size - 1)] -
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

    return new JsNDArray(x.data.map(xi => {
      // Binary search for interval
      if (xi <= xpData[0]) return fpData[0];
      if (xi >= xpData[xpData.length - 1]) return fpData[fpData.length - 1];

      let lo = 0, hi = xpData.length - 1;
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (xpData[mid] <= xi) lo = mid;
        else hi = mid;
      }

      // Linear interpolation
      const t = (xi - xpData[lo]) / (xpData[hi] - xpData[lo]);
      return fpData[lo] + t * (fpData[hi] - fpData[lo]);
    }), x.shape);
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
          let i = lo, j = hi;
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
      const nth = (arr: { value: number; origIdx: number }[], k: number, lo: number, hi: number) => {
        while (lo < hi) {
          const pivot = arr[Math.floor((lo + hi) / 2)].value;
          let i = lo, j = hi;
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

  add(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] + b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  sub(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] - b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  mul(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] * b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  div(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] / b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  pow(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.pow(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  maximum(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.max(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  minimum(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.min(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  // ============ Math - Scalar ============

  addScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x + scalar), arr.shape);
  }

  subScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x - scalar), arr.shape);
  }

  mulScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x * scalar), arr.shape);
  }

  divScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x / scalar), arr.shape);
  }

  powScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.pow(x, scalar)), arr.shape);
  }

  clip(arr: NDArray, min: number, max: number): NDArray {
    return new JsNDArray(
      arr.data.map((x) => Math.min(Math.max(x, min), max)),
      arr.shape
    );
  }

  // ============ Stats ============

  sum(arr: NDArray): number {
    return arr.data.reduce((a, b) => a + b, 0);
  }

  prod(arr: NDArray): number {
    return arr.data.reduce((a, b) => a * b, 1);
  }

  mean(arr: NDArray): number {
    if (arr.data.length === 0) return NaN;
    return this.sum(arr) / arr.data.length;
  }

  var(arr: NDArray, ddof: number = 0): number {
    if (arr.data.length === 0) return NaN;
    const m = this.mean(arr);
    const sumSq = arr.data.reduce((acc, x) => acc + (x - m) ** 2, 0);
    return sumSq / (arr.data.length - ddof);
  }

  std(arr: NDArray, ddof: number = 0): number {
    return Math.sqrt(this.var(arr, ddof));
  }

  min(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.min(...arr.data);
  }

  max(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.max(...arr.data);
  }

  argmin(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let minIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < arr.data[minIdx]) minIdx = i;
    }
    return minIdx;
  }

  argmax(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let maxIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > arr.data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  cumsum(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      sum += arr.data[i];
      data[i] = sum;
    }
    return new JsNDArray(data, arr.shape);
  }

  cumprod(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      prod *= arr.data[i];
      data[i] = prod;
    }
    return new JsNDArray(data, arr.shape);
  }

  all(arr: NDArray): boolean {
    return arr.data.every((x) => x !== 0);
  }

  any(arr: NDArray): boolean {
    return arr.data.some((x) => x !== 0);
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
      sumResult.data.map((x) => x / divisor),
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
          if (arr.data[i * cols + j] === 0) { data[j] = 0; break; }
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(1);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (arr.data[i * cols + j] === 0) { data[i] = 0; break; }
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
          if (arr.data[i * cols + j] !== 0) { data[j] = 1; break; }
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      const data = new Float64Array(rows).fill(0);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (arr.data[i * cols + j] !== 0) { data[i] = 1; break; }
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

  inner(a: NDArray, b: NDArray): number {
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

  transpose(arr: NDArray): NDArray {
    if (arr.shape.length === 1) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }
    if (arr.shape.length !== 2) throw new Error('transpose requires 1D or 2D');
    const [rows, cols] = arr.shape;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[j * rows + i] = arr.data[i * cols + j];
      }
    }
    return new JsNDArray(data, [cols, rows]);
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

  norm(arr: NDArray, ord: number = 2): number {
    if (ord === 1) {
      return arr.data.reduce((acc, x) => acc + Math.abs(x), 0);
    }
    if (ord === Infinity) {
      return Math.max(...arr.data.map(Math.abs));
    }
    if (ord === -Infinity) {
      return Math.min(...arr.data.map(Math.abs));
    }
    // Default L2 norm
    return Math.sqrt(arr.data.reduce((acc, x) => acc + x * x, 0));
  }

  qr(arr: NDArray): { q: NDArray; r: NDArray } {
    if (arr.shape.length !== 2) throw new Error('qr requires 2D');
    const [m, n] = arr.shape;

    // Modified Gram-Schmidt
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
      for (let k = j + 1; k < n; k++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += q[i * n + j] * q[i * n + k];
        }
        r[j * n + k] = dot;
        for (let i = 0; i < m; i++) {
          q[i * n + k] -= dot * q[i * n + j];
        }
      }
    }

    return {
      q: new JsNDArray(q, [m, n]),
      r: new JsNDArray(r, [n, n]),
    };
  }

  svd(arr: NDArray): { u: NDArray; s: NDArray; vt: NDArray } {
    // Full SVD using power iteration with deflation
    // Computes A = U * Σ * V^T
    if (arr.shape.length !== 2) throw new Error('svd requires 2D');
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

    return {
      u: new JsNDArray(uData, [m, k]),
      s: new JsNDArray(sortedS, [k]),
      vt: new JsNDArray(vtData, [k, n]),
    };
  }

  // ============ Helpers ============

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

  zerosLike(arr: NDArray): NDArray {
    return this.zeros(arr.shape);
  }

  onesLike(arr: NDArray): NDArray {
    return this.ones(arr.shape);
  }

  emptyLike(arr: NDArray): NDArray {
    // In JS, we can't have uninitialized memory, so same as zeros
    return this.zeros(arr.shape);
  }

  fullLike(arr: NDArray, value: number): NDArray {
    return this.full(arr.shape, value);
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

  moveaxis(arr: NDArray, source: number, destination: number): NDArray {
    const ndim = arr.shape.length;
    source = this._normalizeAxis(source, ndim);
    destination = this._normalizeAxis(destination, ndim);

    if (source === destination) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }

    // Build permutation
    const perm: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i !== source) perm.push(i);
    }
    perm.splice(destination, 0, source);

    const newShape = perm.map(i => arr.shape[i]);
    return this._transposeGeneral(arr, perm, newShape);
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

  squeeze(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) {
      const normalizedAxis = this._normalizeAxis(axis, arr.shape.length);
      if (arr.shape[normalizedAxis] !== 1) {
        throw new Error(`cannot squeeze axis ${axis} with size ${arr.shape[normalizedAxis]}`);
      }
      const newShape = arr.shape.filter((_, i) => i !== normalizedAxis);
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

  flatten(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.slice(), [arr.data.length]);
  }

  concatenate(arrays: NDArray[], axis: number = 0): NDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');
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

  where(condition: NDArray, x: NDArray, y: NDArray): NDArray {
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
      throw new Error(`einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`);
    }

    // Map each label to its dimension size
    const labelSizes: Map<string, number> = new Map();
    const inputLabels: string[][] = [];

    for (let i = 0; i < operands.length; i++) {
      const labels = inputSubscripts[i].split('');
      inputLabels.push(labels);
      if (labels.length !== operands[i].shape.length) {
        throw new Error(`einsum: operand ${i} has ${operands[i].shape.length} dimensions but subscripts specify ${labels.length}`);
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
    const contractedTotal = contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

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
          const size = contractedSizes[d];
          const stride = d < contractedSizes.length - 1
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

  diff(arr: NDArray, n: number = 1, axis: number = -1): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    let result = arr;
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

      let srcIdx1 = 0, srcIdx2 = 0;
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

  gradient(arr: NDArray, axis: number = -1): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const shape = arr.shape;
    const axisLen = shape[axis];

    if (axisLen < 2) {
      throw new Error('gradient requires at least 2 elements along axis');
    }

    const result = new Float64Array(arr.data.length);
    const strides = this._computeStrides(shape);

    // For each position, compute central difference (or forward/backward at edges)
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
        // Forward difference
        const nextCoords = [...coords];
        nextCoords[axis] = 1;
        let nextIdx = 0;
        for (let d = 0; d < ndim; d++) nextIdx += nextCoords[d] * strides[d];
        grad = arr.data[nextIdx] - arr.data[i];
      } else if (axisCoord === axisLen - 1) {
        // Backward difference
        const prevCoords = [...coords];
        prevCoords[axis] = axisLen - 2;
        let prevIdx = 0;
        for (let d = 0; d < ndim; d++) prevIdx += prevCoords[d] * strides[d];
        grad = arr.data[i] - arr.data[prevIdx];
      } else {
        // Central difference
        const prevCoords = [...coords];
        const nextCoords = [...coords];
        prevCoords[axis] = axisCoord - 1;
        nextCoords[axis] = axisCoord + 1;
        let prevIdx = 0, nextIdx = 0;
        for (let d = 0; d < ndim; d++) {
          prevIdx += prevCoords[d] * strides[d];
          nextIdx += nextCoords[d] * strides[d];
        }
        grad = (arr.data[nextIdx] - arr.data[prevIdx]) / 2;
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
      new Float64Array([
        a2 * b3 - a3 * b2,
        a3 * b1 - a1 * b3,
        a1 * b2 - a2 * b1,
      ]),
      [3]
    );
  }

  // ============ Statistics ============

  cov(x: NDArray, y?: NDArray): NDArray {
    if (y === undefined) {
      // Compute covariance matrix for rows of x
      if (x.shape.length !== 2) {
        throw new Error('cov requires 2D array when y is not provided');
      }
      const [nVars, nObs] = x.shape;
      const result = new Float64Array(nVars * nVars);

      // Compute means for each row
      const means = new Float64Array(nVars);
      for (let i = 0; i < nVars; i++) {
        let sum = 0;
        for (let j = 0; j < nObs; j++) {
          sum += x.data[i * nObs + j];
        }
        means[i] = sum / nObs;
      }

      // Compute covariance matrix
      for (let i = 0; i < nVars; i++) {
        for (let j = 0; j < nVars; j++) {
          let cov = 0;
          for (let k = 0; k < nObs; k++) {
            cov += (x.data[i * nObs + k] - means[i]) * (x.data[j * nObs + k] - means[j]);
          }
          result[i * nVars + j] = cov / (nObs - 1);
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
      const xMean = this.mean(xFlat);
      const yMean = this.mean(yFlat);

      let cov = 0;
      for (let i = 0; i < n; i++) {
        cov += (xFlat.data[i] - xMean) * (yFlat.data[i] - yMean);
      }
      cov /= n - 1;

      // Return 2x2 covariance matrix
      const xVar = this.var(xFlat, 1);
      const yVar = this.var(yFlat, 1);

      return new JsNDArray(
        new Float64Array([xVar, cov, cov, yVar]),
        [2, 2]
      );
    }
  }

  corrcoef(x: NDArray, y?: NDArray): NDArray {
    const covMatrix = this.cov(x, y);
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
    const vReversed = new JsNDArray(
      new Float64Array([...vFlat.data].reverse()),
      vFlat.shape
    );
    return this.convolve(a, vReversed, mode);
  }

  // ============ Matrix Creation ============

  identity(n: number): NDArray {
    return this.eye(n);
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

  meshgrid(x: NDArray, y: NDArray): { X: NDArray; Y: NDArray } {
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
      X: new JsNDArray(X, [ny, nx]),
      Y: new JsNDArray(Y, [ny, nx]),
    };
  }

  logspace(start: number, stop: number, num: number, base: number = 10): NDArray {
    const linear = this.linspace(start, stop, num);
    const result = new Float64Array(num);
    for (let i = 0; i < num; i++) {
      result[i] = Math.pow(base, linear.data[i]);
    }
    return new JsNDArray(result, [num]);
  }

  geomspace(start: number, stop: number, num: number): NDArray {
    if (start === 0 || stop === 0) {
      throw new Error('geomspace: start and stop must be non-zero');
    }
    if ((start < 0) !== (stop < 0)) {
      throw new Error('geomspace: start and stop must have same sign');
    }

    const logStart = Math.log(Math.abs(start));
    const logStop = Math.log(Math.abs(stop));
    const linear = this.linspace(logStart, logStop, num);
    const result = new Float64Array(num);
    const sign = start < 0 ? -1 : 1;

    for (let i = 0; i < num; i++) {
      result[i] = sign * Math.exp(linear.data[i]);
    }

    return new JsNDArray(result, [num]);
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

  sort(arr: NDArray, axis: number = -1): NDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const result = new Float64Array(arr.data);
    const shape = arr.shape;
    const strides = this._computeStrides(shape);
    const axisLen = shape[axis];

    // Number of 1D subarrays to sort
    const nSlices = arr.data.length / axisLen;

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

  argsort(arr: NDArray, axis: number = -1): NDArray {
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
        const va = values[a], vb = values[b];
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

  searchsorted(arr: NDArray, v: number | NDArray, side: 'left' | 'right' = 'left'): NDArray | number {
    const flat = this.flatten(arr);
    const data = flat.data;

    const search = (val: number): number => {
      let lo = 0, hi = data.length;
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

  unique(arr: NDArray): NDArray {
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

    // Sort the result
    result.sort((a, b) => {
      if (Number.isNaN(a) && Number.isNaN(b)) return 0;
      if (Number.isNaN(a)) return 1;
      if (Number.isNaN(b)) return -1;
      return a - b;
    });

    return new JsNDArray(new Float64Array(result), [result.length]);
  }

  // No-op for JS backend (all data is already on CPU)
  async materializeAll(): Promise<void> {
    // Nothing to do
  }

  // ============ NaN-aware Stats ============

  nansum(arr: NDArray): number {
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) sum += arr.data[i];
    }
    return sum;
  }

  nanmean(arr: NDArray): number {
    let sum = 0, count = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) { sum += arr.data[i]; count++; }
    }
    return count > 0 ? sum / count : NaN;
  }

  nanvar(arr: NDArray, ddof: number = 0): number {
    const mean = this.nanmean(arr);
    if (Number.isNaN(mean)) return NaN;
    let sumSq = 0, count = 0;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i])) {
        const diff = arr.data[i] - mean;
        sumSq += diff * diff;
        count++;
      }
    }
    return count > ddof ? sumSq / (count - ddof) : NaN;
  }

  nanstd(arr: NDArray, ddof: number = 0): number {
    return Math.sqrt(this.nanvar(arr, ddof));
  }

  nanmin(arr: NDArray): number {
    let min = Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] < min) min = arr.data[i];
    }
    return min === Infinity ? NaN : min;
  }

  nanmax(arr: NDArray): number {
    let max = -Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] > max) max = arr.data[i];
    }
    return max === -Infinity ? NaN : max;
  }

  nanargmin(arr: NDArray): number {
    let minIdx = -1, minVal = Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] < minVal) {
        minVal = arr.data[i]; minIdx = i;
      }
    }
    return minIdx;
  }

  nanargmax(arr: NDArray): number {
    let maxIdx = -1, maxVal = -Infinity;
    for (let i = 0; i < arr.data.length; i++) {
      if (!Number.isNaN(arr.data[i]) && arr.data[i] > maxVal) {
        maxVal = arr.data[i]; maxIdx = i;
      }
    }
    return maxIdx;
  }

  nanprod(arr: NDArray): number {
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
    return Array.from(arr.data).filter(x => !Number.isNaN(x)).sort((a, b) => a - b);
  }

  median(arr: NDArray): number {
    const sorted = this._sortedData(arr);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  percentile(arr: NDArray, q: number): number {
    if (q < 0 || q > 100) throw new Error('percentile must be 0-100');
    return this.quantile(arr, q / 100);
  }

  quantile(arr: NDArray, q: number): number {
    if (q < 0 || q > 1) throw new Error('quantile must be 0-1');
    const sorted = this._sortedData(arr);
    if (sorted.length === 0) return NaN;
    if (sorted.length === 1) return sorted[0];
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    const frac = pos - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }

  nanmedian(arr: NDArray): number {
    const sorted = this._sortedNonNaN(arr);
    if (sorted.length === 0) return NaN;
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  nanpercentile(arr: NDArray, q: number): number {
    if (q < 0 || q > 100) throw new Error('percentile must be 0-100');
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

  histogram(arr: NDArray, bins: number = 10): { hist: NDArray; binEdges: NDArray } {
    const data = arr.data;
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (!Number.isNaN(data[i])) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
      }
    }
    if (min === Infinity) {
      // All NaN
      return {
        hist: new JsNDArray(new Float64Array(bins), [bins]),
        binEdges: new JsNDArray(new Float64Array(bins + 1), [bins + 1])
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
      hist: new JsNDArray(hist, [bins]),
      binEdges: new JsNDArray(edges, [bins + 1])
    };
  }

  histogramBinEdges(arr: NDArray, bins: number = 10): NDArray {
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
    return (this._rngState >>> 0) / 0xFFFFFFFF;
  }

  seed(s: number): void {
    this._rngState = s >>> 0;
    if (this._rngState === 0) this._rngState = 1; // Avoid zero state
  }

  rand(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift();
    }
    return new JsNDArray(data, shape);
  }

  randn(shape: number[]): NDArray {
    // Box-Muller transform for normal distribution
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
    return new JsNDArray(data, shape);
  }

  randint(low: number, high: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(this._xorshift() * range) + low;
    }
    return new JsNDArray(data, shape);
  }

  uniform(low: number, high: number, shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = this._xorshift() * range + low;
    }
    return new JsNDArray(data, shape);
  }

  normal(loc: number, scale: number, shape: number[]): NDArray {
    const arr = this.randn(shape);
    const data = arr.data;
    for (let i = 0; i < data.length; i++) {
      data[i] = data[i] * scale + loc;
    }
    return new JsNDArray(data, shape);
  }

  shuffle(arr: NDArray): NDArray {
    // Fisher-Yates shuffle on first axis
    const data = new Float64Array(arr.data);
    const shape = [...arr.shape];
    if (shape.length === 1) {
      // 1D shuffle
      for (let i = data.length - 1; i > 0; i--) {
        const j = Math.floor(this._xorshift() * (i + 1));
        [data[i], data[j]] = [data[j], data[i]];
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
    return new JsNDArray(data, shape);
  }

  choice(arr: NDArray, size: number, replace: boolean = true): NDArray {
    const n = arr.data.length;
    const data = new Float64Array(size);
    if (replace) {
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
    return this.shuffle(arr);
  }
}

export function createJsBackend(): Backend {
  return new JsBackend();
}
