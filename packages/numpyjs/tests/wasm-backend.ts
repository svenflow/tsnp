/**
 * WASM Backend adapter for tests
 *
 * This wraps the rumpy-wasm module to implement the Backend interface.
 */

import { Backend, NDArray as IFaceNDArray } from './test-utils';

// These will be populated by initWasmBackend
let wasmModule: any = null;

class WasmNDArray implements IFaceNDArray {
  private _inner: any;

  constructor(inner: any) {
    this._inner = inner;
  }

  get shape(): number[] {
    return Array.from(this._inner.shape);
  }

  get data(): Float64Array {
    return this._inner.toTypedArray();
  }

  toArray(): number[] {
    return Array.from(this._inner.toTypedArray());
  }

  get inner(): any {
    return this._inner;
  }
}

export class WasmBackend implements Backend {
  name = 'wasm';
  private wasm: any;

  constructor(wasm: any) {
    this.wasm = wasm;
  }

  private wrap(inner: any): WasmNDArray {
    return new WasmNDArray(inner);
  }

  private unwrap(arr: IFaceNDArray): any {
    return (arr as WasmNDArray).inner;
  }

  // ============ Creation ============

  zeros(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.zeros(new Uint32Array(shape)));
  }

  ones(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.ones(new Uint32Array(shape)));
  }

  full(shape: number[], value: number): IFaceNDArray {
    return this.wrap(this.wasm.full(new Uint32Array(shape), value));
  }

  arange(start: number, stop: number, step: number): IFaceNDArray {
    if (step === 0) {
      throw new Error('step cannot be zero');
    }
    return this.wrap(this.wasm.arange(start, stop, step));
  }

  linspace(start: number, stop: number, num: number): IFaceNDArray {
    return this.wrap(this.wasm.linspace(start, stop, num));
  }

  eye(n: number): IFaceNDArray {
    return this.wrap(this.wasm.eye(n));
  }

  diag(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    return this.wrap(this.unwrap(arr).diag(k));
  }

  array(data: number[], shape?: number[]): IFaceNDArray {
    const s = shape || [data.length];
    return this.wrap(this.wasm.arrayFromTyped(new Float64Array(data), new Uint32Array(s)));
  }

  // ============ Math - Unary ============

  sin(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinArr(this.unwrap(arr)));
  }

  cos(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.cosArr(this.unwrap(arr)));
  }

  tan(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanArr(this.unwrap(arr)));
  }

  arcsin(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arcsinArr(this.unwrap(arr)));
  }

  arccos(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arccosArr(this.unwrap(arr)));
  }

  arctan(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arctanArr(this.unwrap(arr)));
  }

  sinh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinhArr(this.unwrap(arr)));
  }

  cosh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.coshArr(this.unwrap(arr)));
  }

  tanh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanhArr(this.unwrap(arr)));
  }

  exp(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.expArr(this.unwrap(arr)));
  }

  log(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.logArr(this.unwrap(arr)));
  }

  log2(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.log2Arr(this.unwrap(arr)));
  }

  log10(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.log10Arr(this.unwrap(arr)));
  }

  sqrt(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sqrtArr(this.unwrap(arr)));
  }

  cbrt(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.cbrtArr(this.unwrap(arr)));
  }

  abs(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.absArr(this.unwrap(arr)));
  }

  sign(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.signArr(this.unwrap(arr)));
  }

  floor(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.floorArr(this.unwrap(arr)));
  }

  ceil(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.ceilArr(this.unwrap(arr)));
  }

  round(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.roundArr(this.unwrap(arr)));
  }

  neg(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.negArr(this.unwrap(arr)));
  }

  reciprocal(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.reciprocalArr(this.unwrap(arr)));
  }

  square(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.squareArr(this.unwrap(arr)));
  }

  // ============ Math - Unary (Extended) ============

  arcsinh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arcsinhArr(this.unwrap(arr)));
  }

  arccosh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arccoshArr(this.unwrap(arr)));
  }

  arctanh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.arctanhArr(this.unwrap(arr)));
  }

  expm1(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.expm1Arr(this.unwrap(arr)));
  }

  log1p(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.log1pArr(this.unwrap(arr)));
  }

  trunc(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.truncArr(this.unwrap(arr)));
  }

  fix(arr: IFaceNDArray): IFaceNDArray {
    // Same as trunc - round toward zero
    return this.trunc(arr);
  }

  sinc(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sincArr(this.unwrap(arr)));
  }

  deg2rad(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.deg2radArr(this.unwrap(arr)));
  }

  rad2deg(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.rad2degArr(this.unwrap(arr)));
  }

  heaviside(arr: IFaceNDArray, h0: number): IFaceNDArray {
    return this.wrap(this.wasm.heavisideArr(this.unwrap(arr), h0));
  }

  signbit(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.signbitArr(this.unwrap(arr)));
  }

  // ============ Math - Decomposition ============

  modf(arr: IFaceNDArray): { frac: IFaceNDArray; integ: IFaceNDArray } {
    const result = this.wasm.modfArr(this.unwrap(arr));
    return {
      frac: this.wrap(result[0]),
      integ: this.wrap(result[1]),
    };
  }

  frexp(arr: IFaceNDArray): { mantissa: IFaceNDArray; exponent: IFaceNDArray } {
    const result = this.wasm.frexpArr(this.unwrap(arr));
    return {
      mantissa: this.wrap(result[0]),
      exponent: this.wrap(result[1]),
    };
  }

  ldexp(arr: IFaceNDArray, exp: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.ldexpArr(this.unwrap(arr), this.unwrap(exp)));
  }

  divmod(a: IFaceNDArray, b: IFaceNDArray): { quotient: IFaceNDArray; remainder: IFaceNDArray } {
    const result = this.wasm.divmodArr(this.unwrap(a), this.unwrap(b));
    return {
      quotient: this.wrap(result[0]),
      remainder: this.wrap(result[1]),
    };
  }

  // ============ Math - Binary (Extended) ============

  mod(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.modArr(this.unwrap(a), this.unwrap(b)));
  }

  fmod(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // C-style fmod: result has same sign as dividend (a)
    // This is different from Python-style mod (divisor sign)
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] % b.data[i];
    }
    return this.array(Array.from(result), a.shape);
  }

  remainder(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.mod(a, b);
  }

  copysign(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.copysignArr(this.unwrap(a), this.unwrap(b)));
  }

  hypot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.hypotArr(this.unwrap(a), this.unwrap(b)));
  }

  arctan2(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Note: WASM export is atan2Arr with args (y, x) - same as Math.atan2(y, x)
    return this.wrap(this.wasm.atan2Arr(this.unwrap(a), this.unwrap(b)));
  }

  logaddexp(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.logaddexpArr(this.unwrap(a), this.unwrap(b)));
  }

  logaddexp2(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.logaddexp2Arr(this.unwrap(a), this.unwrap(b)));
  }

  fmax(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.fmaxArr(this.unwrap(a), this.unwrap(b)));
  }

  fmin(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.fminArr(this.unwrap(a), this.unwrap(b)));
  }

  // ============ Math - Binary ============

  add(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).add(this.unwrap(b)));
  }

  sub(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).sub(this.unwrap(b)));
  }

  mul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).mul(this.unwrap(b)));
  }

  div(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).div(this.unwrap(b)));
  }

  pow(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.powArr(this.unwrap(a), this.unwrap(b)));
  }

  maximum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.maximum(this.unwrap(a), this.unwrap(b)));
  }

  minimum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.minimum(this.unwrap(a), this.unwrap(b)));
  }

  // ============ Math - Scalar ============

  addScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).addScalar(scalar));
  }

  subScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).subScalar(scalar));
  }

  mulScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).mulScalar(scalar));
  }

  divScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).divScalar(scalar));
  }

  powScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).powScalar(scalar));
  }

  clip(arr: IFaceNDArray, min: number, max: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).clip(min, max));
  }

  // ============ Stats ============

  sum(arr: IFaceNDArray): number {
    return this.unwrap(arr).sum();
  }

  prod(arr: IFaceNDArray): number {
    return this.unwrap(arr).prod();
  }

  mean(arr: IFaceNDArray): number {
    return this.unwrap(arr).mean();
  }

  var(arr: IFaceNDArray, ddof: number = 0): number {
    if (ddof === 0) {
      return this.unwrap(arr).var();
    }
    return this.unwrap(arr).varDdof(ddof);
  }

  std(arr: IFaceNDArray, ddof: number = 0): number {
    return this.unwrap(arr).std(ddof);
  }

  min(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).min();
  }

  max(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).max();
  }

  argmin(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmin();
  }

  argmax(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmax();
  }

  cumsum(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumsum(0));
  }

  cumprod(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumprod(0));
  }

  all(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).all() !== 0;
  }

  any(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).any() !== 0;
  }

  sumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).sumAxis(axis, false));
  }

  meanAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).meanAxis(axis, false));
  }

  minAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).minAxis(axis, false));
  }

  maxAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).maxAxis(axis, false));
  }

  argminAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).argminAxis(axis));
  }

  argmaxAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).argmaxAxis(axis));
  }

  varAxis(arr: IFaceNDArray, axis: number, ddof: number = 0): IFaceNDArray {
    return this.wrap(this.unwrap(arr).varAxis(axis, ddof, false));
  }

  stdAxis(arr: IFaceNDArray, axis: number, ddof: number = 0): IFaceNDArray {
    return this.wrap(this.unwrap(arr).stdAxis(axis, ddof, false));
  }

  prodAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).prodAxis(axis, false));
  }

  allAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).allAxis(axis, false));
  }

  anyAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).anyAxis(axis, false));
  }

  cumsumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumsum(axis));
  }

  cumprodAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumprod(axis));
  }

  // ============ Comparison Operations ============

  private _checkSameShape(a: IFaceNDArray, b: IFaceNDArray): void {
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
    return this.wrap(this.unwrap(a).eq(this.unwrap(b)));
  }

  notEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).ne(this.unwrap(b)));
  }

  less(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).lt(this.unwrap(b)));
  }

  lessEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).le(this.unwrap(b)));
  }

  greater(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).gt(this.unwrap(b)));
  }

  greaterEqual(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).ge(this.unwrap(b)));
  }

  isnan(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).isNan());
  }

  isinf(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).isInf());
  }

  isfinite(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).isFinite());
  }

  // ============ Set Operations ============

  setdiff1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const setB = new Set(b.data);
    const result = Array.from(a.data).filter(x => !setB.has(x));
    const unique = [...new Set(result)].sort((x, y) => x - y);
    return this.array(unique, [unique.length]);
  }

  union1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const combined = new Set([...a.data, ...b.data]);
    const result = [...combined].sort((x, y) => x - y);
    return this.array(result, [result.length]);
  }

  intersect1d(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    const setB = new Set(b.data);
    const result = [...new Set(Array.from(a.data).filter(x => setB.has(x)))].sort((x, y) => x - y);
    return this.array(result, [result.length]);
  }

  isin(element: IFaceNDArray, testElements: IFaceNDArray): IFaceNDArray {
    const testSet = new Set(testElements.data);
    const data = element.data.map(x => testSet.has(x) ? 1 : 0);
    return this.array(Array.from(data), element.shape);
  }

  // ============ Extended Array Manipulation ============

  insert(arr: IFaceNDArray, index: number, values: IFaceNDArray | number, axis?: number): IFaceNDArray {
    if (axis === undefined) {
      const flat = Array.from(this.flatten(arr).data);
      const toInsert = typeof values === 'number' ? [values] : Array.from(values.data);
      if (index < 0) index = flat.length + index + 1;
      flat.splice(index, 0, ...toInsert);
      return this.array(flat, [flat.length]);
    }
    throw new Error('insert with axis not yet implemented');
  }

  deleteArr(arr: IFaceNDArray, index: number | number[], axis?: number): IFaceNDArray {
    if (axis === undefined) {
      const flat = Array.from(this.flatten(arr).data);
      const indices = Array.isArray(index) ? index : [index];
      const normalized = indices.map(i => i < 0 ? flat.length + i : i).sort((a, b) => b - a);
      for (const i of normalized) {
        flat.splice(i, 1);
      }
      return this.array(flat, [flat.length]);
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
      return this.array(Array.from(result), [result.length]);
    }
    return this.concatenate([arr, values], axis);
  }

  atleast1d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.array(Array.from(arr.data), [1]);
    }
    return arr;
  }

  atleast2d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.array(Array.from(arr.data), [1, 1]);
    }
    if (arr.shape.length === 1) {
      return this.array(Array.from(arr.data), [1, arr.shape[0]]);
    }
    return arr;
  }

  atleast3d(arr: IFaceNDArray): IFaceNDArray {
    if (arr.shape.length === 0) {
      return this.array(Array.from(arr.data), [1, 1, 1]);
    }
    if (arr.shape.length === 1) {
      return this.array(Array.from(arr.data), [1, arr.shape[0], 1]);
    }
    if (arr.shape.length === 2) {
      return this.array(Array.from(arr.data), [arr.shape[0], arr.shape[1], 1]);
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

    return this.array(Array.from(result), outShape);
  }

  // ============ Linalg ============

  matmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Use optimized f32 parallel matmul for better performance
    // The generic wasm.matmul goes through f64 which is much slower
    const aShape = a.shape;
    const bShape = b.shape;

    // Handle different cases
    if (aShape.length === 2 && bShape.length === 2) {
      // Validate dimensions: a is [m, k], b is [k2, n] - k must equal k2
      const m = aShape[0];
      const k = aShape[1];
      const k2 = bShape[0];
      const n = bShape[1];

      if (k !== k2) {
        throw new Error(`dimension mismatch for matmul: a.shape[1] (${k}) != b.shape[0] (${k2})`);
      }

      // Convert to f32 for optimized kernel
      const aData = new Float32Array(a.data);
      const bData = new Float32Array(b.data);

      // Use the optimized parallel f32 matmul (XNNPACK-style 6x8 kernel with packing)
      const cData = this.wasm.matmulF32OptimizedParallelV3(aData, bData, m, n, k);

      // Convert result back and wrap
      return this.array(Array.from(cData), [m, n]);
    }

    // Fallback to generic matmul for non-2D cases
    return this.wrap(this.wasm.matmul(this.unwrap(a), this.unwrap(b)));
  }

  dot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.dot(this.unwrap(a), this.unwrap(b)));
  }

  inner(a: IFaceNDArray, b: IFaceNDArray): number {
    // Inner product: sum of elementwise multiplication of flattened arrays
    const aFlat = this.flatten(a).data;
    const bFlat = this.flatten(b).data;
    if (aFlat.length !== bFlat.length) {
      throw new Error('inner product requires arrays of same size');
    }
    let sum = 0;
    for (let i = 0; i < aFlat.length; i++) {
      sum += aFlat[i] * bFlat[i];
    }
    return sum;
  }

  outer(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Outer product: a[i] * b[j] for all i, j
    const aFlat = this.flatten(a).data;
    const bFlat = this.flatten(b).data;
    const m = aFlat.length;
    const n = bFlat.length;
    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aFlat[i] * bFlat[j];
      }
    }
    return this.array(Array.from(result), [m, n]);
  }

  transpose(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).transpose());
  }

  trace(arr: IFaceNDArray): number {
    // Sum of diagonal elements
    if (arr.shape.length !== 2) throw new Error('trace requires 2D array');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);
    let sum = 0;
    for (let i = 0; i < k; i++) {
      sum += arr.data[i * n + i];
    }
    return sum;
  }

  det(arr: IFaceNDArray): number {
    return this.wasm.det(this.unwrap(arr));
  }

  inv(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.inv(this.unwrap(arr)));
  }

  solve(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.solve(this.unwrap(a), this.unwrap(b)));
  }

  norm(arr: IFaceNDArray, ord: number = 2): number {
    return this.wasm.norm(this.unwrap(arr), ord);
  }

  qr(arr: IFaceNDArray): { q: IFaceNDArray; r: IFaceNDArray } {
    const result = this.wasm.qr(this.unwrap(arr));
    return {
      q: this.wrap(result[0]),
      r: this.wrap(result[1]),
    };
  }

  svd(arr: IFaceNDArray): { u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray } {
    const result = this.wasm.svd(this.unwrap(arr));
    return {
      u: this.wrap(result[0]),
      s: this.wrap(result[1]),
      vt: this.wrap(result[2]),
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

    return this.array(Array.from(result), outShape);
  }

  cond(arr: IFaceNDArray, _p: number | 'fro' = 2): number {
    // Condition number via SVD: max(s) / min(s)
    const { s } = this.svd(arr);
    const sData = s.data;
    if (sData.length === 0) return Infinity;

    let maxS = -Infinity;
    let minS = Infinity;
    for (let i = 0; i < sData.length; i++) {
      if (sData[i] > maxS) maxS = sData[i];
      if (sData[i] < minS) minS = sData[i];
    }

    if (minS < 1e-14) return Infinity;
    return maxS / minS;
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
      return this.array(Array.from(arrays[0].data), arrays[0].shape);
    }
    let result = arrays[0];
    for (let i = 1; i < arrays.length; i++) {
      result = this.matmul(result, arrays[i]);
    }
    return result;
  }

  // ============ Polynomial ============

  polyval(p: IFaceNDArray, x: IFaceNDArray): IFaceNDArray {
    const coeffs = this.flatten(p).data;
    const result = x.data.map(xi => {
      let res = 0;
      for (let i = 0; i < coeffs.length; i++) {
        res = res * xi + coeffs[i];
      }
      return res;
    });
    return this.array(Array.from(result), x.shape);
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
    return this.array(Array.from(result), [maxLen]);
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
    return this.array(Array.from(result), [resultLen]);
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

    return this.array(Array.from(c), [m]);
  }

  roots(p: IFaceNDArray): IFaceNDArray {
    const coeffs = Array.from(this.flatten(p).data);

    while (coeffs.length > 0 && coeffs[0] === 0) {
      coeffs.shift();
    }

    if (coeffs.length <= 1) {
      return this.array([], [0]);
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
        const shift = H[(size - 1) * n + (size - 1)] -
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

    return this.array(roots, [roots.length]);
  }

  // ============ Interpolation ============

  interp(x: IFaceNDArray, xp: IFaceNDArray, fp: IFaceNDArray): IFaceNDArray {
    const xpData = this.flatten(xp).data;
    const fpData = this.flatten(fp).data;

    const result = x.data.map(xi => {
      if (xi <= xpData[0]) return fpData[0];
      if (xi >= xpData[xpData.length - 1]) return fpData[fpData.length - 1];

      let lo = 0, hi = xpData.length - 1;
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (xpData[mid] <= xi) lo = mid;
        else hi = mid;
      }

      const t = (xi - xpData[lo]) / (xpData[hi] - xpData[lo]);
      return fpData[lo] + t * (fpData[hi] - fpData[lo]);
    });
    return this.array(Array.from(result), x.shape);
  }

  // ============ Histogram ============

  bincount(x: IFaceNDArray, weights?: IFaceNDArray, minlength?: number): IFaceNDArray {
    const xFlat = this.flatten(x).data;
    const wFlat = weights ? this.flatten(weights).data : null;

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

    return this.array(Array.from(result), [outLen]);
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

    return this.array(Array.from(result), shape);
  }

  broadcastArrays(...arrays: IFaceNDArray[]): IFaceNDArray[] {
    if (arrays.length === 0) return [];
    if (arrays.length === 1) return [this.array(Array.from(arrays[0].data), arrays[0].shape)];

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

    return this.array(Array.from(result), newShape);
  }

  swapaxes(arr: IFaceNDArray, axis1: number, axis2: number): IFaceNDArray {
    const ndim = arr.shape.length;
    axis1 = this._normalizeAxis(axis1, ndim);
    axis2 = this._normalizeAxis(axis2, ndim);

    if (axis1 === axis2) {
      return this.array(Array.from(arr.data), arr.shape);
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
      return this.array(Array.from(arr.data), arr.shape);
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
      return this.array(Array.from(arr.data), newShape.length === 0 ? [1] : newShape);
    }

    const newShape = arr.shape.filter(d => d !== 1);
    return this.array(Array.from(arr.data), newShape.length === 0 ? [1] : newShape);
  }

  expandDims(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const ndim = arr.shape.length + 1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    const newShape = [...arr.shape];
    newShape.splice(axis, 0, 1);
    return this.array(Array.from(arr.data), newShape);
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

    return this.array(Array.from(arr.data), newShape);
  }

  flatten(arr: IFaceNDArray): IFaceNDArray {
    return this.array(Array.from(arr.data), [arr.data.length]);
  }

  concatenate(arrays: IFaceNDArray[], axis: number = 0): IFaceNDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');
    if (arrays.length === 1) return this.array(Array.from(arrays[0].data), arrays[0].shape);

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
      return this.array(Array.from(result), outShape);
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

    return this.array(Array.from(result), outShape);
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

      return this.array(Array.from(sliceData), sliceShape);
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

    return this.array(Array.from(result), condBcast.shape);
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
      return this.array(Array.from(result), [indexArray.length]);
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

    return this.array(Array.from(result), outShape);
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

    return this.array(Array.from(result), arr.shape);
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

    return this.array(Array.from(result), arr.shape);
  }

  lexsort(keys: IFaceNDArray[]): IFaceNDArray {
    if (keys.length === 0) {
      return this.array([], [0]);
    }

    const n = keys[0].data.length;
    for (const key of keys) {
      if (key.data.length !== n) {
        throw new Error('all keys must have the same length');
      }
    }

    const indices = Array.from({ length: n }, (_, i) => i);

    indices.sort((a, b) => {
      for (let k = keys.length - 1; k >= 0; k--) {
        const va = keys[k].data[a];
        const vb = keys[k].data[b];
        if (va < vb) return -1;
        if (va > vb) return 1;
      }
      return 0;
    });

    return this.array(indices, [n]);
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
      return this.array(result, [result.length]);
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

    return this.array(Array.from(result), outShape);
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

    return this.array(result, [result.length]);
  }

  place(arr: IFaceNDArray, mask: IFaceNDArray, vals: IFaceNDArray): void {
    const maskFlat = this.flatten(mask).data;
    const valsFlat = this.flatten(vals).data;
    const arrData = arr.data; // Get a copy of the current data

    let valIdx = 0;
    for (let i = 0; i < arrData.length && i < maskFlat.length; i++) {
      if (maskFlat[i] !== 0) {
        arrData[i] = valsFlat[valIdx % valsFlat.length];
        valIdx++;
      }
    }

    // For WASM backend, we need to update the underlying array
    // by creating a new WASM array and replacing the inner reference
    const newInner = this.wasm.arrayFromTyped(new Float64Array(arrData), new Uint32Array(arr.shape));
    (arr as any)._inner = newInner;
  }

  select(condlist: IFaceNDArray[], choicelist: IFaceNDArray[], defaultVal: number = 0): IFaceNDArray {
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

    return this.array(Array.from(result), shape);
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

    return this.array(Array.from(result), outShape);
  }

  // ============ Einstein Summation ============

  einsum(subscripts: string, ...operands: IFaceNDArray[]): IFaceNDArray {
    const [inputStr, outputStr] = subscripts.split('->').map(s => s.trim());
    const inputSubscripts = inputStr.split(',').map(s => s.trim());

    if (inputSubscripts.length !== operands.length) {
      throw new Error(`einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`);
    }

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

    let outputLabels: string[];
    if (outputStr !== undefined) {
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
    const contractedTotal = contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

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
          const size = contractedSizes[d];
          const stride = d < contractedSizes.length - 1
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

    return this.array(Array.from(result), outputShape.length === 0 ? [1] : outputShape);
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

    return this.array(Array.from(result), newShape);
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
        let prevIdx = 0, nextIdx = 0;
        for (let d = 0; d < ndim; d++) {
          prevIdx += prevCoords[d] * strides[d];
          nextIdx += nextCoords[d] * strides[d];
        }
        grad = (arr.data[nextIdx] - arr.data[prevIdx]) / 2;
      }

      result[i] = grad;
    }

    return this.array(Array.from(result), shape);
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

    return this.array([
      a2 * b3 - a3 * b2,
      a3 * b1 - a1 * b3,
      a1 * b2 - a2 * b1,
    ], [3]);
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

      return this.array(Array.from(result), [nVars, nVars]);
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

      return this.array([xVar, cov, cov, yVar], [2, 2]);
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

    return this.array(Array.from(result), [n, n]);
  }

  // ============ Convolution ============

  convolve(a: IFaceNDArray, v: IFaceNDArray, mode: 'full' | 'same' | 'valid' = 'full'): IFaceNDArray {
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
      outLen = Math.max(aLen - vLen + 1, 0);
      startIdx = vLen - 1;
    }

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

    const result = new Float64Array(outLen);
    for (let i = 0; i < outLen; i++) {
      result[i] = full[startIdx + i];
    }

    return this.array(Array.from(result), [outLen]);
  }

  correlate(a: IFaceNDArray, v: IFaceNDArray, mode: 'full' | 'same' | 'valid' = 'valid'): IFaceNDArray {
    const vFlat = this.flatten(v);
    const vReversed = this.array([...vFlat.data].reverse(), vFlat.shape);
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

    return this.array(Array.from(result), [rows, cols]);
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

    return this.array(Array.from(result), [rows, cols]);
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
      X: this.array(Array.from(X), [ny, nx]),
      Y: this.array(Array.from(Y), [ny, nx]),
    };
  }

  logspace(start: number, stop: number, num: number, base: number = 10): IFaceNDArray {
    return this.wrap(this.wasm.logspaceArr(start, stop, num, base));
  }

  geomspace(start: number, stop: number, num: number): IFaceNDArray {
    return this.wrap(this.wasm.geomspaceArr(start, stop, num));
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

    return this.array(Array.from(result), outShape);
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
      return this.array(Array.from(result), [result.length]);
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

    return this.array(Array.from(result), outShape);
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

    return indices.map(idx => this.array(idx, [idx.length]));
  }

  argwhere(arr: IFaceNDArray): IFaceNDArray {
    const indices = this.nonzero(arr);
    if (indices.length === 0 || indices[0].data.length === 0) {
      return this.array([], [0, arr.shape.length]);
    }

    const nNonzero = indices[0].data.length;
    const ndim = arr.shape.length;
    const result = new Float64Array(nNonzero * ndim);

    for (let i = 0; i < nNonzero; i++) {
      for (let d = 0; d < ndim; d++) {
        result[i * ndim + d] = indices[d].data[i];
      }
    }

    return this.array(Array.from(result), [nNonzero, ndim]);
  }

  flatnonzero(arr: IFaceNDArray): IFaceNDArray {
    const indices: number[] = [];
    for (let i = 0; i < arr.data.length; i++) {
      if (arr.data[i] !== 0) {
        indices.push(i);
      }
    }
    return this.array(indices, [indices.length]);
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

    return this.array(Array.from(result), arr.shape);
  }

  // ============ Sorting ============

  sort(arr: IFaceNDArray, axis: number = -1): IFaceNDArray {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const result = new Float64Array(arr.data);
    const shape = arr.shape;
    const strides = this._computeStrides(shape);
    const axisLen = shape[axis];

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

      for (let i = 0; i < axisLen; i++) {
        const coords = [...baseCoords];
        coords[axis] = i;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          idx += coords[d] * strides[d];
        }
        slice[i] = arr.data[idx];
      }

      const sorted = Array.from(slice).sort((a, b) => {
        if (Number.isNaN(a) && Number.isNaN(b)) return 0;
        if (Number.isNaN(a)) return 1;
        if (Number.isNaN(b)) return -1;
        return a - b;
      });

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

    return this.array(Array.from(result), shape);
  }

  argsort(arr: IFaceNDArray, axis: number = -1): IFaceNDArray {
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

    return this.array(Array.from(result), shape);
  }

  searchsorted(arr: IFaceNDArray, v: number | IFaceNDArray, side: 'left' | 'right' = 'left'): IFaceNDArray | number {
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
      return this.array(Array.from(result), vFlat.shape);
    }
  }

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

    result.sort((a, b) => {
      if (Number.isNaN(a) && Number.isNaN(b)) return 0;
      if (Number.isNaN(a)) return 1;
      if (Number.isNaN(b)) return -1;
      return a - b;
    });

    return this.array(result, [result.length]);
  }

  // No-op for WASM backend (all data is already on CPU)
  async materializeAll(): Promise<void> {
    // Nothing to do
  }

  // ============ NaN-aware Stats ============
  // These delegate to Rust WASM implementations

  nansum(arr: IFaceNDArray): number {
    return this.unwrap(arr).nansum();
  }

  nanmean(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanmean();
  }

  nanvar(arr: IFaceNDArray, ddof: number = 0): number {
    return this.unwrap(arr).nanvar(ddof);
  }

  nanstd(arr: IFaceNDArray, ddof: number = 0): number {
    return this.unwrap(arr).nanstd(ddof);
  }

  nanmin(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanmin();
  }

  nanmax(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanmax();
  }

  nanargmin(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanargmin();
  }

  nanargmax(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanargmax();
  }

  nanprod(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanprod();
  }

  // ============ Order Statistics ============
  // These delegate to Rust WASM implementations

  median(arr: IFaceNDArray): number {
    return this.unwrap(arr).median();
  }

  percentile(arr: IFaceNDArray, q: number): number {
    return this.unwrap(arr).percentile(q);
  }

  quantile(arr: IFaceNDArray, q: number): number {
    return this.unwrap(arr).quantile(q);
  }

  nanmedian(arr: IFaceNDArray): number {
    return this.unwrap(arr).nanmedian();
  }

  nanpercentile(arr: IFaceNDArray, q: number): number {
    return this.unwrap(arr).nanpercentile(q);
  }

  // ============ Histogram ============

  histogram(arr: IFaceNDArray, bins: number = 10): { hist: IFaceNDArray; binEdges: IFaceNDArray } {
    const data = arr.data;
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (!Number.isNaN(data[i])) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
      }
    }
    if (min === Infinity) {
      return {
        hist: this.array(Array(bins).fill(0), [bins]),
        binEdges: this.array(Array(bins + 1).fill(0), [bins + 1])
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
      binEdges: this.array(Array.from(edges), [bins + 1])
    };
  }

  histogramBinEdges(arr: IFaceNDArray, bins: number = 10): IFaceNDArray {
    const { binEdges } = this.histogram(arr, bins);
    return binEdges;
  }

  // ============ Random ============

  seed(s: number): void {
    this.wasm.randomSeed(BigInt(s >>> 0));
  }

  rand(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.randomRand(new Uint32Array(shape)));
  }

  randn(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.randomRandn(new Uint32Array(shape)));
  }

  randint(low: number, high: number, shape: number[]): IFaceNDArray {
    // WASM doesn't have randint, implement using uniform
    const size = shape.reduce((a, b) => a * b, 1);
    const uniform = this.wasm.randomUniform(0, 1, new Uint32Array(shape));
    const data = uniform.toTypedArray();
    const range = high - low;
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(data[i] * range) + low;
    }
    return this.array(Array.from(data), shape);
  }

  uniform(low: number, high: number, shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.randomUniform(low, high, new Uint32Array(shape)));
  }

  normal(loc: number, scale: number, shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.randomNormal(loc, scale, new Uint32Array(shape)));
  }

  shuffle(arr: IFaceNDArray): IFaceNDArray {
    // Fisher-Yates shuffle on first axis
    const data = new Float64Array(arr.data);
    const shape = [...arr.shape];
    if (shape.length === 1) {
      // 1D shuffle
      for (let i = data.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [data[i], data[j]] = [data[j], data[i]];
      }
    } else {
      // Shuffle along first axis
      const stride = shape.slice(1).reduce((a, b) => a * b, 1);
      const n = shape[0];
      const temp = new Float64Array(stride);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        temp.set(data.subarray(i * stride, (i + 1) * stride));
        data.copyWithin(i * stride, j * stride, (j + 1) * stride);
        data.set(temp, j * stride);
      }
    }
    return this.array(Array.from(data), shape);
  }

  choice(arr: IFaceNDArray, size: number, replace: boolean = true): IFaceNDArray {
    const n = arr.data.length;
    const data = new Float64Array(size);
    if (replace) {
      for (let i = 0; i < size; i++) {
        const idx = Math.floor(Math.random() * n);
        data[i] = arr.data[idx];
      }
    } else {
      if (size > n) throw new Error('Cannot sample more than array size without replacement');
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = 0; i < size; i++) {
        const j = i + Math.floor(Math.random() * (n - i));
        [indices[i], indices[j]] = [indices[j], indices[i]];
        data[i] = arr.data[indices[i]];
      }
    }
    return this.array(Array.from(data), [size]);
  }

  permutation(n: number | IFaceNDArray): IFaceNDArray {
    let arr: IFaceNDArray;
    if (typeof n === 'number') {
      arr = this.arange(0, n, 1);
    } else {
      arr = this.array(Array.from(n.data), [...n.shape]);
    }
    return this.shuffle(arr);
  }
}

/**
 * Initialize the WASM backend
 *
 * Must be called before createWasmBackend()
 */
export async function initWasmBackend(): Promise<void> {
  // Import the WASM module
  const module = await import('./wasm-pkg/rumpy_wasm.js');

  // In browser context, fetch the WASM file
  const wasmUrl = new URL('./wasm-pkg/rumpy_wasm_bg.wasm', import.meta.url);
  const wasmResponse = await fetch(wasmUrl);
  const wasmBytes = await wasmResponse.arrayBuffer();

  // Initialize the module with wasm bytes
  await module.default(wasmBytes);

  // Initialize thread pool (wasm-bindgen-rayon)
  // In browser, this sets up Web Workers
  await module.initThreadPool(navigator.hardwareConcurrency || 4);

  wasmModule = module;
}

/**
 * Create a WASM backend instance
 *
 * Requires initWasmBackend() to have been called first
 */
export function createWasmBackend(): Backend {
  if (!wasmModule) {
    throw new Error('WASM module not initialized. Call initWasmBackend() first.');
  }
  return new WasmBackend(wasmModule);
}
