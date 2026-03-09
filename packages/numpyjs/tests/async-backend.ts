/**
 * Async Backend Interface for GPU/WASM backends
 *
 * This interface mirrors the sync Backend interface but with async methods.
 * GPU operations are inherently async (buffer readback), so this is the native interface.
 */

import { NDArray } from './test-utils';

/** Async NDArray with getData() method */
export interface AsyncNDArray {
  shape: number[];
  /** Get data from GPU/WASM memory (async) */
  getData(): Promise<Float64Array>;
  /** Convert to JS array (async) */
  toArray(): Promise<number[]>;
}

/** Async Backend interface - all ops return Promise */
export interface AsyncBackend {
  name: string;

  // ============ Creation ============
  zeros(shape: number[]): Promise<AsyncNDArray>;
  ones(shape: number[]): Promise<AsyncNDArray>;
  full(shape: number[], value: number): Promise<AsyncNDArray>;
  arange(start: number, stop: number, step: number): Promise<AsyncNDArray>;
  linspace(start: number, stop: number, num: number): Promise<AsyncNDArray>;
  eye(n: number): Promise<AsyncNDArray>;
  diag(arr: AsyncNDArray, k?: number): Promise<AsyncNDArray>;
  array(data: number[], shape?: number[]): Promise<AsyncNDArray>;

  // ============ Math - Unary ============
  sin(arr: AsyncNDArray): Promise<AsyncNDArray>;
  cos(arr: AsyncNDArray): Promise<AsyncNDArray>;
  tan(arr: AsyncNDArray): Promise<AsyncNDArray>;
  arcsin(arr: AsyncNDArray): Promise<AsyncNDArray>;
  arccos(arr: AsyncNDArray): Promise<AsyncNDArray>;
  arctan(arr: AsyncNDArray): Promise<AsyncNDArray>;
  sinh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  cosh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  tanh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  exp(arr: AsyncNDArray): Promise<AsyncNDArray>;
  log(arr: AsyncNDArray): Promise<AsyncNDArray>;
  log2(arr: AsyncNDArray): Promise<AsyncNDArray>;
  log10(arr: AsyncNDArray): Promise<AsyncNDArray>;
  sqrt(arr: AsyncNDArray): Promise<AsyncNDArray>;
  cbrt(arr: AsyncNDArray): Promise<AsyncNDArray>;
  abs(arr: AsyncNDArray): Promise<AsyncNDArray>;
  sign(arr: AsyncNDArray): Promise<AsyncNDArray>;
  floor(arr: AsyncNDArray): Promise<AsyncNDArray>;
  ceil(arr: AsyncNDArray): Promise<AsyncNDArray>;
  round(arr: AsyncNDArray): Promise<AsyncNDArray>;
  neg(arr: AsyncNDArray): Promise<AsyncNDArray>;
  reciprocal(arr: AsyncNDArray): Promise<AsyncNDArray>;
  square(arr: AsyncNDArray): Promise<AsyncNDArray>;

  // ============ Math - Unary (Extended) ============
  arcsinh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  arccosh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  arctanh(arr: AsyncNDArray): Promise<AsyncNDArray>;
  expm1(arr: AsyncNDArray): Promise<AsyncNDArray>;
  log1p(arr: AsyncNDArray): Promise<AsyncNDArray>;
  trunc(arr: AsyncNDArray): Promise<AsyncNDArray>;
  fix(arr: AsyncNDArray): Promise<AsyncNDArray>;
  sinc(arr: AsyncNDArray): Promise<AsyncNDArray>;
  deg2rad(arr: AsyncNDArray): Promise<AsyncNDArray>;
  rad2deg(arr: AsyncNDArray): Promise<AsyncNDArray>;
  heaviside(arr: AsyncNDArray, h0: number): Promise<AsyncNDArray>;
  signbit(arr: AsyncNDArray): Promise<AsyncNDArray>;

  // ============ Math - Binary ============
  add(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  sub(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  mul(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  div(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  pow(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  maximum(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  minimum(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  mod(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  fmod(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  copysign(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  hypot(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  arctan2(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;

  // ============ Linear Algebra ============
  matmul(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  dot(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  transpose(arr: AsyncNDArray): Promise<AsyncNDArray>;

  // ============ Reductions ============
  sum(arr: AsyncNDArray): Promise<number>;
  mean(arr: AsyncNDArray): Promise<number>;
  min(arr: AsyncNDArray): Promise<number>;
  max(arr: AsyncNDArray): Promise<number>;

  // ============ Phase 2 Ops ============
  matrixPower(arr: AsyncNDArray, n: number): Promise<AsyncNDArray>;
  kron(a: AsyncNDArray, b: AsyncNDArray): Promise<AsyncNDArray>;
  polyval(p: AsyncNDArray, x: AsyncNDArray): Promise<AsyncNDArray>;
  interp(x: AsyncNDArray, xp: AsyncNDArray, fp: AsyncNDArray): Promise<AsyncNDArray>;
  bincount(x: AsyncNDArray, weights?: AsyncNDArray, minlength?: number): Promise<AsyncNDArray>;
  partition(arr: AsyncNDArray, kth: number, axis?: number): Promise<AsyncNDArray>;
  argpartition(arr: AsyncNDArray, kth: number, axis?: number): Promise<AsyncNDArray>;
  compress(condition: AsyncNDArray, arr: AsyncNDArray, axis?: number): Promise<AsyncNDArray>;
  extract(condition: AsyncNDArray, arr: AsyncNDArray): Promise<AsyncNDArray>;
  select(condlist: AsyncNDArray[], choicelist: AsyncNDArray[], defaultVal?: number): Promise<AsyncNDArray>;
}

/**
 * Wrap a sync NDArray as AsyncNDArray
 */
export function wrapSyncArray(arr: NDArray): AsyncNDArray {
  return {
    shape: arr.shape,
    async getData() { return arr.data; },
    async toArray() { return arr.toArray(); },
  };
}

/**
 * Wrap a sync Backend as AsyncBackend
 * Useful for JS backend which doesn't need async
 */
export function wrapSyncBackend(backend: any): AsyncBackend {
  const wrapped: any = { name: backend.name };

  // For each method, wrap return value in Promise
  for (const key of Object.keys(backend)) {
    if (typeof backend[key] === 'function') {
      wrapped[key] = async (...args: any[]) => {
        const result = backend[key](...args);
        if (result && typeof result === 'object' && 'shape' in result) {
          return wrapSyncArray(result);
        }
        return result;
      };
    }
  }

  return wrapped as AsyncBackend;
}
