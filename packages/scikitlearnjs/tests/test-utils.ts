/**
 * Test utilities for scikitlearnjs
 *
 * Provides a simple JS backend for testing without WASM dependencies.
 */

import type { NDArray, Backend } from 'numpyjs';

/**
 * Simple NDArray implementation for testing
 */
class TestNDArray implements NDArray {
  data: Float64Array;
  shape: number[];

  constructor(data: number[] | Float64Array, shape: number[]) {
    this.data = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = [...shape];
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }
}

/**
 * Minimal backend for testing - implements only what scikitlearnjs needs
 */
export class TestBackend implements Backend {
  name = 'test';

  array(data: number[], shape: number[]): NDArray {
    return new TestNDArray(data, shape);
  }

  zeros(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new TestNDArray(new Float64Array(size), shape);
  }

  ones(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new TestNDArray(new Float64Array(size).fill(1), shape);
  }

  transpose(a: NDArray): NDArray {
    const [rows, cols] = a.shape;
    const result = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = a.data[i * cols + j];
      }
    }
    return new TestNDArray(result, [cols, rows]);
  }

  matmul(a: NDArray, b: NDArray): NDArray {
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error(`Incompatible shapes for matmul: ${a.shape} and ${b.shape}`);
    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          sum += a.data[i * k1 + k] * b.data[k * n + j];
        }
        result[i * n + j] = sum;
      }
    }
    return new TestNDArray(result, [m, n]);
  }

  reshape(a: NDArray, shape: number[]): NDArray {
    return new TestNDArray(a.data, shape);
  }

  sub(a: NDArray, b: NDArray): NDArray {
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] - b.data[i];
    }
    return new TestNDArray(result, [...a.shape]);
  }

  mulScalar(a: NDArray, scalar: number): NDArray {
    const result = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      result[i] = a.data[i] * scalar;
    }
    return new TestNDArray(result, [...a.shape]);
  }

  flatten(a: NDArray): NDArray {
    return new TestNDArray(a.data, [a.data.length]);
  }
}

/**
 * Create an NDArray from a 2D JS array
 */
export function toNDArray(arr: number[][]): NDArray {
  const shape = [arr.length, arr[0].length];
  const flat = arr.flat();
  return new TestNDArray(flat, shape);
}

/**
 * Create a 1D NDArray from a JS array
 */
export function to1DArray(arr: number[]): NDArray {
  return new TestNDArray(arr, [arr.length]);
}

/**
 * Convert NDArray to 2D JS array
 */
export function to2DArray(arr: NDArray): number[][] {
  const [rows, cols] = arr.shape;
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result.push(Array.from(arr.data.slice(i * cols, (i + 1) * cols)));
  }
  return result;
}

/**
 * Compare arrays with tolerance
 */
export function expectArraysClose(
  actual: number[] | Float64Array | NDArray,
  expected: number[],
  decimals = 6,
) {
  let arr: number[];
  if (actual instanceof Float64Array) {
    arr = Array.from(actual);
  } else if (Array.isArray(actual)) {
    arr = actual;
  } else {
    // NDArray
    arr = Array.from(actual.data);
  }

  if (arr.length !== expected.length) {
    throw new Error(`Length mismatch: ${arr.length} vs ${expected.length}`);
  }

  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(arr[i] - expected[i]);
    const tolerance = Math.pow(10, -decimals);
    if (diff > tolerance) {
      throw new Error(
        `Value mismatch at index ${i}: ${arr[i]} vs ${expected[i]} (diff: ${diff})`,
      );
    }
  }
}

/**
 * Create test backend instance
 */
export function createTestBackend(): Backend {
  return new TestBackend();
}
