/**
 * Statistics tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/stats.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

export function statsTests(getBackend: () => Backend) {
  describe('stats', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[]) => B.array(data, [data.length]);
    const mat = (data: number[], rows: number, cols: number) =>
      B.array(data, [rows, cols]);

    // ============ sum ============

    describe('sum', () => {
      it('computes sum of 1D array', () => {
        const a = arr([1.0, 2.0, 3.0, 4.0, 5.0]);
        expect(B.sum(a)).toBe(15.0);
      });

      it('computes sum of 2D array', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        expect(B.sum(a)).toBe(21.0);
      });

      it('computes sum of single element', () => {
        const a = arr([42.0]);
        expect(B.sum(a)).toBe(42.0);
      });

      it('computes sum of empty array', () => {
        const a = arr([]);
        expect(B.sum(a)).toBe(0.0);
      });

      it('computes sum with negative values', () => {
        const a = arr([-5.0, 10.0, -3.0]);
        expect(B.sum(a)).toBe(2.0);
      });
    });

    // ============ prod ============

    describe('prod', () => {
      it('computes product', () => {
        const a = arr([1.0, 2.0, 3.0, 4.0]);
        expect(B.prod(a)).toBe(24.0);
      });

      it('computes product with zero', () => {
        const a = arr([1.0, 2.0, 0.0, 4.0]);
        expect(B.prod(a)).toBe(0.0);
      });
    });

    // ============ mean ============

    describe('mean', () => {
      it('computes mean of array', () => {
        const a = arr([1.0, 2.0, 3.0, 4.0, 5.0]);
        expect(B.mean(a)).toBe(3.0);
      });

      it('computes mean of single element', () => {
        const a = arr([42.0]);
        expect(B.mean(a)).toBe(42.0);
      });

      it('computes mean with decimals', () => {
        const a = arr([1.0, 2.0, 3.0, 4.0]);
        expect(B.mean(a)).toBe(2.5);
      });

      it('returns NaN for empty array', () => {
        const a = arr([]);
        expect(Number.isNaN(B.mean(a))).toBe(true);
      });
    });

    // ============ var ============

    describe('var', () => {
      it('computes variance', () => {
        const a = arr([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        // Population variance = 4.0
        expect(approxEq(B.var(a, 0), 4.0, RELAXED_TOL)).toBe(true);
      });

      it('computes variance of constant array', () => {
        const a = arr([5.0, 5.0, 5.0, 5.0]);
        expect(B.var(a, 0)).toBe(0.0);
      });
    });

    // ============ std ============

    describe('std', () => {
      it('computes standard deviation', () => {
        const a = arr([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        // Population std = 2.0
        expect(approxEq(B.std(a, 0), 2.0, RELAXED_TOL)).toBe(true);
      });

      it('computes std of constant array', () => {
        const a = arr([5.0, 5.0, 5.0, 5.0]);
        expect(B.std(a, 0)).toBe(0.0);
      });
    });

    // ============ min/max ============

    describe('min/max', () => {
      it('computes min', () => {
        const a = arr([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        expect(B.min(a)).toBe(1.0);
      });

      it('computes max', () => {
        const a = arr([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        expect(B.max(a)).toBe(9.0);
      });

      it('handles negative values', () => {
        const a = arr([-5.0, -2.0, -8.0, -1.0]);
        expect(B.min(a)).toBe(-8.0);
        expect(B.max(a)).toBe(-1.0);
      });

      it('handles single element', () => {
        const a = arr([42.0]);
        expect(B.min(a)).toBe(42.0);
        expect(B.max(a)).toBe(42.0);
      });

      it('throws for empty array', () => {
        const a = arr([]);
        expect(() => B.min(a)).toThrow();
        expect(() => B.max(a)).toThrow();
      });
    });

    // ============ argmin/argmax ============

    describe('argmin/argmax', () => {
      it('computes argmin', () => {
        const a = arr([3.0, 1.0, 4.0, 1.0, 5.0]);
        expect(B.argmin(a)).toBe(1); // First occurrence of minimum
      });

      it('computes argmax', () => {
        const a = arr([3.0, 1.0, 4.0, 5.0, 2.0]);
        expect(B.argmax(a)).toBe(3);
      });

      it('throws for empty array', () => {
        const a = arr([]);
        expect(() => B.argmin(a)).toThrow();
        expect(() => B.argmax(a)).toThrow();
      });
    });

    // ============ axis operations ============

    describe('axis operations', () => {
      it('computes sum along axis 0', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const result = B.sumAxis(a, 0);
        expect(await getData(result, B)).toEqual([5.0, 7.0, 9.0]);
      });

      it('computes sum along axis 1', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const result = B.sumAxis(a, 1);
        expect(await getData(result, B)).toEqual([6.0, 15.0]);
      });

      it('throws on invalid axis', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        expect(() => B.sumAxis(a, 5)).toThrow();
      });

      it('computes mean along axis 0', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const result = B.meanAxis(a, 0);
        expect(await getData(result, B)).toEqual([2.5, 3.5, 4.5]);
      });
    });

    // ============ cumsum/cumprod ============

    describe('cumulative operations', () => {
      it('computes cumsum', async () => {
        const a = arr([1.0, 2.0, 3.0, 4.0, 5.0]);
        const result = B.cumsum(a);
        expect(await getData(result, B)).toEqual([1.0, 3.0, 6.0, 10.0, 15.0]);
      });

      it('computes cumsum of single element', async () => {
        const a = arr([42.0]);
        const result = B.cumsum(a);
        expect(await getData(result, B)).toEqual([42.0]);
      });

      it('computes cumprod', async () => {
        const a = arr([1.0, 2.0, 3.0, 4.0]);
        const result = B.cumprod(a);
        expect(await getData(result, B)).toEqual([1.0, 2.0, 6.0, 24.0]);
      });
    });

    // ============ all/any ============

    describe('boolean reductions', () => {
      it('all returns true for all non-zero', () => {
        const a = arr([1.0, 2.0, 3.0]);
        expect(B.all(a)).toBe(true);
      });

      it('all returns false if any zero', () => {
        const a = arr([1.0, 0.0, 3.0]);
        expect(B.all(a)).toBe(false);
      });

      it('any returns true if any non-zero', () => {
        const a = arr([0.0, 0.0, 1.0]);
        expect(B.any(a)).toBe(true);
      });

      it('any returns false for all zeros', () => {
        const a = arr([0.0, 0.0, 0.0]);
        expect(B.any(a)).toBe(false);
      });

      it('all returns true for empty array', () => {
        const a = arr([]);
        expect(B.all(a)).toBe(true); // NumPy behavior
      });

      it('any returns false for empty array', () => {
        const a = arr([]);
        expect(B.any(a)).toBe(false); // NumPy behavior
      });
    });
  });
}
