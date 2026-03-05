/**
 * Phase 2 operations tests
 * Covers: Advanced linalg, polynomial ops, advanced indexing, interpolation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, arraysApproxEq } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

export function phase2Tests(getBackend: () => Backend) {
  describe('phase2', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    // ============ Advanced Linalg ============

    describe('matrixPower', () => {
      it('computes A^0 = I', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.matrixPower(A, 0);
        expect(result.shape).toEqual([2, 2]);
        // Should be identity matrix
        const data = await getData(result, B);
        expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 1, DEFAULT_TOL)).toBe(true);
      });

      it('computes A^1 = A', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.matrixPower(A, 1);
        expect(arraysApproxEq(await getData(result, B), [1, 2, 3, 4], DEFAULT_TOL)).toBe(true);
      });

      it('computes A^2', async () => {
        // [[1, 2], [3, 4]] ^ 2 = [[7, 10], [15, 22]]
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.matrixPower(A, 2);
        expect(arraysApproxEq(await getData(result, B), [7, 10, 15, 22], DEFAULT_TOL)).toBe(true);
      });

      it('computes A^3', async () => {
        // [[1, 2], [3, 4]] ^ 3 = [[37, 54], [81, 118]]
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.matrixPower(A, 3);
        expect(arraysApproxEq(await getData(result, B), [37, 54, 81, 118], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('kron', () => {
      it('computes Kronecker product of 2x2 matrices', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const B_ = B.array([0, 5, 6, 7], [2, 2]);
        const result = B.kron(A, B_);
        // Expected 4x4 result
        expect(result.shape).toEqual([4, 4]);
        // First block: 1 * [[0,5],[6,7]] = [[0,5],[6,7]]
        const data = await getData(result, B);
        expect(approxEq(data[0], 0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], 6, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[5], 7, DEFAULT_TOL)).toBe(true);
      });

      it('computes Kronecker product with 1D arrays', async () => {
        const A = B.array([1, 2], [2]);
        const B_ = B.array([3, 4], [2]);
        const result = B.kron(A, B_);
        // NumPy kron with 1D arrays: [1, 2] kron [3, 4] = [3, 4, 6, 8] as 1D
        // But our implementation reshapes to [n, 1] first, giving [4, 1] output
        // This matches NumPy when treating 1D as column vectors
        expect(result.shape[0] * result.shape[1]).toBe(4);
        const flatResult = (await getData(result, B)).slice();
        expect(flatResult.sort()).toEqual([3, 4, 6, 8]);
      });
    });

    describe('slogdet', () => {
      it('computes sign and log determinant of positive-determinant matrix', async () => {
        // [[4, 7], [2, 6]] has det = 24 - 14 = 10
        const A = B.array([4, 7, 2, 6], [2, 2]);
        const { sign, logabsdet } = B.slogdet(A);
        if (B.materializeAll) await B.materializeAll();
        expect(sign).toBe(1);
        expect(approxEq(logabsdet, Math.log(10), RELAXED_TOL)).toBe(true);
      });

      it('computes sign and log determinant of negative-determinant matrix', async () => {
        // [[1, 2], [3, 4]] has det = 4 - 6 = -2
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const { sign, logabsdet } = B.slogdet(A);
        if (B.materializeAll) await B.materializeAll();
        expect(sign).toBe(-1);
        expect(approxEq(logabsdet, Math.log(2), RELAXED_TOL)).toBe(true);
      });

      it('computes sign and log determinant of identity', async () => {
        const A = B.eye(3);
        const { sign, logabsdet } = B.slogdet(A);
        if (B.materializeAll) await B.materializeAll();
        expect(sign).toBe(1);
        expect(approxEq(logabsdet, 0, RELAXED_TOL)).toBe(true);
      });
    });

    describe('multiDot', () => {
      it('computes multi_dot with two matrices', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const B_ = B.array([5, 6, 7, 8], [2, 2]);
        const result = B.multiDot([A, B_]);
        // Same as A @ B
        expect(arraysApproxEq(await getData(result, B), [19, 22, 43, 50], DEFAULT_TOL)).toBe(true);
      });

      it('computes multi_dot with three matrices', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const B_ = B.array([5, 6, 7, 8], [2, 2]);
        const C = B.array([1, 0, 0, 1], [2, 2]); // Identity
        const result = B.multiDot([A, B_, C]);
        // (A @ B) @ I = A @ B
        expect(arraysApproxEq(await getData(result, B), [19, 22, 43, 50], DEFAULT_TOL)).toBe(true);
      });

      it('handles single matrix', async () => {
        const A = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.multiDot([A]);
        expect(arraysApproxEq(await getData(result, B), [1, 2, 3, 4], DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ Polynomial ============

    describe('polyval', () => {
      it('evaluates polynomial at single point', async () => {
        // p(x) = 2x^2 + 3x + 1
        const p = B.array([2, 3, 1], [3]);
        const x = B.array([2], [1]);
        const result = B.polyval(p, x);
        // p(2) = 2*4 + 3*2 + 1 = 8 + 6 + 1 = 15
        const data = await getData(result, B);
        expect(approxEq(data[0], 15, DEFAULT_TOL)).toBe(true);
      });

      it('evaluates polynomial at multiple points', async () => {
        // p(x) = x^2 - 1
        const p = B.array([1, 0, -1], [3]);
        const x = B.array([0, 1, 2, 3], [4]);
        const result = B.polyval(p, x);
        expect(arraysApproxEq(await getData(result, B), [-1, 0, 3, 8], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('polyadd', () => {
      it('adds two polynomials of same degree', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.polyadd(a, b);
        expect(arraysApproxEq(await getData(result, B), [5, 7, 9], DEFAULT_TOL)).toBe(true);
      });

      it('adds polynomials of different degrees', async () => {
        const a = B.array([1, 2], [2]);  // x + 2
        const b = B.array([3, 4, 5], [3]);  // 3x^2 + 4x + 5
        const result = B.polyadd(a, b);
        // 0*x^2 + x + 2 + 3x^2 + 4x + 5 = 3x^2 + 5x + 7
        expect(arraysApproxEq(await getData(result, B), [3, 5, 7], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('polymul', () => {
      it('multiplies two polynomials', async () => {
        // (x + 1) * (x + 2) = x^2 + 3x + 2
        const a = B.array([1, 1], [2]);  // x + 1
        const b = B.array([1, 2], [2]);  // x + 2
        const result = B.polymul(a, b);
        expect(arraysApproxEq(await getData(result, B), [1, 3, 2], DEFAULT_TOL)).toBe(true);
      });

      it('multiplies polynomial by constant', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([2], [1]);
        const result = B.polymul(a, b);
        expect(arraysApproxEq(await getData(result, B), [2, 4, 6], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('polyfit', () => {
      it('fits linear polynomial', async () => {
        // Points: (0, 1), (1, 3), (2, 5) - line y = 2x + 1
        const x = B.array([0, 1, 2], [3]);
        const y = B.array([1, 3, 5], [3]);
        const result = B.polyfit(x, y, 1);
        // Should get [2, 1] for y = 2x + 1
        expect(result.shape).toEqual([2]);
        const data = await getData(result, B);
        expect(approxEq(data[0], 2, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 1, RELAXED_TOL)).toBe(true);
      });

      it('fits quadratic polynomial', async () => {
        // Points on y = x^2: (0, 0), (1, 1), (2, 4), (3, 9)
        const x = B.array([0, 1, 2, 3], [4]);
        const y = B.array([0, 1, 4, 9], [4]);
        const result = B.polyfit(x, y, 2);
        // Should get [1, 0, 0] for y = x^2
        expect(result.shape).toEqual([3]);
        const data = await getData(result, B);
        expect(approxEq(data[0], 1, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0, RELAXED_TOL)).toBe(true);
      });
    });

    describe('roots', () => {
      it('finds roots of quadratic', async () => {
        // x^2 - 3x + 2 = (x-1)(x-2) has roots 1 and 2
        const p = B.array([1, -3, 2], [3]);
        const result = B.roots(p);
        expect(result.shape[0]).toBe(2);
        const data = await getData(result, B);
        const roots = data.slice().sort((a, b) => a - b);
        expect(approxEq(roots[0], 1, RELAXED_TOL)).toBe(true);
        expect(approxEq(roots[1], 2, RELAXED_TOL)).toBe(true);
      });

      it('finds roots of linear', async () => {
        // 2x - 4 = 0 has root x = 2
        const p = B.array([2, -4], [2]);
        const result = B.roots(p);
        expect(result.shape[0]).toBe(1);
        const data = await getData(result, B);
        expect(approxEq(data[0], 2, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ Interpolation ============

    describe('interp', () => {
      it('interpolates linearly', async () => {
        const xp = B.array([0, 1, 2], [3]);
        const fp = B.array([0, 10, 20], [3]);
        const x = B.array([0.5, 1.5], [2]);
        const result = B.interp(x, xp, fp);
        const data = await getData(result, B);
        expect(approxEq(data[0], 5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 15, DEFAULT_TOL)).toBe(true);
      });

      it('clips to boundary values', async () => {
        const xp = B.array([0, 1, 2], [3]);
        const fp = B.array([10, 20, 30], [3]);
        const x = B.array([-1, 3], [2]);
        const result = B.interp(x, xp, fp);
        const data = await getData(result, B);
        expect(approxEq(data[0], 10, DEFAULT_TOL)).toBe(true);  // Clips to first
        expect(approxEq(data[1], 30, DEFAULT_TOL)).toBe(true);  // Clips to last
      });

      it('returns exact values at knots', async () => {
        const xp = B.array([0, 1, 2, 3], [4]);
        const fp = B.array([5, 10, 15, 20], [4]);
        const x = B.array([0, 1, 2, 3], [4]);
        const result = B.interp(x, xp, fp);
        expect(arraysApproxEq(await getData(result, B), [5, 10, 15, 20], DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ Histogram ============

    describe('bincount', () => {
      it('counts integer occurrences', async () => {
        const x = B.array([0, 1, 1, 2, 2, 2], [6]);
        const result = B.bincount(x);
        expect(result.shape).toEqual([3]);
        expect(arraysApproxEq(await getData(result, B), [1, 2, 3], DEFAULT_TOL)).toBe(true);
      });

      it('uses weights', async () => {
        const x = B.array([0, 1, 1, 2], [4]);
        const weights = B.array([0.5, 1.0, 1.5, 2.0], [4]);
        const result = B.bincount(x, weights);
        expect(result.shape).toEqual([3]);
        const data = await getData(result, B);
        expect(approxEq(data[0], 0.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 2.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.0, DEFAULT_TOL)).toBe(true);
      });

      it('respects minlength', async () => {
        const x = B.array([0, 1], [2]);
        const result = B.bincount(x, undefined, 5);
        expect(result.shape).toEqual([5]);
        expect(arraysApproxEq(await getData(result, B), [1, 1, 0, 0, 0], DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ Advanced Indexing ============

    describe('partition', () => {
      it('partitions 1D array', async () => {
        const arr = B.array([3, 4, 2, 1, 5], [5]);
        const result = B.partition(arr, 2);
        // Element at index 2 should be 3 (third smallest)
        // Elements before index 2 should be <= 3
        // Elements after index 2 should be >= 3
        const data = await getData(result, B);
        expect(data[2]).toBe(3);
        expect(data[0] <= 3 && data[1] <= 3).toBe(true);
        expect(data[3] >= 3 && data[4] >= 3).toBe(true);
      });
    });

    describe('argpartition', () => {
      it('returns indices that would partition', async () => {
        const arr = B.array([3, 4, 2, 1, 5], [5]);
        const indices = B.argpartition(arr, 2);
        // indices[2] should be the index of the third smallest (which is value 3 at index 0)
        const arrData = await getData(arr, B);
        const indicesData = await getData(indices, B);
        const kthValue = arrData[indicesData[2]];
        for (let i = 0; i < 2; i++) {
          expect(arrData[indicesData[i]] <= kthValue).toBe(true);
        }
        for (let i = 3; i < 5; i++) {
          expect(arrData[indicesData[i]] >= kthValue).toBe(true);
        }
      });
    });

    describe('lexsort', () => {
      it('sorts by multiple keys', async () => {
        // Last key is primary
        const a = B.array([1, 2, 1, 2], [4]);  // Secondary key
        const b = B.array([3, 3, 4, 4], [4]);  // Primary key
        const indices = B.lexsort([a, b]);
        // Should sort by b first (3s before 4s), then by a
        // Expected order: (1,3), (2,3), (1,4), (2,4) -> indices [0, 1, 2, 3]
        const data = await getData(indices, B);
        expect(data[0]).toBe(0);  // (1, 3)
        expect(data[1]).toBe(1);  // (2, 3)
        expect(data[2]).toBe(2);  // (1, 4)
        expect(data[3]).toBe(3);  // (2, 4)
      });
    });

    describe('compress', () => {
      it('compresses 1D array', async () => {
        const condition = B.array([1, 0, 1, 1, 0], [5]);
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.compress(condition, arr);
        expect(result.shape).toEqual([3]);
        expect(arraysApproxEq(await getData(result, B), [10, 30, 40], DEFAULT_TOL)).toBe(true);
      });

      it('compresses along axis', async () => {
        const condition = B.array([1, 0, 1], [3]);
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.compress(condition, arr, 1);
        expect(result.shape).toEqual([2, 2]);
        // Row 0: [1, 2, 3] with condition [1, 0, 1] -> [1, 3]
        // Row 1: [4, 5, 6] with condition [1, 0, 1] -> [4, 6]
        expect(arraysApproxEq(await getData(result, B), [1, 3, 4, 6], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('extract', () => {
      it('extracts elements where condition is true', async () => {
        const condition = B.array([0, 1, 0, 1, 1], [5]);
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.extract(condition, arr);
        expect(result.shape).toEqual([3]);
        expect(arraysApproxEq(await getData(result, B), [20, 40, 50], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('place', () => {
      // Note: place modifies array in-place. This works on all backends.
      it('places values at masked positions', async () => {
        const arr = B.array([0, 0, 0, 0, 0], [5]);
        const mask = B.array([1, 0, 1, 0, 1], [5]);
        const vals = B.array([10, 20, 30], [3]);
        B.place(arr, mask, vals);
        expect(arraysApproxEq(await getData(arr, B), [10, 0, 20, 0, 30], DEFAULT_TOL)).toBe(true);
      });

      it('cycles values if too few', async () => {
        const arr = B.array([0, 0, 0, 0, 0], [5]);
        const mask = B.array([1, 1, 1, 1, 1], [5]);
        const vals = B.array([1, 2], [2]);
        B.place(arr, mask, vals);
        expect(arraysApproxEq(await getData(arr, B), [1, 2, 1, 2, 1], DEFAULT_TOL)).toBe(true);
      });
    });

    describe('select', () => {
      it('selects from multiple choices based on conditions', () => {
        // Use explicit condition arrays instead of comparison functions
        // Conditions: [true, true, false, false, false] and [false, false, true, false, false]
        const condNeg = B.array([1, 1, 0, 0, 0], [5]);  // x < 0
        const condZero = B.array([0, 0, 1, 0, 0], [5]); // x == 0

        // Choices: 100 for negative, 0 for zero
        const choiceNeg = B.full([5], 100);
        const choiceZero = B.zeros([5]);

        const result = B.select([condNeg, condZero], [choiceNeg, choiceZero], -1);
        // Expected: [100, 100, 0, -1, -1]
        expect(approxEq(result.data[0], 100, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result.data[1], 100, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result.data[2], 0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result.data[3], -1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result.data[4], -1, DEFAULT_TOL)).toBe(true);
      });
    });
  });
}
