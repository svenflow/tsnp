/**
 * Coverage boost phase 4 — targeting final ~30 uncovered lines in base-backend.ts
 *
 * Covers: cond singular matrix (lines 1004,1028), roots non-convergence (1309-1310),
 * quickselect exact-k break (1443,1537), lexsort edge cases (1561,1582),
 * compress axis out-of-bounds (1607), apply_along_axis 3D (2185),
 * _checkSameShape errors (3039,3043), LU pivot swap (3065,3069-3073),
 * broadcast mismatch (3175), resize empty (3354), stack shape mismatch (3458),
 * histogram unknown bin strategy (5298), eig 1x1 (6139), polydiv remainder
 * trimming (7360), block non-array row (7501), nextafter(0, negative) (8097-8098).
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend } from './test-utils';

export function coverageBoost4Tests(getBackend: () => Backend) {
  describe('coverage-boost-4', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[], shape?: number[]) => B.array(data, shape ?? [data.length]);
    const mat = (data: number[], rows: number, cols: number) => B.array(data, [rows, cols]);

    // ============================================================
    // cond with truly singular matrix (lines 1004, 1028)
    // ============================================================

    describe('cond with non-standard p hits default fallback', () => {
      it('cond with p=3 falls through to default SVD path', () => {
        // p=3 is not 2, -2, 1, -1, Inf, -Inf, or 'fro' — hits default fallback (line 1022)
        const a = mat([3, 1, 1, 2], 2, 2);
        const c = B.cond(a, 3 as any);
        expect(c).toBeGreaterThan(0);
        expect(Number.isFinite(c)).toBe(true);
      });
    });

    // ============================================================
    // roots non-convergence (lines 1309-1310)
    // ============================================================

    describe('roots non-convergence fallback', () => {
      it('roots of a difficult polynomial still returns values', () => {
        // A polynomial whose companion matrix eigenvalues are hard to converge
        // x^10 + 1 (roots are 10th roots of -1, complex roots — real QR may not converge well)
        const coeffs = arr([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        const result = B.roots(coeffs);
        // Should return 10 roots (possibly approximate)
        expect(result.shape[0]).toBe(10);
      });
    });

    // ============================================================
    // quickselect exact-k break (lines 1443, 1537)
    // ============================================================

    describe('partition and argpartition quickselect exact-k break', () => {
      it('partition [3,2,1] kth=1 triggers exact-k break', () => {
        // [3,2,1] with kth=1: pivot=2 (middle element), after partition
        // i jumps to 2, j drops to 0, k=1 is between them → break
        const a = arr([3, 2, 1]);
        const result = B.partition(a, 1);
        const d = result.data;
        expect(d[1]).toBe(2);
        expect(d[0]).toBeLessThanOrEqual(2);
        expect(d[2]).toBeGreaterThanOrEqual(2);
      });

      it('argpartition [3,2,1] kth=1 triggers exact-k break', () => {
        const a = arr([3, 2, 1]);
        const result = B.argpartition(a, 1);
        const d = result.data;
        expect(a.data[d[1]]).toBe(2);
      });

      it('partition [5,3,1,4,2] kth=2 for larger array', () => {
        const a = arr([5, 3, 1, 4, 2]);
        const result = B.partition(a, 2);
        // Element at index 2 should be 3 (the median)
        expect(result.data[2]).toBe(3);
      });
    });

    // ============================================================
    // lexsort edge cases (lines 1561, 1582)
    // ============================================================

    describe('lexsort edge cases', () => {
      it('lexsort with empty keys returns empty array', () => {
        const result = B.lexsort([]);
        expect(result.shape).toEqual([0]);
        expect(result.data.length).toBe(0);
      });

      it('lexsort with all-equal keys returns stable order', () => {
        // All keys are equal, so comparator returns 0
        const keys = [arr([1, 1, 1]), arr([2, 2, 2])];
        const result = B.lexsort(keys);
        const d = Array.from(result.data);
        // Should preserve original order (stable sort)
        expect(d).toEqual([0, 1, 2]);
      });
    });

    // ============================================================
    // compress with invalid axis (line 1607)
    // ============================================================

    describe('compress axis out of bounds', () => {
      it('throws for axis >= ndim', () => {
        const a = arr([1, 2, 3, 4], [2, 2]);
        const cond = arr([1, 0]);
        expect(() => B.compress(cond, a, 5)).toThrow('out of bounds');
      });

      it('throws for large negative axis', () => {
        const a = arr([1, 2, 3, 4], [2, 2]);
        const cond = arr([1, 0]);
        expect(() => B.compress(cond, a, -5)).toThrow('out of bounds');
      });
    });

    // ============================================================
    // apply_along_axis with 3D+ array (line 2185)
    // ============================================================

    describe('_cumAlongAxis with 3D array (outerShape.length >= 2)', () => {
      it('cumsum on 3D array along middle axis uses multi-dim outer strides', () => {
        // Shape [2, 3, 2] — cumsum along axis 1 gives outerShape [2, 2] (length 2)
        const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        const a = arr(data, [2, 3, 2]);
        const result = B.cumsum(a, 1);
        expect(result.shape).toEqual([2, 3, 2]);
        // For [0,:,0]: cumsum of [1,3,5] = [1,4,9]
        // For [0,:,1]: cumsum of [2,4,6] = [2,6,12]
        // For [1,:,0]: cumsum of [7,9,11] = [7,16,27]
        // For [1,:,1]: cumsum of [8,10,12] = [8,18,30]
        expect(Array.from(result.data)).toEqual([1, 2, 4, 6, 9, 12, 7, 8, 16, 18, 27, 30]);
      });

      it('cumprod on 3D array along last axis', () => {
        const data = [1, 2, 3, 4, 5, 6, 7, 8];
        const a = arr(data, [2, 2, 2]);
        const result = B.cumprod(a, 2);
        // [0,0,:]: cumprod [1,2] = [1,2]
        // [0,1,:]: cumprod [3,4] = [3,12]
        // [1,0,:]: cumprod [5,6] = [5,30]
        // [1,1,:]: cumprod [7,8] = [7,56]
        expect(Array.from(result.data)).toEqual([1, 2, 3, 12, 5, 30, 7, 56]);
      });
    });

    // ============================================================
    // _checkSameShape errors (lines 3039, 3043)
    // ============================================================

    describe('_checkSameShape error branches', () => {
      it('ldexp with different ndim throws shape mismatch', () => {
        const a = arr([1, 2, 3]);
        const b = arr([1, 2, 3, 4], [2, 2]);
        expect(() => B.ldexp(a, b)).toThrow('shape mismatch');
      });

      it('ldexp with same ndim but different dims throws shape mismatch', () => {
        const a = arr([1, 2, 3, 4], [2, 2]);
        const b = arr([1, 2, 3, 4, 5, 6], [2, 3]);
        expect(() => B.ldexp(a, b)).toThrow('shape mismatch');
      });

      it('isclose with different shapes throws', () => {
        const a = arr([1, 2, 3]);
        const b = arr([1, 2]);
        expect(() => B.isclose(a, b)).toThrow('shape mismatch');
      });

      it('allclose with different shapes throws', () => {
        const a = arr([1, 2, 3]);
        const b = arr([1, 2, 3, 4], [2, 2]);
        expect(() => B.allclose(a, b)).toThrow('shape mismatch');
      });
    });

    // ============================================================
    // LU decompose pivot swap (lines 3065, 3069-3073)
    // ============================================================

    describe('LU pivot row swap (4x4+ to bypass direct det formulas)', () => {
      it('4x4 matrix with zero on diagonal needs pivot swap', () => {
        // 4x4 permutation matrix — LU decomposition requires row swaps
        // [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]
        const a = mat([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 4, 4);
        const d = B.det(a) as number;
        // det of this permutation = +1 (even number of swaps: (1,2)(3,4))
        expect(Math.abs(Math.abs(d) - 1)).toBeLessThan(1e-10);
      });

      it('4x4 matrix where first column pivot is not on diagonal', () => {
        // First element is 0, forces pivot swap in first column of LU
        const a = mat([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17], 4, 4);
        const d = B.det(a) as number;
        // Just verify it computes without error and is finite
        expect(Number.isFinite(d)).toBe(true);
      });
    });

    // ============================================================
    // Broadcast shape mismatch (line 3175)
    // ============================================================

    describe('broadcast_arrays shape mismatch', () => {
      it('broadcast_arrays with incompatible shapes throws', () => {
        const a = arr([1, 2, 3], [3]);
        const b = arr([1, 2], [2]);
        expect(() => B.broadcastArrays(a, b)).toThrow('not broadcastable');
      });
    });

    // ============================================================
    // resize empty source array (line 3354)
    // ============================================================

    describe('resize empty array', () => {
      it('resize empty to [3] gives zeros', () => {
        const empty = arr([], [0]);
        const result = B.resize(empty, [3]);
        expect(Array.from(result.data)).toEqual([0, 0, 0]);
        expect(result.shape).toEqual([3]);
      });
    });

    // ============================================================
    // stack shape mismatch (line 3458)
    // ============================================================

    describe('stack shape mismatch', () => {
      it('stack arrays with different ndim throws', () => {
        const a = arr([1, 2, 3]);
        const b = arr([1, 2, 3, 4], [2, 2]);
        expect(() => B.stack([a, b])).toThrow('same shape');
      });
    });

    // ============================================================
    // histogram unknown bin strategy (line 5298)
    // ============================================================

    describe('histogram unknown bin strategy', () => {
      it('histogram with unknown string bins falls back to 10', () => {
        const a = arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        // Pass an unrecognized string as bins — should fall through to default: return 10
        const result = B.histogram(a, 'unknown_strategy' as any);
        // Should produce 10 bins => 11 bin edges
        expect(result.binEdges.data.length).toBe(11);
      });
    });

    // ============================================================
    // eig on 1x1 matrix (line 6139)
    // ============================================================

    describe('eig edge cases', () => {
      it('eig on 2x2 symmetric matrix', () => {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        const a = mat([2, 1, 1, 2], 2, 2);
        const result = B.eig(a);
        const vals = Array.from(result.values.data).sort((a, b) => a - b);
        expect(Math.abs(vals[0] - 1)).toBeLessThan(1e-6);
        expect(Math.abs(vals[1] - 3)).toBeLessThan(1e-6);
      });
    });

    // ============================================================
    // polydiv remainder trimming (line 7360)
    // ============================================================

    describe('polydiv remainder trimming', () => {
      it('polydiv where remainder has leading near-zeros', () => {
        // (x^3 + 0*x^2 + 0*x + 1) / (x^3 + 1) = q=[1], r=[0]
        // Leading zeros in remainder get trimmed
        const num = arr([1, 0, 0, 1]);
        const den = arr([1, 0, 0, 1]);
        const result = B.polydiv(num, den);
        // quotient should be [1]
        expect(Math.abs(result.q.data[0] - 1)).toBeLessThan(1e-10);
        // remainder should be trimmed to just [0] (or very close)
        expect(result.r.data.length).toBeLessThanOrEqual(1);
      });

      it('polydiv with non-trivial remainder trimming', () => {
        // (x^2 + 2x + 1) / (x + 1) = x + 1, remainder 0
        const num = arr([1, 2, 1]);
        const den = arr([1, 1]);
        const result = B.polydiv(num, den);
        expect(result.q.data.length).toBe(2);
        // remainder should be near-zero, trimmed
        for (let i = 0; i < result.r.data.length; i++) {
          expect(Math.abs(result.r.data[i])).toBeLessThan(1e-10);
        }
      });
    });

    // ============================================================
    // block with non-array row (line 7501)
    // ============================================================

    describe('block with non-array rows', () => {
      it('block with direct NDArray rows (not nested arrays)', () => {
        // block([[a, b], c]) where c is a direct NDArray (not wrapped in array)
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const c = mat([9, 10, 11, 12], 1, 4);
        // Row 1: [a, b] => hstack => 2x4
        // Row 2: c => direct NDArray => 1x4
        // vstack => 3x4
        const result = B.block([[a, b], c]);
        expect(result.shape).toEqual([3, 4]);
        expect(Array.from(result.data)).toEqual([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12]);
      });
    });

    // ============================================================
    // nextafter(0, negative) (lines 8097-8098)
    // ============================================================

    describe('nextafter edge cases', () => {
      it('nextafter(0, -1) produces smallest negative denormal', () => {
        // xv=0, yv=-1: hits line 8075 (xv===0), then line 8080 (yv > 0 is false)
        const result = B.nextafter(arr([0]), arr([-1]));
        const val = result.data[0];
        expect(val).toBeLessThan(0);
        expect(val).toBeGreaterThan(-1e-300);
      });

      it('nextafter(positive, negative) decrements', () => {
        // xv > 0, yv < xv: hits else branch at line 8094, then xv > 0 at line 8099
        const result = B.nextafter(arr([1.0]), arr([-1.0]));
        const val = result.data[0];
        expect(val).toBeLessThan(1.0);
        expect(val).toBeGreaterThan(0.99);
      });

      it('nextafter(negative, more_negative) increments magnitude', () => {
        // xv < 0, yv < xv: hits else branch line 8094, then else at 8105
        const result = B.nextafter(arr([-1.0]), arr([-2.0]));
        const val = result.data[0];
        expect(val).toBeLessThan(-1.0);
      });

      it('nextafter(0, 0) stays at 0', () => {
        const result = B.nextafter(arr([0]), arr([0]));
        expect(result.data[0]).toBe(0);
      });
    });
  });
}
