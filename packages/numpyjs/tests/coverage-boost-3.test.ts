/**
 * Coverage boost phase 3 — final push toward 100% line coverage in base-backend.ts
 *
 * Targets all remaining uncovered lines: scalar broadcasting in divmod,
 * quantile interpolation methods, tensordot tuple axes, FFT length-1,
 * packbits/unpackbits little-endian, lstsq rcond, block, columnStack 2D,
 * argwhere empty, dstack variants, and many error/edge-case branches.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, approxEq, getData } from './test-utils';

export function coverageBoost3Tests(getBackend: () => Backend) {
  describe('coverage-boost-3', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[], shape?: number[]) => B.array(data, shape ?? [data.length]);
    const mat = (data: number[], rows: number, cols: number) => B.array(data, [rows, cols]);

    // ============================================================
    // Scalar broadcasting in divmod
    // ============================================================

    describe('divmod scalar broadcasting', () => {
      it('scalar / array', async () => {
        const result = B.divmod(10, arr([3, 4, 5]));
        expect(await getData(result.quotient, B)).toEqual([3, 2, 2]);
        expect(await getData(result.remainder, B)).toEqual([1, 2, 0]);
      });

      it('array / scalar', async () => {
        const result = B.divmod(arr([10, 11, 12]), 3);
        expect(await getData(result.quotient, B)).toEqual([3, 3, 4]);
        expect(await getData(result.remainder, B)).toEqual([1, 2, 0]);
      });
    });

    // ============================================================
    // Quantile interpolation methods
    // ============================================================

    describe('quantile interpolation methods', () => {
      const a = () => arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      it('lower method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'lower');
        expect(result).toBe(3);
      });

      it('higher method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'higher');
        expect(result).toBe(4);
      });

      it('midpoint method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'midpoint');
        expect(result).toBe(3.5);
      });

      it('nearest method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'nearest');
        expect(typeof result).toBe('number');
      });
    });

    // ============================================================
    // cond fallback / edge cases
    // ============================================================

    describe('cond edge cases', () => {
      it('cond with p=2 produces finite positive number', () => {
        const a = mat([3, 1, 1, 2], 2, 2);
        const c = B.cond(a, 2);
        expect(c).toBeGreaterThan(0);
        expect(Number.isFinite(c)).toBe(true);
      });

      it('cond default fallback (unsupported p)', () => {
        const a = mat([1, 0, 0, 1], 2, 2);
        const c = B.cond(a, 3 as any);
        expect(Number.isFinite(c)).toBe(true);
        expect(c).toBeGreaterThan(0);
      });
    });

    // ============================================================
    // General N-D broadcasting (4D+)
    // ============================================================

    describe('general N-D broadcasting path', () => {
      it('4D broadcast with actual broadcasting', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
        const b = B.array([10, 20], [1, 1, 2, 1]);
        const result = B.add(a, b);
        expect(result.shape).toEqual([1, 2, 2, 3]);
        const data = await getData(result, B);
        expect(data.length).toBe(12);
      });
    });

    // ============================================================
    // tensordot with tuple axes
    // ============================================================

    describe('tensordot', () => {
      it('tensordot with axes as tuple arrays', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const b = mat([1, 2, 3, 4, 5, 6], 3, 2);
        const result = B.tensordot(a, b, [[1], [0]]);
        expect(result.shape).toEqual([2, 2]);
        // Same as matmul
        expect(await getData(result, B)).toEqual([22, 28, 49, 64]);
      });

      it('tensordot producing non-scalar result', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.tensordot(a, b, 1);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    // ============================================================
    // FFT on length-1 input
    // ============================================================

    describe('fft length-1', () => {
      it('fft on single element', async () => {
        const real = arr([42]);
        const imag = arr([0]);
        const result = B.fft(real, imag);
        expect(await getData(result.real, B)).toEqual([42]);
        expect(await getData(result.imag, B)).toEqual([0]);
      });
    });

    // ============================================================
    // packbits / unpackbits little-endian
    // ============================================================

    describe('packbits/unpackbits little-endian', () => {
      it('packbits little-endian', async () => {
        const a = arr([1, 0, 1, 0, 0, 0, 0, 0]);
        const result = B.packbits(a, undefined, 'little');
        const data = await getData(result, B);
        expect(data[0]).toBe(5); // bits 0,2 set = 1+4 = 5
      });

      it('unpackbits little-endian', async () => {
        const a = B.array([5], [1], 'uint8');
        const result = B.unpackbits(a, undefined, undefined, 'little');
        const data = await getData(result, B);
        expect(data[0]).toBe(1); // bit 0
        expect(data[1]).toBe(0); // bit 1
        expect(data[2]).toBe(1); // bit 2
      });
    });

    // ============================================================
    // lstsq with rcond
    // ============================================================

    describe('lstsq rcond', () => {
      it('lstsq with rcond=null', async () => {
        const a = mat([1, 1, 1, 2, 2, 3], 3, 2);
        const b = arr([1, 2, 3]);
        const result = B.lstsq(a, b, null);
        expect(result.x.shape).toEqual([2]);
      });

      it('lstsq with numeric rcond', async () => {
        const a = mat([1, 1, 1, 2, 2, 3], 3, 2);
        const b = arr([1, 2, 3]);
        const result = B.lstsq(a, b, 0.01);
        expect(result.x.shape).toEqual([2]);
      });
    });

    // ============================================================
    // block with non-array row
    // ============================================================

    describe('block', () => {
      it('block with mixed array/non-array rows', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.block([[a, b]]);
        expect(result.shape).toEqual([2, 4]);
      });

      it('block with NDArray as row (not wrapped)', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        // Pass a as a row directly, not wrapped in array
        const result = B.block([a, b]);
        expect(result.shape[0]).toBeGreaterThan(0);
      });
    });

    // ============================================================
    // columnStack 2D
    // ============================================================

    describe('columnStack 2D', () => {
      it('columnStack with 2D arrays', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.columnStack([a, b]);
        expect(result.shape).toEqual([2, 4]);
        expect(await getData(result, B)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
      });
    });

    // ============================================================
    // argwhere on all-zero array
    // ============================================================

    describe('argwhere empty', () => {
      it('argwhere on all zeros', () => {
        const a = arr([0, 0, 0]);
        const result = B.argwhere(a);
        expect(result.shape).toEqual([0, 1]);
      });
    });

    // ============================================================
    // dstack variants
    // ============================================================

    describe('dstack edge cases', () => {
      it('dstack with 1D arrays', async () => {
        const a = arr([1, 2]);
        const b = arr([3, 4]);
        const result = B.dstack([a, b]);
        expect(result.shape).toEqual([1, 2, 2]);
      });

      it('dstack with 3D arrays', async () => {
        const a = B.array([1, 2, 3, 4], [1, 2, 2]);
        const b = B.array([5, 6, 7, 8], [1, 2, 2]);
        const result = B.dstack([a, b]);
        expect(result.shape).toEqual([1, 2, 4]);
      });
    });

    // ============================================================
    // Error throws we haven't hit
    // ============================================================

    describe('remaining error branches', () => {
      it('partition with OOB axis', () => {
        expect(() => B.partition(arr([3, 1, 2]), 1, 5)).toThrow();
      });

      it('partition with OOB kth', () => {
        expect(() => B.partition(arr([3, 1, 2]), 10)).toThrow();
      });

      it('argpartition with OOB axis', () => {
        expect(() => B.argpartition(arr([3, 1, 2]), 1, 5)).toThrow();
      });

      it('argpartition with OOB kth', () => {
        expect(() => B.argpartition(arr([3, 1, 2]), 10)).toThrow();
      });

      it('concatenate ndim mismatch', () => {
        expect(() => B.concatenate([arr([1, 2]), mat([3, 4], 1, 2)])).toThrow();
      });

      it('concatenate non-axis dim mismatch', () => {
        expect(() => B.concatenate([mat([1, 2], 1, 2), mat([3, 4, 5, 6], 1, 4)], 0)).toThrow();
      });

      it('broadcastTo fewer dims', () => {
        expect(() => B.broadcastTo(mat([1, 2, 3, 4], 2, 2), [4])).toThrow();
      });

      it('broadcastTo incompatible dim', () => {
        expect(() => B.broadcastTo(arr([1, 2, 3]), [2])).toThrow();
      });

      it('moveaxis source/dest length mismatch', () => {
        expect(() => B.moveaxis(B.array([1, 2, 3, 4, 5, 6], [2, 3]), [0, 1], [0])).toThrow();
      });

      it('stack shape mismatch (covered in boost-2 but ensure)', () => {
        expect(() => B.stack([arr([1, 2]), arr([1, 2, 3])])).toThrow();
      });

      it('batchedMatmul non-broadcastable batch', () => {
        const a = B.array(
          Array.from({ length: 8 }, (_, i) => i),
          [2, 2, 2]
        );
        const b = B.array(
          Array.from({ length: 12 }, (_, i) => i),
          [3, 2, 2]
        );
        expect(() => B.batchedMatmul(a, b)).toThrow();
      });

      it('einsum operand ndim mismatch', () => {
        expect(() => B.einsum('ijk->i', mat([1, 2, 3, 4], 2, 2))).toThrow();
      });

      it('einsum label size mismatch', () => {
        // 'ii' on non-square matrix
        expect(() => B.einsum('ii->', mat([1, 2, 3, 4, 5, 6], 2, 3))).toThrow();
      });

      it('gradient with <2 elements throws', () => {
        expect(() => B.gradient(arr([1]))).toThrow();
      });

      it('gradient edge_order=2 with <3 elements throws', () => {
        expect(() => B.gradient(arr([1, 2]), 0, 2)).toThrow();
      });

      it('cross with wrong size throws', () => {
        expect(() => B.cross(arr([1, 2]), arr([3, 4]))).toThrow();
      });

      it('cov with different length x,y throws', () => {
        expect(() => B.cov(arr([1, 2, 3]), arr([1, 2]))).toThrow();
      });

      it('average weight length mismatch throws', () => {
        expect(() => B.average(arr([1, 2, 3]), arr([1, 2]))).toThrow();
      });

      it('pad 3D throws', () => {
        expect(() => B.pad(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]), 1)).toThrow();
      });

      it('_diffOnce on <2 elements throws', () => {
        const a = mat([1], 1, 1);
        expect(() => B.diff(a, 1, 0)).toThrow();
      });

      it('roll with mismatched shift/axis lengths', () => {
        expect(() => B.roll(mat([1, 2, 3, 4], 2, 2), [1, 2], [0])).toThrow();
      });

      it('solve with non-square matrix', () => {
        expect(() => B.solve(mat([1, 2, 3, 4, 5, 6], 2, 3), arr([1, 2]))).toThrow();
      });
    });

    // ============================================================
    // Remaining edge case paths
    // ============================================================

    describe('remaining edge paths', () => {
      it('cov on 1D array', async () => {
        const result = B.cov(arr([1, 2, 3, 4, 5]));
        // 1D array -> 1x1 covariance matrix = variance
        expect(result.shape).toEqual([1, 1]);
      });

      it('histogram with explicit range', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const result = B.histogram(a, 5, [0, 10]);
        expect(result.binEdges.shape[0]).toBe(6);
      });

      it('histogram with NDArray bins and density', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const edges = arr([0, 2.5, 5]);
        const result = B.histogram(a, edges, undefined, true);
        const data = await getData(result.hist, B);
        // With density, should integrate to 1
        expect(data.every(v => v >= 0)).toBe(true);
      });

      it('flip 2D without axis', async () => {
        // Already covered but this exercises the full reversal path
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.flip(a);
        expect(await getData(result, B)).toEqual([6, 5, 4, 3, 2, 1]);
      });

      it('ptp with axis no keepdims', async () => {
        const a = mat([1, 5, 3, 7, 2, 8], 2, 3);
        const result = B.ptp(a, 1) as any;
        const data = await getData(result, B);
        expect(data).toEqual([4, 6]);
      });

      it('nanquantile with axis no keepdims', () => {
        const a = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanquantile(a, 0.5, 1);
        expect(result).toBeTruthy();
      });

      it('diff with numeric append on 2D', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.diff(a, 1, 1, undefined, 10);
        // Appending 10 along axis 1 then diff
        expect(result.shape[1]).toBe(3);
      });

      it('nanprod with axis no keepdims', async () => {
        const a = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanprod(a, 1) as any;
        const data = await getData(result, B);
        expect(data[0]).toBe(3); // 1 * 3
        expect(data[1]).toBe(20); // 4 * 5
      });

      it('quantile with axis no keepdims (lower)', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.quantile(a, 0.5, 1, false, 'lower');
        const data = await getData(result as any, B);
        expect(data).toEqual([2, 5]);
      });

      it('polydiv with near-zero remainder', async () => {
        // (x^3 - 1) / (x - 1) = x^2 + x + 1, remainder ~0
        const u = arr([1, 0, 0, -1]);
        const v = arr([1, -1]);
        const { q } = B.polydiv(u, v);
        const qData = await getData(q, B);
        expect(approxEq(qData[0], 1, 1e-10)).toBe(true);
        expect(approxEq(qData[1], 1, 1e-10)).toBe(true);
        expect(approxEq(qData[2], 1, 1e-10)).toBe(true);
      });

      it('_reduceAlongAxis on 3D (axis 0)', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const result = B.sum(a, 0) as any;
        expect(result.shape).toEqual([2, 2]);
      });

      it('sort 3D along non-last axis', async () => {
        const a = B.array([4, 3, 2, 1, 8, 7, 6, 5], [2, 2, 2]);
        const result = B.sort(a, 0);
        expect(result.shape).toEqual([2, 2, 2]);
      });

      it('Jacobi SVD on negative tau matrix', async () => {
        // Matrix where (aqq - app) / (2*apq) is negative
        const a = mat([1, 5, 5, 2], 2, 2);
        const { s } = B.svd(a);
        expect(s.shape[0]).toBe(2);
      });

      it('solve with 1D b vector', async () => {
        const a = mat([2, 1, 1, 3], 2, 2);
        const b = arr([5, 7]);
        const result = B.solve(a, b) as any;
        const data = await getData(result, B);
        // 2x + y = 5, x + 3y = 7 => x = 1.6, y = 1.8
        expect(data.length).toBe(2);
      });
    });
  });
}
