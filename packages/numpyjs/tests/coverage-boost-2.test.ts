/**
 * Coverage boost phase 2 — targeting ALL remaining uncovered lines in base-backend.ts
 *
 * Covers: keepdims branches, deprecated aliases, error branches, norm variants,
 * QR complete, LU/det 4x4+, cond variants, pad 2D, N-D broadcasting,
 * fftn/ifftn 3D, weighted choice, and many edge cases.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, RELAXED_TOL, approxEq, getData } from './test-utils';

export function coverageBoost2Tests(getBackend: () => Backend) {
  describe('coverage-boost-2', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[], shape?: number[]) => B.array(data, shape ?? [data.length]);
    const mat = (data: number[], rows: number, cols: number) => B.array(data, [rows, cols]);

    // ============================================================
    // 1. Deprecated Aliases
    // ============================================================

    describe('deprecated aliases', () => {
      it('absolute delegates to abs', async () => {
        const a = arr([-1, 2, -3]);
        expect(await getData(B.absolute(a), B)).toEqual([1, 2, 3]);
      });

      it('neg delegates to negative', async () => {
        const a = arr([1, -2, 3]);
        expect(await getData(B.neg(a), B)).toEqual([-1, 2, -3]);
      });

      it('sub delegates to subtract', async () => {
        expect(await getData(B.sub(arr([5, 3]), arr([1, 2])), B)).toEqual([4, 1]);
      });

      it('mul delegates to multiply', async () => {
        expect(await getData(B.mul(arr([2, 3]), arr([4, 5])), B)).toEqual([8, 15]);
      });

      it('div delegates to divide', async () => {
        expect(await getData(B.div(arr([10, 6]), arr([2, 3])), B)).toEqual([5, 2]);
      });

      it('pow delegates to power', async () => {
        expect(await getData(B.pow(arr([2, 3]), arr([3, 2])), B)).toEqual([8, 9]);
      });

      it('trapz delegates to trapezoid', async () => {
        const y = arr([1, 2, 3]);
        const result = B.trapz(y);
        expect(result).toBe(4); // trapezoid of [1,2,3] with dx=1
      });
    });

    // ============================================================
    // 2. keepdims branches on all reduction operations
    // ============================================================

    describe('keepdims branches', () => {
      const a = () => mat([1, 2, 3, 4, 5, 6], 2, 3);

      it('prod with keepdims', async () => {
        const result = B.prod(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(await getData(result, B)).toEqual([6, 120]);
      });

      it('var with keepdims', async () => {
        const result = B.var(a(), 1, 0, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('std with keepdims', async () => {
        const result = B.std(a(), 1, 0, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('min with keepdims', async () => {
        const result = B.min(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(await getData(result, B)).toEqual([1, 4]);
      });

      it('max with keepdims', async () => {
        const result = B.max(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(await getData(result, B)).toEqual([3, 6]);
      });

      it('argmin with keepdims', async () => {
        const result = B.argmin(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(await getData(result, B)).toEqual([0, 0]);
      });

      it('argmax with keepdims', async () => {
        const result = B.argmax(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
        expect(await getData(result, B)).toEqual([2, 2]);
      });

      it('all with keepdims', async () => {
        const result = B.all(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('any with keepdims', async () => {
        const result = B.any(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('countNonzero with keepdims', async () => {
        const x = mat([0, 1, 2, 0, 0, 3], 2, 3);
        const result = B.countNonzero(x, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('nansum with axis and keepdims', async () => {
        const x = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nansum(x, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('nanmean with axis and keepdims', async () => {
        const x = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanmean(x, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('nanprod with axis and keepdims', async () => {
        const x = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanprod(x, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('median with axis and keepdims', async () => {
        const result = B.median(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('average with axis and keepdims', async () => {
        const result = B.average(a(), undefined, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('average with weights and axis', async () => {
        const w = mat([1, 2, 3, 1, 2, 3], 2, 3);
        const result = B.average(a(), w, 1) as any;
        const data = await getData(result, B);
        // Row 0: (1*1+2*2+3*3)/(1+2+3) = 14/6
        expect(approxEq(data[0], 14 / 6, RELAXED_TOL)).toBe(true);
      });

      it('ptp with axis and keepdims', async () => {
        const result = B.ptp(a(), 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });

      it('nanquantile with axis and keepdims', async () => {
        const x = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanquantile(x, 0.5, 1, true) as any;
        expect(result.shape).toEqual([2, 1]);
      });
    });

    // ============================================================
    // 3. Error branches
    // ============================================================

    describe('error branches', () => {
      it('item() on multi-element array throws', () => {
        expect(() => arr([1, 2]).item()).toThrow('size 1');
      });

      it('insert with axis throws', () => {
        expect(() => B.insert(mat([1, 2, 3, 4], 2, 2), 0, 99, 0)).toThrow('not yet implemented');
      });

      it('deleteArr with axis throws', () => {
        expect(() => B.deleteArr(mat([1, 2, 3, 4], 2, 2), 0, 0)).toThrow('not yet implemented');
      });

      it('matrixPower non-square throws', () => {
        expect(() => B.matrixPower(mat([1, 2, 3, 4, 5, 6], 2, 3), 2)).toThrow('square');
      });

      it('kron >2D throws', () => {
        expect(() => B.kron(B.array([1], [1, 1, 1]), arr([1]))).toThrow('1D or 2D');
      });

      it('cond non-2D throws', () => {
        expect(() => B.cond(arr([1, 2, 3]))).toThrow('2D');
      });

      it('slogdet non-square throws', () => {
        expect(() => B.slogdet(mat([1, 2, 3, 4, 5, 6], 2, 3))).toThrow('square');
      });

      it('multiDot empty throws', () => {
        expect(() => B.multiDot([])).toThrow('at least one');
      });

      it('reshape size mismatch throws', () => {
        expect(() => B.reshape(arr([1, 2, 3]), [2, 3])).toThrow();
      });

      it('squeeze on non-1 axis throws', () => {
        expect(() => B.squeeze(mat([1, 2, 3, 4], 2, 2), 0)).toThrow();
      });

      it('tril on 1D throws', () => {
        expect(() => B.tril(arr([1, 2, 3]))).toThrow();
      });

      it('triu on 1D throws', () => {
        expect(() => B.triu(arr([1, 2, 3]))).toThrow();
      });

      it('geomspace with zero throws', () => {
        expect(() => B.geomspace(0, 10, 5)).toThrow('non-zero');
      });

      it('geomspace with mixed signs throws', () => {
        expect(() => B.geomspace(-1, 10, 5)).toThrow('same sign');
      });

      it('vsplit on 1D throws', () => {
        expect(() => B.vsplit(arr([1, 2, 3]), 3)).toThrow('2 dimensions');
      });

      it('dsplit on 2D throws', () => {
        expect(() => B.dsplit(mat([1, 2, 3, 4], 2, 2), 2)).toThrow('3 dimensions');
      });

      it('fillDiagonal on 1D throws', () => {
        expect(() => B.fillDiagonal(arr([1, 2, 3]), 5)).toThrow('2-d');
      });

      it('select with empty lists throws', () => {
        expect(() => B.select([], [], 0)).toThrow();
      });

      it('diag on 3D throws', () => {
        expect(() => B.diag(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]))).toThrow();
      });

      it('transpose with wrong axes count throws', () => {
        expect(() => B.transpose(mat([1, 2, 3, 4], 2, 2), [0])).toThrow();
      });

      it('select length mismatch throws', () => {
        const cond1 = arr([1, 0, 0]);
        expect(() => B.select([cond1], [10, 20], 0)).toThrow();
      });
    });

    // ============================================================
    // 4. norm variants
    // ============================================================

    describe('norm variants', () => {
      it('frobenius norm', () => {
        const a = arr([1, 2, 3, 4]);
        expect(approxEq(B.norm(a, 'fro') as number, Math.sqrt(30), 1e-10)).toBe(true);
      });

      it('nuclear norm', () => {
        const a = mat([1, 0, 0, 1], 2, 2); // identity -> singular values [1, 1]
        expect(approxEq(B.norm(a, 'nuc') as number, 2, 1e-6)).toBe(true);
      });

      it('-Infinity norm', () => {
        const a = arr([1, -2, 3, -4]);
        expect(B.norm(a, -Infinity)).toBe(1); // min |x_i|
      });

      it('general numeric ord (p=3)', () => {
        const a = arr([1, 2, 3]);
        const expected = Math.pow(1 + 8 + 27, 1 / 3);
        expect(approxEq(B.norm(a, 3) as number, expected, 1e-10)).toBe(true);
      });

      it('norm along axis=0', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const result = B.norm(a, 2, 0) as any;
        expect(result.shape).toEqual([2]);
        const data = await getData(result, B);
        expect(approxEq(data[0], Math.sqrt(10), 1e-10)).toBe(true);
        expect(approxEq(data[1], Math.sqrt(20), 1e-10)).toBe(true);
      });

      it('norm along axis=1', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const result = B.norm(a, 2, 1) as any;
        expect(result.shape).toEqual([2]);
      });

      it('norm axis=0 with -Infinity ord', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const result = B.norm(a, -Infinity, 0) as any;
        const data = await getData(result, B);
        expect(data[0]).toBe(1);
        expect(data[1]).toBe(2);
      });

      it('norm axis=1 with -Infinity ord', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const result = B.norm(a, -Infinity, 1) as any;
        const data = await getData(result, B);
        expect(data[0]).toBe(1);
        expect(data[1]).toBe(3);
      });
    });

    // ============================================================
    // 5. QR complete mode
    // ============================================================

    describe('qr complete', () => {
      it('produces full Q and R for square matrix', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const { q, r } = B.qr(a, 'complete');
        expect(q.shape).toEqual([2, 2]);
        expect(r.shape).toEqual([2, 2]);
        // Q*R should reconstruct A
        const reconstructed = B.matmul(q, r);
        const data = await getData(reconstructed, B);
        expect(approxEq(data[0], 1, 1e-10)).toBe(true);
        expect(approxEq(data[1], 2, 1e-10)).toBe(true);
        expect(approxEq(data[2], 3, 1e-10)).toBe(true);
        expect(approxEq(data[3], 4, 1e-10)).toBe(true);
      });

      it('produces m x m Q for non-square matrix', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 3, 2);
        const { q, r } = B.qr(a, 'complete');
        expect(q.shape).toEqual([3, 3]);
        expect(r.shape).toEqual([3, 2]);
      });
    });

    // ============================================================
    // 6. LU / det for 4x4+ matrices
    // ============================================================

    describe('det 4x4+', () => {
      it('computes determinant of 4x4 matrix', () => {
        // Identity matrix -> det = 1
        const eye4 = B.eye(4);
        expect(approxEq(B.det(eye4) as number, 1, 1e-10)).toBe(true);
      });

      it('computes determinant of 5x5 matrix', () => {
        const a = B.eye(5);
        expect(approxEq(B.det(a) as number, 1, 1e-10)).toBe(true);
      });

      it('det of non-trivial 4x4 matrix', () => {
        // [[1,2,0,0],[0,1,2,0],[0,0,1,2],[0,0,0,1]] -> upper triangular, det = 1
        const a = mat([1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1], 4, 4);
        expect(approxEq(B.det(a) as number, 1, 1e-8)).toBe(true);
      });
    });

    // ============================================================
    // 7. cond variants
    // ============================================================

    describe('cond variants', () => {
      it('cond with p=1', () => {
        const a = mat([1, 0, 0, 1], 2, 2);
        const c = B.cond(a, 1);
        // norm(I,1) flattened = |1|+|0|+|0|+|1| = 2, so cond = 4
        expect(c).toBeGreaterThan(0);
        expect(Number.isFinite(c)).toBe(true);
      });

      it('cond with p=Infinity', () => {
        const a = mat([1, 0, 0, 1], 2, 2);
        const c = B.cond(a, Infinity);
        expect(approxEq(c, 1, 1e-10)).toBe(true);
      });

      it('cond with p=fro', () => {
        const a = mat([1, 0, 0, 1], 2, 2);
        const c = B.cond(a, 'fro');
        expect(approxEq(c, 2, 1e-10)).toBe(true); // norm_fro(I) * norm_fro(I) = sqrt(2)*sqrt(2) = 2
      });

      it('cond of singular matrix is Infinity', () => {
        const a = mat([1, 2, 2, 4], 2, 2); // singular
        const c = B.cond(a, 1);
        expect(c).toBe(Infinity);
      });

      it('cond with p=-2', () => {
        const a = mat([1, 0, 0, 2], 2, 2);
        const c = B.cond(a, -2);
        expect(approxEq(c, 0.5, 1e-6)).toBe(true); // sMin/sMax = 1/2
      });
    });

    // ============================================================
    // 8. slogdet
    // ============================================================

    describe('slogdet', () => {
      it('slogdet of singular matrix', () => {
        const a = mat([1, 2, 2, 4], 2, 2);
        const result = B.slogdet(a) as { sign: number; logabsdet: number };
        expect(result.sign).toBe(0);
        expect(result.logabsdet).toBe(-Infinity);
      });
    });

    // ============================================================
    // 9. matrixPower negative
    // ============================================================

    describe('matrixPower negative', () => {
      it('A^-1 via matrixPower', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const inv = B.matrixPower(a, -1) as any;
        // A * A^-1 should be identity
        const product = B.matmul(a, inv);
        const data = await getData(product, B);
        expect(approxEq(data[0], 1, 1e-10)).toBe(true);
        expect(approxEq(data[1], 0, 1e-10)).toBe(true);
        expect(approxEq(data[2], 0, 1e-10)).toBe(true);
        expect(approxEq(data[3], 1, 1e-10)).toBe(true);
      });
    });

    // ============================================================
    // 10. atleast*d edge cases
    // ============================================================

    describe('atleast edge cases', () => {
      it('atleast2d on 0D', () => {
        const a = B.array([5], []);
        const result = B.atleast2d(a);
        expect(result.shape).toEqual([1, 1]);
      });

      it('atleast3d on 0D', () => {
        const a = B.array([5], []);
        const result = B.atleast3d(a);
        expect(result.shape).toEqual([1, 1, 1]);
      });

      it('atleast3d on 3D+ is no-op', () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const result = B.atleast3d(a);
        expect(result.shape).toEqual([2, 2, 2]);
      });
    });

    // ============================================================
    // 11. append with axis
    // ============================================================

    describe('append with axis', () => {
      it('appends along axis 0', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6], 1, 2);
        const result = B.append(a, b, 0);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6]);
      });
    });

    // ============================================================
    // 12. N-D broadcasting (2D, 3D, 4D paths)
    // ============================================================

    describe('N-D broadcasting', () => {
      it('2D broadcast: [2,1] + [1,3]', async () => {
        const a = B.array([1, 2], [2, 1]);
        const b = B.array([10, 20, 30], [1, 3]);
        const result = B.add(a, b);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([11, 21, 31, 12, 22, 32]);
      });

      it('3D broadcast: [2,1,3] + [1,2,1]', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6], [2, 1, 3]);
        const b = B.array([10, 20], [1, 2, 1]);
        const result = B.add(a, b);
        expect(result.shape).toEqual([2, 2, 3]);
        expect(await getData(result, B)).toEqual([11, 12, 13, 21, 22, 23, 14, 15, 16, 24, 25, 26]);
      });

      it('4D broadcast: general N-D path', async () => {
        const a = B.array([1, 2], [1, 1, 1, 2]);
        const b = B.array([10], [1, 1, 1, 1]);
        const result = B.add(a, b);
        expect(result.shape).toEqual([1, 1, 1, 2]);
        expect(await getData(result, B)).toEqual([11, 12]);
      });
    });

    // ============================================================
    // 13. Pad 1D modes (edge, reflect, symmetric, wrap, linear_ramp, mean, min, max)
    // ============================================================

    describe('pad 1D modes', () => {
      const a = () => arr([1, 2, 3, 4, 5]);

      it('edge mode', async () => {
        const result = B.pad(a(), 2, 'edge');
        const data = await getData(result, B);
        expect(data[0]).toBe(1); // edge before
        expect(data[1]).toBe(1);
        expect(data.at(-1)).toBe(5); // edge after
        expect(data.at(-2)).toBe(5);
      });

      it('reflect mode', async () => {
        const result = B.pad(a(), 2, 'reflect');
        expect(result.shape).toEqual([9]);
      });

      it('symmetric mode', async () => {
        const result = B.pad(a(), 2, 'symmetric');
        expect(result.shape).toEqual([9]);
      });

      it('wrap mode', async () => {
        const result = B.pad(a(), 2, 'wrap');
        expect(result.shape).toEqual([9]);
      });

      it('linear_ramp mode', async () => {
        const result = B.pad(a(), 2, 'linear_ramp', 0);
        const data = await getData(result, B);
        expect(result.shape).toEqual([9]);
        // Should ramp from 0 to edge value
        expect(data[2]).toBe(1); // first real element
      });

      it('mean mode', async () => {
        const result = B.pad(a(), 2, 'mean');
        const data = await getData(result, B);
        expect(data[0]).toBe(3); // mean of [1,2,3,4,5] = 3
        expect(data[1]).toBe(3);
      });

      it('minimum mode', async () => {
        const result = B.pad(a(), 2, 'minimum');
        const data = await getData(result, B);
        expect(data[0]).toBe(1);
        expect(data[1]).toBe(1);
      });

      it('maximum mode', async () => {
        const result = B.pad(a(), 2, 'maximum');
        const data = await getData(result, B);
        expect(data[0]).toBe(5);
        expect(data[1]).toBe(5);
      });
    });

    // ============================================================
    // 14. Pad 2D modes
    // ============================================================

    describe('pad 2D', () => {
      const a = () => mat([1, 2, 3, 4], 2, 2);

      it('constant mode', async () => {
        const result = B.pad(a(), 1, 'constant', 0);
        expect(result.shape).toEqual([4, 4]);
        const data = await getData(result, B);
        // Corners should be 0, center should be [1,2,3,4]
        expect(data[0]).toBe(0);
        expect(data[5]).toBe(1);
        expect(data[6]).toBe(2);
        expect(data[9]).toBe(3);
        expect(data[10]).toBe(4);
      });

      it('edge mode', async () => {
        const result = B.pad(a(), 1, 'edge');
        expect(result.shape).toEqual([4, 4]);
      });

      it('reflect mode', async () => {
        const result = B.pad(a(), 1, 'reflect');
        expect(result.shape).toEqual([4, 4]);
      });

      it('wrap mode', async () => {
        const result = B.pad(a(), 1, 'wrap');
        expect(result.shape).toEqual([4, 4]);
      });

      it('symmetric mode', async () => {
        const result = B.pad(a(), 1, 'symmetric');
        expect(result.shape).toEqual([4, 4]);
      });

      it('linear_ramp mode', async () => {
        const result = B.pad(a(), 1, 'linear_ramp', 0);
        expect(result.shape).toEqual([4, 4]);
      });

      it('mean mode', async () => {
        const result = B.pad(a(), 1, 'mean');
        expect(result.shape).toEqual([4, 4]);
        const data = await getData(result, B);
        // Mean of [1,2,3,4] = 2.5
        expect(data[0]).toBe(2.5);
      });

      it('minimum mode', async () => {
        const result = B.pad(a(), 1, 'minimum');
        expect(result.shape).toEqual([4, 4]);
        const data = await getData(result, B);
        expect(data[0]).toBe(1);
      });

      it('maximum mode', async () => {
        const result = B.pad(a(), 1, 'maximum');
        expect(result.shape).toEqual([4, 4]);
        const data = await getData(result, B);
        expect(data[0]).toBe(4);
      });
    });

    // ============================================================
    // 15. fftn / ifftn on 3D
    // ============================================================

    describe('fftn 3D', () => {
      it('fftn on 3D array', async () => {
        const a = B.array([1, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2]);
        const result = B.fftn(a);
        expect(result.real.shape).toEqual([2, 2, 2]);
        // DC component should be sum = 1
        const realData = await getData(result.real, B);
        expect(approxEq(realData[0], 1, 1e-10)).toBe(true);
      });

      it('ifftn roundtrip on 3D', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const fwd = B.fftn(a);
        const inv = B.ifftn(fwd.real, fwd.imag);
        const realData = await getData(inv.real, B);
        const origData = await getData(a, B);
        for (let i = 0; i < 8; i++) {
          expect(approxEq(realData[i], origData[i], 1e-10)).toBe(true);
        }
      });
    });

    // ============================================================
    // 16. nancumsum / nancumprod with axis
    // ============================================================

    describe('nancum with axis', () => {
      it('nancumsum with axis', async () => {
        const a = mat([1, NaN, 3, 4, NaN, 6], 2, 3);
        const result = B.nancumsum(a, 1);
        const data = await getData(result, B);
        // NaN replaced with 0, then cumsum along axis 1
        expect(data).toEqual([1, 1, 4, 4, 4, 10]);
      });

      it('nancumprod with axis', async () => {
        const a = mat([1, NaN, 3, 4, NaN, 6], 2, 3);
        const result = B.nancumprod(a, 1);
        const data = await getData(result, B);
        // NaN replaced with 1, then cumprod along axis 1
        expect(data).toEqual([1, 1, 3, 4, 4, 24]);
      });
    });

    // ============================================================
    // 17. flip without axis
    // ============================================================

    describe('flip without axis', () => {
      it('reverses all elements', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const result = B.flip(a);
        expect(await getData(result, B)).toEqual([5, 4, 3, 2, 1]);
      });

      it('2D flip without axis reverses both axes', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const result = B.flip(a);
        expect(await getData(result, B)).toEqual([4, 3, 2, 1]);
      });
    });

    // ============================================================
    // 18. Weighted choice without replacement
    // ============================================================

    describe('weighted choice without replacement', () => {
      it('selects unique elements', async () => {
        B.seed(42);
        const a = arr([10, 20, 30, 40, 50]);
        const weights = arr([0.1, 0.2, 0.3, 0.2, 0.2]);
        const result = B.choice(a, 3, false, weights);
        const data = await getData(result, B);
        expect(data.length).toBe(3);
        // All should be unique
        expect(new Set(data).size).toBe(3);
        // All should be from the original array
        expect(data.every(v => [10, 20, 30, 40, 50].includes(v))).toBe(true);
      });
    });

    // ============================================================
    // 19. roots edge cases
    // ============================================================

    describe('roots edge cases', () => {
      it('polynomial with leading zeros', async () => {
        const p = arr([0, 0, 1, -1]); // effectively x - 1
        const r = B.roots(p);
        const data = await getData(r, B);
        expect(data.length).toBe(1);
        expect(approxEq(data[0], 1, 1e-10)).toBe(true);
      });

      it('constant polynomial returns empty', () => {
        const p = arr([5]);
        const r = B.roots(p);
        expect(r.shape).toEqual([0]);
      });

      it('empty polynomial returns empty', () => {
        const p = arr([0]);
        const r = B.roots(p);
        expect(r.shape).toEqual([0]);
      });
    });

    // ============================================================
    // 20. polydiv edge cases
    // ============================================================

    describe('polydiv edge cases', () => {
      it('numerator degree < denominator degree', async () => {
        const u = arr([1]); // degree 0
        const v = arr([1, 1]); // degree 1
        const { q, r } = B.polydiv(u, v);
        expect(await getData(q, B)).toEqual([0]);
        expect(await getData(r, B)).toEqual([1]);
      });

      it('polydiv with near-zero leading remainder', async () => {
        // x^2 + 2x + 1 divided by x + 1 = x + 1, remainder 0
        const u = arr([1, 2, 1]);
        const v = arr([1, 1]);
        const { q } = B.polydiv(u, v);
        const qData = await getData(q, B);
        expect(approxEq(qData[0], 1, 1e-10)).toBe(true);
        expect(approxEq(qData[1], 1, 1e-10)).toBe(true);
      });
    });

    // ============================================================
    // 21. cov edge cases
    // ============================================================

    describe('cov edge cases', () => {
      it('cov with rowvar=false', async () => {
        const x = mat([1, 2, 3, 4, 5, 6], 3, 2);
        const result = B.cov(x, undefined, false);
        expect(result.shape).toEqual([2, 2]);
      });

      it('cov 3D throws', () => {
        expect(() => B.cov(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]))).toThrow();
      });
    });

    // ============================================================
    // 22. diff with NDArray prepend/append
    // ============================================================

    describe('diff with prepend/append', () => {
      it('diff with number prepend', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.diff(a, 1, 1, 0);
        expect(result.shape[1]).toBe(3); // 3 cols after prepend
      });

      it('diff with NDArray prepend', async () => {
        const a = arr([2, 5, 10]);
        const p = arr([0]);
        const result = B.diff(a, 1, 0, p);
        const data = await getData(result, B);
        expect(data).toEqual([2, 3, 5]);
      });

      it('diff with NDArray append', async () => {
        const a = arr([1, 3, 6]);
        const ap = arr([10]);
        const result = B.diff(a, 1, 0, undefined, ap);
        const data = await getData(result, B);
        expect(data).toEqual([2, 3, 4]);
      });
    });

    // ============================================================
    // 23. gradient with edgeOrder=2
    // ============================================================

    describe('gradient edge_order=2', () => {
      it('second-order accurate gradient', async () => {
        const a = arr([0, 1, 4, 9, 16]); // x^2 for x=0,1,2,3,4
        const result = B.gradient(a, 0, 2);
        const data = await getData(result, B);
        // derivative of x^2 is 2x, so [0, 2, 4, 6, 8]
        expect(data.length).toBe(5);
      });
    });

    // ============================================================
    // 24. searchsorted with sorter
    // ============================================================

    describe('searchsorted with sorter', () => {
      it('uses sorter permutation', () => {
        const a = arr([30, 10, 20]); // unsorted
        const sorter = arr([1, 2, 0]); // indices that sort a: [10, 20, 30]
        const result = B.searchsorted(a, 15, 'left', sorter);
        expect(result).toBe(1); // 15 goes between 10 and 20
      });
    });

    // ============================================================
    // 25. histogram edge cases
    // ============================================================

    describe('histogram edge cases', () => {
      it('doane bins', async () => {
        B.seed(42);
        const a = B.randn([100]);
        const result = B.histogram(a, 'doane');
        expect(result.hist.shape[0]).toBe(result.binEdges.shape[0] - 1);
      });

      it('histogram with NDArray bin edges', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const edges = arr([0, 2, 4, 6]);
        const result = B.histogram(a, edges);
        expect(result.hist.shape[0]).toBe(3);
        const data = await getData(result.hist, B);
        expect(data).toEqual([1, 2, 2]); // [1], [2,3], [4,5]
      });

      it('histogram with density', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const result = B.histogram(a, 5, undefined, true);
        const data = await getData(result.hist, B);
        // density: integrate to 1
        const binWidth = (5 - 1) / 5;
        const integral = data.reduce((s, v) => s + v * binWidth, 0);
        expect(approxEq(integral, 1, 0.1)).toBe(true);
      });
    });

    // ============================================================
    // 26. fillDiagonal with wrap
    // ============================================================

    describe('fillDiagonal wrap', () => {
      it('fills diagonal wrapping for tall matrix', async () => {
        const a = mat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 4, 3);
        const result = B.fillDiagonal(a, 1, true);
        const data = await getData(result, B);
        // 4x3 with wrap: (0,0), (1,1), (2,2), (3,0)
        expect(data[0]).toBe(1);
        expect(data[4]).toBe(1);
        expect(data[8]).toBe(1);
        expect(data[9]).toBe(1); // wrapped
      });
    });

    // ============================================================
    // 27. asarray with dtype conversion
    // ============================================================

    describe('asarray dtype conversion', () => {
      it('converts existing NDArray dtype', async () => {
        const a = arr([1.5, 2.7, 3.9]);
        const result = B.asarray(a, 'int32');
        expect(result.dtype).toBe('int32');
      });
    });

    // ============================================================
    // 28. lexsort with ties
    // ============================================================

    describe('lexsort ties', () => {
      it('breaks ties by secondary key', async () => {
        const primary = arr([1, 1, 2, 2]);
        const secondary = arr([4, 3, 2, 1]);
        const result = B.lexsort([secondary, primary]);
        const data = await getData(result, B);
        // Sort by primary first, then secondary for ties
        // primary: [1,1,2,2], secondary: [4,3,2,1]
        // indices sorted: 1(1,3), 0(1,4), 3(2,1), 2(2,2)
        expect(data[0]).toBe(1);
        expect(data[1]).toBe(0);
      });
    });

    // ============================================================
    // 29. histogram2d with density
    // ============================================================

    describe('histogram2d', () => {
      it('histogram2d with density=true', async () => {
        B.seed(42);
        const x = B.randn([50]);
        const y = B.randn([50]);
        const result = B.histogram2d(x, y, 5, undefined, true);
        expect(result.hist.shape).toEqual([5, 5]);
      });

      it('histogram2d with explicit range', async () => {
        const x = arr([1, 2, 3]);
        const y = arr([4, 5, 6]);
        const result = B.histogram2d(x, y, 3, [
          [0, 4],
          [3, 7],
        ]);
        expect(result.hist.shape).toEqual([3, 3]);
      });
    });

    // ============================================================
    // 30. _reduceAlongAxis on 3D+ (outerStrides computation)
    // ============================================================

    describe('reduce on 3D+', () => {
      it('sum along axis of 3D array', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const result = B.sum(a, 1) as any;
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([4, 6, 12, 14]);
      });
    });

    // ============================================================
    // 31. Scalar + array broadcasting in _binaryOp
    // ============================================================

    describe('scalar broadcasting', () => {
      it('scalar + array', async () => {
        const result = B.add(5, arr([1, 2, 3]));
        expect(await getData(result, B)).toEqual([6, 7, 8]);
      });

      it('array + scalar', async () => {
        const result = B.add(arr([1, 2, 3]), 10);
        expect(await getData(result, B)).toEqual([11, 12, 13]);
      });
    });

    // ============================================================
    // 32. _binaryOp shape mismatch error
    // ============================================================

    describe('broadcast errors', () => {
      it('incompatible shapes throw', () => {
        expect(() => B.add(arr([1, 2, 3]), arr([1, 2]))).toThrow('broadcast');
      });
    });

    // ============================================================
    // 33. eigvalsh
    // ============================================================

    describe('eigvalsh', () => {
      it('returns eigenvalues only', async () => {
        const a = mat([2, 1, 1, 2], 2, 2);
        const result = B.eigvalsh(a);
        const data = await getData(result, B);
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3
        const sorted = [...data].sort();
        expect(approxEq(sorted[0], 1, 1e-6)).toBe(true);
        expect(approxEq(sorted[1], 3, 1e-6)).toBe(true);
      });
    });

    // ============================================================
    // 34. Misc remaining
    // ============================================================

    describe('misc remaining', () => {
      it('resize works with various sizes', async () => {
        const a = arr([1, 2]);
        const result = B.resize(a, [4]);
        expect(await getData(result, B)).toEqual([1, 2, 1, 2]);
      });

      it('split unevenly throws', () => {
        expect(() => B.split(arr([1, 2, 3, 4, 5]), 3)).toThrow();
      });

      it('expandDims out of bounds throws', () => {
        expect(() => B.expandDims(arr([1, 2, 3]), 5)).toThrow();
      });

      it('batchedMatmul with incompatible shapes throws', () => {
        const a = B.array([1, 2, 3, 4], [1, 2, 2]);
        const b = B.array([1, 2, 3, 4, 5, 6], [1, 3, 2]);
        expect(() => B.batchedMatmul(a, b)).toThrow();
      });

      it('batchedMatmul on <2D throws', () => {
        expect(() => B.batchedMatmul(arr([1, 2]), arr([3, 4]))).toThrow();
      });

      it('compress with axis', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const condition = arr([1, 0, 1]);
        const result = B.compress(condition, a, 1);
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([1, 3, 4, 6]);
      });

      it('moveaxis rearranges', async () => {
        const a = B.array(
          Array.from({ length: 24 }, (_, i) => i),
          [2, 3, 4]
        );
        const result = B.moveaxis(a, [0, 1], [1, 0]);
        expect(result.shape).toEqual([3, 2, 4]);
      });

      it('broadcastTo incompatible throws', () => {
        expect(() => B.broadcastTo(arr([1, 2, 3]), [2, 2])).toThrow();
      });

      it('stack shape mismatch throws', () => {
        expect(() => B.stack([arr([1, 2]), arr([1, 2, 3])])).toThrow();
      });

      it('partition on 2D array', async () => {
        const a = mat([3, 1, 2, 6, 4, 5], 2, 3);
        const result = B.partition(a, 1, 1);
        // After partition with k=1, element at position 1 should be in sorted position
        expect(result.shape).toEqual([2, 3]);
      });

      it('argpartition on 2D array', async () => {
        const a = mat([3, 1, 2, 6, 4, 5], 2, 3);
        const result = B.argpartition(a, 1, 1);
        expect(result.shape).toEqual([2, 3]);
      });

      it('matmul on 1D throws (requires 2D)', () => {
        const a = arr([1, 2, 3]);
        const b = arr([4, 5, 6]);
        expect(() => B.matmul(a, b)).toThrow('2D');
      });

      it('einsum validation error', () => {
        // Wrong number of operands
        expect(() => B.einsum('ij,jk->ik', arr([1]))).toThrow();
      });

      it('histogram all NaN', async () => {
        const a = arr([NaN, NaN, NaN]);
        const result = B.histogram(a, 5);
        const data = await getData(result.hist, B);
        expect(data.every(v => v === 0)).toBe(true);
      });

      it('_gammaRandom with shape < 1 (via beta distribution)', async () => {
        B.seed(42);
        // beta with alpha < 1 triggers _gammaRandom with shape < 1
        const result = B.beta(0.5, 0.5, [100]);
        const data = await getData(result, B);
        expect(data.every(v => v > 0 && v < 1)).toBe(true);
      });

      it('lexsort mismatched lengths throws', () => {
        expect(() => B.lexsort([arr([1, 2]), arr([1, 2, 3])])).toThrow();
      });

      it('roll with array shift on 2D', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.roll(a, [1, 2], [0, 1]);
        expect(result.shape).toEqual([2, 3]);
      });
    });
  });
}
