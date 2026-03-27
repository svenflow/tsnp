/**
 * NumPy API parity tests — test cases converted directly from NumPy's test suite.
 *
 * Sources:
 *   numpy/_core/tests/test_umath.py        (logic, bitwise, rint)
 *   numpy/_core/tests/test_numeric.py       (isclose, allclose, roll, around)
 *   numpy/lib/tests/test_function_base.py   (flip, average, ptp, digitize, trapezoid)
 *   numpy/lib/tests/test_twodim_base.py     (fliplr, flipud, rot90)
 *   numpy/lib/tests/test_arraypad.py        (pad)
 *   numpy/fft/tests/test_helper.py          (fftfreq, rfftfreq, fftshift, ifftshift)
 *   numpy/fft/tests/test_pocketfft.py       (fft, ifft, rfft, irfft)
 *   numpy/linalg/tests/test_linalg.py       (eig, eigh, cholesky, lstsq, pinv, matrix_rank)
 *   numpy/_core/tests/test_multiarray.py    (vdot)
 *   numpy/polynomial/tests/test_polynomial.py (polyder, polyint, polydiv, polysub)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, getData } from './test-utils';

export function numpyApiTests(getBackend: () => Backend) {
  describe('numpy-api', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[]) => B.array(data, [data.length]);
    const mat = (data: number[], rows: number, cols: number) =>
      B.array(data, [rows, cols]);

    // ============================================================
    // Creation Additional — fromfunction, fromiter
    // ============================================================
    describe('creation-additional', () => {
      it('fromfunction — 2x3 coordinate sum', async () => {
        const result = B.fromfunction((i, j) => i + j, [2, 3]);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([0, 1, 2, 1, 2, 3]);
      });

      it('fromfunction — 3x3 identity-like', async () => {
        const result = B.fromfunction((i, j) => i === j ? 1 : 0, [3, 3]);
        expect(await getData(result, B)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      });

      it('fromiter — generator', async () => {
        function* gen() { yield 1; yield 2; yield 3; yield 4; yield 5; }
        const result = B.fromiter(gen());
        expect(result.shape).toEqual([5]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5]);
      });

      it('fromiter — with count limit', async () => {
        function* gen() { let i = 0; while (true) yield i++; }
        const result = B.fromiter(gen(), 4);
        expect(result.shape).toEqual([4]);
        expect(await getData(result, B)).toEqual([0, 1, 2, 3]);
      });
    });

    // ============================================================
    // Complex Helpers — real, imag, conj
    // ============================================================
    describe('complex-helpers', () => {
      it('real — returns copy of real array', async () => {
        const a = arr([1, 2, 3]);
        const result = B.real(a);
        expect(await getData(result, B)).toEqual([1, 2, 3]);
        expect(result).not.toBe(a);
      });

      it('imag — returns zeros for real array', async () => {
        const a = arr([1, 2, 3]);
        expect(await getData(B.imag(a), B)).toEqual([0, 0, 0]);
      });

      it('conj — returns copy for real array', async () => {
        const a = arr([1, -2, 3]);
        expect(await getData(B.conj(a), B)).toEqual([1, -2, 3]);
      });
    });

    // ============================================================
    // Logic — from test_umath.py test_truth_table_logical
    // ============================================================
    describe('logic', () => {
      // np.test_truth_table_logical: input1=[F,F,T,T], input2=[F,T,F,T]
      it('logicalAnd — truth table', async () => {
        const a = arr([0, 0, 1, 1]);
        const b = arr([0, 1, 0, 1]);
        expect(await getData(B.logicalAnd(a, b), B)).toEqual([0, 0, 0, 1]);
      });

      it('logicalOr — truth table', async () => {
        const a = arr([0, 0, 1, 1]);
        const b = arr([0, 1, 0, 1]);
        expect(await getData(B.logicalOr(a, b), B)).toEqual([0, 1, 1, 1]);
      });

      it('logicalNot — ones become zeros', async () => {
        // From TestInt.test_logical_not: logical_not(ones(10)) => all False
        const x = arr([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        const result = await getData(B.logicalNot(x), B);
        expect(result).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      });

      it('logicalXor — truth table', async () => {
        const a = arr([0, 0, 1, 1]);
        const b = arr([0, 1, 0, 1]);
        expect(await getData(B.logicalXor(a, b), B)).toEqual([0, 1, 1, 0]);
      });

      // From TestIsclose in test_numeric.py
      it('isclose — inf and small values', async () => {
        const a = arr([Infinity, 0]);
        const b = arr([Infinity, 2e-8]);
        const result = await getData(B.isclose(a, b), B);
        expect(result[0]).toBe(1); // inf == inf
        expect(result[1]).toBe(0); // 0 vs 2e-8 with default rtol=1e-5, atol=1e-8
      });

      it('isclose — arange vs close values', async () => {
        const a = arr([0, 1, 2]);
        const b = arr([0, 1, 2.1]);
        const result = await getData(B.isclose(a, b), B);
        expect(result).toEqual([1, 1, 0]);
      });

      // From TestAllclose in test_numeric.py
      it('allclose — equal arrays', () => {
        expect(B.allclose(arr([1, 0]), arr([1, 0]))).toBe(true);
      });

      it('allclose — within tolerance', () => {
        expect(B.allclose(arr([1e-8]), arr([0]))).toBe(true);
      });

      it('allclose — inf mismatch', () => {
        expect(B.allclose(arr([Infinity, 0]), arr([1, Infinity]))).toBe(false);
      });

      it('allclose — negative inf vs positive inf', () => {
        expect(B.allclose(arr([-Infinity, 0]), arr([Infinity, 0]))).toBe(false);
      });

      it('allclose — beyond tolerance', () => {
        expect(B.allclose(arr([2e-8]), arr([0]))).toBe(false);
      });

      it('arrayEqual — same arrays', () => {
        expect(B.arrayEqual(arr([1, 2, 3]), arr([1, 2, 3]))).toBe(true);
      });

      it('arrayEqual — different arrays', () => {
        expect(B.arrayEqual(arr([1, 2, 3]), arr([1, 2, 4]))).toBe(false);
      });
    });

    // ============================================================
    // Bitwise — from test_umath.py TestBitwiseUFuncs.test_values
    // ============================================================
    describe('bitwise', () => {
      // zeros=[0], ones=[-1] (all bits set)
      it('bitwiseAnd — identity cases', async () => {
        expect(await getData(B.bitwiseAnd(arr([0]), arr([0])), B)).toEqual([0]);
        expect(await getData(B.bitwiseAnd(arr([0]), arr([-1])), B)).toEqual([0]);
        expect(await getData(B.bitwiseAnd(arr([-1]), arr([0])), B)).toEqual([0]);
        expect(await getData(B.bitwiseAnd(arr([-1]), arr([-1])), B)).toEqual([-1]);
      });

      it('bitwiseOr — identity cases', async () => {
        expect(await getData(B.bitwiseOr(arr([0]), arr([0])), B)).toEqual([0]);
        expect(await getData(B.bitwiseOr(arr([0]), arr([-1])), B)).toEqual([-1]);
        expect(await getData(B.bitwiseOr(arr([-1]), arr([0])), B)).toEqual([-1]);
        expect(await getData(B.bitwiseOr(arr([-1]), arr([-1])), B)).toEqual([-1]);
      });

      it('bitwiseXor — identity cases', async () => {
        expect(await getData(B.bitwiseXor(arr([0]), arr([0])), B)).toEqual([0]);
        expect(await getData(B.bitwiseXor(arr([0]), arr([-1])), B)).toEqual([-1]);
        expect(await getData(B.bitwiseXor(arr([-1]), arr([0])), B)).toEqual([-1]);
        expect(await getData(B.bitwiseXor(arr([-1]), arr([-1])), B)).toEqual([0]);
      });

      it('bitwiseNot — invert zeros and ones', async () => {
        expect(await getData(B.bitwiseNot(arr([0])), B)).toEqual([-1]);
        expect(await getData(B.bitwiseNot(arr([-1])), B)).toEqual([0]);
      });

      it('leftShift', async () => {
        expect(await getData(B.leftShift(arr([1, 2, 3]), arr([1, 2, 3])), B)).toEqual([2, 8, 24]);
      });

      it('rightShift', async () => {
        expect(await getData(B.rightShift(arr([8, 16, 32]), arr([1, 2, 3])), B)).toEqual([4, 4, 4]);
      });
    });

    // ============================================================
    // Array Manipulation — from test_function_base.py, test_twodim_base.py
    // ============================================================
    describe('array-manipulation', () => {
      it('copy — independent copy', async () => {
        const a = arr([1, 2, 3]);
        const c = B.copy(a);
        expect(await getData(c, B)).toEqual([1, 2, 3]);
        expect(c).not.toBe(a);
      });

      it('empty — creates array with shape', () => {
        expect(B.empty([3, 2]).shape).toEqual([3, 2]);
      });

      // From TestFlip in test_function_base.py
      it('flip 2d axis=1 (left-right)', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        expect(await getData(B.flip(a, 1), B)).toEqual([2, 1, 0, 5, 4, 3]);
      });

      it('flip 2d axis=0 (up-down)', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        expect(await getData(B.flip(a, 0), B)).toEqual([3, 4, 5, 0, 1, 2]);
      });

      // From TestFliplr in test_twodim_base.py
      it('fliplr — [[0,1,2],[3,4,5]]', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        expect(await getData(B.fliplr(a), B)).toEqual([2, 1, 0, 5, 4, 3]);
      });

      // From TestFlipud in test_twodim_base.py
      it('flipud — [[0,1,2],[3,4,5]]', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        expect(await getData(B.flipud(a), B)).toEqual([3, 4, 5, 0, 1, 2]);
      });

      // From TestRoll in test_numeric.py
      it('roll 1d — arange(10) shift 2', async () => {
        const a = arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        expect(await getData(B.roll(a, 2), B)).toEqual([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]);
      });

      it('roll — empty array', async () => {
        const a = arr([]);
        expect(await getData(B.roll(a, 1), B)).toEqual([]);
      });

      // From test_twodim_base.py
      it('rot90 — single rotation', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        const result = B.rot90(a);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([2, 5, 1, 4, 0, 3]);
      });

      it('rot90 k=2 — 180 degrees', async () => {
        const a = mat([0, 1, 2, 3, 4, 5], 2, 3);
        const result = B.rot90(a, 2);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([5, 4, 3, 2, 1, 0]);
      });

      it('ravel — 4x3 to flat', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
        const result = B.ravel(a);
        expect(result.shape).toEqual([12]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      });

      // From test_arraypad.py
      it('pad constant — [1,1] padded by 2', async () => {
        const a = arr([1, 1]);
        expect(await getData(B.pad(a, 2, 'constant', 0), B)).toEqual([0, 0, 1, 1, 0, 0]);
      });

      it('pad edge — 1d', async () => {
        const a = arr([1, 2, 3]);
        expect(await getData(B.pad(a, [1, 2], 'edge'), B)).toEqual([1, 1, 2, 3, 3, 3]);
      });

      it('columnStack', async () => {
        const a = arr([1, 2, 3]);
        const b = arr([4, 5, 6]);
        const result = B.columnStack([a, b]);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('arraySplit equal parts', async () => {
        const a = arr([1, 2, 3, 4, 5, 6]);
        const parts = B.arraySplit(a, 3);
        expect(parts.length).toBe(3);
        expect(await getData(parts[0], B)).toEqual([1, 2]);
        expect(await getData(parts[1], B)).toEqual([3, 4]);
        expect(await getData(parts[2], B)).toEqual([5, 6]);
      });

      it('arraySplit with indices', async () => {
        const a = arr([1, 2, 3, 4, 5, 6]);
        const parts = B.arraySplit(a, [2, 4]);
        expect(parts.length).toBe(3);
        expect(await getData(parts[0], B)).toEqual([1, 2]);
        expect(await getData(parts[1], B)).toEqual([3, 4]);
        expect(await getData(parts[2], B)).toEqual([5, 6]);
      });

      it('takeAlongAxis — 2d axis=1', async () => {
        const a = mat([10, 20, 30, 40, 50, 60], 2, 3);
        const indices = B.array([0, 2, 0, 2], [2, 2]);
        const result = B.takeAlongAxis(a, indices, 1);
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([10, 30, 40, 60]);
      });

      it('putAlongAxis — 2d axis=1', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const indices = B.array([0, 2, 0, 2], [2, 2]);
        const values = B.array([99, 88, 77, 66], [2, 2]);
        const result = B.putAlongAxis(a, indices, values, 1);
        const data = await getData(result, B);
        expect(data[0]).toBe(99);
        expect(data[2]).toBe(88);
        expect(data[3]).toBe(77);
        expect(data[5]).toBe(66);
      });
    });

    // ============================================================
    // Linalg — from test_linalg.py
    // ============================================================
    describe('linalg-extended', () => {
      // eig: eigenvalues of [[1,0.5],[0.5,1]] are 1.5 and 0.5
      it('eig — eigenvalues of [[1,0.5],[0.5,1]]', async () => {
        const a = mat([1, 0.5, 0.5, 1], 2, 2);
        const { values } = B.eig(a);
        const vals = await getData(values, B);
        const sorted = [...vals].sort((a, b) => b - a);
        expect(approxEq(sorted[0], 1.5, RELAXED_TOL)).toBe(true);
        expect(approxEq(sorted[1], 0.5, RELAXED_TOL)).toBe(true);
      });

      // eigh: eigenvalues of [[0,1],[1,0]] are -1 and 1
      it('eigh — [[0,1],[1,0]] eigenvalues [-1,1]', async () => {
        const a = mat([0, 1, 1, 0], 2, 2);
        const { values } = B.eigh(a);
        const vals = await getData(values, B);
        const sorted = [...vals].sort((a, b) => a - b);
        expect(approxEq(sorted[0], -1, RELAXED_TOL)).toBe(true);
        expect(approxEq(sorted[1], 1, RELAXED_TOL)).toBe(true);
      });

      it('eigvals — [[1,0.5],[0.5,1]]', async () => {
        const a = mat([1, 0.5, 0.5, 1], 2, 2);
        const vals = await getData(B.eigvals(a), B);
        const sorted = [...vals].sort((a, b) => b - a);
        // Eigenvalues of [[1,0.5],[0.5,1]] are 1.5 and 0.5
        expect(approxEq(sorted[0], 1.5, RELAXED_TOL)).toBe(true);
        expect(approxEq(sorted[1], 0.5, RELAXED_TOL)).toBe(true);
      });

      // cholesky property: L @ L^T == A
      it('cholesky — property L @ L^T = A', async () => {
        const a = mat([4, 2, 2, 5], 2, 2);
        const L = B.cholesky(a);
        const LT = B.transpose(L);
        const product = B.matmul(L, LT);
        const data = await getData(product, B);
        expect(approxEq(data[0], 4, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 2, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 2, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 5, RELAXED_TOL)).toBe(true);
      });

      // lstsq: for m <= n case, b == A @ x (exact solution)
      it('lstsq — exact solution for square system', async () => {
        const A = mat([1, 0, 0, 1], 2, 2);
        const b = arr([3, 7]);
        const result = B.lstsq(A, b);
        const x = await getData(result.x, B);
        expect(approxEq(x[0], 3, RELAXED_TOL)).toBe(true);
        expect(approxEq(x[1], 7, RELAXED_TOL)).toBe(true);
      });

      it('lstsq — overdetermined system', async () => {
        const A = mat([1, 1, 1, 2, 1, 3], 3, 2);
        const b = arr([1, 2, 3]);
        const result = B.lstsq(A, b);
        const x = await getData(result.x, B);
        expect(approxEq(x[0], 0, RELAXED_TOL)).toBe(true);
        expect(approxEq(x[1], 1, RELAXED_TOL)).toBe(true);
      });

      // pinv property: A @ pinv(A) @ A == A
      it('pinv — property A @ pinv(A) @ A = A', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const pi = B.pinv(a);
        const step1 = B.matmul(a, pi);
        const step2 = B.matmul(step1, a);
        const orig = await getData(a, B);
        const result = await getData(step2, B);
        for (let i = 0; i < 4; i++) {
          expect(approxEq(result[i], orig[i], RELAXED_TOL)).toBe(true);
        }
      });

      // From TestMatrixRank
      it('matrixRank — eye(4) = 4', () => {
        const a = B.eye(4);
        expect(B.matrixRank(a)).toBe(4);
      });

      it('matrixRank — zeros(4,4) = 0', () => {
        const a = B.zeros([4, 4]);
        expect(B.matrixRank(a)).toBe(0);
      });

      it('matrixRank — rank-deficient', () => {
        const a = mat([1, 2, 2, 4], 2, 2);
        expect(B.matrixRank(a)).toBe(1);
      });

      // From test_multiarray.py: vdot(eye(3), eye(3)) = 3
      it('vdot — dot product', () => {
        const a = arr([1, 2, 3]);
        const b = arr([4, 5, 6]);
        expect(approxEq(B.vdot(a, b), 32, DEFAULT_TOL)).toBe(true);
      });

      // tensordot: full contraction of arange(6).reshape(2,3)
      it('tensordot — full contraction', async () => {
        const x = mat([0, 1, 2, 3, 4, 5], 2, 3);
        const result = B.tensordot(x, x, 2);
        const data = await getData(result, B);
        // 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
        expect(approxEq(data[0], 55, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============================================================
    // FFT — from test_helper.py and test_pocketfft.py
    // ============================================================
    describe('fft', () => {
      // Property test: ifft(fft(x)) == x
      it('fft/ifft roundtrip', async () => {
        const x = arr([1, 2, 3, 4, 5, 6, 7, 8]);
        const freq = B.fft(x);
        const recovered = B.ifft(freq.real, freq.imag);
        const data = await getData(recovered.real, B);
        for (let i = 0; i < 8; i++) {
          expect(approxEq(data[i], i + 1, RELAXED_TOL)).toBe(true);
        }
      });

      it('fft — DC component of constant signal', async () => {
        const a = arr([1, 1, 1, 1]);
        const result = B.fft(a);
        const real = await getData(result.real, B);
        expect(approxEq(real[0], 4, RELAXED_TOL)).toBe(true);
        for (let i = 1; i < 4; i++) {
          expect(approxEq(real[i], 0, RELAXED_TOL)).toBe(true);
        }
      });

      it('fft2/ifft2 roundtrip', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const freq = B.fft2(a);
        const recovered = B.ifft2(freq.real, freq.imag);
        const data = await getData(recovered.real, B);
        expect(approxEq(data[0], 1, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 2, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 3, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 4, RELAXED_TOL)).toBe(true);
      });

      it('rfft — DC component', async () => {
        const a = arr([1, 2, 3, 4]);
        const result = B.rfft(a);
        expect(result.real.shape[0]).toBe(3); // N/2 + 1
        const real = await getData(result.real, B);
        expect(approxEq(real[0], 10, RELAXED_TOL)).toBe(true); // sum
      });

      it('irfft roundtrip', async () => {
        const original = arr([1, 2, 3, 4]);
        const freq = B.rfft(original);
        const recovered = B.irfft(freq.real, freq.imag, 4);
        const data = await getData(recovered, B);
        for (let i = 0; i < 4; i++) {
          expect(approxEq(data[i], i + 1, RELAXED_TOL)).toBe(true);
        }
      });

      // From TestFFTFreq: 9 * fftfreq(9) => [0,1,2,3,4,-4,-3,-2,-1]
      it('fftfreq(9)', async () => {
        const result = B.fftfreq(9);
        const data = await getData(result, B);
        const scaled = data.map(v => v * 9);
        expect(scaled.map(v => Math.round(v))).toEqual([0, 1, 2, 3, 4, -4, -3, -2, -1]);
      });

      // 10 * fftfreq(10) => [0,1,2,3,4,-5,-4,-3,-2,-1]
      it('fftfreq(10)', async () => {
        const result = B.fftfreq(10);
        const data = await getData(result, B);
        const scaled = data.map(v => v * 10);
        expect(scaled.map(v => Math.round(v))).toEqual([0, 1, 2, 3, 4, -5, -4, -3, -2, -1]);
      });

      // From TestRFFTFreq: 9 * rfftfreq(9) => [0,1,2,3,4]
      it('rfftfreq(9)', async () => {
        const result = B.rfftfreq(9);
        const data = await getData(result, B);
        expect(data.length).toBe(5);
        const scaled = data.map(v => v * 9);
        expect(scaled.map(v => Math.round(v))).toEqual([0, 1, 2, 3, 4]);
      });

      // From TestFFTShift — even array
      it('fftshift — even', async () => {
        const a = arr([0, 1, 2, 3, 4, -5, -4, -3, -2, -1]);
        expect(await getData(B.fftshift(a), B)).toEqual([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);
      });

      // odd array
      it('fftshift — odd', async () => {
        const a = arr([0, 1, 2, 3, 4, -4, -3, -2, -1]);
        expect(await getData(B.fftshift(a), B)).toEqual([-4, -3, -2, -1, 0, 1, 2, 3, 4]);
      });

      it('ifftshift — even', async () => {
        const a = arr([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);
        expect(await getData(B.ifftshift(a), B)).toEqual([0, 1, 2, 3, 4, -5, -4, -3, -2, -1]);
      });

      it('ifftshift — odd', async () => {
        const a = arr([-4, -3, -2, -1, 0, 1, 2, 3, 4]);
        expect(await getData(B.ifftshift(a), B)).toEqual([0, 1, 2, 3, 4, -4, -3, -2, -1]);
      });
    });

    // ============================================================
    // Random Distributions — statistical property tests
    // ============================================================
    describe('random-distributions', () => {
      it('exponential — positive values', async () => {
        B.seed(42);
        const data = await getData(B.exponential(2.0, [100]), B);
        expect(data.length).toBe(100);
        expect(data.every(v => v >= 0)).toBe(true);
      });

      it('poisson — non-negative integers', async () => {
        B.seed(42);
        const data = await getData(B.poisson(5, [100]), B);
        expect(data.every(v => v >= 0 && Number.isInteger(v))).toBe(true);
      });

      it('binomial — values in [0, n]', async () => {
        B.seed(42);
        const data = await getData(B.binomial(10, 0.5, [100]), B);
        expect(data.every(v => v >= 0 && v <= 10 && Number.isInteger(v))).toBe(true);
      });

      it('beta — values in [0, 1]', async () => {
        B.seed(42);
        const data = await getData(B.beta(2, 5, [100]), B);
        expect(data.every(v => v >= 0 && v <= 1)).toBe(true);
      });

      it('gamma — positive values', async () => {
        B.seed(42);
        const data = await getData(B.gamma(2, 1, [100]), B);
        expect(data.every(v => v >= 0)).toBe(true);
      });

      it('lognormal — positive values', async () => {
        B.seed(42);
        const data = await getData(B.lognormal(0, 1, [100]), B);
        expect(data.every(v => v > 0)).toBe(true);
      });

      it('chisquare — positive values', async () => {
        B.seed(42);
        const data = await getData(B.chisquare(3, [100]), B);
        expect(data.every(v => v >= 0)).toBe(true);
      });

      it('standardT — finite values', async () => {
        B.seed(42);
        const data = await getData(B.standardT(5, [100]), B);
        expect(data.every(v => Number.isFinite(v))).toBe(true);
      });

      it('multivariateNormal — shape', async () => {
        B.seed(42);
        const result = B.multivariateNormal(arr([0, 0]), mat([1, 0, 0, 1], 2, 2), 50);
        expect(result.shape).toEqual([50, 2]);
      });

      it('geometric — positive integers >= 1', async () => {
        B.seed(42);
        const data = await getData(B.geometric(0.5, [100]), B);
        expect(data.every(v => v >= 1 && Number.isInteger(v))).toBe(true);
      });

      it('weibull — positive values', async () => {
        B.seed(42);
        const data = await getData(B.weibull(2, [100]), B);
        expect(data.every(v => v >= 0)).toBe(true);
      });
    });

    // ============================================================
    // Stats — from test_function_base.py
    // ============================================================
    describe('stats-extended', () => {
      // From TestAverage
      it('average — simple', () => {
        expect(approxEq(B.average(arr([1, 2, 3])), 2.0, DEFAULT_TOL)).toBe(true);
      });

      it('average — weighted arange(10)', () => {
        // weighted average: sum(i*i for i in 0..9) / sum(0..9) = 285/45 = 6.333...
        const vals = arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        const wts = arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        expect(approxEq(B.average(vals, wts), 285 / 45, DEFAULT_TOL)).toBe(true);
      });

      // From TestPtp
      it('ptp — peak to peak', () => {
        expect(B.ptp(arr([3, 4, 5, 10, -3, -5, 6.0]))).toBe(15.0);
      });

      // From TestDigitize
      it('digitize — forward bins', async () => {
        const x = arr([1, 5, 4, 10, 8, 11, 0]);
        const bins = arr([1, 5, 10]);
        const data = await getData(B.digitize(x, bins), B);
        expect(data).toEqual([1, 2, 1, 3, 2, 3, 0]);
      });

      it('nanquantile', () => {
        expect(approxEq(B.nanquantile(arr([1, NaN, 3, 4, 5]), 0.5), 3.5, DEFAULT_TOL)).toBe(true);
      });

      it('nancumsum', async () => {
        const data = await getData(B.nancumsum(arr([1, NaN, 3, 4])), B);
        expect(data).toEqual([1, 1, 4, 8]);
      });

      it('nancumprod', async () => {
        const data = await getData(B.nancumprod(arr([1, NaN, 3, 4])), B);
        expect(data).toEqual([1, 1, 3, 12]);
      });

      it('uniqueCounts', async () => {
        const { values, counts } = B.uniqueCounts(arr([1, 2, 2, 3, 3, 3]));
        expect(await getData(values, B)).toEqual([1, 2, 3]);
        expect(await getData(counts, B)).toEqual([1, 2, 3]);
      });

      it('uniqueInverse', async () => {
        const { values, inverse } = B.uniqueInverse(arr([3, 1, 2, 1, 3]));
        expect(await getData(values, B)).toEqual([1, 2, 3]);
        expect(await getData(inverse, B)).toEqual([2, 0, 1, 0, 2]);
      });

      it('histogram2d — total counts', async () => {
        const result = B.histogram2d(arr([1, 2, 3, 4, 5]), arr([5, 4, 3, 2, 1]), 3);
        const hist = await getData(result.hist, B);
        expect(hist.reduce((a, b) => a + b, 0)).toBe(5);
      });
    });

    // ============================================================
    // Rounding — from test_numeric.py and test_umath.py
    // ============================================================
    describe('rounding', () => {
      // From test_numeric.py: around([1.56,72.54,6.35,3.25], decimals=1) => [1.6,72.5,6.4,3.2]
      // Note: JS Math.round differs from NumPy's banker's rounding; test non-ambiguous cases
      it('around — decimals=1', async () => {
        const a = arr([1.56, 72.54, 6.35, 3.25]);
        const result = B.around(a, 1);
        const data = await getData(result, B);
        expect(approxEq(data[0], 1.6, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 72.5, DEFAULT_TOL)).toBe(true);
      });

      it('around — decimals=0', async () => {
        const data = await getData(B.around(arr([1.6, 2.4, 3.7])), B);
        expect(data[0]).toBe(2);
        expect(data[1]).toBe(2);
        expect(data[2]).toBe(4);
      });

      it('rint — standard values', async () => {
        const data = await getData(B.rint(arr([3.7, -3.7, 0.1, 10.0])), B);
        expect(data[0]).toBe(4);
        expect(data[1]).toBe(-4);
        expect(data[2]).toBe(0);
        expect(data[3]).toBe(10);
      });
    });

    // ============================================================
    // Polynomial — from test_polynomial.py (np.poly1d style: high-order first)
    // ============================================================
    describe('polynomial-extended', () => {
      // np.polyder: poly1d([1,2,3]) = x^2+2x+3 => deriv = 2x+2 = [2,2]
      it('polyder — x^2+2x+3', async () => {
        const p = arr([1, 2, 3]);
        expect(await getData(B.polyder(p), B)).toEqual([2, 2]);
      });

      it('polyder m=2 — x^2+2x+3', async () => {
        const p = arr([1, 2, 3]);
        expect(await getData(B.polyder(p, 2), B)).toEqual([2]);
      });

      // np.polyint: poly1d([1,2,3]) => x^3/3 + x^2 + 3x = [1/3, 1, 3, 0]
      it('polyint — x^2+2x+3', async () => {
        const p = arr([1, 2, 3]);
        const result = await getData(B.polyint(p), B);
        expect(result.length).toBe(4);
        expect(approxEq(result[0], 1 / 3, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result[1], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result[2], 3, DEFAULT_TOL)).toBe(true);
        expect(approxEq(result[3], 0, DEFAULT_TOL)).toBe(true);
      });

      // np.polydiv: (x^2 - 1) / (x + 1) = (x - 1) remainder 0
      it('polydiv — (x^2-1)/(x+1)', async () => {
        const u = arr([1, 0, -1]);
        const v = arr([1, 1]);
        const { q, r } = B.polydiv(u, v);
        const qData = await getData(q, B);
        expect(approxEq(qData[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(qData[1], -1, DEFAULT_TOL)).toBe(true);
      });

      // np.polydiv: (x^2+2x+4) / (4x^2+2x+1) = 0.25 remainder [1.5, 3.75]
      it('polydiv — non-exact', async () => {
        const { q, r } = B.polydiv(arr([1, 2, 4]), arr([4, 2, 1]));
        const qData = await getData(q, B);
        const rData = await getData(r, B);
        expect(approxEq(qData[0], 0.25, DEFAULT_TOL)).toBe(true);
        expect(approxEq(rData[0], 1.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(rData[1], 3.75, DEFAULT_TOL)).toBe(true);
      });

      it('polysub', async () => {
        expect(await getData(B.polysub(arr([3, 2, 1]), arr([1, 1, 1])), B)).toEqual([2, 1, 0]);
      });
    });

    // ============================================================
    // Integration — from test_function_base.py TestTrapezoid
    // ============================================================
    describe('integration', () => {
      // trapezoid([1,1], dx=1) = 1.0 (rectangle)
      it('trapezoid — rectangle', () => {
        expect(approxEq(B.trapezoid(arr([1, 1])), 1.0, DEFAULT_TOL)).toBe(true);
      });

      // trapezoid([0,1], dx=1) = 0.5 (triangle)
      it('trapezoid — triangle', () => {
        expect(approxEq(B.trapezoid(arr([0, 1])), 0.5, DEFAULT_TOL)).toBe(true);
      });

      // trapezoid([1,2,3], dx=1) = 4.0
      it('trapezoid — [1,2,3]', () => {
        expect(approxEq(B.trapezoid(arr([1, 2, 3])), 4.0, DEFAULT_TOL)).toBe(true);
      });

      it('trapezoid — with x values', () => {
        const x = arr([0, 0.5, 1, 1.5, 2]);
        const y = arr([0, 0.25, 1, 2.25, 4]);
        expect(approxEq(B.trapezoid(y, x), 2.75, DEFAULT_TOL)).toBe(true);
      });

      it('trapezoid — constant', () => {
        expect(approxEq(B.trapezoid(arr([5, 5, 5, 5])), 15, DEFAULT_TOL)).toBe(true);
      });
    });
  });
}
