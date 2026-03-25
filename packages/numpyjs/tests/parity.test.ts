/**
 * NumPy parity tests — unified axis API + new functions
 *
 * Tests for:
 * 1. Unified axis params (sum(arr, axis) instead of sumAxis)
 * 2. New NumPy functions: gcd, lcm, tri, diagflat, block, indices,
 *    diagIndices, trilIndices, triuIndices, window functions,
 *    packbits, unpackbits, eigvalsh, unravelIndex, ravelMultiIndex,
 *    fftn, ifftn
 *
 * All test values sourced from NumPy.
 */

import { describe, it, expect } from 'vitest';
import type { Backend, NDArray } from './test-utils';
import { getData, getTol, approxEq } from './test-utils';

export function parityTests(getBackend: () => Backend) {
  describe('unified axis API (NumPy-style)', () => {
    // >>> np.sum(np.array([[1,2],[3,4]]), axis=0)
    // array([4, 6])
    it('sum with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.sum(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([4, 6]);
    });

    // >>> np.sum(np.array([[1,2],[3,4]]), axis=1)
    // array([3, 7])
    it('sum with axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.sum(a, 1) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([3, 7]);
    });

    // >>> np.sum(np.array([[1,2],[3,4]]))
    // 10
    it('sum without axis (scalar)', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.sum(a);
      expect(r).toBe(10);
    });

    // >>> np.mean(np.array([[1,2],[3,4]]), axis=0)
    // array([2., 3.])
    it('mean with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.mean(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([2, 3]);
    });

    // >>> np.min(np.array([[1,2],[3,4]]), axis=1)
    // array([1, 3])
    it('min with axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.min(a, 1) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([1, 3]);
    });

    // >>> np.max(np.array([[1,2],[3,4]]), axis=0)
    // array([3, 4])
    it('max with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.max(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([3, 4]);
    });

    // >>> np.prod(np.array([[1,2],[3,4]]), axis=0)
    // array([3, 8])
    it('prod with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.prod(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([3, 8]);
    });

    // >>> np.argmin(np.array([[3,1],[2,4]]), axis=1)
    // array([1, 0])
    it('argmin with axis=1', async () => {
      const B = getBackend();
      const a = B.array([3, 1, 2, 4], [2, 2]);
      const r = B.argmin(a, 1) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([1, 0]);
    });

    // >>> np.argmax(np.array([[3,1],[2,4]]), axis=0)
    // array([0, 1])
    it('argmax with axis=0', async () => {
      const B = getBackend();
      const a = B.array([3, 1, 2, 4], [2, 2]);
      const r = B.argmax(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([0, 1]);
    });

    // >>> np.var(np.array([[1,2],[3,4]]), ddof=0, axis=0)
    // array([1., 1.])
    it('var with ddof and axis', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.var(a, 0) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 1, tol)).toBe(true);
      expect(approxEq(d[1], 1, tol)).toBe(true);
    });

    // >>> np.std(np.array([[1,2],[3,4]]), ddof=0, axis=1)
    // array([0.5, 0.5])
    it('std with ddof and axis', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.std(a, 1) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 0.5, tol)).toBe(true);
      expect(approxEq(d[1], 0.5, tol)).toBe(true);
    });

    // >>> np.all(np.array([[True, False],[True, True]]), axis=0)
    // array([ True, False])
    it('all with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 0, 1, 1], [2, 2]);
      const r = B.all(a, 0) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([1, 0]);
    });

    // >>> np.any(np.array([[0, 0],[1, 0]]), axis=1)
    // array([False,  True])
    it('any with axis=1', async () => {
      const B = getBackend();
      const a = B.array([0, 0, 1, 0], [2, 2]);
      const r = B.any(a, 1) as NDArray;
      const d = await getData(r, B);
      expect(d).toEqual([0, 1]);
    });

    // >>> np.cumsum(np.array([[1,2],[3,4]]), axis=0)
    // array([[1, 2],[4, 6]])
    it('cumsum with axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.cumsum(a, 0);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 4, 6]);
    });

    // >>> np.cumprod(np.array([[1,2],[3,4]]), axis=1)
    // array([[ 1,  2],[ 3, 12]])
    it('cumprod with axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.cumprod(a, 1);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3, 12]);
    });
  });

  describe('gcd / lcm', () => {
    // >>> np.gcd(np.array([12, 18, 24]), np.array([8, 12, 36]))
    // array([ 4,  6, 12])
    it('gcd element-wise', async () => {
      const B = getBackend();
      const a = B.array([12, 18, 24]);
      const b = B.array([8, 12, 36]);
      const r = B.gcd(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([4, 6, 12]);
    });

    // >>> np.gcd(np.array([0, 5]), np.array([0, 0]))
    // array([0, 5])
    it('gcd with zeros', async () => {
      const B = getBackend();
      const r = B.gcd(B.array([0, 5]), B.array([0, 0]));
      const d = await getData(r, B);
      expect(d).toEqual([0, 5]);
    });

    // >>> np.lcm(np.array([4, 6]), np.array([6, 15]))
    // array([12, 30])
    it('lcm element-wise', async () => {
      const B = getBackend();
      const r = B.lcm(B.array([4, 6]), B.array([6, 15]));
      const d = await getData(r, B);
      expect(d).toEqual([12, 30]);
    });

    // Scalar broadcasting
    // >>> np.gcd(np.array([12, 18, 24]), 6)
    // array([6, 6, 6])
    it('gcd with scalar', async () => {
      const B = getBackend();
      const r = B.gcd(B.array([12, 18, 24]), 6);
      const d = await getData(r, B);
      expect(d).toEqual([6, 6, 6]);
    });
  });

  describe('tri', () => {
    // >>> np.tri(3)
    // array([[1., 0., 0.],
    //        [1., 1., 0.],
    //        [1., 1., 1.]])
    it('tri(3)', async () => {
      const B = getBackend();
      const r = B.tri(3);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1]);
      expect(r.shape).toEqual([3, 3]);
    });

    // >>> np.tri(3, 4, 1)
    // array([[1., 1., 0., 0.],
    //        [1., 1., 1., 0.],
    //        [1., 1., 1., 1.]])
    it('tri(3, 4, 1)', async () => {
      const B = getBackend();
      const r = B.tri(3, 4, 1);
      const d = await getData(r, B);
      expect(d).toEqual([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]);
      expect(r.shape).toEqual([3, 4]);
    });

    // >>> np.tri(3, dtype=np.int32)
    it('tri with dtype', async () => {
      const B = getBackend();
      const r = B.tri(2, undefined, undefined, 'int32');
      expect(r.dtype).toBe('int32');
    });
  });

  describe('diagflat', () => {
    // >>> np.diagflat([1, 2, 3])
    // array([[1, 0, 0],
    //        [0, 2, 0],
    //        [0, 0, 3]])
    it('diagflat basic', async () => {
      const B = getBackend();
      const r = B.diagflat(B.array([1, 2, 3]));
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3]);
      expect(r.shape).toEqual([3, 3]);
    });

    // >>> np.diagflat([1, 2], 1)
    // array([[0, 1, 0],
    //        [0, 0, 2],
    //        [0, 0, 0]])
    it('diagflat with k=1', async () => {
      const B = getBackend();
      const r = B.diagflat(B.array([1, 2]), 1);
      const d = await getData(r, B);
      expect(d).toEqual([0, 1, 0, 0, 0, 2, 0, 0, 0]);
      expect(r.shape).toEqual([3, 3]);
    });
  });

  describe('block', () => {
    // >>> np.block([np.array([1,2]), np.array([3,4])])
    // array([1, 2, 3, 4])
    it('block 1D', async () => {
      const B = getBackend();
      const r = B.block([B.array([1, 2]), B.array([3, 4])]);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3, 4]);
    });

    // >>> np.block([[np.array([[1,2]]), np.array([[3]])],
    //              [np.array([[4,5]]), np.array([[6]])]])
    // array([[1, 2, 3],
    //        [4, 5, 6]])
    it('block 2D', async () => {
      const B = getBackend();
      const a = B.array([1, 2], [1, 2]);
      const b = B.array([3], [1, 1]);
      const c = B.array([4, 5], [1, 2]);
      const dd = B.array([6], [1, 1]);
      const r = B.block([
        [a, b],
        [c, dd],
      ]);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3, 4, 5, 6]);
      expect(r.shape).toEqual([2, 3]);
    });
  });

  describe('indices', () => {
    // >>> np.indices((2, 3))
    // array([[[0, 0, 0],
    //         [1, 1, 1]],
    //        [[0, 1, 2],
    //         [0, 1, 2]]])
    it('indices (2,3)', async () => {
      const B = getBackend();
      const [rows, cols] = B.indices([2, 3]);
      const rd = await getData(rows, B);
      const cd = await getData(cols, B);
      expect(rd).toEqual([0, 0, 0, 1, 1, 1]);
      expect(cd).toEqual([0, 1, 2, 0, 1, 2]);
      expect(rows.shape).toEqual([2, 3]);
      expect(cols.shape).toEqual([2, 3]);
    });
  });

  describe('diagIndices', () => {
    // >>> np.diag_indices(3)
    // (array([0, 1, 2]), array([0, 1, 2]))
    it('diagIndices(3)', async () => {
      const B = getBackend();
      const result = B.diagIndices(3);
      expect(result.length).toBe(2);
      const d0 = await getData(result[0], B);
      const d1 = await getData(result[1], B);
      expect(d0).toEqual([0, 1, 2]);
      expect(d1).toEqual([0, 1, 2]);
    });
  });

  describe('trilIndices / triuIndices', () => {
    // >>> np.tril_indices(3)
    // (array([0, 1, 1, 2, 2, 2]), array([0, 0, 1, 0, 1, 2]))
    it('trilIndices(3)', async () => {
      const B = getBackend();
      const [rows, cols] = B.trilIndices(3);
      const rd = await getData(rows, B);
      const cd = await getData(cols, B);
      expect(rd).toEqual([0, 1, 1, 2, 2, 2]);
      expect(cd).toEqual([0, 0, 1, 0, 1, 2]);
    });

    // >>> np.triu_indices(3)
    // (array([0, 0, 0, 1, 1, 2]), array([0, 1, 2, 1, 2, 2]))
    it('triuIndices(3)', async () => {
      const B = getBackend();
      const [rows, cols] = B.triuIndices(3);
      const rd = await getData(rows, B);
      const cd = await getData(cols, B);
      expect(rd).toEqual([0, 0, 0, 1, 1, 2]);
      expect(cd).toEqual([0, 1, 2, 1, 2, 2]);
    });
  });

  describe('window functions', () => {
    // >>> np.bartlett(5)
    // array([0. , 0.5, 1. , 0.5, 0. ])
    it('bartlett(5)', async () => {
      const B = getBackend();
      const r = B.bartlett(5);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 0, tol)).toBe(true);
      expect(approxEq(d[1], 0.5, tol)).toBe(true);
      expect(approxEq(d[2], 1, tol)).toBe(true);
      expect(approxEq(d[3], 0.5, tol)).toBe(true);
      expect(approxEq(d[4], 0, tol)).toBe(true);
    });

    // >>> np.hamming(5)
    // array([0.08, 0.54, 1.  , 0.54, 0.08])
    it('hamming(5)', async () => {
      const B = getBackend();
      const r = B.hamming(5);
      const d = await getData(r, B);
      const tol = getTol(B, true);
      expect(approxEq(d[0], 0.08, tol)).toBe(true);
      expect(approxEq(d[2], 1.0, tol)).toBe(true);
      expect(approxEq(d[4], 0.08, tol)).toBe(true);
    });

    // >>> np.hanning(5)
    // array([0. , 0.5, 1. , 0.5, 0. ])
    it('hanning(5)', async () => {
      const B = getBackend();
      const r = B.hanning(5);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 0, tol)).toBe(true);
      expect(approxEq(d[1], 0.5, tol)).toBe(true);
      expect(approxEq(d[2], 1, tol)).toBe(true);
    });

    // >>> np.blackman(5)
    // array([-1.38777878e-17,  3.40000000e-01,  1.00000000e+00, ...])
    it('blackman(5)', async () => {
      const B = getBackend();
      const r = B.blackman(5);
      const d = await getData(r, B);
      const tol = getTol(B, true);
      expect(approxEq(d[0], 0, tol)).toBe(true);
      expect(approxEq(d[1], 0.34, tol)).toBe(true);
      expect(approxEq(d[2], 1, tol)).toBe(true);
    });

    // >>> np.kaiser(5, 14)
    // array([7.72686684e-06, 1.58631610e-01, 1.00000000e+00, ...])
    it('kaiser(5, 14)', async () => {
      const B = getBackend();
      const r = B.kaiser(5, 14);
      const d = await getData(r, B);
      const tol = getTol(B, true);
      // The center value should be 1.0
      expect(approxEq(d[2], 1.0, tol)).toBe(true);
      // Symmetric
      expect(approxEq(d[0], d[4], tol)).toBe(true);
      expect(approxEq(d[1], d[3], tol)).toBe(true);
    });

    // Edge case: window of 1
    it('hamming(1)', async () => {
      const B = getBackend();
      const r = B.hamming(1);
      const d = await getData(r, B);
      expect(d).toEqual([1]);
    });
  });

  describe('unravelIndex / ravelMultiIndex', () => {
    // >>> np.unravel_index(5, (2, 3))
    // (1, 2)
    it('unravelIndex scalar', async () => {
      const B = getBackend();
      const result = B.unravelIndex(5, [2, 3]);
      const d0 = await getData(result[0], B);
      const d1 = await getData(result[1], B);
      expect(d0).toEqual([1]);
      expect(d1).toEqual([2]);
    });

    // >>> np.unravel_index(np.array([0, 3, 5]), (2, 3))
    // (array([0, 1, 1]), array([0, 0, 2]))
    it('unravelIndex array', async () => {
      const B = getBackend();
      const result = B.unravelIndex(B.array([0, 3, 5]), [2, 3]);
      const d0 = await getData(result[0], B);
      const d1 = await getData(result[1], B);
      expect(d0).toEqual([0, 1, 1]);
      expect(d1).toEqual([0, 0, 2]);
    });

    // >>> np.ravel_multi_index([[0,1,1],[0,0,2]], (2, 3))
    // array([0, 3, 5])
    it('ravelMultiIndex', async () => {
      const B = getBackend();
      const rows = B.array([0, 1, 1]);
      const cols = B.array([0, 0, 2]);
      const r = B.ravelMultiIndex([rows, cols], [2, 3]);
      const d = await getData(r, B);
      expect(d).toEqual([0, 3, 5]);
    });
  });

  describe('packbits / unpackbits', () => {
    // >>> np.packbits(np.array([1,0,1,0,0,0,0,1], dtype=np.uint8))
    // array([161], dtype=uint8)  # 10100001 = 161
    it('packbits basic', async () => {
      const B = getBackend();
      const a = B.array([1, 0, 1, 0, 0, 0, 0, 1], undefined, 'uint8');
      const r = B.packbits(a);
      const d = await getData(r, B);
      expect(d).toEqual([161]);
    });

    // >>> np.unpackbits(np.array([161], dtype=np.uint8))
    // array([1, 0, 1, 0, 0, 0, 0, 1], dtype=uint8)
    it('unpackbits basic', async () => {
      const B = getBackend();
      const a = B.array([161], undefined, 'uint8');
      const r = B.unpackbits(a);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 1, 0, 0, 0, 0, 1]);
    });

    // Round trip
    it('packbits/unpackbits round trip', async () => {
      const B = getBackend();
      const original = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0];
      const a = B.array(original, undefined, 'uint8');
      const packed = B.packbits(a);
      const unpacked = B.unpackbits(packed);
      const d = await getData(unpacked, B);
      expect(d).toEqual(original);
    });
  });

  describe('eigvalsh', () => {
    // >>> np.linalg.eigvalsh(np.array([[2, 1], [1, 2]]))
    // array([1., 3.])
    it('eigvalsh symmetric 2x2', async () => {
      const B = getBackend();
      const a = B.array([2, 1, 1, 2], [2, 2]);
      const r = B.eigvalsh(a);
      const d = await getData(r, B);
      const tol = getTol(B, true);
      const sorted = [...d].sort((a, b) => a - b);
      expect(approxEq(sorted[0], 1, tol)).toBe(true);
      expect(approxEq(sorted[1], 3, tol)).toBe(true);
    });
  });

  describe('fftn / ifftn', () => {
    // For 1D input, fftn should match fft
    it('fftn 1D matches fft', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4]);
      const fftResult = B.fft(a);
      const fftnResult = B.fftn(a);
      const tol = getTol(B, true);
      const fftReal = await getData(fftResult.real, B);
      const fftnReal = await getData(fftnResult.real, B);
      for (let i = 0; i < fftReal.length; i++) {
        expect(approxEq(fftReal[i], fftnReal[i], tol)).toBe(true);
      }
    });

    // fftn/ifftn round trip
    it('fftn/ifftn round trip', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4]);
      const forward = B.fftn(a);
      const back = B.ifftn(forward.real, forward.imag);
      const d = await getData(back.real, B);
      const tol = getTol(B, true);
      const expected = [1, 2, 3, 4];
      for (let i = 0; i < expected.length; i++) {
        expect(approxEq(d[i], expected[i], tol)).toBe(true);
      }
    });
  });

  // ============ P0 Parity: NumPy-canonical names ============

  describe('NumPy-canonical arithmetic names', () => {
    // >>> np.subtract(np.array([10, 20, 30]), np.array([1, 2, 3]))
    // array([ 9, 18, 27])
    it('subtract', async () => {
      const B = getBackend();
      const a = B.array([10, 20, 30]);
      const b = B.array([1, 2, 3]);
      const r = B.subtract(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([9, 18, 27]);
    });

    // >>> np.multiply(np.array([1, 2, 3]), np.array([4, 5, 6]))
    // array([ 4, 10, 18])
    it('multiply', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const b = B.array([4, 5, 6]);
      const r = B.multiply(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([4, 10, 18]);
    });

    // >>> np.divide(np.array([10, 20, 30]), np.array([2, 4, 5]))
    // array([5., 5., 6.])
    it('divide', async () => {
      const B = getBackend();
      const a = B.array([10, 20, 30]);
      const b = B.array([2, 4, 5]);
      const r = B.divide(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([5, 5, 6]);
    });

    // >>> np.power(np.array([2, 3, 4]), np.array([2, 3, 2]))
    // array([ 4, 27, 16])
    it('power', async () => {
      const B = getBackend();
      const a = B.array([2, 3, 4]);
      const b = B.array([2, 3, 2]);
      const r = B.power(a, b);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 4, tol)).toBe(true);
      expect(approxEq(d[1], 27, tol)).toBe(true);
      expect(approxEq(d[2], 16, tol)).toBe(true);
    });

    // >>> np.negative(np.array([1, -2, 3]))
    // array([-1,  2, -3])
    it('negative', async () => {
      const B = getBackend();
      const a = B.array([1, -2, 3]);
      const r = B.negative(a);
      const d = await getData(r, B);
      expect(d).toEqual([-1, 2, -3]);
    });
  });

  describe('floor_divide', () => {
    // >>> np.floor_divide(np.array([7, 8, 9]), np.array([2, 3, 4]))
    // array([3, 2, 2])
    it('floorDivide element-wise', async () => {
      const B = getBackend();
      const a = B.array([7, 8, 9]);
      const b = B.array([2, 3, 4]);
      const r = B.floorDivide(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([3, 2, 2]);
    });

    // >>> np.floor_divide(np.array([7, -7, 7, -7]), np.array([3, 3, -3, -3]))
    // array([ 2, -3, -3,  2])
    it('floorDivide with negatives matches NumPy', async () => {
      const B = getBackend();
      const a = B.array([7, -7, 7, -7]);
      const b = B.array([3, 3, -3, -3]);
      const r = B.floorDivide(a, b);
      const d = await getData(r, B);
      expect(d).toEqual([2, -3, -3, 2]);
    });

    // >>> np.floor_divide(10, np.array([3, 4, 5]))
    // array([3, 2, 2])
    it('floorDivide scalar broadcasting', async () => {
      const B = getBackend();
      const b = B.array([3, 4, 5]);
      const r = B.floorDivide(10, b);
      const d = await getData(r, B);
      expect(d).toEqual([3, 2, 2]);
    });
  });

  describe('asarray', () => {
    // >>> np.asarray([1, 2, 3])
    // array([1., 2., 3.])
    it('from number[]', async () => {
      const B = getBackend();
      const r = B.asarray([1, 2, 3]);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3]);
      expect(r.shape).toEqual([3]);
    });

    // >>> np.asarray(5)
    // array(5)
    it('from scalar', async () => {
      const B = getBackend();
      const r = B.asarray(5);
      const d = await getData(r, B);
      expect(d).toEqual([5]);
    });

    // asarray with existing NDArray returns same object (no copy)
    it('no-copy when dtype matches', () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.asarray(a);
      expect(r).toBe(a);
    });

    // asarray with dtype cast
    it('dtype cast', async () => {
      const B = getBackend();
      const r = B.asarray([1, 2, 3], 'int32');
      expect(r.dtype).toBe('int32');
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3]);
    });
  });

  describe('var/std NumPy param order', () => {
    // >>> np.var(np.array([1, 2, 3, 4, 5]))
    // 2.0
    it('var scalar default ddof=0', () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4, 5]);
      const r = B.var(a);
      expect(r).toBe(2.0);
    });

    // >>> np.var(np.array([1, 2, 3, 4, 5]), ddof=1)
    // 2.5
    it('var scalar ddof=1 via third param', () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4, 5]);
      const r = B.var(a, null, 1);
      expect(r).toBe(2.5);
    });

    // >>> np.std(np.array([1, 2, 3, 4, 5]))
    // ~1.4142
    it('std scalar default ddof=0', () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4, 5]);
      const r = B.std(a) as number;
      const tol = getTol(B);
      expect(approxEq(r, Math.sqrt(2), tol)).toBe(true);
    });
  });

  // ============ P1 Parity: Enhanced signatures ============

  describe('eye with M and k params', () => {
    // >>> np.eye(3, 4)
    // array([[1., 0., 0., 0.],
    //        [0., 1., 0., 0.],
    //        [0., 0., 1., 0.]])
    it('eye with M != N', async () => {
      const B = getBackend();
      const r = B.eye(3, 4);
      const d = await getData(r, B);
      expect(r.shape).toEqual([3, 4]);
      expect(d).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    });

    // >>> np.eye(3, k=1)
    // array([[0., 1., 0.],
    //        [0., 0., 1.],
    //        [0., 0., 0.]])
    it('eye with k=1 (superdiagonal)', async () => {
      const B = getBackend();
      const r = B.eye(3, undefined, 1);
      const d = await getData(r, B);
      expect(r.shape).toEqual([3, 3]);
      expect(d).toEqual([0, 1, 0, 0, 0, 1, 0, 0, 0]);
    });

    // >>> np.eye(3, k=-1)
    // array([[0., 0., 0.],
    //        [1., 0., 0.],
    //        [0., 1., 0.]])
    it('eye with k=-1 (subdiagonal)', async () => {
      const B = getBackend();
      const r = B.eye(3, undefined, -1);
      const d = await getData(r, B);
      expect(r.shape).toEqual([3, 3]);
      expect(d).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    });
  });

  describe('linspace with endpoint', () => {
    // >>> np.linspace(0, 1, 5, endpoint=False)
    // array([0. , 0.2, 0.4, 0.6, 0.8])
    it('endpoint=false excludes stop', async () => {
      const B = getBackend();
      const r = B.linspace(0, 1, 5, false);
      const d = await getData(r, B);
      const tol = getTol(B);
      const expected = [0, 0.2, 0.4, 0.6, 0.8];
      for (let i = 0; i < 5; i++) {
        expect(approxEq(d[i], expected[i], tol)).toBe(true);
      }
    });
  });

  describe('round with decimals', () => {
    // >>> np.round_(np.array([1.234, 2.567, 3.891]), 2)
    // array([1.23, 2.57, 3.89])
    it('round to 2 decimals', async () => {
      const B = getBackend();
      const a = B.array([1.234, 2.567, 3.891]);
      const r = B.round(a, 2);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 1.23, tol)).toBe(true);
      expect(approxEq(d[1], 2.57, tol)).toBe(true);
      expect(approxEq(d[2], 3.89, tol)).toBe(true);
    });

    // >>> np.round_(np.array([123.4, 567.8]), -1)
    // array([120., 570.])
    it('round to negative decimals', async () => {
      const B = getBackend();
      const a = B.array([123.4, 567.8]);
      const r = B.round(a, -1);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 120, tol)).toBe(true);
      expect(approxEq(d[1], 570, tol)).toBe(true);
    });
  });

  describe('transpose with axes', () => {
    // >>> np.transpose(np.arange(24).reshape(2,3,4), (2,0,1)).shape
    // (4, 2, 3)
    it('3D transpose with axes permutation', async () => {
      const B = getBackend();
      const a = B.array(
        Array.from({ length: 24 }, (_, i) => i),
        [2, 3, 4]
      );
      const r = B.transpose(a, [2, 0, 1]);
      expect(r.shape).toEqual([4, 2, 3]);
    });
  });

  describe('clip with null bounds', () => {
    // >>> np.clip(np.array([-1, 0, 1, 5, 10]), None, 5)
    // array([-1,  0,  1,  5,  5])
    it('clip with null min', async () => {
      const B = getBackend();
      const a = B.array([-1, 0, 1, 5, 10]);
      const r = B.clip(a, null, 5);
      const d = await getData(r, B);
      expect(d).toEqual([-1, 0, 1, 5, 5]);
    });

    // >>> np.clip(np.array([-1, 0, 1, 5, 10]), 0, None)
    // array([ 0,  0,  1,  5, 10])
    it('clip with null max', async () => {
      const B = getBackend();
      const a = B.array([-1, 0, 1, 5, 10]);
      const r = B.clip(a, 0, null);
      const d = await getData(r, B);
      expect(d).toEqual([0, 0, 1, 5, 10]);
    });
  });

  describe('meshgrid indexing', () => {
    // >>> x = np.array([1, 2, 3])
    // >>> y = np.array([4, 5])
    // >>> X, Y = np.meshgrid(x, y, indexing='ij')
    // >>> X.shape
    // (3, 2)
    it('ij indexing swaps shape', async () => {
      const B = getBackend();
      const x = B.array([1, 2, 3]);
      const y = B.array([4, 5]);
      const [X, Y] = B.meshgrid(x, y, 'ij');
      expect(X.shape).toEqual([3, 2]);
      expect(Y.shape).toEqual([3, 2]);
    });
  });

  describe('norm with axis', () => {
    // >>> np.linalg.norm(np.array([[1, 2], [3, 4]]), axis=1)
    // array([2.23606798, 5.])
    it('norm along axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const r = B.norm(a, undefined, 1) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], Math.sqrt(5), tol)).toBe(true);
      expect(approxEq(d[1], 5, tol)).toBe(true);
    });
  });

  // ============ P0 Round 2: arange, nested array, shuffle, nan axis, unique flags ============

  describe('arange NumPy-style overloads', () => {
    // >>> np.arange(5)
    // array([0, 1, 2, 3, 4])
    it('arange(stop)', async () => {
      const B = getBackend();
      const r = B.arange(5);
      const d = await getData(r, B);
      expect(d).toEqual([0, 1, 2, 3, 4]);
    });

    // >>> np.arange(2, 5)
    // array([2, 3, 4])
    it('arange(start, stop)', async () => {
      const B = getBackend();
      const r = B.arange(2, 5);
      const d = await getData(r, B);
      expect(d).toEqual([2, 3, 4]);
    });

    // >>> np.arange(1, 10, 2)
    // array([1, 3, 5, 7, 9])
    it('arange(start, stop, step)', async () => {
      const B = getBackend();
      const r = B.arange(1, 10, 2);
      const d = await getData(r, B);
      expect(d).toEqual([1, 3, 5, 7, 9]);
    });
  });

  describe('nested array creation', () => {
    // >>> np.array([[1, 2], [3, 4]])
    // array([[1, 2], [3, 4]])  shape=(2, 2)
    it('array from 2D nested list', async () => {
      const B = getBackend();
      const r = B.array([
        [1, 2],
        [3, 4],
      ]);
      expect(r.shape).toEqual([2, 2]);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3, 4]);
    });

    // >>> np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    // shape=(2, 2, 2)
    it('array from 3D nested list', async () => {
      const B = getBackend();
      const r = B.array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(r.shape).toEqual([2, 2, 2]);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });
  });

  describe('shuffle in-place', () => {
    it('shuffle modifies array in-place', async () => {
      const B = getBackend();
      B.seed(42);
      const a = B.arange(10);
      const originalData = [...(await getData(a, B))];
      B.shuffle(a); // void return
      const shuffledData = await getData(a, B);
      // All elements still present
      expect([...shuffledData].sort((a, b) => a - b)).toEqual(originalData.sort((a, b) => a - b));
    });
  });

  describe('nansum with axis', () => {
    // >>> np.nansum(np.array([[1, np.nan], [3, 4]]), axis=0)
    // array([4., 4.])
    it('nansum axis=0', async () => {
      const B = getBackend();
      const a = B.array([1, NaN, 3, 4], [2, 2]);
      const r = B.nansum(a, 0) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 4, tol)).toBe(true);
      expect(approxEq(d[1], 4, tol)).toBe(true);
    });
  });

  describe('nanmean with axis', () => {
    // >>> np.nanmean(np.array([[1, np.nan], [3, 4]]), axis=1)
    // array([1., 3.5])
    it('nanmean axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, NaN, 3, 4], [2, 2]);
      const r = B.nanmean(a, 1) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 1, tol)).toBe(true);
      expect(approxEq(d[1], 3.5, tol)).toBe(true);
    });
  });

  describe('median with axis', () => {
    // >>> np.median(np.array([[1, 2, 3], [4, 5, 6]]), axis=1)
    // array([2., 5.])
    it('median axis=1', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
      const r = B.median(a, 1) as NDArray;
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 2, tol)).toBe(true);
      expect(approxEq(d[1], 5, tol)).toBe(true);
    });
  });

  describe('unique with return flags', () => {
    // >>> np.unique([3, 1, 2, 1, 3], return_counts=True)
    // (array([1, 2, 3]), array([2, 1, 2]))
    it('unique with returnCounts', async () => {
      const B = getBackend();
      const a = B.array([3, 1, 2, 1, 3]);
      const result = B.unique(a, false, false, true) as { values: NDArray; counts?: NDArray };
      const vals = await getData(result.values, B);
      const counts = await getData(result.counts!, B);
      expect(vals).toEqual([1, 2, 3]);
      expect(counts).toEqual([2, 1, 2]);
    });

    // >>> np.unique([3, 1, 2, 1, 3], return_inverse=True)
    // (array([1, 2, 3]), array([2, 0, 1, 0, 2]))
    it('unique with returnInverse', async () => {
      const B = getBackend();
      const a = B.array([3, 1, 2, 1, 3]);
      const result = B.unique(a, false, true) as { values: NDArray; inverse?: NDArray };
      const vals = await getData(result.values, B);
      const inv = await getData(result.inverse!, B);
      expect(vals).toEqual([1, 2, 3]);
      expect(inv).toEqual([2, 0, 1, 0, 2]);
    });
  });
}
