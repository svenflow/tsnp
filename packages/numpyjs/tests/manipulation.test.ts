/**
 * Array manipulation and high-priority ops tests
 * Covers: zeros_like, broadcast_to, swapaxes, where, einsum, batched matmul, etc.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, arraysApproxEq } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

export function manipulationTests(getBackend: () => Backend) {
  describe('manipulation', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    // ============ Like Functions ============

    describe('zerosLike', () => {
      it('creates zeros with same shape', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const zeros = B.zerosLike(arr);
        expect(zeros.shape).toEqual([2, 3]);
        const data = await getData(zeros, B);
        expect(data.every(x => x === 0)).toBe(true);
      });
    });

    describe('onesLike', () => {
      it('creates ones with same shape', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const ones = B.onesLike(arr);
        expect(ones.shape).toEqual([2, 3]);
        const data = await getData(ones, B);
        expect(data.every(x => x === 1)).toBe(true);
      });
    });

    describe('fullLike', () => {
      it('creates full array with same shape', async () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const full = B.fullLike(arr, 7.5);
        expect(full.shape).toEqual([2, 2]);
        const data = await getData(full, B);
        expect(data.every(x => x === 7.5)).toBe(true);
      });
    });

    // ============ Broadcasting ============

    describe('broadcastTo', () => {
      it('broadcasts scalar to shape', async () => {
        const arr = B.array([5], [1]);
        const result = B.broadcastTo(arr, [3]);
        expect(result.shape).toEqual([3]);
        expect(await getData(result, B)).toEqual([5, 5, 5]);
      });

      it('broadcasts 1D to 2D', async () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.broadcastTo(arr, [2, 3]);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 1, 2, 3]);
      });

      it('broadcasts with leading dimensions', async () => {
        const arr = B.array([1, 2], [1, 2]);
        const result = B.broadcastTo(arr, [3, 2]);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 2, 1, 2, 1, 2]);
      });
    });

    describe('broadcastArrays', () => {
      it('broadcasts two arrays to common shape', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4], [1]);
        const [aBcast, bBcast] = B.broadcastArrays(a, b);
        expect(aBcast.shape).toEqual([3]);
        expect(bBcast.shape).toEqual([3]);
        expect(await getData(aBcast, B)).toEqual([1, 2, 3]);
        expect(await getData(bBcast, B)).toEqual([4, 4, 4]);
      });

      it('broadcasts 1D and 2D arrays', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([10, 20], [2, 1]);
        const [aBcast, bBcast] = B.broadcastArrays(a, b);
        expect(aBcast.shape).toEqual([2, 3]);
        expect(bBcast.shape).toEqual([2, 3]);
      });
    });

    // ============ Shape Manipulation ============

    describe('swapaxes', () => {
      it('swaps axes in 2D array', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.swapaxes(arr, 0, 1);
        expect(result.shape).toEqual([3, 2]);
        // Original: [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        expect(await getData(result, B)).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('handles negative axis', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.swapaxes(arr, 0, -1);
        expect(result.shape).toEqual([3, 2]);
      });

      it('same axis returns copy', async () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.swapaxes(arr, 0, 0);
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4]);
      });
    });

    describe('moveaxis', () => {
      it('moves axis to new position', async () => {
        const arr = B.array(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
        const result = B.moveaxis(arr, 0, -1);
        expect(result.shape).toEqual([3, 4, 2]);
      });

      it('moves axis forward', async () => {
        const arr = B.array(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
        const result = B.moveaxis(arr, 2, 0);
        expect(result.shape).toEqual([4, 2, 3]);
      });
    });

    describe('squeeze', () => {
      it('removes all size-1 dimensions', async () => {
        const arr = B.array([1, 2, 3], [1, 3, 1]);
        const result = B.squeeze(arr);
        expect(result.shape).toEqual([3]);
      });

      it('removes specific axis', async () => {
        const arr = B.array([1, 2, 3], [1, 3]);
        const result = B.squeeze(arr, 0);
        expect(result.shape).toEqual([3]);
      });

      it('handles negative axis', async () => {
        const arr = B.array([1, 2, 3], [3, 1]);
        const result = B.squeeze(arr, -1);
        expect(result.shape).toEqual([3]);
      });
    });

    describe('expandDims', () => {
      it('adds dimension at start', async () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, 0);
        expect(result.shape).toEqual([1, 3]);
      });

      it('adds dimension at end', async () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, 1);
        expect(result.shape).toEqual([3, 1]);
      });

      it('handles negative axis', async () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, -1);
        expect(result.shape).toEqual([3, 1]);
      });
    });

    describe('reshape', () => {
      it('reshapes to new shape', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const result = B.reshape(arr, [2, 3]);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('handles -1 dimension inference', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const result = B.reshape(arr, [2, -1]);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('flatten', () => {
      it('flattens 2D array', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.flatten(arr);
        expect(result.shape).toEqual([6]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6]);
      });
    });

    describe('concatenate', () => {
      it('concatenates along axis 0', async () => {
        const a = B.array([1, 2], [2]);
        const b = B.array([3, 4, 5], [3]);
        const result = B.concatenate([a, b], 0);
        expect(result.shape).toEqual([5]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5]);
      });

      it('concatenates 2D arrays along axis 0', async () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6], [1, 2]);
        const result = B.concatenate([a, b], 0);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('concatenates along axis 1', async () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6], [2, 1]);
        const result = B.concatenate([a, b], 1);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([1, 2, 5, 3, 4, 6]);
      });
    });

    describe('stack', () => {
      it('stacks along axis 0', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.stack([a, b], 0);
        expect(result.shape).toEqual([2, 3]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('stacks along axis 1', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.stack([a, b], 1);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 4, 2, 5, 3, 6]);
      });
    });

    describe('split', () => {
      it('splits into equal parts', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const parts = B.split(arr, 3, 0);
        expect(parts.length).toBe(3);
        expect(await getData(parts[0], B)).toEqual([1, 2]);
        expect(await getData(parts[1], B)).toEqual([3, 4]);
        expect(await getData(parts[2], B)).toEqual([5, 6]);
      });

      it('splits at indices', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const parts = B.split(arr, [2, 4], 0);
        expect(parts.length).toBe(3);
        expect(await getData(parts[0], B)).toEqual([1, 2]);
        expect(await getData(parts[1], B)).toEqual([3, 4]);
        expect(await getData(parts[2], B)).toEqual([5, 6]);
      });
    });

    // ============ Conditional ============

    describe('where', () => {
      it('selects based on condition', async () => {
        const cond = B.array([1, 0, 1, 0], [4]); // truthy, falsy, truthy, falsy
        const x = B.array([1, 2, 3, 4], [4]);
        const y = B.array([10, 20, 30, 40], [4]);
        const result = B.where(cond, x, y);
        expect(await getData(result, B)).toEqual([1, 20, 3, 40]);
      });

      it('broadcasts condition', async () => {
        const cond = B.array([1, 0], [2]);
        const x = B.array([1, 2, 3, 4, 5, 6], [3, 2]);
        const y = B.array([10, 20, 30, 40, 50, 60], [3, 2]);
        const result = B.where(cond, x, y);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 20, 3, 40, 5, 60]);
      });
    });

    // ============ Advanced Indexing ============

    describe('take', () => {
      it('takes elements by indices', async () => {
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.take(arr, [0, 2, 4]);
        expect(await getData(result, B)).toEqual([10, 30, 50]);
      });

      it('takes along axis', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.take(arr, [0, 2], 1);
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([1, 3, 4, 6]);
      });

      it('handles negative indices', async () => {
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.take(arr, [-1, -2]);
        expect(await getData(result, B)).toEqual([50, 40]);
      });
    });

    // ============ Batched Matmul ============

    describe('batchedMatmul', () => {
      it('performs batched matrix multiplication', async () => {
        // Two 2x2 matrices in batch
        const a = B.array([
          1, 2, 3, 4,  // First 2x2
          5, 6, 7, 8   // Second 2x2
        ], [2, 2, 2]);
        const b = B.array([
          1, 0, 0, 1,  // Identity
          1, 0, 0, 1   // Identity
        ], [2, 2, 2]);
        const result = B.batchedMatmul(a, b);
        expect(result.shape).toEqual([2, 2, 2]);
        // A @ I = A
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      });

      it('broadcasts batch dimensions', async () => {
        // (2, 2, 2) @ (2, 2) -> (2, 2, 2)
        const a = B.array([
          1, 2, 3, 4,
          5, 6, 7, 8
        ], [2, 2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]); // Single 2x2 identity
        const result = B.batchedMatmul(a, b);
        expect(result.shape).toEqual([2, 2, 2]);
      });
    });

    // ============ Einstein Summation ============

    describe('einsum', () => {
      it('computes matrix multiplication: ij,jk->ik', async () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]);
        const result = B.einsum('ij,jk->ik', a, b);
        expect(result.shape).toEqual([2, 2]);
        // A @ I = A
        expect(await getData(result, B)).toEqual([1, 2, 3, 4]);
      });

      it('computes trace: ii->', async () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.einsum('ii->', a);
        expect(await getData(result, B)).toEqual([5]); // 1 + 4
      });

      it('computes transpose: ij->ji', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.einsum('ij->ji', a);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('computes outer product: i,j->ij', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5], [2]);
        const result = B.einsum('i,j->ij', a, b);
        expect(result.shape).toEqual([3, 2]);
        expect(await getData(result, B)).toEqual([4, 5, 8, 10, 12, 15]);
      });

      it('computes dot product: i,i->', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.einsum('i,i->', a, b);
        expect(await getData(result, B)).toEqual([32]); // 1*4 + 2*5 + 3*6
      });

      it('computes batch matmul: bij,bjk->bik', async () => {
        // Two 2x2 matrices in batch
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const b = B.array([1, 0, 0, 1, 1, 0, 0, 1], [2, 2, 2]); // Two identity matrices
        const result = B.einsum('bij,bjk->bik', a, b);
        expect(result.shape).toEqual([2, 2, 2]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      });

      it('computes element-wise multiply and sum: ij,ij->', async () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 1, 1, 1], [2, 2]);
        const result = B.einsum('ij,ij->', a, b);
        expect(await getData(result, B)).toEqual([10]); // 1+2+3+4
      });

      it('implicit output (sum repeated indices)', async () => {
        // ij,jk with no explicit output -> ik
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]);
        const result = B.einsum('ij,jk', a, b);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    // ============ Differences ============

    describe('diff', () => {
      it('computes first difference', async () => {
        const arr = B.array([1, 2, 4, 7, 11], [5]);
        const result = B.diff(arr);
        expect(result.shape).toEqual([4]);
        expect(await getData(result, B)).toEqual([1, 2, 3, 4]);
      });

      it('computes second difference', async () => {
        const arr = B.array([1, 2, 4, 7, 11], [5]);
        const result = B.diff(arr, 2);
        expect(result.shape).toEqual([3]);
        expect(await getData(result, B)).toEqual([1, 1, 1]);
      });

      it('computes diff along axis', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.diff(arr, 1, 1);
        expect(result.shape).toEqual([2, 2]);
        expect(await getData(result, B)).toEqual([1, 1, 1, 1]);
      });
    });

    describe('gradient', () => {
      it('computes gradient of 1D array', async () => {
        const arr = B.array([1, 2, 4, 7, 11], [5]);
        const result = B.gradient(arr);
        expect(result.shape).toEqual([5]);
        // Forward: 2-1=1, Central: (4-1)/2=1.5, (7-2)/2=2.5, (11-4)/2=3.5, Backward: 11-7=4
        const data = await getData(result, B);
        expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 3.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], 4, DEFAULT_TOL)).toBe(true);
      });
    });

    describe('ediff1d', () => {
      it('computes differences on flattened array', async () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.ediff1d(arr);
        expect(result.shape).toEqual([5]);
        expect(await getData(result, B)).toEqual([1, 1, 1, 1, 1]);
      });
    });

    // ============ Cross Product ============

    describe('cross', () => {
      it('computes 3D cross product', async () => {
        const a = B.array([1, 0, 0], [3]);
        const b = B.array([0, 1, 0], [3]);
        const result = B.cross(a, b);
        expect(result.shape).toEqual([3]);
        expect(await getData(result, B)).toEqual([0, 0, 1]);
      });

      it('computes another cross product', async () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.cross(a, b);
        // (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
        expect(await getData(result, B)).toEqual([-3, 6, -3]);
      });
    });

    // ============ Statistics ============

    describe('cov', () => {
      it('computes covariance matrix', async () => {
        // Two variables, 4 observations each
        const x = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        const result = B.cov(x);
        expect(result.shape).toEqual([2, 2]);
        // Each row has variance 5/3 (sample var)
        const data = await getData(result, B);
        expect(approxEq(data[0], 5 / 3, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 5 / 3, RELAXED_TOL)).toBe(true);
      });

      it('computes covariance between two vectors', async () => {
        const x = B.array([1, 2, 3], [3]);
        const y = B.array([4, 5, 6], [3]);
        const result = B.cov(x, y);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    describe('corrcoef', () => {
      it('computes correlation coefficients', async () => {
        const x = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        const result = B.corrcoef(x);
        expect(result.shape).toEqual([2, 2]);
        // Diagonal should be 1
        const data = await getData(result, B);
        expect(approxEq(data[0], 1, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ Convolution ============

    describe('convolve', () => {
      it('computes full convolution', async () => {
        const a = B.array([1, 2, 3], [3]);
        const v = B.array([0, 1, 0.5], [3]);
        const result = B.convolve(a, v, 'full');
        expect(result.shape).toEqual([5]);
        // [0*1, 1*1+0*2, 0.5*1+1*2+0*3, 0.5*2+1*3, 0.5*3] = [0, 1, 2.5, 4, 1.5]
        const data = await getData(result, B);
        expect(approxEq(data[0], 0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.5, DEFAULT_TOL)).toBe(true);
      });

      it('computes same mode convolution', async () => {
        const a = B.array([1, 2, 3, 4, 5], [5]);
        const v = B.array([1, 1, 1], [3]);
        const result = B.convolve(a, v, 'same');
        expect(result.shape).toEqual([5]);
      });

      it('computes valid mode convolution', async () => {
        const a = B.array([1, 2, 3, 4, 5], [5]);
        const v = B.array([1, 1, 1], [3]);
        const result = B.convolve(a, v, 'valid');
        expect(result.shape).toEqual([3]);
        expect(await getData(result, B)).toEqual([6, 9, 12]);
      });
    });

    describe('correlate', () => {
      it('computes correlation', async () => {
        const a = B.array([1, 2, 3, 4, 5], [5]);
        const v = B.array([1, 1, 1], [3]);
        const result = B.correlate(a, v, 'valid');
        expect(result.shape).toEqual([3]);
        expect(await getData(result, B)).toEqual([6, 9, 12]);
      });
    });

    // ============ Matrix Creation ============

    describe('identity', () => {
      it('creates identity matrix', async () => {
        const result = B.identity(3);
        expect(result.shape).toEqual([3, 3]);
        expect(await getData(result, B)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      });
    });

    describe('tril', () => {
      it('extracts lower triangle', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
        const result = B.tril(arr);
        expect(result.toArray()).toEqual([1, 0, 0, 4, 5, 0, 7, 8, 9]);
      });

      it('extracts lower triangle with k=1', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
        const result = B.tril(arr, 1);
        expect(result.toArray()).toEqual([1, 2, 0, 4, 5, 6, 7, 8, 9]);
      });
    });

    describe('triu', () => {
      it('extracts upper triangle', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
        const result = B.triu(arr);
        expect(result.toArray()).toEqual([1, 2, 3, 0, 5, 6, 0, 0, 9]);
      });

      it('extracts upper triangle with k=-1', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
        const result = B.triu(arr, -1);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6, 0, 8, 9]);
      });
    });

    // ============ Grid Creation ============

    describe('meshgrid', () => {
      it('creates coordinate matrices', () => {
        const x = B.array([1, 2, 3], [3]);
        const y = B.array([4, 5], [2]);
        const { X, Y } = B.meshgrid(x, y);
        expect(X.shape).toEqual([2, 3]);
        expect(Y.shape).toEqual([2, 3]);
        expect(X.toArray()).toEqual([1, 2, 3, 1, 2, 3]);
        expect(Y.toArray()).toEqual([4, 4, 4, 5, 5, 5]);
      });
    });

    describe('logspace', () => {
      it('creates log-spaced array', () => {
        const result = B.logspace(0, 2, 3);
        expect(result.shape).toEqual([3]);
        const data = result.toArray();
        expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 10, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 100, DEFAULT_TOL)).toBe(true);
      });

      it('uses custom base', () => {
        const result = B.logspace(0, 3, 4, 2);
        const data = result.toArray();
        expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 2, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 4, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 8, DEFAULT_TOL)).toBe(true);
      });
    });

    describe('geomspace', () => {
      it('creates geometrically-spaced array', () => {
        const result = B.geomspace(1, 1000, 4);
        expect(result.shape).toEqual([4]);
        const data = result.toArray();
        expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 10, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 100, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 1000, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ Stacking Shortcuts ============

    describe('vstack', () => {
      it('stacks 1D arrays vertically', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.vstack([a, b]);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('stacks 2D arrays vertically', () => {
        const a = B.array([1, 2], [1, 2]);
        const b = B.array([3, 4], [1, 2]);
        const result = B.vstack([a, b]);
        expect(result.shape).toEqual([2, 2]);
      });
    });

    describe('hstack', () => {
      it('stacks 1D arrays horizontally', () => {
        const a = B.array([1, 2], [2]);
        const b = B.array([3, 4, 5], [3]);
        const result = B.hstack([a, b]);
        expect(result.shape).toEqual([5]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
      });

      it('stacks 2D arrays horizontally', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6], [2, 1]);
        const result = B.hstack([a, b]);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('dstack', () => {
      it('stacks arrays along third axis', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6, 7, 8], [2, 2]);
        const result = B.dstack([a, b]);
        expect(result.shape).toEqual([2, 2, 2]);
      });
    });

    // ============ Split Shortcuts ============

    describe('vsplit', () => {
      it('splits array vertically', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [3, 2]);
        const parts = B.vsplit(arr, 3);
        expect(parts.length).toBe(3);
        expect(parts[0].shape).toEqual([1, 2]);
      });
    });

    describe('hsplit', () => {
      it('splits array horizontally', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const parts = B.hsplit(arr, 3);
        expect(parts.length).toBe(3);
        expect(parts[0].shape).toEqual([2, 1]);
      });

      it('splits 1D array', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const parts = B.hsplit(arr, 3);
        expect(parts.length).toBe(3);
        expect(parts[0].toArray()).toEqual([1, 2]);
      });
    });

    describe('dsplit', () => {
      it('splits array along third axis', () => {
        const arr = B.array(Array.from({ length: 12 }, (_, i) => i), [2, 2, 3]);
        const parts = B.dsplit(arr, 3);
        expect(parts.length).toBe(3);
        expect(parts[0].shape).toEqual([2, 2, 1]);
      });
    });

    // ============ Array Replication ============

    describe('tile', () => {
      it('tiles array', () => {
        const arr = B.array([1, 2], [2]);
        const result = B.tile(arr, 3);
        expect(result.shape).toEqual([6]);
        expect(result.toArray()).toEqual([1, 2, 1, 2, 1, 2]);
      });

      it('tiles array with multiple reps', () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.tile(arr, [2, 3]);
        expect(result.shape).toEqual([4, 6]);
      });
    });

    describe('repeat', () => {
      it('repeats elements', () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.repeat(arr, 2);
        expect(result.shape).toEqual([6]);
        expect(result.toArray()).toEqual([1, 1, 2, 2, 3, 3]);
      });

      it('repeats along axis', () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.repeat(arr, 2, 0);
        expect(result.shape).toEqual([4, 2]);
      });
    });

    // ============ Index Finding ============

    describe('nonzero', () => {
      it('finds non-zero indices', () => {
        const arr = B.array([0, 1, 0, 2, 3, 0], [6]);
        const result = B.nonzero(arr);
        expect(result.length).toBe(1);
        expect(result[0].toArray()).toEqual([1, 3, 4]);
      });

      it('finds non-zero indices in 2D', () => {
        const arr = B.array([1, 0, 0, 2, 0, 3], [2, 3]);
        const result = B.nonzero(arr);
        expect(result.length).toBe(2);
        expect(result[0].toArray()).toEqual([0, 1, 1]); // row indices
        expect(result[1].toArray()).toEqual([0, 0, 2]); // col indices
      });
    });

    describe('argwhere', () => {
      it('returns indices as rows', () => {
        const arr = B.array([1, 0, 0, 2, 0, 3], [2, 3]);
        const result = B.argwhere(arr);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([0, 0, 1, 0, 1, 2]);
      });
    });

    describe('flatnonzero', () => {
      it('returns flat indices', () => {
        const arr = B.array([0, 1, 0, 2, 3, 0], [6]);
        const result = B.flatnonzero(arr);
        expect(result.toArray()).toEqual([1, 3, 4]);
      });
    });

    // ============ Value Handling ============

    describe('nanToNum', () => {
      it('replaces NaN with 0', () => {
        const arr = B.array([1, NaN, 3], [3]);
        const result = B.nanToNum(arr);
        expect(result.toArray()).toEqual([1, 0, 3]);
      });

      it('replaces Inf values', () => {
        const arr = B.array([1, Infinity, -Infinity], [3]);
        const result = B.nanToNum(arr, 0, 999, -999);
        expect(result.toArray()).toEqual([1, 999, -999]);
      });
    });

    // ============ Sorting ============

    describe('sort', () => {
      it('sorts 1D array', () => {
        const arr = B.array([3, 1, 4, 1, 5, 9, 2, 6], [8]);
        const result = B.sort(arr);
        expect(result.toArray()).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
      });

      it('sorts 2D array along axis', () => {
        const arr = B.array([3, 1, 2, 6, 5, 4], [2, 3]);
        const result = B.sort(arr, 1);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('handles NaN (sorts to end)', () => {
        const arr = B.array([3, NaN, 1, 2], [4]);
        const result = B.sort(arr);
        const data = result.toArray();
        expect(data[0]).toBe(1);
        expect(data[1]).toBe(2);
        expect(data[2]).toBe(3);
        expect(Number.isNaN(data[3])).toBe(true);
      });
    });

    describe('argsort', () => {
      it('returns sort indices', () => {
        const arr = B.array([3, 1, 2], [3]);
        const result = B.argsort(arr);
        expect(result.toArray()).toEqual([1, 2, 0]);
      });

      it('returns sort indices for 2D along axis', () => {
        const arr = B.array([3, 1, 2, 6, 5, 4], [2, 3]);
        const result = B.argsort(arr, 1);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 0, 2, 1, 0]);
      });
    });

    describe('searchsorted', () => {
      it('finds insertion points', () => {
        const arr = B.array([1, 2, 3, 4, 5], [5]);
        expect(B.searchsorted(arr, 3)).toBe(2);
        expect(B.searchsorted(arr, 3, 'right')).toBe(3);
      });

      it('finds insertion points for array', () => {
        const arr = B.array([1, 2, 3, 4, 5], [5]);
        const v = B.array([0, 3, 6], [3]);
        const result = B.searchsorted(arr, v) as any;
        expect(result.toArray()).toEqual([0, 2, 5]);
      });
    });

    describe('unique', () => {
      it('returns unique values sorted', () => {
        const arr = B.array([3, 1, 2, 1, 3, 2, 4], [7]);
        const result = B.unique(arr);
        expect(result.toArray()).toEqual([1, 2, 3, 4]);
      });
    });
  });
}
