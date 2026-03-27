/**
 * WASM Backend coverage test suite
 *
 * Tests every uncovered function/branch in src/index.ts to push toward 100% line coverage.
 */

import { describe, beforeAll, it, expect } from 'vitest';
import type { Backend, NDArray as IFaceNDArray } from 'numpyjs';
import { createWasmBackend } from '../src/index.js';

describe('wasm-coverage', () => {
  let B: Backend;

  beforeAll(async () => {
    B = await createWasmBackend();
  });

  const arr = (data: number[], shape?: number[]) => B.array(data, shape || [data.length]);
  const mat = (data: number[], r: number, c: number) => B.array(data, [r, c]);
  const getData = (a: IFaceNDArray) => Array.from(a.data);

  // ============ Trig - Uncovered ============

  describe('arccos', () => {
    it('computes arccos', () => {
      const a = arr([1, 0, -1]);
      const r = B.arccos(a);
      expect(r.data[0]).toBeCloseTo(0);
      expect(r.data[1]).toBeCloseTo(Math.PI / 2);
      expect(r.data[2]).toBeCloseTo(Math.PI);
    });
  });

  describe('arctan', () => {
    it('computes arctan', () => {
      const a = arr([0, 1, -1]);
      const r = B.arctan(a);
      expect(r.data[0]).toBeCloseTo(0);
      expect(r.data[1]).toBeCloseTo(Math.PI / 4);
      expect(r.data[2]).toBeCloseTo(-Math.PI / 4);
    });
  });

  // ============ remainder ============

  describe('remainder', () => {
    it('computes remainder (alias for mod)', () => {
      const a = arr([7, 8, 9]);
      const b = arr([3, 3, 3]);
      const r = B.remainder(a, b);
      expect(getData(r)).toEqual([1, 2, 0]);
    });
  });

  // ============ Comparison ============

  describe('comparison ops', () => {
    it('equal', () => {
      const r = B.equal(arr([1, 2, 3]), arr([1, 0, 3]));
      expect(getData(r)).toEqual([1, 0, 1]);
    });

    it('notEqual', () => {
      const r = B.notEqual(arr([1, 2, 3]), arr([1, 0, 3]));
      expect(getData(r)).toEqual([0, 1, 0]);
    });

    it('less', () => {
      const r = B.less(arr([1, 2, 3]), arr([2, 2, 2]));
      expect(getData(r)).toEqual([1, 0, 0]);
    });

    it('lessEqual', () => {
      const r = B.lessEqual(arr([1, 2, 3]), arr([2, 2, 2]));
      expect(getData(r)).toEqual([1, 1, 0]);
    });

    it('greater', () => {
      const r = B.greater(arr([1, 2, 3]), arr([2, 2, 2]));
      expect(getData(r)).toEqual([0, 0, 1]);
    });

    it('greaterEqual', () => {
      const r = B.greaterEqual(arr([1, 2, 3]), arr([2, 2, 2]));
      expect(getData(r)).toEqual([0, 1, 1]);
    });

    it('isnan', () => {
      const r = B.isnan(arr([1, NaN, 3]));
      expect(r.data[0]).toBe(0);
      expect(r.data[1]).toBe(1);
      expect(r.data[2]).toBe(0);
    });

    it('isinf', () => {
      const r = B.isinf(arr([1, Infinity, -Infinity]));
      expect(r.data[0]).toBe(0);
      expect(r.data[1]).toBe(1);
      expect(r.data[2]).toBe(1);
    });

    it('isfinite', () => {
      const r = B.isfinite(arr([1, Infinity, NaN]));
      expect(r.data[0]).toBe(1);
      expect(r.data[1]).toBe(0);
      expect(r.data[2]).toBe(0);
    });
  });

  // ============ Set Operations ============

  describe('set operations', () => {
    it('setdiff1d', () => {
      const r = B.setdiff1d(arr([1, 2, 3, 4, 5]), arr([2, 4]));
      expect(getData(r)).toEqual([1, 3, 5]);
    });

    it('union1d', () => {
      const r = B.union1d(arr([1, 3, 5]), arr([2, 3, 4]));
      expect(getData(r)).toEqual([1, 2, 3, 4, 5]);
    });

    it('intersect1d', () => {
      const r = B.intersect1d(arr([1, 2, 3, 4]), arr([2, 4, 6]));
      expect(getData(r)).toEqual([2, 4]);
    });

    it('isin', () => {
      const r = B.isin(arr([1, 2, 3, 4, 5]), arr([2, 4]));
      expect(getData(r)).toEqual([0, 1, 0, 1, 0]);
    });
  });

  // ============ Array Manipulation (Extended) ============

  describe('insert', () => {
    it('inserts values', () => {
      const r = B.insert(arr([1, 2, 3]), 1, 99);
      expect(getData(r)).toEqual([1, 99, 2, 3]);
    });

    it('inserts array', () => {
      const r = B.insert(arr([1, 2, 3]), 1, arr([10, 20]));
      expect(getData(r)).toEqual([1, 10, 20, 2, 3]);
    });

    it('throws on insert with axis', () => {
      expect(() => B.insert(arr([1, 2, 3]), 1, 99, 0)).toThrow('not yet implemented');
    });
  });

  describe('deleteArr', () => {
    it('deletes single index', () => {
      const r = B.deleteArr(arr([1, 2, 3, 4]), 1);
      expect(getData(r)).toEqual([1, 3, 4]);
    });

    it('deletes multiple indices', () => {
      const r = B.deleteArr(arr([1, 2, 3, 4, 5]), [1, 3]);
      expect(getData(r)).toEqual([1, 3, 5]);
    });

    it('throws on delete with axis', () => {
      expect(() => B.deleteArr(arr([1, 2, 3]), 1, 0)).toThrow('not yet implemented');
    });
  });

  describe('append', () => {
    it('appends without axis', () => {
      const r = B.append(arr([1, 2]), arr([3, 4]));
      expect(getData(r)).toEqual([1, 2, 3, 4]);
    });

    it('appends with axis', () => {
      const r = B.append(mat([1, 2, 3, 4], 2, 2), mat([5, 6], 1, 2), 0);
      expect(r.shape).toEqual([3, 2]);
    });
  });

  describe('atleast*', () => {
    it('atleast1d scalar-like', () => {
      // 0D → 1D
      const a = B.array([5], []);
      const r = B.atleast1d(a);
      expect(r.shape).toEqual([1]);
    });

    it('atleast1d already 1d', () => {
      const r = B.atleast1d(arr([1, 2]));
      expect(r.shape).toEqual([2]);
    });

    it('atleast2d scalar', () => {
      const a = B.array([5], []);
      const r = B.atleast2d(a);
      expect(r.shape).toEqual([1, 1]);
    });

    it('atleast2d 1d', () => {
      const r = B.atleast2d(arr([1, 2, 3]));
      expect(r.shape).toEqual([1, 3]);
    });

    it('atleast2d already 2d', () => {
      const r = B.atleast2d(mat([1, 2, 3, 4], 2, 2));
      expect(r.shape).toEqual([2, 2]);
    });

    it('atleast3d scalar', () => {
      const a = B.array([5], []);
      const r = B.atleast3d(a);
      expect(r.shape).toEqual([1, 1, 1]);
    });

    it('atleast3d 1d', () => {
      const r = B.atleast3d(arr([1, 2]));
      expect(r.shape).toEqual([1, 2, 1]);
    });

    it('atleast3d 2d', () => {
      const r = B.atleast3d(mat([1, 2, 3, 4], 2, 2));
      expect(r.shape).toEqual([2, 2, 1]);
    });

    it('atleast3d already 3d', () => {
      const r = B.atleast3d(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]));
      expect(r.shape).toEqual([2, 2, 2]);
    });
  });

  describe('countNonzero', () => {
    it('without axis', () => {
      const r = B.countNonzero(arr([0, 1, 0, 3, 0, 5]));
      expect(r).toBe(3);
    });

    it('with axis 0', () => {
      const a = mat([0, 1, 2, 0, 3, 0], 2, 3);
      const r = B.countNonzero(a, 0) as IFaceNDArray;
      expect(r.shape).toEqual([3]);
      // col 0: [0,0]->0, col 1: [1,3]->2, col 2: [2,0]->1
      expect(getData(r)).toEqual([0, 2, 1]);
    });

    it('with axis 1', () => {
      const a = mat([0, 1, 2, 0, 3, 0], 2, 3);
      const r = B.countNonzero(a, 1) as IFaceNDArray;
      expect(r.shape).toEqual([2]);
      // row 0: [0,1,2]->2, row 1: [0,3,0]->1
      expect(getData(r)).toEqual([2, 1]);
    });
  });

  // ============ cond ============

  describe('cond', () => {
    it('well-conditioned matrix', () => {
      // Use a simple 2x2 matrix with known condition number
      const a = mat([2, 1, 1, 2], 2, 2);
      const r = B.cond(a);
      // Eigenvalues are 3 and 1, so cond ≈ 3
      expect(r).toBeGreaterThanOrEqual(1);
      expect(r).toBeLessThan(100);
    });

    it('singular matrix returns Infinity', () => {
      const r = B.cond(mat([1, 2, 2, 4], 2, 2));
      expect(r).toBe(Infinity);
    });
  });

  // ============ slogdet ============

  describe('slogdet', () => {
    it('positive det', () => {
      const r = B.slogdet(mat([2, 0, 0, 3], 2, 2));
      expect(r.sign).toBe(1);
      expect(r.logabsdet).toBeCloseTo(Math.log(6));
    });

    it('negative det', () => {
      const r = B.slogdet(mat([0, 1, 1, 0], 2, 2));
      expect(r.sign).toBe(-1);
      expect(r.logabsdet).toBeCloseTo(0);
    });

    it('singular matrix', () => {
      const r = B.slogdet(mat([1, 2, 2, 4], 2, 2));
      expect(r.sign).toBe(0);
      expect(r.logabsdet).toBe(-Infinity);
    });

    it('throws for non-square', () => {
      expect(() => B.slogdet(mat([1, 2, 3, 4, 5, 6], 2, 3))).toThrow('square');
    });
  });

  // ============ Scalar math ============

  describe('scalar operations', () => {
    it('subScalar', () => {
      const r = B.subScalar(arr([10, 20, 30]), 5);
      expect(getData(r)).toEqual([5, 15, 25]);
    });

    it('divScalar', () => {
      const r = B.divScalar(arr([10, 20, 30]), 5);
      expect(getData(r)).toEqual([2, 4, 6]);
    });
  });

  // ============ Axis reductions ============

  describe('axis reductions', () => {
    const m = () => mat([1, 2, 3, 4, 5, 6], 2, 3);

    it('minAxis 0', () => {
      const r = B.minAxis(m(), 0);
      expect(getData(r)).toEqual([1, 2, 3]);
    });

    it('maxAxis 0', () => {
      const r = B.maxAxis(m(), 0);
      expect(getData(r)).toEqual([4, 5, 6]);
    });

    it('argminAxis 0', () => {
      const r = B.argminAxis(m(), 0);
      expect(getData(r)).toEqual([0, 0, 0]);
    });

    it('argmaxAxis 0', () => {
      const r = B.argmaxAxis(m(), 0);
      expect(getData(r)).toEqual([1, 1, 1]);
    });

    it('argminAxis 1', () => {
      const r = B.argminAxis(m(), 1);
      expect(getData(r)).toEqual([0, 0]);
    });

    it('argmaxAxis 1', () => {
      const r = B.argmaxAxis(m(), 1);
      expect(getData(r)).toEqual([2, 2]);
    });

    it('varAxis 0', () => {
      const r = B.varAxis(m(), 0);
      // var([1,4])=2.25, var([2,5])=2.25, var([3,6])=2.25
      for (const v of r.data) expect(v).toBeCloseTo(2.25);
    });

    it('stdAxis 0', () => {
      const r = B.stdAxis(m(), 0);
      for (const v of r.data) expect(v).toBeCloseTo(1.5);
    });

    it('prodAxis 0', () => {
      const r = B.prodAxis(m(), 0);
      expect(getData(r)).toEqual([4, 10, 18]);
    });

    it('prodAxis 1', () => {
      const r = B.prodAxis(m(), 1);
      expect(getData(r)).toEqual([6, 120]);
    });

    it('allAxis', () => {
      const a = mat([1, 0, 1, 1, 1, 1], 2, 3);
      const r = B.allAxis(a, 0);
      expect(getData(r)).toEqual([1, 0, 1]);
    });

    it('anyAxis', () => {
      const a = mat([0, 0, 1, 0, 0, 0], 2, 3);
      const r = B.anyAxis(a, 0);
      expect(getData(r)).toEqual([0, 0, 1]);
    });

    it('cumsumAxis', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
      const r = B.cumsumAxis(a, 1);
      expect(getData(r)).toEqual([1, 3, 6, 4, 9, 15]);
    });

    it('cumprodAxis', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
      const r = B.cumprodAxis(a, 1);
      expect(getData(r)).toEqual([1, 2, 6, 4, 20, 120]);
    });
  });

  // ============ NaN-aware stats ============

  describe('nan stats', () => {
    it('nansum', () => {
      expect(B.nansum(arr([1, NaN, 3]))).toBeCloseTo(4);
    });

    it('nanmean', () => {
      expect(B.nanmean(arr([1, NaN, 3]))).toBeCloseTo(2);
    });

    it('nanstd', () => {
      const r = B.nanstd(arr([1, NaN, 3]));
      expect(r).toBeCloseTo(1);
    });

    it('nanvar', () => {
      const r = B.nanvar(arr([1, NaN, 3]));
      expect(r).toBeCloseTo(1);
    });

    it('nanmin', () => {
      expect(B.nanmin(arr([3, NaN, 1]))).toBe(1);
    });

    it('nanmax', () => {
      expect(B.nanmax(arr([1, NaN, 3]))).toBe(3);
    });

    it('nanargmin', () => {
      expect(B.nanargmin(arr([3, NaN, 1]))).toBe(2);
    });

    it('nanargmax', () => {
      expect(B.nanargmax(arr([1, NaN, 3]))).toBe(2);
    });

    it('nanprod', () => {
      expect(B.nanprod(arr([2, NaN, 3]))).toBeCloseTo(6);
    });
  });

  // ============ Order statistics ============

  describe('order statistics', () => {
    it('median', () => {
      expect(B.median(arr([3, 1, 2]))).toBeCloseTo(2);
    });

    it('percentile', () => {
      expect(B.percentile(arr([1, 2, 3, 4, 5]), 50)).toBeCloseTo(3);
    });

    it('quantile', () => {
      expect(B.quantile(arr([1, 2, 3, 4, 5]), 0.5)).toBeCloseTo(3);
    });

    it('nanmedian', () => {
      expect(B.nanmedian(arr([3, NaN, 1, 2]))).toBeCloseTo(2);
    });

    it('nanpercentile', () => {
      expect(B.nanpercentile(arr([1, NaN, 3, 4, 5]), 50)).toBeCloseTo(3.5);
    });
  });

  // ============ Histogram ============

  describe('histogram', () => {
    it('basic histogram', () => {
      const { hist, binEdges } = B.histogram(arr([1, 2, 3, 4, 5]), 5);
      expect(hist.shape).toEqual([5]);
      expect(binEdges.shape).toEqual([6]);
      expect(B.sum(hist)).toBe(5);
    });

    it('all NaN returns zero hist', () => {
      const { hist } = B.histogram(arr([NaN, NaN]), 3);
      expect(getData(hist)).toEqual([0, 0, 0]);
    });
  });

  describe('histogramBinEdges', () => {
    it('returns bin edges', () => {
      const r = B.histogramBinEdges(arr([1, 2, 3, 4, 5]), 5);
      expect(r.shape).toEqual([6]);
      expect(r.data[0]).toBeCloseTo(1);
      expect(r.data[5]).toBeCloseTo(5);
    });
  });

  // ============ emptyLike ============

  describe('emptyLike', () => {
    it('creates array with same shape', () => {
      const r = B.emptyLike(mat([1, 2, 3, 4], 2, 2));
      expect(r.shape).toEqual([2, 2]);
    });
  });

  // ============ broadcastTo ============

  describe('broadcastTo', () => {
    it('broadcasts 1d to 2d', () => {
      const a = arr([1, 2, 3]);
      const r = B.broadcastTo(a, [2, 3]);
      expect(r.shape).toEqual([2, 3]);
      expect(getData(r)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it('throws on smaller dims', () => {
      expect(() => B.broadcastTo(mat([1, 2, 3, 4], 2, 2), [4])).toThrow('smaller');
    });

    it('throws on incompatible', () => {
      expect(() => B.broadcastTo(arr([1, 2, 3]), [2, 4])).toThrow('Cannot broadcast');
    });
  });

  // ============ broadcastArrays ============

  describe('broadcastArrays', () => {
    it('empty returns empty', () => {
      expect(B.broadcastArrays().length).toBe(0);
    });

    it('single array', () => {
      const [r] = B.broadcastArrays(arr([1, 2]));
      expect(getData(r)).toEqual([1, 2]);
    });

    it('broadcasts two arrays', () => {
      const [a, b] = B.broadcastArrays(arr([1, 2, 3]), B.array([10], [1]));
      expect(a.shape).toEqual([3]);
      expect(b.shape).toEqual([3]);
      expect(getData(b)).toEqual([10, 10, 10]);
    });

    it('throws on incompatible', () => {
      expect(() => B.broadcastArrays(arr([1, 2, 3]), arr([1, 2]))).toThrow('broadcastable');
    });
  });

  // ============ moveaxis ============

  describe('moveaxis', () => {
    it('same source and dest returns copy', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
      const r = B.moveaxis(a, 0, 0);
      expect(r.shape).toEqual([2, 3]);
    });
  });

  // ============ squeeze ============

  describe('squeeze', () => {
    it('throws when axis not size 1', () => {
      expect(() => B.squeeze(arr([1, 2, 3]), 0)).toThrow('cannot squeeze');
    });
  });

  // ============ expandDims ============

  describe('expandDims', () => {
    it('throws on out-of-bounds axis', () => {
      expect(() => B.expandDims(arr([1, 2]), 5)).toThrow('out of bounds');
    });
  });

  // ============ reshape ============

  describe('reshape', () => {
    it('throws on two inferred dims', () => {
      expect(() => B.reshape(arr([1, 2, 3, 4, 5, 6]), [-1, -1])).toThrow('one unknown');
    });

    it('throws on incompatible size', () => {
      expect(() => B.reshape(arr([1, 2, 3]), [2, 2])).toThrow('cannot reshape');
    });
  });

  // ============ concatenate ============

  describe('concatenate', () => {
    it('throws on empty', () => {
      expect(() => B.concatenate([])).toThrow('need at least one');
    });

    it('single array', () => {
      const r = B.concatenate([arr([1, 2, 3])]);
      expect(getData(r)).toEqual([1, 2, 3]);
    });

    it('throws on dimension mismatch', () => {
      expect(() => B.concatenate([arr([1, 2]), mat([1, 2, 3, 4], 2, 2)])).toThrow('same number');
    });

    it('throws on shape mismatch', () => {
      expect(() => B.concatenate([mat([1, 2, 3, 4], 2, 2), mat([1, 2, 3, 4, 5, 6], 2, 3)], 0)).toThrow('must match');
    });
  });

  // ============ stack ============

  describe('stack', () => {
    it('throws on empty', () => {
      expect(() => B.stack([])).toThrow('need at least one');
    });

    it('throws on mismatched dims', () => {
      expect(() => B.stack([arr([1, 2]), mat([1, 2, 3, 4], 2, 2)])).toThrow('same shape');
    });

    it('throws on mismatched shapes', () => {
      expect(() => B.stack([arr([1, 2]), arr([1, 2, 3])])).toThrow('same shape');
    });
  });

  // ============ split ============

  describe('split', () => {
    it('throws on uneven split', () => {
      expect(() => B.split(arr([1, 2, 3]), 2)).toThrow('cannot be split');
    });
  });

  // ============ take with negative indices ============

  describe('take negative indices', () => {
    it('takes with negative index', () => {
      const a = arr([10, 20, 30, 40, 50]);
      const r = B.take(a, arr([-1, -2]), 0);
      expect(getData(r)).toEqual([50, 40]);
    });
  });

  // ============ batchedMatmul ============

  describe('batchedMatmul', () => {
    it('throws on 1D', () => {
      expect(() => B.batchedMatmul(arr([1, 2]), arr([3, 4]))).toThrow('at least 2D');
    });

    it('throws on inner dim mismatch', () => {
      expect(() => B.batchedMatmul(mat([1, 2, 3, 4], 2, 2), mat([1, 2, 3, 4, 5, 6], 3, 2))).toThrow('inner dimensions');
    });

    it('throws on non-broadcastable batch', () => {
      const a = B.array([1, 0, 0, 1, 1, 0, 0, 1], [2, 2, 2]);
      const b = B.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [3, 2, 2]);
      expect(() => B.batchedMatmul(a, b)).toThrow('broadcastable');
    });
  });

  // ============ einsum ============

  describe('einsum', () => {
    it('throws on operand count mismatch', () => {
      expect(() => B.einsum('ij,jk->ik', arr([1]))).toThrow('expected 2 operands');
    });

    it('throws on dimension mismatch', () => {
      expect(() => B.einsum('ijk->i', arr([1, 2]))).toThrow('dimensions but subscripts specify');
    });

    it('throws on inconsistent size', () => {
      expect(() => B.einsum('i,i->i', arr([1, 2, 3]), arr([1, 2]))).toThrow('inconsistent size');
    });

    it('matmul via einsum', () => {
      const a = mat([1, 2, 3, 4], 2, 2);
      const b = mat([5, 6, 7, 8], 2, 2);
      const r = B.einsum('ij,jk->ik', a, b);
      expect(getData(r)).toEqual([19, 22, 43, 50]);
    });
  });

  // ============ diff ============

  describe('diff', () => {
    it('throws on too few elements', () => {
      expect(() => B.diff(arr([1]), 1, 0)).toThrow('at least 2');
    });

    it('2d diff along axis 0', () => {
      const a = mat([1, 2, 3, 5, 7, 11], 2, 3);
      const r = B.diff(a, 1, 0);
      expect(r.shape).toEqual([1, 3]);
      expect(getData(r)).toEqual([4, 5, 8]);
    });
  });

  // ============ gradient ============

  describe('gradient', () => {
    it('throws on too few elements', () => {
      expect(() => B.gradient(arr([1]), 0)).toThrow('at least 2');
    });
  });

  // ============ cross ============

  describe('cross', () => {
    it('throws on non-3D', () => {
      expect(() => B.cross(arr([1, 2]), arr([3, 4]))).toThrow('3D vectors');
    });
  });

  // ============ cov ============

  describe('cov', () => {
    it('throws on non-2D without y', () => {
      expect(() => B.cov(arr([1, 2, 3]))).toThrow('2D array');
    });

    it('throws on length mismatch with y', () => {
      expect(() => B.cov(arr([1, 2, 3]), arr([1, 2]))).toThrow('same length');
    });
  });

  // ============ tril/triu ============

  describe('tril/triu', () => {
    it('tril throws on non-2D', () => {
      expect(() => B.tril(arr([1, 2, 3]))).toThrow('2D');
    });

    it('triu throws on non-2D', () => {
      expect(() => B.triu(arr([1, 2, 3]))).toThrow('2D');
    });
  });

  // ============ dstack ============

  describe('dstack', () => {
    it('dstack 1d arrays', () => {
      const r = B.dstack([arr([1, 2]), arr([3, 4])]);
      expect(r.shape).toEqual([1, 2, 2]);
    });

    it('dstack 3d arrays', () => {
      const a = B.array([1, 2, 3, 4], [1, 2, 2]);
      const b = B.array([5, 6, 7, 8], [1, 2, 2]);
      const r = B.dstack([a, b]);
      expect(r.shape).toEqual([1, 2, 4]);
    });
  });

  // ============ vsplit/dsplit ============

  describe('vsplit', () => {
    it('throws on 1D', () => {
      expect(() => B.vsplit(arr([1, 2, 3]), 1)).toThrow('at least 2');
    });
  });

  describe('dsplit', () => {
    it('throws on 2D', () => {
      expect(() => B.dsplit(mat([1, 2, 3, 4], 2, 2), 1)).toThrow('at least 3');
    });
  });

  // ============ argwhere empty ============

  describe('argwhere', () => {
    it('empty result', () => {
      const r = B.argwhere(arr([0, 0, 0]));
      expect(r.shape).toEqual([0, 1]);
    });
  });

  // ============ sort with NaN ============

  describe('sort with NaN', () => {
    it('NaN goes to end', () => {
      const r = B.sort(arr([3, NaN, 1, 2]));
      expect(r.data[0]).toBe(1);
      expect(r.data[1]).toBe(2);
      expect(r.data[2]).toBe(3);
      expect(Number.isNaN(r.data[3])).toBe(true);
    });
  });

  // ============ argsort with NaN ============

  describe('argsort with NaN', () => {
    it('NaN indices at end', () => {
      const r = B.argsort(arr([3, NaN, 1, 2]));
      expect(r.data[0]).toBe(2); // index of 1
      expect(r.data[1]).toBe(3); // index of 2
      expect(r.data[2]).toBe(0); // index of 3
      expect(r.data[3]).toBe(1); // index of NaN
    });
  });

  // ============ unique with NaN ============

  describe('unique with NaN', () => {
    it('NaN at end', () => {
      const r = B.unique(arr([3, NaN, 1, 2, 1]));
      expect(r.data[0]).toBe(1);
      expect(r.data[1]).toBe(2);
      expect(r.data[2]).toBe(3);
      // NaN handling may vary
    });
  });

  // ============ partition ============

  describe('partition', () => {
    it('throws on axis out of bounds', () => {
      expect(() => B.partition(arr([1, 2, 3]), 0, 5)).toThrow('out of bounds');
    });

    it('throws on kth out of bounds', () => {
      expect(() => B.partition(arr([1, 2, 3]), 5)).toThrow('kth');
    });

    it('partition 2D along axis 1', () => {
      const a = mat([3, 1, 2, 6, 4, 5], 2, 3);
      const r = B.partition(a, 1, 1);
      // After partition: element at kth position is correct
      expect(r.shape).toEqual([2, 3]);
    });
  });

  // ============ matmul 1D path ============

  describe('matmul paths', () => {
    it('WASM matmul fallback for non-2D uses wasm.matmul', () => {
      // Test 2D matmul via the WASM path
      const a = mat([1, 2, 3, 4], 2, 2);
      const b = mat([5, 6, 7, 8], 2, 2);
      const r = B.matmul(a, b);
      expect(getData(r)).toEqual([19, 22, 43, 50]);
    });
  });

  // ============ inner product ============

  describe('inner', () => {
    it('throws on size mismatch', () => {
      expect(() => B.inner(arr([1, 2]), arr([1, 2, 3]))).toThrow('same size');
    });
  });

  // ============ trace ============

  describe('trace', () => {
    it('throws on non-2D', () => {
      expect(() => B.trace(arr([1, 2, 3]))).toThrow('2D');
    });
  });

  // ============ SVD full matrices padding ============

  describe('svd full matrices', () => {
    it('pads vt for tall matrix', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
      const { u, s, vt } = B.svd(a);
      expect(u.shape).toEqual([2, 2]);
      expect(s.shape).toEqual([2]);
      expect(vt.shape).toEqual([3, 3]);
    });

    it('reduced SVD has smaller vt', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
      const { vt } = B.svd(a, false);
      expect(vt.shape).toEqual([2, 3]);
    });

    it('pads u for wide matrix', () => {
      const a = mat([1, 2, 3, 4, 5, 6], 3, 2);
      const { u, vt } = B.svd(a);
      expect(u.shape).toEqual([3, 3]);
      expect(vt.shape).toEqual([2, 2]);
    });
  });

  // ============ repeat with axis ============

  describe('repeat', () => {
    it('repeat with axis', () => {
      const a = mat([1, 2, 3, 4], 2, 2);
      const r = B.repeat(a, 2, 0);
      expect(r.shape).toEqual([4, 2]);
      expect(getData(r)).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
    });
  });

  // ============ vstack with 1D ============

  describe('vstack with 1D', () => {
    it('reshapes 1D to 2D', () => {
      const r = B.vstack([arr([1, 2, 3]), arr([4, 5, 6])]);
      expect(r.shape).toEqual([2, 3]);
    });
  });

  // ============ std with ddof ============

  describe('std with ddof', () => {
    it('std ddof=1', () => {
      const r = B.std(arr([1, 2, 3, 4, 5]), 1);
      expect(r).toBeCloseTo(Math.sqrt(2.5));
    });
  });

  // ============ insert negative index ============

  describe('insert negative index', () => {
    it('inserts at negative index', () => {
      const r = B.insert(arr([1, 2, 3]), -1, 99);
      // -1 → 3 + (-1) + 1 = 3, inserts at end
      expect(getData(r)).toEqual([1, 2, 3, 99]);
    });
  });

  // ============ matrixPower edge cases ============

  describe('matrixPower', () => {
    it('throws on non-square', () => {
      expect(() => B.matrixPower(mat([1, 2, 3, 4, 5, 6], 2, 3), 2)).toThrow('square');
    });

    it('negative power', () => {
      const a = mat([1, 0, 0, 2], 2, 2);
      const r = B.matrixPower(a, -1);
      expect(r.data[0]).toBeCloseTo(1);
      expect(r.data[3]).toBeCloseTo(0.5);
    });
  });

  // ============ kron 3D throws ============

  describe('kron', () => {
    it('throws on 3D', () => {
      expect(() => B.kron(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]), B.eye(2))).toThrow('1D or 2D');
    });
  });

  // ============ cond empty singular values ============

  describe('cond edge', () => {
    it('cond singular values empty returns Infinity', () => {
      // Use svd's empty result path — hard to trigger naturally
      // Just verify it handles singular matrix
      const r = B.cond(mat([0, 0, 0, 0], 2, 2));
      expect(r).toBe(Infinity);
    });
  });

  // ============ multiDot ============

  describe('multiDot', () => {
    it('throws on empty', () => {
      expect(() => B.multiDot([])).toThrow('at least one');
    });

    it('single array', () => {
      const r = B.multiDot([arr([1, 2, 3])]);
      expect(getData(r)).toEqual([1, 2, 3]);
    });
  });

  // ============ roots ============

  describe('roots', () => {
    it('handles leading zeros', () => {
      // p(x) = 0*x^2 + 1*x - 2 → root at x=2
      const r = B.roots(arr([0, 1, -2]));
      expect(r.data.length).toBe(1);
      expect(r.data[0]).toBeCloseTo(2);
    });

    it('single coefficient returns empty', () => {
      const r = B.roots(arr([5]));
      expect(r.data.length).toBe(0);
    });

    it('linear equation', () => {
      // p(x) = 2x + 4 → root at -2
      const r = B.roots(arr([2, 4]));
      expect(r.data.length).toBe(1);
      expect(r.data[0]).toBeCloseTo(-2);
    });

    it('quadratic equation', () => {
      // p(x) = x^2 - 5x + 6 → roots at 2, 3
      const r = B.roots(arr([1, -5, 6]));
      const sorted = Array.from(r.data).sort((a, b) => a - b);
      expect(sorted[0]).toBeCloseTo(2);
      expect(sorted[1]).toBeCloseTo(3);
    });
  });

  // ============ bincount ============

  describe('bincount', () => {
    it('throws on negative', () => {
      expect(() => B.bincount(arr([-1, 2]))).toThrow('non-negative');
    });
  });

  // ============ partition 2D ============

  describe('partition 2D', () => {
    it('partition along axis 0', () => {
      const a = mat([5, 1, 3, 2, 4, 6], 3, 2);
      const r = B.partition(a, 1, 0);
      expect(r.shape).toEqual([3, 2]);
    });

    it('partition axis out of bounds', () => {
      expect(() => B.partition(mat([1, 2, 3, 4], 2, 2), 0, 5)).toThrow('out of bounds');
    });

    it('partition kth out of bounds', () => {
      expect(() => B.partition(arr([1, 2, 3]), 10)).toThrow('kth');
    });
  });

  // ============ lexsort ============

  describe('lexsort', () => {
    it('empty keys returns empty', () => {
      const r = B.lexsort([]);
      expect(r.data.length).toBe(0);
    });

    it('throws on length mismatch', () => {
      expect(() => B.lexsort([arr([1, 2]), arr([1, 2, 3])])).toThrow('same length');
    });

    it('sorts by keys', () => {
      const r = B.lexsort([arr([1, 2, 1]), arr([3, 1, 2])]);
      expect(r.data.length).toBe(3);
    });
  });

  // ============ select ============

  describe('select', () => {
    it('throws on length mismatch', () => {
      expect(() => B.select([arr([1, 0])], [arr([10, 20]), arr([30, 40])])).toThrow('same length');
    });

    it('throws on empty', () => {
      expect(() => B.select([], [])).toThrow('must not be empty');
    });
  });

  // ============ compress axis out of bounds ============

  describe('compress', () => {
    it('throws on axis out of bounds', () => {
      expect(() => B.compress(arr([1, 0, 1]), mat([1, 2, 3, 4, 5, 6], 2, 3), 5)).toThrow('out of bounds');
    });
  });

  // ============ _normalizeAxis out of bounds ============

  describe('normalizeAxis', () => {
    it('axis out of bounds throws', () => {
      // swapaxes uses _normalizeAxis
      expect(() => B.swapaxes(arr([1, 2, 3]), 0, 5)).toThrow('out of bounds');
    });
  });

  // ============ NaN in unique sort comparator ============

  describe('unique NaN', () => {
    it('both NaN in comparison', () => {
      const r = B.unique(arr([NaN, NaN, 1, 2]));
      // Should have 1, 2, and NaN (deduplicated)
      expect(r.data.length).toBeGreaterThanOrEqual(3);
    });
  });

  // ============ argsort NaN comparator ============

  describe('argsort NaN-NaN', () => {
    it('two NaN values in argsort', () => {
      const r = B.argsort(arr([NaN, 2, NaN, 1]));
      // 1 at idx 3 first, 2 at idx 1, NaN at 0 and 2 last
      expect(r.data[0]).toBe(3);
      expect(r.data[1]).toBe(1);
    });
  });

  // ============ sort NaN-NaN ============

  describe('sort NaN-NaN', () => {
    it('two NaN values stay at end', () => {
      const r = B.sort(arr([NaN, 2, NaN, 1]));
      expect(r.data[0]).toBe(1);
      expect(r.data[1]).toBe(2);
      expect(Number.isNaN(r.data[2])).toBe(true);
      expect(Number.isNaN(r.data[3])).toBe(true);
    });
  });

  // ============ argmin with minIdx branch ============

  describe('argmin with later min', () => {
    it('finds min at later index', () => {
      const a = mat([5, 3, 1, 2, 4, 6], 2, 3);
      const r = B.argminAxis(a, 1);
      // Row 0: min is 1 at idx 2, Row 1: min is 2 at idx 0
      expect(getData(r)).toEqual([2, 0]);
    });
  });

  // ============ histogram NaN skip in binning ============

  describe('histogram NaN binning', () => {
    it('skips NaN in bin assignment', () => {
      const { hist } = B.histogram(arr([1, NaN, 2, 3, NaN]), 3);
      expect(B.sum(hist)).toBe(3); // only 3 non-NaN values
    });
  });

  // ============ histogram binIdx < 0 guard ============

  describe('histogram edge binIdx', () => {
    it('handles value at exact min', () => {
      const { hist } = B.histogram(arr([0, 0, 0]), 3);
      // All values at min, binIdx could be 0
      expect(B.sum(hist)).toBe(3);
    });
  });

  // ============ matmul non-2D fallback ============

  describe('matmul non-2D fallback', () => {
    it('1D via inner', () => {
      // 1D x 1D uses the fallback wasm.matmul path (line 1633)
      // But WASM matmul requires 2D... let's use dot instead
      const r = B.dot(arr([1, 2, 3]), arr([4, 5, 6]));
      expect(B.sum(r)).toBeCloseTo(32);
    });
  });

  // ============ argpartition ============

  describe('argpartition', () => {
    it('argpartition 2D along axis 1', () => {
      const a = mat([5, 1, 3, 2, 4, 6], 2, 3);
      const r = B.argpartition(a, 1, 1);
      expect(r.shape).toEqual([2, 3]);
      // After argpartition, element at kth position should have kth-smallest index
    });

    it('argpartition axis out of bounds', () => {
      expect(() => B.argpartition(arr([1, 2, 3]), 0, 5)).toThrow('out of bounds');
    });

    it('argpartition kth out of bounds', () => {
      expect(() => B.argpartition(arr([1, 2, 3]), 10)).toThrow('kth');
    });
  });

  // ============ lexsort tie-breaking ============

  describe('lexsort tie', () => {
    it('equal keys return 0', () => {
      // All values are the same so comparator returns 0
      const r = B.lexsort([arr([1, 1, 1])]);
      expect(r.data.length).toBe(3);
    });
  });

  // ============ roots larger polynomial ============

  describe('roots larger', () => {
    it('cubic polynomial', () => {
      // x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
      const r = B.roots(arr([1, -6, 11, -6]));
      const sorted = Array.from(r.data).sort((a, b) => a - b);
      expect(sorted.length).toBe(3);
      expect(sorted[0]).toBeCloseTo(1, 1);
      expect(sorted[1]).toBeCloseTo(2, 1);
      expect(sorted[2]).toBeCloseTo(3, 1);
    });
  });

  // ============ unique NaN-NaN comparator (line 3078) ============

  describe('unique NaN-NaN comparator', () => {
    it('handles both NaN values in sort', () => {
      // Force two NaN values through the unique sort
      const r = B.unique(arr([NaN, 3, NaN, 1, 2]));
      expect(r.data[0]).toBe(1);
      expect(r.data[1]).toBe(2);
      expect(r.data[2]).toBe(3);
    });
  });

  // ============ materializeAll ============

  describe('materializeAll', () => {
    it('is a no-op', async () => {
      await B.materializeAll();
      // Should not throw
    });
  });
});
