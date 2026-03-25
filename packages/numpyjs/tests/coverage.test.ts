/**
 * Coverage tests for all 54 untested Backend methods
 * Covers: Comparison, Set Ops, Array Manipulation Extended, NaN-aware Stats,
 * Order Statistics, Histogram, Axis Operations, and Misc methods
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, getData } from './test-utils';

export function coverageTests(getBackend: () => Backend) {
  describe('coverage', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[]) => B.array(data, [data.length]);
    const mat = (data: number[], rows: number, cols: number) => B.array(data, [rows, cols]);

    // ============================================================
    // Comparison (9)
    // ============================================================

    describe('comparison', () => {
      describe('equal', () => {
        it('compares equal elements', async () => {
          const a = arr([1, 2, 3, 4]);
          const b = arr([1, 5, 3, 6]);
          const result = B.equal(a, b);
          expect(await getData(result, B)).toEqual([1, 0, 1, 0]);
        });

        it('compares all equal', async () => {
          const a = arr([7, 7, 7]);
          const b = arr([7, 7, 7]);
          const result = B.equal(a, b);
          expect(await getData(result, B)).toEqual([1, 1, 1]);
        });
      });

      describe('notEqual', () => {
        it('compares not-equal elements', async () => {
          const a = arr([1, 2, 3, 4]);
          const b = arr([1, 5, 3, 6]);
          const result = B.notEqual(a, b);
          expect(await getData(result, B)).toEqual([0, 1, 0, 1]);
        });
      });

      describe('less', () => {
        it('compares less-than', async () => {
          const a = arr([1, 5, 3, 4]);
          const b = arr([2, 3, 3, 6]);
          const result = B.less(a, b);
          expect(await getData(result, B)).toEqual([1, 0, 0, 1]);
        });
      });

      describe('lessEqual', () => {
        it('compares less-or-equal', async () => {
          const a = arr([1, 5, 3, 4]);
          const b = arr([2, 3, 3, 6]);
          const result = B.lessEqual(a, b);
          expect(await getData(result, B)).toEqual([1, 0, 1, 1]);
        });
      });

      describe('greater', () => {
        it('compares greater-than', async () => {
          const a = arr([1, 5, 3, 4]);
          const b = arr([2, 3, 3, 6]);
          const result = B.greater(a, b);
          expect(await getData(result, B)).toEqual([0, 1, 0, 0]);
        });
      });

      describe('greaterEqual', () => {
        it('compares greater-or-equal', async () => {
          const a = arr([1, 5, 3, 4]);
          const b = arr([2, 3, 3, 6]);
          const result = B.greaterEqual(a, b);
          expect(await getData(result, B)).toEqual([0, 1, 1, 0]);
        });
      });

      describe('isnan', () => {
        it('detects NaN values', async () => {
          const a = arr([1, NaN, 3, NaN, 5]);
          const result = B.isnan(a);
          expect(await getData(result, B)).toEqual([0, 1, 0, 1, 0]);
        });

        it('returns all zeros for no NaN', async () => {
          const a = arr([1, 2, 3]);
          const result = B.isnan(a);
          expect(await getData(result, B)).toEqual([0, 0, 0]);
        });
      });

      describe('isinf', () => {
        it('detects infinite values', async () => {
          const a = arr([1, Infinity, -Infinity, 0, NaN]);
          const result = B.isinf(a);
          expect(await getData(result, B)).toEqual([0, 1, 1, 0, 0]);
        });
      });

      describe('isfinite', () => {
        it('detects finite values', async () => {
          const a = arr([1, Infinity, -Infinity, 0, NaN]);
          const result = B.isfinite(a);
          expect(await getData(result, B)).toEqual([1, 0, 0, 1, 0]);
        });
      });
    });

    // ============================================================
    // Set Operations (4)
    // ============================================================

    describe('set operations', () => {
      describe('setdiff1d', () => {
        it('computes set difference', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          const b = arr([2, 4]);
          const result = B.setdiff1d(a, b);
          const data = await getData(result, B);
          const sorted = data.slice().sort((x, y) => x - y);
          expect(sorted).toEqual([1, 3, 5]);
        });

        it('returns empty when all elements in b', async () => {
          const a = arr([1, 2, 3]);
          const b = arr([1, 2, 3, 4]);
          const result = B.setdiff1d(a, b);
          const data = await getData(result, B);
          expect(data.length).toBe(0);
        });
      });

      describe('union1d', () => {
        it('computes set union', async () => {
          const a = arr([1, 2, 3]);
          const b = arr([3, 4, 5]);
          const result = B.union1d(a, b);
          const data = await getData(result, B);
          const sorted = data.slice().sort((x, y) => x - y);
          expect(sorted).toEqual([1, 2, 3, 4, 5]);
        });

        it('handles duplicates', async () => {
          const a = arr([1, 1, 2]);
          const b = arr([2, 2, 3]);
          const result = B.union1d(a, b);
          const data = await getData(result, B);
          const sorted = data.slice().sort((x, y) => x - y);
          expect(sorted).toEqual([1, 2, 3]);
        });
      });

      describe('intersect1d', () => {
        it('computes set intersection', async () => {
          const a = arr([1, 2, 3, 4]);
          const b = arr([3, 4, 5, 6]);
          const result = B.intersect1d(a, b);
          const data = await getData(result, B);
          const sorted = data.slice().sort((x, y) => x - y);
          expect(sorted).toEqual([3, 4]);
        });

        it('returns empty for disjoint sets', async () => {
          const a = arr([1, 2]);
          const b = arr([3, 4]);
          const result = B.intersect1d(a, b);
          const data = await getData(result, B);
          expect(data.length).toBe(0);
        });
      });

      describe('isin', () => {
        it('tests membership', async () => {
          const element = arr([1, 2, 3, 4, 5]);
          const testElements = arr([2, 4]);
          const result = B.isin(element, testElements);
          expect(await getData(result, B)).toEqual([0, 1, 0, 1, 0]);
        });

        it('returns all zeros for no matches', async () => {
          const element = arr([1, 2, 3]);
          const testElements = arr([4, 5, 6]);
          const result = B.isin(element, testElements);
          expect(await getData(result, B)).toEqual([0, 0, 0]);
        });
      });
    });

    // ============================================================
    // Array Manipulation Extended (7)
    // ============================================================

    describe('array manipulation extended', () => {
      describe('insert', () => {
        it('inserts a value at an index', async () => {
          const a = arr([1, 2, 3, 4]);
          const result = B.insert(a, 2, 99);
          const data = await getData(result, B);
          expect(data).toEqual([1, 2, 99, 3, 4]);
        });

        it('inserts at the beginning', async () => {
          const a = arr([1, 2, 3]);
          const result = B.insert(a, 0, 10);
          const data = await getData(result, B);
          expect(data).toEqual([10, 1, 2, 3]);
        });

        it('inserts an array of values', async () => {
          const a = arr([1, 2, 3]);
          const vals = arr([10, 20]);
          const result = B.insert(a, 1, vals);
          const data = await getData(result, B);
          expect(data).toEqual([1, 10, 20, 2, 3]);
        });
      });

      describe('deleteArr', () => {
        it('deletes element at index', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          const result = B.deleteArr(a, 2);
          const data = await getData(result, B);
          expect(data).toEqual([1, 2, 4, 5]);
        });

        it('deletes multiple indices', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          const result = B.deleteArr(a, [1, 3]);
          const data = await getData(result, B);
          expect(data).toEqual([1, 3, 5]);
        });
      });

      describe('append', () => {
        it('appends arrays', async () => {
          const a = arr([1, 2, 3]);
          const b = arr([4, 5, 6]);
          const result = B.append(a, b);
          const data = await getData(result, B);
          expect(data).toEqual([1, 2, 3, 4, 5, 6]);
        });
      });

      describe('atleast1d', () => {
        it('ensures at least 1D', async () => {
          const a = B.array([5], []); // scalar-like
          const result = B.atleast1d(a);
          expect(result.shape.length).toBeGreaterThanOrEqual(1);
          const data = await getData(result, B);
          expect(data).toEqual([5]);
        });

        it('preserves 1D array', async () => {
          const a = arr([1, 2, 3]);
          const result = B.atleast1d(a);
          expect(result.shape).toEqual([3]);
          expect(await getData(result, B)).toEqual([1, 2, 3]);
        });
      });

      describe('atleast2d', () => {
        it('promotes 1D to 2D', async () => {
          const a = arr([1, 2, 3]);
          const result = B.atleast2d(a);
          expect(result.shape.length).toBe(2);
          const data = await getData(result, B);
          expect(data).toEqual([1, 2, 3]);
        });

        it('preserves 2D array', async () => {
          const a = mat([1, 2, 3, 4], 2, 2);
          const result = B.atleast2d(a);
          expect(result.shape).toEqual([2, 2]);
        });
      });

      describe('atleast3d', () => {
        it('promotes 1D to 3D', async () => {
          const a = arr([1, 2, 3]);
          const result = B.atleast3d(a);
          expect(result.shape.length).toBe(3);
          const data = await getData(result, B);
          expect(data).toEqual([1, 2, 3]);
        });

        it('promotes 2D to 3D', async () => {
          const a = mat([1, 2, 3, 4], 2, 2);
          const result = B.atleast3d(a);
          expect(result.shape.length).toBe(3);
        });
      });

      describe('countNonzero', () => {
        it('counts non-zero elements', async () => {
          const a = arr([0, 1, 0, 3, 0, 5]);
          const result = B.countNonzero(a);
          expect(result).toBe(3);
        });

        it('returns 0 for all zeros', async () => {
          const a = arr([0, 0, 0]);
          const result = B.countNonzero(a);
          expect(result).toBe(0);
        });

        it('counts all for no zeros', async () => {
          const a = arr([1, 2, 3]);
          const result = B.countNonzero(a);
          expect(result).toBe(3);
        });
      });
    });

    // ============================================================
    // NaN-aware Stats (9)
    // ============================================================

    describe('nan-aware stats', () => {
      describe('nansum', () => {
        it('computes sum ignoring NaN', async () => {
          const a = arr([1, NaN, 3, NaN, 5]);
          expect(B.nansum(a)).toBe(9);
        });

        it('computes sum with no NaN', async () => {
          const a = arr([1, 2, 3]);
          expect(B.nansum(a)).toBe(6);
        });
      });

      describe('nanmean', () => {
        it('computes mean ignoring NaN', async () => {
          const a = arr([1, NaN, 3, NaN, 5]);
          // mean of [1, 3, 5] = 3
          expect(B.nanmean(a)).toBe(3);
        });
      });

      describe('nanstd', () => {
        it('computes std ignoring NaN', async () => {
          const a = arr([2, NaN, 4, NaN, 4, 4, 5, 5, 7, 9]);
          // Non-NaN values: [2, 4, 4, 4, 5, 5, 7, 9], population std = 2.0
          expect(approxEq(B.nanstd(a) as number, 2.0, RELAXED_TOL)).toBe(true);
        });
      });

      describe('nanvar', () => {
        it('computes variance ignoring NaN', async () => {
          const a = arr([2, NaN, 4, NaN, 4, 4, 5, 5, 7, 9]);
          // Non-NaN values: [2, 4, 4, 4, 5, 5, 7, 9], population var = 4.0
          expect(approxEq(B.nanvar(a) as number, 4.0, RELAXED_TOL)).toBe(true);
        });
      });

      describe('nanmin', () => {
        it('computes min ignoring NaN', async () => {
          const a = arr([NaN, 3, 1, NaN, 5]);
          expect(B.nanmin(a)).toBe(1);
        });
      });

      describe('nanmax', () => {
        it('computes max ignoring NaN', async () => {
          const a = arr([NaN, 3, 1, NaN, 5]);
          expect(B.nanmax(a)).toBe(5);
        });
      });

      describe('nanargmin', () => {
        it('finds index of min ignoring NaN', async () => {
          const a = arr([NaN, 3, 1, NaN, 5]);
          expect(B.nanargmin(a)).toBe(2);
        });
      });

      describe('nanargmax', () => {
        it('finds index of max ignoring NaN', async () => {
          const a = arr([NaN, 3, 1, NaN, 5]);
          expect(B.nanargmax(a)).toBe(4);
        });
      });

      describe('nanprod', () => {
        it('computes product ignoring NaN', async () => {
          const a = arr([1, NaN, 3, NaN, 4]);
          expect(B.nanprod(a)).toBe(12);
        });

        it('computes product with no NaN', async () => {
          const a = arr([2, 3, 4]);
          expect(B.nanprod(a)).toBe(24);
        });
      });
    });

    // ============================================================
    // Order Statistics (5)
    // ============================================================

    describe('order statistics', () => {
      describe('median', () => {
        it('computes median of odd-length array', async () => {
          const a = arr([3, 1, 2, 5, 4]);
          expect(B.median(a)).toBe(3);
        });

        it('computes median of even-length array', async () => {
          const a = arr([4, 1, 3, 2]);
          // median = (2 + 3) / 2 = 2.5
          expect(B.median(a)).toBe(2.5);
        });

        it('computes median of single element', async () => {
          const a = arr([42]);
          expect(B.median(a)).toBe(42);
        });
      });

      describe('percentile', () => {
        it('computes 50th percentile (median)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.percentile(a, 50)).toBe(3);
        });

        it('computes 0th percentile (min)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.percentile(a, 0)).toBe(1);
        });

        it('computes 100th percentile (max)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.percentile(a, 100)).toBe(5);
        });

        it('computes 25th percentile', async () => {
          const a = arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
          const result = B.percentile(a, 25);
          expect(approxEq(result, 3.25, RELAXED_TOL)).toBe(true);
        });
      });

      describe('quantile', () => {
        it('computes 0.5 quantile (median)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.quantile(a, 0.5)).toBe(3);
        });

        it('computes 0.0 quantile (min)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.quantile(a, 0.0)).toBe(1);
        });

        it('computes 1.0 quantile (max)', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          expect(B.quantile(a, 1.0)).toBe(5);
        });
      });

      describe('nanmedian', () => {
        it('computes median ignoring NaN', async () => {
          const a = arr([NaN, 3, 1, NaN, 5, 2, 4]);
          // Non-NaN sorted: [1, 2, 3, 4, 5], median = 3
          expect(B.nanmedian(a)).toBe(3);
        });
      });

      describe('nanpercentile', () => {
        it('computes percentile ignoring NaN', async () => {
          const a = arr([NaN, 1, 2, NaN, 3, 4, 5]);
          // Non-NaN: [1, 2, 3, 4, 5], 50th percentile = 3
          expect(B.nanpercentile(a, 50)).toBe(3);
        });

        it('computes 0th percentile ignoring NaN', async () => {
          const a = arr([NaN, 5, 3, 1, NaN]);
          expect(B.nanpercentile(a, 0)).toBe(1);
        });

        it('computes 100th percentile ignoring NaN', async () => {
          const a = arr([NaN, 5, 3, 1, NaN]);
          expect(B.nanpercentile(a, 100)).toBe(5);
        });
      });
    });

    // ============================================================
    // Histogram (2)
    // ============================================================

    describe('histogram', () => {
      describe('histogram', () => {
        it('computes histogram with default bins', async () => {
          const a = arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
          const { hist, binEdges } = B.histogram(a);
          const histData = await getData(hist, B);
          const edgeData = await getData(binEdges, B);
          // Sum of histogram counts should equal number of elements
          const totalCount = histData.reduce((s, v) => s + v, 0);
          expect(totalCount).toBe(10);
          // Bin edges should be sorted ascending
          for (let i = 1; i < edgeData.length; i++) {
            expect(edgeData[i]).toBeGreaterThan(edgeData[i - 1]);
          }
          // Number of bin edges = number of bins + 1
          expect(edgeData.length).toBe(histData.length + 1);
        });

        it('computes histogram with specified bins', async () => {
          const a = arr([1, 1, 2, 2, 2, 3, 3, 3, 3, 4]);
          const { hist, binEdges } = B.histogram(a, 4);
          const histData = await getData(hist, B);
          const edgeData = await getData(binEdges, B);
          expect(histData.length).toBe(4);
          expect(edgeData.length).toBe(5);
          const totalCount = histData.reduce((s, v) => s + v, 0);
          expect(totalCount).toBe(10);
        });
      });

      describe('histogramBinEdges', () => {
        it('computes bin edges', async () => {
          const a = arr([1, 2, 3, 4, 5]);
          const edges = B.histogramBinEdges(a, 5);
          const edgeData = await getData(edges, B);
          expect(edgeData.length).toBe(6); // 5 bins => 6 edges
          // First edge <= min, last edge >= max
          expect(edgeData[0]).toBeLessThanOrEqual(1);
          expect(edgeData[edgeData.length - 1]).toBeGreaterThanOrEqual(5);
          // Edges should be sorted ascending
          for (let i = 1; i < edgeData.length; i++) {
            expect(edgeData[i]).toBeGreaterThan(edgeData[i - 1]);
          }
        });

        it('computes bin edges with default bins', async () => {
          const a = arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
          const edges = B.histogramBinEdges(a);
          const edgeData = await getData(edges, B);
          // Default is typically 10 bins => 11 edges
          expect(edgeData.length).toBeGreaterThan(1);
          expect(edgeData[0]).toBeLessThanOrEqual(1);
          expect(edgeData[edgeData.length - 1]).toBeGreaterThanOrEqual(10);
        });
      });
    });

    // ============================================================
    // Axis Operations (11)
    // ============================================================

    describe('axis operations', () => {
      // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
      // axis 0 reduces rows, axis 1 reduces cols

      describe('minAxis', () => {
        it('computes min along axis 0', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.minAxis(a, 0);
          expect(await getData(result, B)).toEqual([1, 2, 3]);
        });

        it('computes min along axis 1', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.minAxis(a, 1);
          expect(await getData(result, B)).toEqual([1, 2]);
        });
      });

      describe('maxAxis', () => {
        it('computes max along axis 0', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.maxAxis(a, 0);
          expect(await getData(result, B)).toEqual([4, 5, 6]);
        });

        it('computes max along axis 1', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.maxAxis(a, 1);
          expect(await getData(result, B)).toEqual([5, 6]);
        });
      });

      describe('argminAxis', () => {
        it('computes argmin along axis 0', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.argminAxis(a, 0);
          // col 0: min(1,4)=1 at row 0; col 1: min(5,2)=2 at row 1; col 2: min(3,6)=3 at row 0
          expect(await getData(result, B)).toEqual([0, 1, 0]);
        });

        it('computes argmin along axis 1', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.argminAxis(a, 1);
          // row 0: min(1,5,3)=1 at col 0; row 1: min(4,2,6)=2 at col 1
          expect(await getData(result, B)).toEqual([0, 1]);
        });
      });

      describe('argmaxAxis', () => {
        it('computes argmax along axis 0', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.argmaxAxis(a, 0);
          // col 0: max(1,4)=4 at row 1; col 1: max(5,2)=5 at row 0; col 2: max(3,6)=6 at row 1
          expect(await getData(result, B)).toEqual([1, 0, 1]);
        });

        it('computes argmax along axis 1', async () => {
          const a = mat([1, 5, 3, 4, 2, 6], 2, 3);
          const result = B.argmaxAxis(a, 1);
          // row 0: max(1,5,3)=5 at col 1; row 1: max(4,2,6)=6 at col 2
          expect(await getData(result, B)).toEqual([1, 2]);
        });
      });

      describe('varAxis', () => {
        it('computes variance along axis 0', async () => {
          const a = mat([1, 2, 3, 5, 6, 7], 2, 3);
          const result = B.varAxis(a, 0);
          const data = await getData(result, B);
          // col 0: var([1,5]) = ((1-3)^2 + (5-3)^2)/2 = 4
          // col 1: var([2,6]) = ((2-4)^2 + (6-4)^2)/2 = 4
          // col 2: var([3,7]) = ((3-5)^2 + (7-5)^2)/2 = 4
          expect(approxEq(data[0], 4, RELAXED_TOL)).toBe(true);
          expect(approxEq(data[1], 4, RELAXED_TOL)).toBe(true);
          expect(approxEq(data[2], 4, RELAXED_TOL)).toBe(true);
        });
      });

      describe('stdAxis', () => {
        it('computes std along axis 0', async () => {
          const a = mat([1, 2, 3, 5, 6, 7], 2, 3);
          const result = B.stdAxis(a, 0);
          const data = await getData(result, B);
          // std = sqrt(4) = 2 for each column
          expect(approxEq(data[0], 2, RELAXED_TOL)).toBe(true);
          expect(approxEq(data[1], 2, RELAXED_TOL)).toBe(true);
          expect(approxEq(data[2], 2, RELAXED_TOL)).toBe(true);
        });
      });

      describe('prodAxis', () => {
        it('computes product along axis 0', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.prodAxis(a, 0);
          // col 0: 1*4=4, col 1: 2*5=10, col 2: 3*6=18
          expect(await getData(result, B)).toEqual([4, 10, 18]);
        });

        it('computes product along axis 1', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.prodAxis(a, 1);
          // row 0: 1*2*3=6, row 1: 4*5*6=120
          expect(await getData(result, B)).toEqual([6, 120]);
        });
      });

      describe('allAxis', () => {
        it('computes all along axis 0', async () => {
          const a = mat([1, 0, 1, 1, 1, 1], 2, 3);
          const result = B.allAxis(a, 0);
          // col 0: all(1,1)=1, col 1: all(0,1)=0, col 2: all(1,1)=1
          expect(await getData(result, B)).toEqual([1, 0, 1]);
        });

        it('computes all along axis 1', async () => {
          const a = mat([1, 1, 1, 1, 0, 1], 2, 3);
          const result = B.allAxis(a, 1);
          // row 0: all(1,1,1)=1, row 1: all(1,0,1)=0
          expect(await getData(result, B)).toEqual([1, 0]);
        });
      });

      describe('anyAxis', () => {
        it('computes any along axis 0', async () => {
          const a = mat([0, 0, 1, 0, 1, 0], 2, 3);
          const result = B.anyAxis(a, 0);
          // col 0: any(0,0)=0, col 1: any(0,1)=1, col 2: any(1,0)=1
          expect(await getData(result, B)).toEqual([0, 1, 1]);
        });

        it('computes any along axis 1', async () => {
          const a = mat([0, 0, 0, 0, 1, 0], 2, 3);
          const result = B.anyAxis(a, 1);
          // row 0: any(0,0,0)=0, row 1: any(0,1,0)=1
          expect(await getData(result, B)).toEqual([0, 1]);
        });
      });

      describe('cumsumAxis', () => {
        it('computes cumsum along axis 0', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.cumsumAxis(a, 0);
          expect(result.shape).toEqual([2, 3]);
          // row 0: [1, 2, 3], row 1: [1+4, 2+5, 3+6] = [5, 7, 9]
          expect(await getData(result, B)).toEqual([1, 2, 3, 5, 7, 9]);
        });

        it('computes cumsum along axis 1', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.cumsumAxis(a, 1);
          expect(result.shape).toEqual([2, 3]);
          // row 0: [1, 1+2, 1+2+3] = [1, 3, 6]
          // row 1: [4, 4+5, 4+5+6] = [4, 9, 15]
          expect(await getData(result, B)).toEqual([1, 3, 6, 4, 9, 15]);
        });
      });

      describe('cumprodAxis', () => {
        it('computes cumprod along axis 0', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.cumprodAxis(a, 0);
          expect(result.shape).toEqual([2, 3]);
          // row 0: [1, 2, 3], row 1: [1*4, 2*5, 3*6] = [4, 10, 18]
          expect(await getData(result, B)).toEqual([1, 2, 3, 4, 10, 18]);
        });

        it('computes cumprod along axis 1', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.cumprodAxis(a, 1);
          expect(result.shape).toEqual([2, 3]);
          // row 0: [1, 1*2, 1*2*3] = [1, 2, 6]
          // row 1: [4, 4*5, 4*5*6] = [4, 20, 120]
          expect(await getData(result, B)).toEqual([1, 2, 6, 4, 20, 120]);
        });
      });
    });

    // ============================================================
    // Misc (6+1)
    // ============================================================

    describe('misc', () => {
      describe('arccos', () => {
        it('computes arccos', async () => {
          const a = arr([1.0, 0.5, 0.0, -1.0]);
          const result = B.arccos(a);
          const data = await getData(result, B);
          expect(approxEq(data[0], 0.0, RELAXED_TOL)).toBe(true); // arccos(1) = 0
          expect(approxEq(data[1], Math.PI / 3, RELAXED_TOL)).toBe(true); // arccos(0.5) = pi/3
          expect(approxEq(data[2], Math.PI / 2, RELAXED_TOL)).toBe(true); // arccos(0) = pi/2
          expect(approxEq(data[3], Math.PI, RELAXED_TOL)).toBe(true); // arccos(-1) = pi
        });
      });

      describe('arctan', () => {
        it('computes arctan', async () => {
          const a = arr([0.0, 1.0, -1.0]);
          const result = B.arctan(a);
          const data = await getData(result, B);
          expect(approxEq(data[0], 0.0, RELAXED_TOL)).toBe(true); // arctan(0) = 0
          expect(approxEq(data[1], Math.PI / 4, RELAXED_TOL)).toBe(true); // arctan(1) = pi/4
          expect(approxEq(data[2], -Math.PI / 4, RELAXED_TOL)).toBe(true); // arctan(-1) = -pi/4
        });
      });

      describe('remainder', () => {
        it('computes remainder (IEEE 754)', async () => {
          const a = arr([7, -7, 7, -7]);
          const b = arr([3, 3, -3, -3]);
          const result = B.remainder(a, b);
          const data = await getData(result, B);
          // IEEE remainder: sign matches divisor
          // remainder(7, 3) = 1, remainder(-7, 3) = 2, remainder(7, -3) = -2, remainder(-7, -3) = -1
          expect(approxEq(data[0], 1, DEFAULT_TOL)).toBe(true);
          expect(approxEq(data[1], 2, DEFAULT_TOL)).toBe(true);
          expect(approxEq(data[2], -2, DEFAULT_TOL)).toBe(true);
          expect(approxEq(data[3], -1, DEFAULT_TOL)).toBe(true);
        });

        it('computes remainder with exact divisor', async () => {
          const a = arr([6, 9, 12]);
          const b = arr([3, 3, 3]);
          const result = B.remainder(a, b);
          const data = await getData(result, B);
          expect(approxEq(data[0], 0, DEFAULT_TOL)).toBe(true);
          expect(approxEq(data[1], 0, DEFAULT_TOL)).toBe(true);
          expect(approxEq(data[2], 0, DEFAULT_TOL)).toBe(true);
        });
      });

      describe('subScalar', () => {
        it('subtracts scalar from array', async () => {
          const a = arr([10, 20, 30]);
          const result = B.subScalar(a, 5);
          expect(await getData(result, B)).toEqual([5, 15, 25]);
        });

        it('subtracts negative scalar', async () => {
          const a = arr([1, 2, 3]);
          const result = B.subScalar(a, -10);
          expect(await getData(result, B)).toEqual([11, 12, 13]);
        });
      });

      describe('divScalar', () => {
        it('divides array by scalar', async () => {
          const a = arr([10, 20, 30]);
          const result = B.divScalar(a, 5);
          expect(await getData(result, B)).toEqual([2, 4, 6]);
        });

        it('divides by fractional scalar', async () => {
          const a = arr([1, 2, 3]);
          const result = B.divScalar(a, 0.5);
          expect(await getData(result, B)).toEqual([2, 4, 6]);
        });
      });

      describe('emptyLike', () => {
        it('creates array with same shape', async () => {
          const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
          const result = B.emptyLike(a);
          expect(result.shape).toEqual([2, 3]);
          // emptyLike just allocates — shape is what matters
          const data = await getData(result, B);
          expect(data.length).toBe(6);
        });

        it('creates 1D array with same shape', async () => {
          const a = arr([1, 2, 3, 4]);
          const result = B.emptyLike(a);
          expect(result.shape).toEqual([4]);
          const data = await getData(result, B);
          expect(data.length).toBe(4);
        });
      });

      describe('cond', () => {
        it('computes condition number of identity', async () => {
          const a = B.eye(3);
          const result = B.cond(a);
          expect(approxEq(result, 1, RELAXED_TOL)).toBe(true);
        });

        it('computes condition number of well-conditioned matrix', async () => {
          // [[2, 0], [0, 1]] has singular values 2 and 1, cond = 2/1 = 2
          const a = mat([2, 0, 0, 1], 2, 2);
          const result = B.cond(a);
          expect(approxEq(result, 2, RELAXED_TOL)).toBe(true);
        });

        it('computes large condition number for ill-conditioned matrix', async () => {
          // [[1, 0], [0, 1e-10]] has cond = 1/1e-10 = 1e10
          const a = mat([1, 0, 0, 1e-10], 2, 2);
          const result = B.cond(a);
          expect(result).toBeGreaterThan(1e9);
        });
      });
    });
  });
}
