/**
 * Creation function tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/creation.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, approxEq } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

export function creationTests(getBackend: () => Backend) {
  describe('creation', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    // ============ zeros ============

    describe('zeros', () => {
      it('creates 1D zeros array', () => {
        const arr = B.zeros([5]);
        expect(arr.shape).toEqual([5]);
        expect(arr.data.every((x) => x === 0.0)).toBe(true);
      });

      it('creates 2D zeros array', () => {
        const arr = B.zeros([3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        expect(arr.data.length).toBe(12);
        expect(arr.data.every((x) => x === 0.0)).toBe(true);
      });

      it('creates 3D zeros array', () => {
        const arr = B.zeros([2, 3, 4]);
        expect(arr.shape).toEqual([2, 3, 4]);
        expect(arr.data.length).toBe(24);
      });

      it('creates empty zeros array', () => {
        const arr = B.zeros([0]);
        expect(arr.data.length).toBe(0);
      });
    });

    // ============ ones ============

    describe('ones', () => {
      it('creates 1D ones array', () => {
        const arr = B.ones([5]);
        expect(arr.data.every((x) => x === 1.0)).toBe(true);
      });

      it('creates 2D ones array', () => {
        const arr = B.ones([3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        expect(arr.data.every((x) => x === 1.0)).toBe(true);
      });
    });

    // ============ full ============

    describe('full', () => {
      it('creates full array', () => {
        const arr = B.full([3, 3], 7.5);
        expect(arr.data.every((x) => x === 7.5)).toBe(true);
      });

      it('creates full array with negative value', () => {
        const arr = B.full([2, 2], -3.15);
        expect(arr.data.every((x) => approxEq(x, -3.15, DEFAULT_TOL))).toBe(true);
      });
    });

    // ============ arange ============

    describe('arange', () => {
      it('creates basic arange', async () => {
        const arr = B.arange(0.0, 5.0, 1.0);
        expect(await getData(arr, B)).toEqual([0.0, 1.0, 2.0, 3.0, 4.0]);
      });

      it('creates arange with step', async () => {
        const arr = B.arange(0.0, 10.0, 2.0);
        expect(await getData(arr, B)).toEqual([0.0, 2.0, 4.0, 6.0, 8.0]);
      });

      it('creates arange with float step', async () => {
        const arr = B.arange(0.0, 1.0, 0.25);
        const expected = [0.0, 0.25, 0.5, 0.75];
        const data = await getData(arr, B);
        for (let i = 0; i < expected.length; i++) {
          expect(approxEq(data[i], expected[i], DEFAULT_TOL)).toBe(true);
        }
      });

      it('creates arange with negative step', async () => {
        const arr = B.arange(5.0, 0.0, -1.0);
        expect(await getData(arr, B)).toEqual([5.0, 4.0, 3.0, 2.0, 1.0]);
      });

      it('throws on zero step', () => {
        expect(() => B.arange(0.0, 5.0, 0.0)).toThrow();
      });

      it('creates empty arange for invalid range', () => {
        const arr = B.arange(5.0, 0.0, 1.0);
        expect(arr.data.length).toBe(0);
      });
    });

    // ============ linspace ============

    describe('linspace', () => {
      it('creates basic linspace', async () => {
        const arr = B.linspace(0.0, 1.0, 5);
        const expected = [0.0, 0.25, 0.5, 0.75, 1.0];
        const data = await getData(arr, B);
        for (let i = 0; i < expected.length; i++) {
          expect(approxEq(data[i], expected[i], DEFAULT_TOL)).toBe(true);
        }
      });

      it('creates single point linspace', async () => {
        const arr = B.linspace(5.0, 5.0, 1);
        expect(await getData(arr, B)).toEqual([5.0]);
      });

      it('creates two point linspace', async () => {
        const arr = B.linspace(0.0, 10.0, 2);
        expect(await getData(arr, B)).toEqual([0.0, 10.0]);
      });

      it('creates empty linspace', () => {
        const arr = B.linspace(0.0, 1.0, 0);
        expect(arr.data.length).toBe(0);
      });

      it('creates negative range linspace', async () => {
        const arr = B.linspace(-5.0, 5.0, 11);
        const data = await getData(arr, B);
        expect(approxEq(data[0], -5.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[5], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[10], 5.0, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ eye ============

    describe('eye', () => {
      it('creates 3x3 identity matrix', async () => {
        const arr = B.eye(3);
        expect(arr.shape).toEqual([3, 3]);
        const data = await getData(arr, B);
        expect(data[0]).toBe(1.0); // [0,0]
        expect(data[1]).toBe(0.0); // [0,1]
        expect(data[2]).toBe(0.0); // [0,2]
        expect(data[3]).toBe(0.0); // [1,0]
        expect(data[4]).toBe(1.0); // [1,1]
        expect(data[5]).toBe(0.0); // [1,2]
        expect(data[6]).toBe(0.0); // [2,0]
        expect(data[7]).toBe(0.0); // [2,1]
        expect(data[8]).toBe(1.0); // [2,2]
      });

      it('creates 1x1 identity matrix', async () => {
        const arr = B.eye(1);
        expect(await getData(arr, B)).toEqual([1.0]);
      });
    });

    // ============ diag ============

    describe('diag', () => {
      it('creates diagonal matrix from vector', async () => {
        const vec = B.array([1.0, 2.0, 3.0], [3]);
        const arr = B.diag(vec, 0);
        expect(arr.shape).toEqual([3, 3]);
        const data = await getData(arr, B);
        expect(data[0]).toBe(1.0);
        expect(data[4]).toBe(2.0);
        expect(data[8]).toBe(3.0);
      });

      it('extracts diagonal from matrix', async () => {
        const mat = B.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
        const diag = B.diag(mat, 0);
        expect(await getData(diag, B)).toEqual([1.0, 5.0, 9.0]);
      });

      it('extracts upper diagonal (k=1)', async () => {
        const mat = B.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
        const diag = B.diag(mat, 1);
        expect(await getData(diag, B)).toEqual([2.0, 6.0]);
      });

      it('extracts lower diagonal (k=-1)', async () => {
        const mat = B.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]);
        const diag = B.diag(mat, -1);
        expect(await getData(diag, B)).toEqual([4.0, 8.0]);
      });
    });
  });
}
