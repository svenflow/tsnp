/**
 * Random number generation tests - NumPy compatible
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, getData } from './test-utils';

export function randomTests(getBackend: () => Backend) {
  describe('random', () => {
    let B: Backend;

    beforeAll(() => {
      B = getBackend();
    });
    describe('seed', () => {
      it('produces reproducible results', async () => {
        B.seed(12345);
        const a1 = B.rand([5]);
        B.seed(12345);
        const a2 = B.rand([5]);
        const d1 = await getData(a1, B);
        const d2 = await getData(a2, B);
        for (let i = 0; i < 5; i++) {
          expect(d1[i]).toBe(d2[i]);
        }
      });
    });

    describe('rand', () => {
      it('creates uniform random array in [0, 1)', async () => {
        const arr = B.rand([100]);
        expect(arr.shape).toEqual([100]);
        const data = await getData(arr, B);
        for (let i = 0; i < data.length; i++) {
          expect(data[i]).toBeGreaterThanOrEqual(0);
          expect(data[i]).toBeLessThan(1);
        }
      });

      it('creates 2D random array', () => {
        const arr = B.rand([3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        expect(arr.shape.reduce((a, b) => a * b, 1)).toBe(12);
      });

      it('creates 3D random array', () => {
        const arr = B.rand([2, 3, 4]);
        expect(arr.shape).toEqual([2, 3, 4]);
        expect(arr.shape.reduce((a, b) => a * b, 1)).toBe(24);
      });
    });

    describe('randn', () => {
      it('creates normal distributed array', async () => {
        B.seed(42);
        const arr = B.randn([1000]);
        expect(arr.shape).toEqual([1000]);
        const data = await getData(arr, B);
        // Check mean is approximately 0
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        expect(Math.abs(mean)).toBeLessThan(0.15);
        // Check std is approximately 1
        const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
        expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.15);
      });
    });

    describe('randint', () => {
      it('creates integer array in [low, high)', async () => {
        const arr = B.randint(5, 15, [100]);
        expect(arr.shape).toEqual([100]);
        const data = await getData(arr, B);
        for (let i = 0; i < data.length; i++) {
          expect(data[i]).toBeGreaterThanOrEqual(5);
          expect(data[i]).toBeLessThan(15);
          expect(data[i] % 1).toBe(0); // Integer check
        }
      });

      it('creates 2D integer array', async () => {
        const arr = B.randint(0, 10, [3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        const data = await getData(arr, B);
        for (let i = 0; i < data.length; i++) {
          expect(data[i]).toBeGreaterThanOrEqual(0);
          expect(data[i]).toBeLessThan(10);
        }
      });
    });

    describe('uniform', () => {
      it('creates uniform distribution in [low, high)', async () => {
        const arr = B.uniform(2.5, 7.5, [100]);
        expect(arr.shape).toEqual([100]);
        const data = await getData(arr, B);
        for (let i = 0; i < data.length; i++) {
          expect(data[i]).toBeGreaterThanOrEqual(2.5);
          expect(data[i]).toBeLessThan(7.5);
        }
      });

      it('handles negative range', async () => {
        const arr = B.uniform(-10, -5, [50]);
        const data = await getData(arr, B);
        for (let i = 0; i < data.length; i++) {
          expect(data[i]).toBeGreaterThanOrEqual(-10);
          expect(data[i]).toBeLessThan(-5);
        }
      });
    });

    describe('normal', () => {
      it('creates normal distribution with given mean and std', async () => {
        B.seed(42);
        const arr = B.normal(5.0, 2.0, [1000]);
        expect(arr.shape).toEqual([1000]);
        const data = await getData(arr, B);
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        expect(Math.abs(mean - 5.0)).toBeLessThan(0.3);
        const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
        expect(Math.abs(Math.sqrt(variance) - 2.0)).toBeLessThan(0.3);
      });
    });

    describe('shuffle', () => {
      it('shuffles 1D array', async () => {
        B.seed(42);
        const a = B.arange(0, 10, 1);
        B.shuffle(a); // in-place, returns void
        expect(a.shape).toEqual([10]);
        const data = await getData(a, B);
        // Check all elements are still there
        const sorted = [...data].sort((a, b) => a - b);
        for (let i = 0; i < 10; i++) {
          expect(sorted[i]).toBe(i);
        }
        // Check it's actually shuffled (not in order)
        let inOrder = true;
        for (let i = 0; i < 10; i++) {
          if (data[i] !== i) inOrder = false;
        }
        expect(inOrder).toBe(false);
      });

      it('shuffles along first axis for 2D', async () => {
        B.seed(42);
        const a = B.array([1, 2, 3, 4, 5, 6], [3, 2]);
        B.shuffle(a); // in-place, returns void
        expect(a.shape).toEqual([3, 2]);
        const data = await getData(a, B);
        // Check rows are preserved (pairs stay together)
        const rows: number[][] = [];
        for (let i = 0; i < 3; i++) {
          rows.push([data[i * 2], data[i * 2 + 1]]);
        }
        // Each row should be one of the original rows
        const origRows = [
          [1, 2],
          [3, 4],
          [5, 6],
        ];
        for (const row of rows) {
          const found = origRows.some(orig => orig[0] === row[0] && orig[1] === row[1]);
          expect(found).toBe(true);
        }
      });
    });

    describe('choice', () => {
      it('samples with replacement', async () => {
        B.seed(42);
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const sample = B.choice(arr, 10, true);
        expect(sample.shape).toEqual([10]);
        const data = await getData(sample, B);
        // All values should be from original array
        for (let i = 0; i < data.length; i++) {
          expect([10, 20, 30, 40, 50]).toContain(data[i]);
        }
      });

      it('samples without replacement', async () => {
        B.seed(42);
        const arr = B.array([1, 2, 3, 4, 5], [5]);
        const sample = B.choice(arr, 5, false);
        expect(sample.shape).toEqual([5]);
        const data = await getData(sample, B);
        // Should contain all elements exactly once
        const sorted = [...data].sort((a, b) => a - b);
        for (let i = 0; i < 5; i++) {
          expect(sorted[i]).toBe(i + 1);
        }
      });

      it('throws when sampling too many without replacement', () => {
        const arr = B.array([1, 2, 3], [3]);
        expect(() => B.choice(arr, 5, false)).toThrow();
      });
    });

    describe('permutation', () => {
      it('creates permutation of 0..n-1', async () => {
        B.seed(42);
        const perm = B.permutation(10);
        expect(perm.shape).toEqual([10]);
        const data = await getData(perm, B);
        const sorted = [...data].sort((a, b) => a - b);
        for (let i = 0; i < 10; i++) {
          expect(sorted[i]).toBe(i);
        }
      });

      it('creates permutation of given array', async () => {
        B.seed(42);
        const arr = B.array([5, 10, 15, 20], [4]);
        const perm = B.permutation(arr);
        expect(perm.shape).toEqual([4]);
        const data = await getData(perm, B);
        const sorted = [...data].sort((a, b) => a - b);
        expect(sorted).toEqual([5, 10, 15, 20]);
      });
    });
  });
}
