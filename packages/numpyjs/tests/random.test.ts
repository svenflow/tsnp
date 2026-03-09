/**
 * Random number generation tests - NumPy compatible
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, approxEq, DEFAULT_TOL } from './test-utils';

export function randomTests(getBackend: () => Backend) {
  let B: Backend;

  beforeAll(() => {
    B = getBackend();
  });

  describe('random', () => {
    describe('seed', () => {
      it('produces reproducible results', () => {
        B.seed(12345);
        const a1 = B.rand([5]);
        B.seed(12345);
        const a2 = B.rand([5]);
        for (let i = 0; i < 5; i++) {
          expect(a1.data[i]).toBe(a2.data[i]);
        }
      });
    });

    describe('rand', () => {
      it('creates uniform random array in [0, 1)', () => {
        const arr = B.rand([100]);
        expect(arr.shape).toEqual([100]);
        for (let i = 0; i < arr.data.length; i++) {
          expect(arr.data[i]).toBeGreaterThanOrEqual(0);
          expect(arr.data[i]).toBeLessThan(1);
        }
      });

      it('creates 2D random array', () => {
        const arr = B.rand([3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        expect(arr.data.length).toBe(12);
      });

      it('creates 3D random array', () => {
        const arr = B.rand([2, 3, 4]);
        expect(arr.shape).toEqual([2, 3, 4]);
        expect(arr.data.length).toBe(24);
      });
    });

    describe('randn', () => {
      it('creates normal distributed array', () => {
        B.seed(42);
        const arr = B.randn([1000]);
        expect(arr.shape).toEqual([1000]);
        // Check mean is approximately 0
        const mean = arr.data.reduce((a, b) => a + b, 0) / arr.data.length;
        expect(Math.abs(mean)).toBeLessThan(0.15);
        // Check std is approximately 1
        const variance = arr.data.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.data.length;
        expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.15);
      });
    });

    describe('randint', () => {
      it('creates integer array in [low, high)', () => {
        const arr = B.randint(5, 15, [100]);
        expect(arr.shape).toEqual([100]);
        for (let i = 0; i < arr.data.length; i++) {
          expect(arr.data[i]).toBeGreaterThanOrEqual(5);
          expect(arr.data[i]).toBeLessThan(15);
          expect(arr.data[i] % 1).toBe(0); // Integer check
        }
      });

      it('creates 2D integer array', () => {
        const arr = B.randint(0, 10, [3, 4]);
        expect(arr.shape).toEqual([3, 4]);
        for (let i = 0; i < arr.data.length; i++) {
          expect(arr.data[i]).toBeGreaterThanOrEqual(0);
          expect(arr.data[i]).toBeLessThan(10);
        }
      });
    });

    describe('uniform', () => {
      it('creates uniform distribution in [low, high)', () => {
        const arr = B.uniform(2.5, 7.5, [100]);
        expect(arr.shape).toEqual([100]);
        for (let i = 0; i < arr.data.length; i++) {
          expect(arr.data[i]).toBeGreaterThanOrEqual(2.5);
          expect(arr.data[i]).toBeLessThan(7.5);
        }
      });

      it('handles negative range', () => {
        const arr = B.uniform(-10, -5, [50]);
        for (let i = 0; i < arr.data.length; i++) {
          expect(arr.data[i]).toBeGreaterThanOrEqual(-10);
          expect(arr.data[i]).toBeLessThan(-5);
        }
      });
    });

    describe('normal', () => {
      it('creates normal distribution with given mean and std', () => {
        B.seed(42);
        const arr = B.normal(5.0, 2.0, [1000]);
        expect(arr.shape).toEqual([1000]);
        const mean = arr.data.reduce((a, b) => a + b, 0) / arr.data.length;
        expect(Math.abs(mean - 5.0)).toBeLessThan(0.3);
        const variance = arr.data.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.data.length;
        expect(Math.abs(Math.sqrt(variance) - 2.0)).toBeLessThan(0.3);
      });
    });

    describe('shuffle', () => {
      it('shuffles 1D array', () => {
        B.seed(42);
        const arr = B.arange(0, 10, 1);
        const shuffled = B.shuffle(arr);
        expect(shuffled.shape).toEqual([10]);
        // Check all elements are still there
        const sorted = [...shuffled.data].sort((a, b) => a - b);
        for (let i = 0; i < 10; i++) {
          expect(sorted[i]).toBe(i);
        }
        // Check it's actually shuffled (not in order)
        let inOrder = true;
        for (let i = 0; i < 10; i++) {
          if (shuffled.data[i] !== i) inOrder = false;
        }
        expect(inOrder).toBe(false);
      });

      it('shuffles along first axis for 2D', () => {
        B.seed(42);
        const arr = B.array([1, 2, 3, 4, 5, 6], [3, 2]);
        const shuffled = B.shuffle(arr);
        expect(shuffled.shape).toEqual([3, 2]);
        // Check rows are preserved (pairs stay together)
        const rows: number[][] = [];
        for (let i = 0; i < 3; i++) {
          rows.push([shuffled.data[i * 2], shuffled.data[i * 2 + 1]]);
        }
        // Each row should be one of the original rows
        const origRows = [[1, 2], [3, 4], [5, 6]];
        for (const row of rows) {
          const found = origRows.some(orig => orig[0] === row[0] && orig[1] === row[1]);
          expect(found).toBe(true);
        }
      });
    });

    describe('choice', () => {
      it('samples with replacement', () => {
        B.seed(42);
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const sample = B.choice(arr, 10, true);
        expect(sample.shape).toEqual([10]);
        // All values should be from original array
        for (let i = 0; i < sample.data.length; i++) {
          expect([10, 20, 30, 40, 50]).toContain(sample.data[i]);
        }
      });

      it('samples without replacement', () => {
        B.seed(42);
        const arr = B.array([1, 2, 3, 4, 5], [5]);
        const sample = B.choice(arr, 5, false);
        expect(sample.shape).toEqual([5]);
        // Should contain all elements exactly once
        const sorted = [...sample.data].sort((a, b) => a - b);
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
      it('creates permutation of 0..n-1', () => {
        B.seed(42);
        const perm = B.permutation(10);
        expect(perm.shape).toEqual([10]);
        const sorted = [...perm.data].sort((a, b) => a - b);
        for (let i = 0; i < 10; i++) {
          expect(sorted[i]).toBe(i);
        }
      });

      it('creates permutation of given array', () => {
        B.seed(42);
        const arr = B.array([5, 10, 15, 20], [4]);
        const perm = B.permutation(arr);
        expect(perm.shape).toEqual([4]);
        const sorted = [...perm.data].sort((a, b) => a - b);
        expect(sorted).toEqual([5, 10, 15, 20]);
      });
    });
  });
}
