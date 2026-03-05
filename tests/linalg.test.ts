/**
 * Linear algebra tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/linalg.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, SVD_TOL, approxEq } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

export function linalgTests(getBackend: () => Backend) {
  describe('linalg', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const mat = (data: number[], rows: number, cols: number) =>
      B.array(data, [rows, cols]);
    const vec1d = (data: number[]) => B.array(data, [data.length]);

    // ============ matmul ============

    describe('matmul', () => {
      it('multiplies 2x2 matrices', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([5.0, 6.0, 7.0, 8.0], 2, 2);
        const c = B.matmul(a, b);

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        expect(await getData(c, B)).toEqual([19.0, 22.0, 43.0, 50.0]);
      });

      it('multiplies 2x3 and 3x2 matrices', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const b = mat([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        const c = B.matmul(a, b);

        expect(c.shape).toEqual([2, 2]);
        expect(await getData(c, B)).toEqual([58.0, 64.0, 139.0, 154.0]);
      });

      it('throws on dimension mismatch', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        expect(() => B.matmul(a, b)).toThrow();
      });
    });

    // ============ dot ============

    describe('dot', () => {
      it('computes dot product of vectors', async () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const b = vec1d([4.0, 5.0, 6.0]);
        const c = B.dot(a, b);
        // 1*4 + 2*5 + 3*6 = 32
        expect((await getData(c, B))[0]).toBe(32.0);
      });

      it('computes matmul for 2D arrays', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([5.0, 6.0, 7.0, 8.0], 2, 2);
        const c = B.dot(a, b);
        expect(await getData(c, B)).toEqual([19.0, 22.0, 43.0, 50.0]);
      });
    });

    // ============ inner ============

    describe('inner', () => {
      it('computes inner product', async () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const b = vec1d([4.0, 5.0, 6.0]);
        const result = B.inner(a, b);
        expect(result).toBe(32.0);
      });
    });

    // ============ outer ============

    describe('outer', () => {
      it('computes outer product', async () => {
        const a = vec1d([1.0, 2.0]);
        const b = vec1d([3.0, 4.0, 5.0]);
        const c = B.outer(a, b);

        expect(c.shape).toEqual([2, 3]);
        expect(await getData(c, B)).toEqual([3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
      });
    });

    // ============ inv ============

    describe('inv', () => {
      it('computes inverse of 2x2 matrix', async () => {
        const a = mat([4.0, 7.0, 2.0, 6.0], 2, 2);
        const aInv = B.inv(a);

        // A @ A^-1 should be identity
        const identity = B.matmul(a, aInv);
        const data = await getData(identity, B);
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('computes inverse of 3x3 matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], 3, 3);
        const aInv = B.inv(a);

        // A @ A^-1 should be identity
        const identity = B.matmul(a, aInv);
        const data = await getData(identity, B);
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[4], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[8], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('throws for non-square matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        expect(() => B.inv(a)).toThrow();
      });
    });

    // ============ det ============

    describe('det', () => {
      it('computes determinant of 2x2 matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const det = B.det(a);
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        expect(approxEq(det, -2.0, RELAXED_TOL)).toBe(true);
      });

      it('computes determinant of 3x3 singular matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        const det = B.det(a);
        // This matrix is singular, det = 0
        expect(approxEq(det, 0.0, RELAXED_TOL)).toBe(true);
      });

      it('computes determinant of identity matrix', async () => {
        const a = B.eye(3);
        const det = B.det(a);
        expect(approxEq(det, 1.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ trace ============

    describe('trace', () => {
      it('computes trace of matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        const tr = B.trace(a);
        expect(tr).toBe(15.0); // 1 + 5 + 9
      });

      it('computes trace of identity', async () => {
        const a = B.eye(5);
        const tr = B.trace(a);
        expect(tr).toBe(5.0);
      });
    });

    // ============ norm ============

    describe('norm', () => {
      it('computes L2 norm', async () => {
        const a = vec1d([3.0, 4.0]);
        const n = B.norm(a, 2);
        expect(approxEq(n, 5.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes L1 norm', async () => {
        const a = vec1d([-3.0, 4.0]);
        const n = B.norm(a, 1);
        expect(approxEq(n, 7.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes L-infinity norm', async () => {
        const a = vec1d([-3.0, 4.0, 2.0]);
        const n = B.norm(a, Infinity);
        expect(approxEq(n, 4.0, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ solve ============

    describe('solve', () => {
      it('solves linear system', async () => {
        // Solve Ax = b where A = [[3,1],[1,2]], b = [[9],[8]]
        // Solution: x = [[2],[3]]
        const a = mat([3.0, 1.0, 1.0, 2.0], 2, 2);
        const b = mat([9.0, 8.0], 2, 1);
        const x = B.solve(a, b);

        const xData = await getData(x, B);
        expect(approxEq(xData[0], 2.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(xData[1], 3.0, RELAXED_TOL)).toBe(true);
      });

      it('verifies A @ x = b', async () => {
        const a = mat([3.0, 1.0, 1.0, 2.0], 2, 2);
        const b = mat([9.0, 8.0], 2, 1);
        const x = B.solve(a, b);

        // Reshape x for matmul (solve may return 1D)
        const xData = await getData(x, B);
        const x2d = B.array(xData, [xData.length, 1]);

        // Verify A @ x = b
        const ax = B.matmul(a, x2d);
        const axData = await getData(ax, B);
        expect(approxEq(axData[0], 9.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(axData[1], 8.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ qr ============

    describe('qr', () => {
      it('computes QR decomposition (Q @ R = A)', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        const { q, r } = B.qr(a);

        // Q @ R should equal A (approximately)
        const reconstructed = B.matmul(q, r);
        const aData = await getData(a, B);
        const recData = await getData(reconstructed, B);
        for (let i = 0; i < aData.length; i++) {
          expect(approxEq(aData[i], recData[i], RELAXED_TOL)).toBe(true);
        }
      });

      it('Q is orthogonal (Q^T @ Q = I)', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const { q } = B.qr(a);

        // Q^T @ Q should be identity (for square Q)
        const qt = B.transpose(q);
        const qtq = B.matmul(qt, q);
        const data = await getData(qtq, B);
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('R is upper triangular', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        const { r } = B.qr(a);
        const rData = await getData(r, B);

        // Check lower triangular elements are ~0
        expect(approxEq(rData[3], 0.0, RELAXED_TOL)).toBe(true); // r[1,0]
        expect(approxEq(rData[6], 0.0, RELAXED_TOL)).toBe(true); // r[2,0]
        expect(approxEq(rData[7], 0.0, RELAXED_TOL)).toBe(true); // r[2,1]
      });
    });

    // ============ svd ============

    describe('svd', () => {
      it('computes SVD with correct shapes', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const { u, s, vt } = B.svd(a);

        // Verify shapes: A is 2x3, so U is 2x2, S is 2, Vt is 2x3
        expect(u.shape).toEqual([2, 2]);
        expect(s.shape).toEqual([2]);
        expect(vt.shape).toEqual([2, 3]);

        // Singular values should be non-negative and sorted descending
        const sData = await getData(s, B);
        expect(sData.every((x: number) => x >= 0)).toBe(true);
        expect(sData[0] >= sData[1]).toBe(true);
      });

      it('reconstructs A from U @ diag(S) @ Vt', async () => {
        // Simple 2x2 matrix
        const a = mat([3.0, 1.0, 1.0, 3.0], 2, 2);
        const { u, s, vt } = B.svd(a);

        // Reconstruct: A = U @ diag(S) @ Vt
        const sData = await getData(s, B);
        const uData = await getData(u, B);
        const vtData = await getData(vt, B);

        // Manually compute U @ diag(S) @ Vt
        // For 2x2: result[i,j] = sum_k u[i,k] * s[k] * vt[k,j]
        const reconstructed = new Float64Array(4);
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 2; j++) {
            let sum = 0;
            for (let k = 0; k < 2; k++) {
              sum += uData[i * 2 + k] * sData[k] * vtData[k * 2 + j];
            }
            reconstructed[i * 2 + j] = sum;
          }
        }

        // Compare with original (use SVD_TOL because power iteration SVD has limited precision)
        const aData = await getData(a, B);
        for (let i = 0; i < 4; i++) {
          expect(approxEq(aData[i], reconstructed[i], SVD_TOL)).toBe(true);
        }
      });

      it('U has orthonormal columns', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const { u } = B.svd(a);

        // U^T @ U should be identity
        const ut = B.transpose(u);
        const utu = B.matmul(ut, u);
        const data = await getData(utu, B);
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('Vt has orthonormal rows', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const { vt } = B.svd(a);

        // Vt @ Vt^T should be identity
        const v = B.transpose(vt);
        const vtv = B.matmul(vt, v);
        const data = await getData(vtv, B);
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });

      // NumPy parity: singular values of [[1,2],[3,4]] are ~5.465 and ~0.366
      it('matches NumPy singular values', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const { s } = B.svd(a);
        const sData = await getData(s, B);

        // NumPy: np.linalg.svd([[1,2],[3,4]]) gives [5.4649857, 0.36596619]
        expect(approxEq(sData[0], 5.4649857, 0.01)).toBe(true);
        expect(approxEq(sData[1], 0.36596619, 0.01)).toBe(true);
      });
    });

    // ============ transpose ============

    describe('transpose', () => {
      it('transposes 2x3 matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const at = B.transpose(a);

        expect(at.shape).toEqual([3, 2]);
        expect(await getData(at, B)).toEqual([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
      });

      it('transposes square matrix', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const at = B.transpose(a);
        expect(await getData(at, B)).toEqual([1.0, 3.0, 2.0, 4.0]);
      });

      it('transpose is no-op for 1D', async () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const at = B.transpose(a);
        expect(at.shape).toEqual(a.shape);
        expect(await getData(at, B)).toEqual(await getData(a, B));
      });

      it('double transpose returns original', async () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const att = B.transpose(B.transpose(a));
        expect(att.shape).toEqual(a.shape);
        expect(await getData(att, B)).toEqual(await getData(a, B));
      });
    });
  });
}
