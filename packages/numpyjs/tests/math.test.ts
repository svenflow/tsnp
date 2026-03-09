/**
 * Math function tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/math.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, getTol } from './test-utils';

// Helper to get data from arrays (handles GPU materialization)
async function getData(arr: { toArray(): number[] }, B: Backend): Promise<number[]> {
  if (B.materializeAll) await B.materializeAll();
  return arr.toArray();
}

// Get tolerance appropriate for backend (f64 vs f32)
function tol(B: Backend, relaxed: boolean = false): number {
  return getTol(B, relaxed);
}

export function mathTests(getBackend: () => Backend) {
  describe('math', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[]) => B.array(data, [data.length]);

    // ============ Trigonometric ============

    describe('trigonometric', () => {
      it('computes sin', async () => {
        const a = arr([0.0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2, Math.PI]);
        const result = B.sin(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 0.5, tol(B))).toBe(true);
        expect(approxEq(data[2], Math.sqrt(2) / 2, tol(B))).toBe(true);
        expect(approxEq(data[3], Math.sqrt(3) / 2, tol(B))).toBe(true);
        expect(approxEq(data[4], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[5], 0.0, tol(B))).toBe(true);
      });

      it('computes cos', async () => {
        const a = arr([0.0, Math.PI / 3, Math.PI / 2, Math.PI]);
        const result = B.cos(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 0.5, tol(B))).toBe(true);
        expect(approxEq(data[2], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[3], -1.0, tol(B))).toBe(true);
      });

      it('computes tan', async () => {
        const a = arr([0.0, Math.PI / 4]);
        const result = B.tan(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
      });

      it('computes arcsin', async () => {
        const a = arr([0.0, 0.5, 1.0]);
        const result = B.arcsin(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.PI / 6, tol(B))).toBe(true);
        expect(approxEq(data[2], Math.PI / 2, tol(B))).toBe(true);
      });
    });

    // ============ Hyperbolic ============

    describe('hyperbolic', () => {
      it('computes sinh and cosh', async () => {
        const a = arr([0.0, 1.0, 2.0]);
        const sinh = B.sinh(a);
        const cosh = B.cosh(a);

        const sinhData = await getData(sinh, B);
        const coshData = await getData(cosh, B);

        // sinh(0) = 0, cosh(0) = 1
        expect(approxEq(sinhData[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(coshData[0], 1.0, tol(B))).toBe(true);

        // Identity: cosh^2 - sinh^2 = 1
        for (let i = 0; i < 3; i++) {
          expect(approxEq(coshData[i] * coshData[i] - sinhData[i] * sinhData[i], 1.0, tol(B))).toBe(true);
        }
      });

      it('computes tanh', async () => {
        const a = arr([0.0, 1.0, -1.0, 10.0, -10.0]);
        const result = B.tanh(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(data[1] > 0.0 && data[1] < 1.0).toBe(true);
        expect(data[2] < 0.0 && data[2] > -1.0).toBe(true);
        expect(approxEq(data[3], 1.0, tol(B, true))).toBe(true); // tanh saturates
        expect(approxEq(data[4], -1.0, tol(B, true))).toBe(true);
      });
    });

    // ============ Exponential and Logarithmic ============

    describe('exponential and logarithmic', () => {
      it('computes exp', async () => {
        const a = arr([0.0, 1.0, 2.0, -1.0]);
        const result = B.exp(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.E, tol(B))).toBe(true);
        expect(approxEq(data[2], Math.E * Math.E, tol(B))).toBe(true);
        expect(approxEq(data[3], 1.0 / Math.E, tol(B))).toBe(true);
      });

      it('computes log', async () => {
        const a = arr([1.0, Math.E, Math.E * Math.E]);
        const result = B.log(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 2.0, tol(B))).toBe(true);
      });

      it('returns NaN for log of negative', async () => {
        const a = arr([-1.0]);
        const result = B.log(a);
        const data = await getData(result, B);
        expect(Number.isNaN(data[0])).toBe(true);
      });

      it('exp and log are inverse operations', async () => {
        const a = arr([0.5, 1.0, 2.0, 5.0, 10.0]);
        const expA = B.exp(a);
        const logExpA = B.log(expA);

        const aData = await getData(a, B);
        const resultData = await getData(logExpA, B);
        for (let i = 0; i < aData.length; i++) {
          expect(approxEq(aData[i], resultData[i], tol(B))).toBe(true);
        }
      });

      it('computes log2', async () => {
        const a = arr([1.0, 2.0, 4.0, 8.0]);
        const result = B.log2(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 2.0, tol(B))).toBe(true);
        expect(approxEq(data[3], 3.0, tol(B))).toBe(true);
      });

      it('computes log10', async () => {
        const a = arr([1.0, 10.0, 100.0, 1000.0]);
        const result = B.log10(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 2.0, tol(B))).toBe(true);
        expect(approxEq(data[3], 3.0, tol(B))).toBe(true);
      });

      it('computes sqrt', async () => {
        const a = arr([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        const result = B.sqrt(a);
        const data = await getData(result, B);
        expect(data).toEqual([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
      });

      it('returns NaN for sqrt of negative', async () => {
        const a = arr([-1.0]);
        const result = B.sqrt(a);
        const data = await getData(result, B);
        expect(Number.isNaN(data[0])).toBe(true);
      });

      it('computes cbrt', async () => {
        const a = arr([0.0, 1.0, 8.0, 27.0, -8.0]);
        const result = B.cbrt(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 2.0, tol(B))).toBe(true);
        expect(approxEq(data[3], 3.0, tol(B))).toBe(true);
        expect(approxEq(data[4], -2.0, tol(B))).toBe(true);
      });

      it('computes square', async () => {
        const a = arr([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        const result = B.square(a);
        const data = await getData(result, B);
        expect(data).toEqual([9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
      });
    });

    // ============ Rounding ============

    describe('rounding', () => {
      it('computes floor', async () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.floor(a);
        const data = await getData(result, B);
        expect(data).toEqual([-3.0, -1.0, 0.0, 0.0, 2.0]);
      });

      it('computes ceil', async () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.ceil(a);
        const data = await getData(result, B);
        expect(approxEq(data[0], -2.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[3], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[4], 3.0, tol(B))).toBe(true);
      });

      it('computes round', async () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.round(a);
        const data = await getData(result, B);
        expect(data[0]).toBe(-3.0);
        expect(data[2]).toBe(0.0);
        expect(data[4]).toBe(3.0);
      });
    });

    // ============ Other Unary ============

    describe('other unary', () => {
      it('computes abs', async () => {
        const a = arr([-5.0, -2.5, 0.0, 2.5, 5.0]);
        const result = B.abs(a);
        const data = await getData(result, B);
        expect(data).toEqual([5.0, 2.5, 0.0, 2.5, 5.0]);
      });

      it('computes sign', async () => {
        const a = arr([-5.0, -0.5, 0.0, 0.5, 5.0]);
        const result = B.sign(a);
        const data = await getData(result, B);
        expect(data).toEqual([-1.0, -1.0, 0.0, 1.0, 1.0]);
      });

      it('computes neg', async () => {
        const a = arr([-2.0, -1.0, 0.0, 1.0, 2.0]);
        const result = B.neg(a);
        const data = await getData(result, B);
        expect(approxEq(data[0], 2.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 1.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[3], -1.0, tol(B))).toBe(true);
        expect(approxEq(data[4], -2.0, tol(B))).toBe(true);
      });

      it('computes reciprocal', async () => {
        const a = arr([1.0, 2.0, 4.0, 0.5]);
        const result = B.reciprocal(a);
        const data = await getData(result, B);
        expect(data).toEqual([1.0, 0.5, 0.25, 2.0]);
      });
    });

    // ============ Binary Operations ============

    describe('binary operations', () => {
      it('adds arrays', async () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([4.0, 5.0, 6.0]);
        const result = B.add(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([5.0, 7.0, 9.0]);
      });

      it('subtracts arrays', async () => {
        const a = arr([5.0, 7.0, 9.0]);
        const b = arr([1.0, 2.0, 3.0]);
        const result = B.sub(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([4.0, 5.0, 6.0]);
      });

      it('multiplies arrays element-wise', async () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.mul(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([2.0, 6.0, 12.0]);
      });

      it('divides arrays element-wise', async () => {
        const a = arr([4.0, 9.0, 16.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.div(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([2.0, 3.0, 4.0]);
      });

      it('raises to power', async () => {
        const a = arr([2.0, 3.0, 4.0]);
        const b = arr([2.0, 2.0, 2.0]);
        const result = B.pow(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([4.0, 9.0, 16.0]);
      });

      it('computes maximum', async () => {
        const a = arr([1.0, 5.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.maximum(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([2.0, 5.0, 4.0]);
      });

      it('computes minimum', async () => {
        const a = arr([1.0, 5.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.minimum(a, b);
        const data = await getData(result, B);
        expect(data).toEqual([1.0, 3.0, 3.0]);
      });

      it('throws on shape mismatch', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([1.0, 2.0]);
        expect(() => B.add(a, b)).toThrow();
        expect(() => B.sub(a, b)).toThrow();
        expect(() => B.mul(a, b)).toThrow();
        expect(() => B.div(a, b)).toThrow();
      });
    });

    // ============ Scalar Operations ============

    describe('scalar operations', () => {
      it('adds scalar', async () => {
        const a = arr([1.0, 2.0, 3.0]);
        const result = B.addScalar(a, 10.0);
        const data = await getData(result, B);
        expect(data).toEqual([11.0, 12.0, 13.0]);
      });

      it('multiplies by scalar', async () => {
        const a = arr([1.0, 2.0, 3.0]);
        const result = B.mulScalar(a, 2.0);
        const data = await getData(result, B);
        expect(data).toEqual([2.0, 4.0, 6.0]);
      });

      it('raises to scalar power', async () => {
        const a = arr([1.0, 2.0, 3.0, 4.0]);
        const result = B.powScalar(a, 2.0);
        const data = await getData(result, B);
        expect(data).toEqual([1.0, 4.0, 9.0, 16.0]);
      });

      it('clips values to range', async () => {
        const a = arr([-5.0, 0.0, 5.0, 10.0, 15.0]);
        const result = B.clip(a, 0.0, 10.0);
        const data = await getData(result, B);
        expect(data).toEqual([0.0, 0.0, 5.0, 10.0, 10.0]);
      });
    });

    // ============ Extended Unary Operations ============

    describe('extended unary operations', () => {
      it('computes arcsinh', async () => {
        const a = arr([0.0, 1.0, -1.0, 10.0]);
        const result = B.arcsinh(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.asinh(1.0), tol(B))).toBe(true);
        expect(approxEq(data[2], Math.asinh(-1.0), tol(B))).toBe(true);
        expect(approxEq(data[3], Math.asinh(10.0), tol(B))).toBe(true);
      });

      it('computes arccosh', async () => {
        const a = arr([1.0, 2.0, 10.0]);
        const result = B.arccosh(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.acosh(2.0), tol(B))).toBe(true);
        expect(approxEq(data[2], Math.acosh(10.0), tol(B))).toBe(true);
      });

      it('computes arctanh', async () => {
        const a = arr([0.0, 0.5, -0.5, 0.9]);
        const result = B.arctanh(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.atanh(0.5), tol(B))).toBe(true);
        expect(approxEq(data[2], Math.atanh(-0.5), tol(B))).toBe(true);
        expect(approxEq(data[3], Math.atanh(0.9), tol(B))).toBe(true);
      });

      it('computes expm1', async () => {
        const a = arr([0.0, 1.0, -1.0, 0.0001]);
        const result = B.expm1(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.E - 1, tol(B))).toBe(true);
        expect(approxEq(data[2], 1 / Math.E - 1, tol(B))).toBe(true);
        // For small x, expm1 is more accurate than exp(x)-1
        expect(approxEq(data[3], Math.expm1(0.0001), tol(B))).toBe(true);
      });

      it('computes log1p', async () => {
        const a = arr([0.0, 1.0, Math.E - 1, 0.0001]);
        const result = B.log1p(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.log(2), tol(B))).toBe(true);
        expect(approxEq(data[2], 1.0, tol(B))).toBe(true);
        // For small x, log1p is more accurate than log(1+x)
        expect(approxEq(data[3], Math.log1p(0.0001), tol(B))).toBe(true);
      });

      it('computes trunc', async () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.trunc(a);
        const data = await getData(result, B);

        expect(data[0]).toBe(-2.0);
        expect(Object.is(data[1], 0) || Object.is(data[1], -0)).toBe(true);  // trunc(-0.5) = -0 or 0
        expect(data[2]).toBe(0.0);
        expect(data[3]).toBe(0.0);
        expect(data[4]).toBe(2.0);
      });

      it('computes fix (alias for trunc)', async () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.fix(a);
        const data = await getData(result, B);

        expect(data[0]).toBe(-2.0);
        expect(Object.is(data[1], 0) || Object.is(data[1], -0)).toBe(true);  // fix(-0.5) = -0 or 0
        expect(data[2]).toBe(0.0);
        expect(data[3]).toBe(0.0);
        expect(data[4]).toBe(2.0);
      });

      it('computes sinc', async () => {
        const a = arr([0.0, 1.0, -1.0, 0.5]);
        const result = B.sinc(a);
        const data = await getData(result, B);

        // sinc(0) = 1
        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);
        // sinc(1) = sin(pi) / pi = 0
        expect(approxEq(data[1], 0.0, tol(B))).toBe(true);
        // sinc(-1) = sin(-pi) / (-pi) = 0
        expect(approxEq(data[2], 0.0, tol(B))).toBe(true);
        // sinc(0.5) = sin(pi/2) / (pi/2) = 2/pi
        expect(approxEq(data[3], 2 / Math.PI, tol(B))).toBe(true);
      });

      it('computes deg2rad', async () => {
        const a = arr([0.0, 90.0, 180.0, 360.0]);
        const result = B.deg2rad(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], Math.PI / 2, tol(B))).toBe(true);
        expect(approxEq(data[2], Math.PI, tol(B))).toBe(true);
        expect(approxEq(data[3], 2 * Math.PI, tol(B))).toBe(true);
      });

      it('computes rad2deg', async () => {
        const a = arr([0.0, Math.PI / 2, Math.PI, 2 * Math.PI]);
        const result = B.rad2deg(a);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);
        expect(approxEq(data[1], 90.0, tol(B))).toBe(true);
        expect(approxEq(data[2], 180.0, tol(B))).toBe(true);
        expect(approxEq(data[3], 360.0, tol(B))).toBe(true);
      });

      it('computes heaviside', async () => {
        const a = arr([-2.0, -0.5, 0.0, 0.5, 2.0]);
        const result = B.heaviside(a, 0.5);  // h0 = 0.5 at x = 0
        const data = await getData(result, B);
        expect(data).toEqual([0.0, 0.0, 0.5, 1.0, 1.0]);
      });

      it('computes signbit', async () => {
        const a = arr([-2.0, -0.0, 0.0, 0.5, 2.0]);
        const result = B.signbit(a);
        const data = await getData(result, B);

        expect(data[0]).toBe(1.0);  // negative
        // Note: -0 detection varies by implementation
        expect(data[2]).toBe(0.0);  // +0
        expect(data[3]).toBe(0.0);  // positive
        expect(data[4]).toBe(0.0);  // positive
      });

      it('computes signbit with NaN', async () => {
        const a = arr([NaN]);
        const result = B.signbit(a);
        const data = await getData(result, B);
        expect(data[0]).toBe(0.0);  // signbit(NaN) = 0 (NumPy: False)
      });

      it('computes signbit with negative zero', async () => {
        const a = arr([-0.0]);
        const result = B.signbit(a);
        const data = await getData(result, B);
        expect(data[0]).toBe(1.0);  // signbit(-0) = 1 (NumPy: True)
      });
    });

    // ============ Decomposition Operations ============

    describe('decomposition operations', () => {
      it('computes modf', async () => {
        const a = arr([2.5, -2.5, 3.0, -3.0, 0.0]);
        const result = B.modf(a);
        const frac = await getData(result.frac, B);
        const integ = await getData(result.integ, B);

        expect(approxEq(integ[0], 2.0, tol(B))).toBe(true);
        expect(approxEq(frac[0], 0.5, tol(B))).toBe(true);
        expect(approxEq(integ[1], -2.0, tol(B))).toBe(true);
        expect(approxEq(frac[1], -0.5, tol(B))).toBe(true);
        expect(approxEq(integ[2], 3.0, tol(B))).toBe(true);
        expect(approxEq(frac[2], 0.0, tol(B))).toBe(true);
        expect(approxEq(integ[3], -3.0, tol(B))).toBe(true);
        expect(approxEq(frac[3], 0.0, tol(B))).toBe(true);
        expect(approxEq(integ[4], 0.0, tol(B))).toBe(true);
        expect(approxEq(frac[4], 0.0, tol(B))).toBe(true);
      });

      it('computes frexp', async () => {
        const a = arr([0.0, 1.0, 2.0, 4.0, 8.0, -2.0]);
        const result = B.frexp(a);
        const mantissa = await getData(result.mantissa, B);
        const exponent = await getData(result.exponent, B);

        // frexp returns (mantissa, exp) such that x = mantissa * 2^exp
        // where 0.5 <= |mantissa| < 1
        expect(mantissa[0]).toBe(0.0);  // frexp(0) = (0, 0)
        expect(exponent[0]).toBe(0);

        expect(approxEq(mantissa[1], 0.5, tol(B))).toBe(true);  // 1 = 0.5 * 2^1
        expect(exponent[1]).toBe(1);

        expect(approxEq(mantissa[2], 0.5, tol(B))).toBe(true);  // 2 = 0.5 * 2^2
        expect(exponent[2]).toBe(2);

        expect(approxEq(mantissa[3], 0.5, tol(B))).toBe(true);  // 4 = 0.5 * 2^3
        expect(exponent[3]).toBe(3);

        expect(approxEq(mantissa[4], 0.5, tol(B))).toBe(true);  // 8 = 0.5 * 2^4
        expect(exponent[4]).toBe(4);

        expect(approxEq(mantissa[5], -0.5, tol(B))).toBe(true);  // -2 = -0.5 * 2^2
        expect(exponent[5]).toBe(2);
      });

      it('computes ldexp', async () => {
        const mantissa = arr([0.5, 0.5, 0.5, 0.75, -0.5]);
        const exp = arr([1, 2, 3, 2, 3]);
        const result = B.ldexp(mantissa, exp);
        const data = await getData(result, B);

        expect(data[0]).toBe(1.0);   // 0.5 * 2^1 = 1
        expect(data[1]).toBe(2.0);   // 0.5 * 2^2 = 2
        expect(data[2]).toBe(4.0);   // 0.5 * 2^3 = 4
        expect(data[3]).toBe(3.0);   // 0.75 * 2^2 = 3
        expect(data[4]).toBe(-4.0);  // -0.5 * 2^3 = -4
      });

      it('frexp and ldexp are inverse operations', async () => {
        const a = arr([0.5, 1.0, 2.0, 5.0, 10.0, -3.0]);
        const { mantissa, exponent } = B.frexp(a);
        const result = B.ldexp(mantissa, exponent);

        const aData = await getData(a, B);
        const resultData = await getData(result, B);
        for (let i = 0; i < aData.length; i++) {
          expect(approxEq(aData[i], resultData[i], tol(B))).toBe(true);
        }
      });

      it('computes divmod', async () => {
        const a = arr([7.0, -7.0, 7.0, -7.0, 5.5]);
        const b = arr([3.0, 3.0, -3.0, -3.0, 2.0]);
        const result = B.divmod(a, b);
        const quotient = await getData(result.quotient, B);
        const remainder = await getData(result.remainder, B);

        // Python-style floor division and modulo
        expect(quotient[0]).toBe(2.0);   // 7 // 3 = 2
        expect(approxEq(remainder[0], 1.0, tol(B))).toBe(true);   // 7 % 3 = 1

        expect(quotient[1]).toBe(-3.0);  // -7 // 3 = -3 (floor)
        expect(approxEq(remainder[1], 2.0, tol(B))).toBe(true);   // -7 % 3 = 2

        expect(quotient[2]).toBe(-3.0);  // 7 // -3 = -3 (floor)
        expect(approxEq(remainder[2], -2.0, tol(B))).toBe(true);  // 7 % -3 = -2

        expect(quotient[3]).toBe(2.0);   // -7 // -3 = 2 (floor)
        expect(approxEq(remainder[3], -1.0, tol(B))).toBe(true);  // -7 % -3 = -1

        expect(quotient[4]).toBe(2.0);   // 5.5 // 2 = 2
        expect(approxEq(remainder[4], 1.5, tol(B))).toBe(true);   // 5.5 % 2 = 1.5
      });
    });

    // ============ Extended Binary Operations ============

    describe('extended binary operations', () => {
      it('computes mod (Python-style modulo)', async () => {
        const a = arr([7.0, -7.0, 7.0, -7.0]);
        const b = arr([3.0, 3.0, -3.0, -3.0]);
        const result = B.mod(a, b);
        const data = await getData(result, B);

        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);   // 7 % 3 = 1
        expect(approxEq(data[1], 2.0, tol(B))).toBe(true);   // -7 % 3 = 2
        expect(approxEq(data[2], -2.0, tol(B))).toBe(true);  // 7 % -3 = -2
        expect(approxEq(data[3], -1.0, tol(B))).toBe(true);  // -7 % -3 = -1
      });

      it('computes fmod (C-style modulo)', async () => {
        const a = arr([7.0, -7.0, 7.0, -7.0]);
        const b = arr([3.0, 3.0, -3.0, -3.0]);
        const result = B.fmod(a, b);
        const data = await getData(result, B);

        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);   // 7 % 3 = 1
        expect(approxEq(data[1], -1.0, tol(B))).toBe(true);  // -7 % 3 = -1
        expect(approxEq(data[2], 1.0, tol(B))).toBe(true);   // 7 % -3 = 1
        expect(approxEq(data[3], -1.0, tol(B))).toBe(true);  // -7 % -3 = -1
      });

      it('computes copysign', async () => {
        const a = arr([1.0, -1.0, 1.0, -1.0, 0.0]);
        const b = arr([1.0, 1.0, -1.0, -1.0, -1.0]);
        const result = B.copysign(a, b);
        const data = await getData(result, B);

        expect(data[0]).toBe(1.0);   // |1| * sign(1) = 1
        expect(data[1]).toBe(1.0);   // |-1| * sign(1) = 1
        expect(data[2]).toBe(-1.0);  // |1| * sign(-1) = -1
        expect(data[3]).toBe(-1.0);  // |-1| * sign(-1) = -1
        expect(data[4]).toBe(-0.0);  // |0| * sign(-1) = -0
      });

      it('computes copysign with signed zeros', async () => {
        // copysign(5, -0) should return -5 (copy sign bit from -0)
        const a1 = arr([5.0]);
        const b1 = arr([-0.0]);
        const result1 = B.copysign(a1, b1);
        const data1 = await getData(result1, B);
        expect(data1[0]).toBe(-5.0);

        // copysign(-5, 0) should return 5 (copy sign bit from +0)
        const a2 = arr([-5.0]);
        const b2 = arr([0.0]);
        const result2 = B.copysign(a2, b2);
        const data2 = await getData(result2, B);
        expect(data2[0]).toBe(5.0);
      });

      it('computes hypot', async () => {
        const a = arr([3.0, 5.0, 0.0, 1.0]);
        const b = arr([4.0, 12.0, 0.0, 1.0]);
        const result = B.hypot(a, b);
        const data = await getData(result, B);

        expect(approxEq(data[0], 5.0, tol(B))).toBe(true);   // sqrt(9+16) = 5
        expect(approxEq(data[1], 13.0, tol(B))).toBe(true);  // sqrt(25+144) = 13
        expect(approxEq(data[2], 0.0, tol(B))).toBe(true);   // sqrt(0+0) = 0
        expect(approxEq(data[3], Math.sqrt(2), tol(B))).toBe(true);  // sqrt(1+1)
      });

      it('computes arctan2', async () => {
        const y = arr([0.0, 1.0, -1.0, 0.0]);
        const x = arr([1.0, 0.0, 0.0, -1.0]);
        const result = B.arctan2(y, x);
        const data = await getData(result, B);

        expect(approxEq(data[0], 0.0, tol(B))).toBe(true);           // atan2(0, 1) = 0
        expect(approxEq(data[1], Math.PI / 2, tol(B))).toBe(true);   // atan2(1, 0) = pi/2
        expect(approxEq(data[2], -Math.PI / 2, tol(B))).toBe(true);  // atan2(-1, 0) = -pi/2
        expect(approxEq(data[3], Math.PI, tol(B))).toBe(true);       // atan2(0, -1) = pi
      });

      it('computes logaddexp', async () => {
        const a = arr([0.0, 1.0, 2.0, -Infinity]);
        const b = arr([0.0, 2.0, 2.0, 0.0]);
        const result = B.logaddexp(a, b);
        const data = await getData(result, B);

        // logaddexp(a, b) = log(exp(a) + exp(b))
        expect(approxEq(data[0], Math.log(2), tol(B))).toBe(true);      // log(1+1) = ln(2)
        expect(approxEq(data[1], Math.log(Math.E + Math.E * Math.E), tol(B, true))).toBe(true);
        expect(approxEq(data[2], Math.log(2 * Math.E * Math.E), tol(B, true))).toBe(true);
        expect(approxEq(data[3], 0.0, tol(B))).toBe(true);  // log(0 + 1) = 0
      });

      it('computes logaddexp2', async () => {
        const a = arr([0.0, 1.0, 2.0, -Infinity]);
        const b = arr([0.0, 2.0, 2.0, 0.0]);
        const result = B.logaddexp2(a, b);
        const data = await getData(result, B);

        // logaddexp2(a, b) = log2(2^a + 2^b)
        expect(approxEq(data[0], 1.0, tol(B))).toBe(true);      // log2(1+1) = 1
        expect(approxEq(data[1], Math.log2(2 + 4), tol(B, true))).toBe(true);  // log2(2^1 + 2^2) = log2(6)
        expect(approxEq(data[2], 3.0, tol(B, true))).toBe(true);      // log2(4+4) = log2(8) = 3
        expect(approxEq(data[3], 0.0, tol(B))).toBe(true);      // log2(0 + 1) = 0
      });

      it('computes fmax (ignoring NaN)', async () => {
        const a = arr([1.0, NaN, 3.0, NaN]);
        const b = arr([2.0, 2.0, NaN, NaN]);
        const result = B.fmax(a, b);
        const data = await getData(result, B);

        expect(data[0]).toBe(2.0);  // max(1, 2) = 2
        expect(data[1]).toBe(2.0);  // fmax ignores NaN in a
        expect(data[2]).toBe(3.0);  // fmax ignores NaN in b
        expect(Number.isNaN(data[3])).toBe(true);  // both NaN -> NaN
      });

      it('computes fmin (ignoring NaN)', async () => {
        const a = arr([1.0, NaN, 3.0, NaN]);
        const b = arr([2.0, 2.0, NaN, NaN]);
        const result = B.fmin(a, b);
        const data = await getData(result, B);

        expect(data[0]).toBe(1.0);  // min(1, 2) = 1
        expect(data[1]).toBe(2.0);  // fmin ignores NaN in a
        expect(data[2]).toBe(3.0);  // fmin ignores NaN in b
        expect(Number.isNaN(data[3])).toBe(true);  // both NaN -> NaN
      });
    });
  });
}
