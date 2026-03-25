/**
 * DType system and scalar broadcasting tests
 *
 * Tests sourced from NumPy test suite:
 * - numpy/core/tests/test_dtype.py
 * - numpy/core/tests/test_ufunc.py (scalar broadcasting)
 * - numpy/core/tests/test_multiarray.py (astype)
 */

import { describe, it, expect } from 'vitest';
import type { Backend } from './test-utils';
import { getData, getTol, approxEq } from './test-utils';

export function dtypeScalarTests(getBackend: () => Backend) {
  describe('dtype system', () => {
    // === np.zeros with dtype ===
    // >>> np.zeros(3, dtype=np.float32)
    // array([0., 0., 0.], dtype=float32)
    it('zeros float32', async () => {
      const B = getBackend();
      const a = B.zeros([3], 'float32');
      expect(a.dtype).toBe('float32');
      expect(a.shape).toEqual([3]);
      const d = await getData(a, B);
      expect(d).toEqual([0, 0, 0]);
    });

    // >>> np.zeros(3, dtype=np.int32)
    // array([0, 0, 0], dtype=int32)
    it('zeros int32', async () => {
      const B = getBackend();
      const a = B.zeros([3], 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      expect(d).toEqual([0, 0, 0]);
    });

    // >>> np.zeros(3, dtype=np.int8)
    it('zeros int8', async () => {
      const B = getBackend();
      const a = B.zeros([3], 'int8');
      expect(a.dtype).toBe('int8');
      const d = await getData(a, B);
      expect(d).toEqual([0, 0, 0]);
    });

    // >>> np.zeros(3, dtype=np.uint8)
    it('zeros uint8', async () => {
      const B = getBackend();
      const a = B.zeros([4], 'uint8');
      expect(a.dtype).toBe('uint8');
      const d = await getData(a, B);
      expect(d).toEqual([0, 0, 0, 0]);
    });

    // >>> np.zeros(2, dtype=np.bool_)
    it('zeros bool', async () => {
      const B = getBackend();
      const a = B.zeros([2], 'bool');
      expect(a.dtype).toBe('bool');
      const d = await getData(a, B);
      expect(d).toEqual([0, 0]);
    });

    // Default dtype is float64
    // >>> np.zeros(3).dtype
    // dtype('float64')
    it('zeros default dtype is float64', async () => {
      const B = getBackend();
      const a = B.zeros([3]);
      expect(a.dtype).toBe('float64');
    });

    // === np.ones with dtype ===
    // >>> np.ones(3, dtype=np.int32)
    // array([1, 1, 1], dtype=int32)
    it('ones int32', async () => {
      const B = getBackend();
      const a = B.ones([3], 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      expect(d).toEqual([1, 1, 1]);
    });

    // >>> np.ones(3, dtype=np.float32)
    it('ones float32', async () => {
      const B = getBackend();
      const a = B.ones([3], 'float32');
      expect(a.dtype).toBe('float32');
      const d = await getData(a, B);
      expect(d).toEqual([1, 1, 1]);
    });

    // === np.full with dtype ===
    // >>> np.full(3, 7, dtype=np.int16)
    // array([7, 7, 7], dtype=int16)
    it('full int16', async () => {
      const B = getBackend();
      const a = B.full([3], 7, 'int16');
      expect(a.dtype).toBe('int16');
      const d = await getData(a, B);
      expect(d).toEqual([7, 7, 7]);
    });

    // === np.arange with dtype ===
    // >>> np.arange(0, 5, 1, dtype=np.int32)
    // array([0, 1, 2, 3, 4], dtype=int32)
    it('arange int32', async () => {
      const B = getBackend();
      const a = B.arange(0, 5, 1, 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      expect(d).toEqual([0, 1, 2, 3, 4]);
    });

    // === np.linspace with dtype ===
    // >>> np.linspace(0, 1, 5, dtype=np.float32)
    it('linspace float32', async () => {
      const B = getBackend();
      const a = B.linspace(0, 1, 5, 'float32');
      expect(a.dtype).toBe('float32');
      const d = await getData(a, B);
      expect(d.length).toBe(5);
      // float32 precision: values should be approximately [0, 0.25, 0.5, 0.75, 1.0]
      for (let i = 0; i < 5; i++) {
        expect(Math.abs(d[i] - i * 0.25)).toBeLessThan(1e-5);
      }
    });

    // === np.eye with dtype ===
    // >>> np.eye(2, dtype=np.float32)
    it('eye float32', async () => {
      const B = getBackend();
      const a = B.eye(2, 'float32');
      expect(a.dtype).toBe('float32');
      const d = await getData(a, B);
      expect(d).toEqual([1, 0, 0, 1]);
    });

    // === np.array with dtype ===
    // >>> np.array([1.5, 2.7, 3.9], dtype=np.int32)
    // array([1, 2, 3], dtype=int32)
    it('array with dtype int32 truncates floats', async () => {
      const B = getBackend();
      const a = B.array([1.5, 2.7, 3.9], undefined, 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      // Int32Array truncates toward zero
      expect(d).toEqual([1, 2, 3]);
    });

    // >>> np.array([0, 1, 0, 1], dtype=np.bool_)
    it('array with dtype bool', async () => {
      const B = getBackend();
      const a = B.array([0, 1, 0, 1], undefined, 'bool');
      expect(a.dtype).toBe('bool');
      const d = await getData(a, B);
      expect(d).toEqual([0, 1, 0, 1]);
    });

    // === np.identity with dtype ===
    // >>> np.identity(3, dtype=np.int32)
    it('identity int32', async () => {
      const B = getBackend();
      const a = B.identity(3, 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      expect(d).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    });

    // === np.empty with dtype ===
    it('empty float32', async () => {
      const B = getBackend();
      const a = B.empty([2, 3], 'float32');
      expect(a.dtype).toBe('float32');
      expect(a.shape).toEqual([2, 3]);
    });

    // === np.rand with dtype ===
    it('rand float32', async () => {
      const B = getBackend();
      const a = B.rand([10], 'float32');
      expect(a.dtype).toBe('float32');
      const d = await getData(a, B);
      for (const v of d) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('astype', () => {
    // >>> np.array([1.0, 2.5, 3.7]).astype(np.int32)
    // array([1, 2, 3], dtype=int32)
    it('float64 to int32', async () => {
      const B = getBackend();
      const a = B.array([1.0, 2.5, 3.7]);
      const b = B.astype(a, 'int32');
      expect(b.dtype).toBe('int32');
      const d = await getData(b, B);
      expect(d).toEqual([1, 2, 3]);
    });

    // >>> np.array([1, 2, 3], dtype=np.int32).astype(np.float64)
    // array([1., 2., 3.])
    it('int32 to float64', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3], undefined, 'int32');
      const b = B.astype(a, 'float64');
      expect(b.dtype).toBe('float64');
      const d = await getData(b, B);
      expect(d).toEqual([1, 2, 3]);
    });

    // >>> np.array([1.0, 2.0, 3.0]).astype(np.float32)
    it('float64 to float32', async () => {
      const B = getBackend();
      const a = B.array([1.0, 2.0, 3.0]);
      const b = B.astype(a, 'float32');
      expect(b.dtype).toBe('float32');
      const d = await getData(b, B);
      expect(d).toEqual([1, 2, 3]);
    });

    // >>> np.array([0, 1, 2, 255], dtype=np.uint8).astype(np.int8)
    // array([0, 1, 2, -1], dtype=int8)
    it('uint8 to int8 overflow', async () => {
      const B = getBackend();
      const a = B.array([0, 1, 2, 255], undefined, 'uint8');
      const b = B.astype(a, 'int8');
      expect(b.dtype).toBe('int8');
      const d = await getData(b, B);
      expect(d).toEqual([0, 1, 2, -1]);
    });

    // >>> np.array([300, -1], dtype=np.int32).astype(np.uint8)
    // array([44, 255], dtype=uint8)
    it('int32 to uint8 overflow', async () => {
      const B = getBackend();
      const a = B.array([300, -1], undefined, 'int32');
      const b = B.astype(a, 'uint8');
      expect(b.dtype).toBe('uint8');
      const d = await getData(b, B);
      expect(d).toEqual([44, 255]);
    });

    // astype preserves shape
    it('preserves shape', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3, 4], [2, 2]);
      const b = B.astype(a, 'float32');
      expect(b.shape).toEqual([2, 2]);
      expect(b.dtype).toBe('float32');
    });
  });

  describe('*Like functions preserve dtype', () => {
    // >>> a = np.ones(3, dtype=np.float32)
    // >>> np.zeros_like(a).dtype
    // dtype('float32')
    it('zerosLike preserves float32', async () => {
      const B = getBackend();
      const a = B.ones([3], 'float32');
      const b = B.zerosLike(a);
      expect(b.dtype).toBe('float32');
      const d = await getData(b, B);
      expect(d).toEqual([0, 0, 0]);
    });

    it('onesLike preserves int32', async () => {
      const B = getBackend();
      const a = B.zeros([3], 'int32');
      const b = B.onesLike(a);
      expect(b.dtype).toBe('int32');
      const d = await getData(b, B);
      expect(d).toEqual([1, 1, 1]);
    });

    it('fullLike preserves uint8', async () => {
      const B = getBackend();
      const a = B.zeros([3], 'uint8');
      const b = B.fullLike(a, 42);
      expect(b.dtype).toBe('uint8');
      const d = await getData(b, B);
      expect(d).toEqual([42, 42, 42]);
    });

    it('emptyLike preserves float32', async () => {
      const B = getBackend();
      const a = B.ones([2, 3], 'float32');
      const b = B.emptyLike(a);
      expect(b.dtype).toBe('float32');
      expect(b.shape).toEqual([2, 3]);
    });
  });

  describe('scalar broadcasting', () => {
    // === np.add(arr, scalar) ===
    // >>> np.add(np.array([1, 2, 3]), 5)
    // array([6, 7, 8])
    it('add array + scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.add(a, 5);
      const d = await getData(r, B);
      expect(d).toEqual([6, 7, 8]);
    });

    // >>> np.add(5, np.array([1, 2, 3]))
    // array([6, 7, 8])
    it('add scalar + array', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.add(5, a);
      const d = await getData(r, B);
      expect(d).toEqual([6, 7, 8]);
    });

    // >>> np.subtract(np.array([10, 20, 30]), 5)
    // array([ 5, 15, 25])
    it('subtract array - scalar', async () => {
      const B = getBackend();
      const a = B.array([10, 20, 30]);
      const r = B.subtract(a, 5);
      const d = await getData(r, B);
      expect(d).toEqual([5, 15, 25]);
    });

    // >>> np.subtract(100, np.array([10, 20, 30]))
    // array([90, 80, 70])
    it('subtract scalar - array', async () => {
      const B = getBackend();
      const a = B.array([10, 20, 30]);
      const r = B.subtract(100, a);
      const d = await getData(r, B);
      expect(d).toEqual([90, 80, 70]);
    });

    // >>> np.multiply(np.array([1, 2, 3]), 10)
    // array([10, 20, 30])
    it('multiply array * scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.multiply(a, 10);
      const d = await getData(r, B);
      expect(d).toEqual([10, 20, 30]);
    });

    // >>> np.divide(np.array([10, 20, 30]), 5)
    // array([2., 4., 6.])
    it('divide array / scalar', async () => {
      const B = getBackend();
      const a = B.array([10, 20, 30]);
      const r = B.divide(a, 5);
      const d = await getData(r, B);
      expect(d).toEqual([2, 4, 6]);
    });

    // >>> np.divide(100, np.array([2, 4, 5]))
    // array([50., 25., 20.])
    it('divide scalar / array', async () => {
      const B = getBackend();
      const a = B.array([2, 4, 5]);
      const r = B.divide(100, a);
      const d = await getData(r, B);
      expect(d).toEqual([50, 25, 20]);
    });

    // >>> np.power(np.array([2, 3, 4]), 2)
    // array([ 4,  9, 16])
    it('power array ** scalar', async () => {
      const B = getBackend();
      const a = B.array([2, 3, 4]);
      const r = B.power(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([4, 9, 16]);
    });

    // >>> np.power(2, np.array([1, 2, 3]))
    // array([2, 4, 8])
    it('power scalar ** array', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.power(2, a);
      const d = await getData(r, B);
      expect(d).toEqual([2, 4, 8]);
    });

    // >>> np.maximum(np.array([1, 5, 3]), 4)
    // array([4, 5, 4])
    it('maximum array vs scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 5, 3]);
      const r = B.maximum(a, 4);
      const d = await getData(r, B);
      expect(d).toEqual([4, 5, 4]);
    });

    // >>> np.minimum(np.array([1, 5, 3]), 4)
    // array([1, 4, 3])
    it('minimum array vs scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 5, 3]);
      const r = B.minimum(a, 4);
      const d = await getData(r, B);
      expect(d).toEqual([1, 4, 3]);
    });

    // === Comparison ops with scalars ===
    // >>> np.greater(np.array([1, 2, 3]), 2)
    // array([False, False,  True])
    it('greater array > scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.greater(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([0, 0, 1]);
    });

    // >>> np.less(np.array([1, 2, 3]), 2)
    // array([ True, False, False])
    it('less array < scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.less(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 0]);
    });

    // >>> np.equal(np.array([1, 2, 3]), 2)
    // array([False,  True, False])
    it('equal array == scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.equal(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([0, 1, 0]);
    });

    // >>> np.not_equal(np.array([1, 2, 3]), 2)
    // array([ True, False,  True])
    it('notEqual array != scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.notEqual(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 1]);
    });

    // >>> np.less_equal(np.array([1, 2, 3]), 2)
    // array([ True,  True, False])
    it('lessEqual array <= scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.lessEqual(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([1, 1, 0]);
    });

    // >>> np.greater_equal(np.array([1, 2, 3]), 2)
    // array([False,  True,  True])
    it('greaterEqual array >= scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.greaterEqual(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([0, 1, 1]);
    });

    // === Extended binary ops with scalars ===
    // >>> np.mod(np.array([7, 8, 9]), 3)
    // array([1, 2, 0])
    it('mod array % scalar', async () => {
      const B = getBackend();
      const a = B.array([7, 8, 9]);
      const r = B.mod(a, 3);
      const d = await getData(r, B);
      expect(d).toEqual([1, 2, 0]);
    });

    // >>> np.hypot(3, np.array([4, 5, 12]))
    // array([ 5.,  5.83..., 12.37...])
    it('hypot scalar and array', async () => {
      const B = getBackend();
      const a = B.array([4]);
      const r = B.hypot(3, a);
      const d = await getData(r, B);
      const tol = getTol(B);
      expect(approxEq(d[0], 5, tol)).toBe(true);
    });

    // === Logic ops with scalars ===
    // >>> np.logical_and(np.array([1, 0, 1]), 1)
    // array([ True, False,  True])
    it('logicalAnd array & scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 0, 1]);
      const r = B.logicalAnd(a, 1);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 1]);
    });

    // >>> np.logical_or(np.array([1, 0, 1]), 0)
    // array([ True, False,  True])
    it('logicalOr array | scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 0, 1]);
      const r = B.logicalOr(a, 0);
      const d = await getData(r, B);
      expect(d).toEqual([1, 0, 1]);
    });

    // === Bitwise ops with scalars ===
    // >>> np.bitwise_and(np.array([0b1100, 0b1010]), 0b1010)
    // array([8, 10])
    it('bitwiseAnd array & scalar', async () => {
      const B = getBackend();
      const a = B.array([0b1100, 0b1010]);
      const r = B.bitwiseAnd(a, 0b1010);
      const d = await getData(r, B);
      expect(d).toEqual([8, 10]);
    });

    // >>> np.left_shift(np.array([1, 2, 3]), 2)
    // array([ 4,  8, 12])
    it('leftShift array << scalar', async () => {
      const B = getBackend();
      const a = B.array([1, 2, 3]);
      const r = B.leftShift(a, 2);
      const d = await getData(r, B);
      expect(d).toEqual([4, 8, 12]);
    });

    // === Scalar + Scalar ===
    // >>> np.add(3, 5)
    // 8
    it('add scalar + scalar', async () => {
      const B = getBackend();
      const r = B.add(3, 5);
      const d = await getData(r, B);
      expect(d).toEqual([8]);
    });

    // >>> np.multiply(4, 7)
    // 28
    it('multiply scalar * scalar', async () => {
      const B = getBackend();
      const r = B.multiply(4, 7);
      const d = await getData(r, B);
      expect(d).toEqual([28]);
    });
  });

  describe('fromfunction with dtype', () => {
    // >>> np.fromfunction(lambda i, j: i + j, (2, 3), dtype=int)
    // array([[0, 1, 2],
    //        [1, 2, 3]])
    it('fromfunction int32', async () => {
      const B = getBackend();
      const a = B.fromfunction((i: number, j: number) => i + j, [2, 3], 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      expect(d).toEqual([0, 1, 2, 1, 2, 3]);
    });
  });

  describe('fromiter with dtype', () => {
    // >>> np.fromiter(range(5), dtype=np.float32)
    // array([0., 1., 2., 3., 4.], dtype=float32)
    it('fromiter float32', async () => {
      const B = getBackend();
      function* gen() {
        for (let i = 0; i < 5; i++) yield i;
      }
      const a = B.fromiter(gen(), 5, 'float32');
      expect(a.dtype).toBe('float32');
      const d = await getData(a, B);
      expect(d).toEqual([0, 1, 2, 3, 4]);
    });
  });

  describe('rand/randn/randint with dtype', () => {
    it('randint int32', async () => {
      const B = getBackend();
      const a = B.randint(0, 10, [5], 'int32');
      expect(a.dtype).toBe('int32');
      const d = await getData(a, B);
      for (const v of d) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(10);
        expect(Number.isInteger(v)).toBe(true);
      }
    });

    it('randn float32', async () => {
      const B = getBackend();
      const a = B.randn([10], 'float32');
      expect(a.dtype).toBe('float32');
    });
  });

  describe('int dtype overflow behavior (matching NumPy)', () => {
    // >>> np.array([127], dtype=np.int8) + np.array([1], dtype=np.int8)
    // In numpy: -128 (overflow wraps)
    // Our system: binary ops produce float64 so no overflow in the op itself
    // But astype should handle overflow correctly
    it('int8 overflow via astype', async () => {
      const B = getBackend();
      const a = B.array([128], undefined, 'float64');
      const b = B.astype(a, 'int8');
      const d = await getData(b, B);
      expect(d).toEqual([-128]); // 128 wraps to -128 in int8
    });

    it('uint8 overflow via astype', async () => {
      const B = getBackend();
      const a = B.array([256], undefined, 'float64');
      const b = B.astype(a, 'uint8');
      const d = await getData(b, B);
      expect(d).toEqual([0]); // 256 wraps to 0 in uint8
    });
  });
}
