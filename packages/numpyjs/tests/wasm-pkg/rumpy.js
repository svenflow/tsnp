/**
 * rumpy.js - High-level tensor API with automatic B-packing optimization
 *
 * This wrapper adds tfjs-like semantics where weight tensors are automatically
 * packed for fast matmul on first use.
 */

import * as wasm from './rumpy_wasm.js';

// Re-export everything from wasm module
export * from './rumpy_wasm.js';

// WeakMap to cache packed B matrices per Float32Array
// Key: original Float32Array, Value: { packed: Float32Array, k: number, n: number }
const packedBCache = new WeakMap();

// Minimum size to enable packing (packing overhead not worth it below this)
const MIN_PACK_SIZE = 128;

/**
 * Smart matmul that automatically caches packed B matrices.
 *
 * On first call with a given B matrix, packs and caches it.
 * On subsequent calls with the same B, uses the cached packed version.
 *
 * This matches tfjs semantics where weights are optimized on first use.
 *
 * @param {Float32Array} a - Input matrix [m, k]
 * @param {Float32Array} b - Weight matrix [k, n]
 * @param {number} m - Rows of A
 * @param {number} n - Cols of B
 * @param {number} k - Shared dimension
 * @returns {Float32Array} - Result matrix [m, n]
 */
export function matmul(a, b, m, n, k) {
  // Check if packing is worthwhile
  if (k < MIN_PACK_SIZE || n < MIN_PACK_SIZE) {
    // Small matrix - use direct path (packing overhead > benefit)
    return wasm.matmulF32Optimized(a, b, m, n, k);
  }

  // Check cache for packed B
  let cached = packedBCache.get(b);

  if (cached) {
    // Validate cached dimensions match
    if (cached.k === k && cached.n === n) {
      // Use fast pre-packed path
      return wasm.matmulWithPackedB(a, cached.packed, m, n, k);
    }
    // Dimensions changed - invalidate cache
    packedBCache.delete(b);
  }

  // First use of this B - pack and cache
  const packed = wasm.packBForGemm(b, k, n);
  packedBCache.set(b, { packed, k, n });

  // Use packed path
  return wasm.matmulWithPackedB(a, packed, m, n, k);
}

/**
 * Clear the packed B cache for a specific matrix.
 * Call this if you modify a weight matrix in-place.
 *
 * @param {Float32Array} b - The matrix to uncache
 */
export function clearPackedCache(b) {
  packedBCache.delete(b);
}

/**
 * Clear all packed B caches.
 * Useful when switching models or freeing memory.
 */
export function clearAllPackedCaches() {
  // WeakMap doesn't have a clear() method, so we just create a new one
  // The old entries will be GC'd when their keys are no longer referenced
  // For now, this is a no-op since WeakMap handles cleanup automatically
}

// Default export for convenience
export default {
  ...wasm,
  matmul,
  clearPackedCache,
  clearAllPackedCaches,
};
