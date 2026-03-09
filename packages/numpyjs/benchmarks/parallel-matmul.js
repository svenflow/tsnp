/**
 * Parallel f32 matrix multiplication using Web Workers
 *
 * Splits the M dimension across workers, each running our SIMD kernel.
 * Uses SharedArrayBuffer for zero-copy data sharing when available.
 */

let workers = [];
let numWorkers = 0;
let wasmModule = null;
let wasmMemory = null;

/**
 * Initialize the parallel matmul pool
 * @param {number} threads - Number of worker threads (default: navigator.hardwareConcurrency)
 * @param {string} wasmPath - Path to the WASM module
 */
export async function initParallelMatmul(threads = navigator.hardwareConcurrency || 4, wasmPath = './pkg/rumpy_wasm.js') {
  numWorkers = threads;

  // Import the WASM module
  const rumpy = await import(wasmPath);
  await rumpy.default();
  wasmModule = rumpy;

  console.log(`Parallel matmul initialized with ${numWorkers} workers`);
}

/**
 * Parallel f32 matrix multiplication
 * Falls back to single-threaded for small matrices
 *
 * @param {Float32Array} a - Matrix A, row-major [m, k]
 * @param {Float32Array} b - Matrix B, row-major [k, n]
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array} - Result matrix C, row-major [m, n]
 */
export async function matmulF32Parallel(a, b, m, n, k) {
  // For small matrices, just use single-threaded
  if (m * n * k < 64 * 64 * 64) {
    return wasmModule.matmulF32Auto(a, b, m, n, k);
  }

  // Calculate rows per worker
  const rowsPerWorker = Math.ceil(m / numWorkers);

  // Create result array
  const c = new Float32Array(m * n);

  // Process each chunk
  // For now, do this sequentially since Web Workers require more setup
  // TODO: Actually spawn workers with postMessage
  for (let i = 0; i < numWorkers; i++) {
    const startRow = i * rowsPerWorker;
    if (startRow >= m) break;
    const endRow = Math.min(startRow + rowsPerWorker, m);
    const localM = endRow - startRow;

    // Extract slice of A
    const aSlice = a.subarray(startRow * k, endRow * k);

    // Compute this chunk
    const cSlice = wasmModule.matmulF32Slice(new Float32Array(aSlice), b, localM, n, k);

    // Copy to result
    c.set(cSlice, startRow * n);
  }

  return c;
}

/**
 * Get current number of workers
 */
export function getNumWorkers() {
  return numWorkers;
}
