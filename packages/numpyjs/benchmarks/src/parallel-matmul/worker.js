/**
 * Web Worker for parallel matrix multiplication
 *
 * Each worker maintains its own WASM instance and processes a slice of rows.
 * Uses SharedArrayBuffer for zero-copy access to input matrices.
 */

let wasmModule = null;
let workerId = -1;

// Initialize the worker with WASM module
self.onmessage = async function(e) {
  const { type, ...data } = e.data;

  switch (type) {
    case 'init':
      await handleInit(data);
      break;
    case 'matmul':
      handleMatmul(data);
      break;
    default:
      console.error('Unknown message type:', type);
  }
};

async function handleInit({ id, wasmPath }) {
  workerId = id;
  try {
    // Dynamic import of the WASM module
    const module = await import(wasmPath);
    await module.default();
    wasmModule = module;
    self.postMessage({ type: 'ready', id: workerId });
  } catch (err) {
    self.postMessage({ type: 'error', id: workerId, error: err.message });
  }
}

function handleMatmul({ startRow, endRow, m, n, k, aBuffer, bBuffer, cBuffer }) {
  const localM = endRow - startRow;

  // Create views into shared buffers
  const aView = new Float32Array(aBuffer);
  const bView = new Float32Array(bBuffer);
  const cView = new Float32Array(cBuffer);

  // Extract our slice of A
  const aSlice = aView.subarray(startRow * k, endRow * k);

  // Compute using WASM SIMD kernel
  const cSlice = wasmModule.matmulF32Auto(
    new Float32Array(aSlice),  // Copy to non-shared buffer (WASM can't use SharedArrayBuffer directly)
    new Float32Array(bView),
    localM,
    n,
    k
  );

  // Write results back to shared buffer
  cView.set(cSlice, startRow * n);

  self.postMessage({ type: 'done', id: workerId, startRow, endRow });
}
