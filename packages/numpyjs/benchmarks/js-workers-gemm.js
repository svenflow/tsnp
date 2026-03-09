/**
 * JS Web Workers GEMM implementation
 *
 * Avoids SharedArrayBuffer complexity by having each worker
 * load its own WASM instance. Workers receive their portion of
 * matrix A via postMessage (copies, but simple).
 */

// Worker code that will be stringified and run in workers
const workerCode = `
  let rumpy = null;

  self.onmessage = async function(e) {
    const { type, data } = e.data;

    if (type === 'init') {
      // Load WASM module
      const { wasmUrl } = data;
      const module = await import(wasmUrl);
      await module.default();
      rumpy = module;
      self.postMessage({ type: 'ready' });
    } else if (type === 'gemm') {
      // Perform matrix multiplication
      const { a, b, m, n, k, startRow, numRows } = data;

      // a is already the slice for this worker's rows
      const aSlice = new Float32Array(a);
      const bFull = new Float32Array(b);

      // Compute C slice
      const result = rumpy.matmulF32Auto(aSlice, bFull, numRows, n, k);

      self.postMessage({
        type: 'result',
        startRow,
        numRows,
        data: result
      }, [result.buffer]);  // Transfer buffer, don't copy
    }
  };
`;

/**
 * Worker pool for parallel GEMM
 */
export class GemmWorkerPool {
  constructor(wasmUrl, numWorkers = navigator.hardwareConcurrency || 4) {
    this.numWorkers = numWorkers;
    this.wasmUrl = wasmUrl;
    this.workers = [];
    this.ready = false;
  }

  async init() {
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);

    const initPromises = [];
    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(workerUrl, { type: 'module' });
      this.workers.push(worker);

      const initPromise = new Promise((resolve, reject) => {
        const handler = (e) => {
          if (e.data.type === 'ready') {
            worker.removeEventListener('message', handler);
            resolve();
          }
        };
        worker.addEventListener('message', handler);
        worker.addEventListener('error', reject);
      });

      worker.postMessage({
        type: 'init',
        data: { wasmUrl: this.wasmUrl }
      });

      initPromises.push(initPromise);
    }

    await Promise.all(initPromises);
    this.ready = true;
  }

  /**
   * Parallel matrix multiplication
   * @param {Float32Array} a - Matrix A, row-major, shape [m, k]
   * @param {Float32Array} b - Matrix B, row-major, shape [k, n]
   * @param {number} m - Number of rows in A
   * @param {number} n - Number of columns in B
   * @param {number} k - Shared dimension
   * @returns {Float32Array} Result matrix C, shape [m, n]
   */
  async matmul(a, b, m, n, k) {
    if (!this.ready) {
      throw new Error('Worker pool not initialized. Call init() first.');
    }

    // For small matrices, overhead is not worth it
    if (m * n * k < 64 * 64 * 64) {
      // Fall back to single-threaded (first worker)
      return this._singleWorkerMatmul(a, b, m, n, k);
    }

    // Split rows among workers
    const rowsPerWorker = Math.ceil(m / this.numWorkers);
    const results = new Float32Array(m * n);

    const promises = [];
    for (let i = 0; i < this.numWorkers; i++) {
      const startRow = i * rowsPerWorker;
      if (startRow >= m) break;

      const endRow = Math.min(startRow + rowsPerWorker, m);
      const numRows = endRow - startRow;

      // Extract this worker's portion of A
      const aSlice = a.slice(startRow * k, endRow * k);

      const promise = new Promise((resolve) => {
        const handler = (e) => {
          if (e.data.type === 'result') {
            this.workers[i].removeEventListener('message', handler);
            // Copy result into output array
            const { startRow: sr, numRows: nr, data } = e.data;
            results.set(data, sr * n);
            resolve();
          }
        };
        this.workers[i].addEventListener('message', handler);

        this.workers[i].postMessage({
          type: 'gemm',
          data: {
            a: aSlice.buffer,
            b: b.buffer,
            m, n, k,
            startRow,
            numRows
          }
        }, [aSlice.buffer]);  // Transfer A slice
      });

      promises.push(promise);
    }

    await Promise.all(promises);
    return results;
  }

  async _singleWorkerMatmul(a, b, m, n, k) {
    return new Promise((resolve) => {
      const handler = (e) => {
        if (e.data.type === 'result') {
          this.workers[0].removeEventListener('message', handler);
          resolve(new Float32Array(e.data.data));
        }
      };
      this.workers[0].addEventListener('message', handler);

      this.workers[0].postMessage({
        type: 'gemm',
        data: {
          a: a.buffer,
          b: b.buffer,
          m, n, k,
          startRow: 0,
          numRows: m
        }
      });
    });
  }

  terminate() {
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
    this.ready = false;
  }
}
