/**
 * Web Worker script for WASM thread pool
 *
 * This worker receives a compiled WASM module and shared memory,
 * instantiates the module with the shared memory, and enters the
 * worker entry point which blocks forever waiting for tasks.
 *
 * IMPORTANT: We do NOT call _start or _initialize - that would reset
 * the shared memory state. Instead we instantiate with the existing
 * memory and jump directly to the worker entry function.
 */

let wasmInstance = null;
let threadIdx = -1;

self.onmessage = async (event) => {
  const { type, module, memory, stackBase, stackSize, idx, imports } = event.data;

  if (type === 'init') {
    threadIdx = idx;

    try {
      // Create import object with shared memory
      // The memory is already initialized by the main thread
      const importObject = {
        env: {
          memory: memory,
          // Standard WASI-like imports that workers might need
          // These should match what the main WASM module expects
        },
        wasi_snapshot_preview1: {
          // Minimal WASI stubs - workers shouldn't do I/O
          proc_exit: (code) => {
            console.error(`Worker ${threadIdx}: proc_exit called with code ${code}`);
            throw new Error(`Worker exit: ${code}`);
          },
          fd_write: () => 0,
          fd_read: () => 0,
          fd_close: () => 0,
          fd_seek: () => 0,
          environ_get: () => 0,
          environ_sizes_get: () => 0,
          args_get: () => 0,
          args_sizes_get: () => 0,
          clock_time_get: () => 0,
          random_get: (buf, len) => {
            const view = new Uint8Array(memory.buffer, buf, len);
            crypto.getRandomValues(view);
            return 0;
          },
        },
        // Atomics support through shared memory is automatic
        // when using SharedArrayBuffer
      };

      // Merge any additional imports provided by main thread
      if (imports) {
        for (const [namespace, funcs] of Object.entries(imports)) {
          importObject[namespace] = { ...importObject[namespace], ...funcs };
        }
      }

      // Instantiate the module with shared memory
      // Do NOT call _start - memory is already initialized
      wasmInstance = await WebAssembly.instantiate(module, importObject);

      // Set up worker's stack pointer if the WASM module supports it
      if (wasmInstance.exports.__stack_pointer) {
        // Each worker gets its own stack region
        // Stack grows downward, so we set SP to the top of our region
        const stackTop = stackBase + stackSize;
        wasmInstance.exports.__stack_pointer.value = stackTop;
      }

      // Signal ready to main thread
      self.postMessage({ type: 'ready', idx: threadIdx });

      // Enter the worker loop - this should block forever
      // using Atomics.wait internally
      if (wasmInstance.exports.threadpool_worker_entry) {
        wasmInstance.exports.threadpool_worker_entry(threadIdx);
        // If we return, the worker is shutting down
        self.postMessage({ type: 'exited', idx: threadIdx });
      } else {
        throw new Error('WASM module missing threadpool_worker_entry export');
      }

    } catch (err) {
      self.postMessage({
        type: 'error',
        idx: threadIdx,
        error: err.message,
        stack: err.stack
      });
    }
  } else if (type === 'shutdown') {
    // Clean shutdown requested
    // The Rust side should handle this via atomic flag
    self.postMessage({ type: 'shutdown_ack', idx: threadIdx });
    self.close();
  }
};

// Handle errors
self.onerror = (event) => {
  self.postMessage({
    type: 'error',
    idx: threadIdx,
    error: event.message,
    filename: event.filename,
    lineno: event.lineno
  });
};
