/**
 * WASM Thread Pool Coordinator
 *
 * Creates and manages Web Workers that share WASM memory for parallel
 * computation. Uses SharedArrayBuffer and Atomics for synchronization.
 *
 * Requirements:
 * - Must be served with COOP/COEP headers for SharedArrayBuffer
 * - WASM module must export: threadpool_worker_entry, threadpool_dispatch
 * - WASM module must be compiled with shared-memory and atomics features
 */

const DEFAULT_STACK_SIZE = 1024 * 1024; // 1MB per worker stack
const DEFAULT_MEMORY_PAGES = 256;       // 16MB initial (256 * 64KB)
const MAX_MEMORY_PAGES = 16384;         // 1GB max

export class ThreadPool {
  /**
   * @private
   */
  constructor() {
    this.workers = [];
    this.memory = null;
    this.module = null;
    this.instance = null;
    this.numWorkers = 0;
    this.readyCount = 0;
    this.readyPromise = null;
    this.readyResolve = null;
    this.isShutdown = false;
    this.workerScriptUrl = null;
  }

  /**
   * Create a new thread pool with the specified number of workers
   *
   * @param {number} numWorkers - Number of worker threads
   * @param {Object} options - Configuration options
   * @param {string} options.wasmUrl - URL to the WASM module
   * @param {ArrayBuffer} [options.wasmBytes] - Pre-loaded WASM bytes (alternative to wasmUrl)
   * @param {number} [options.initialMemoryPages] - Initial memory pages (64KB each)
   * @param {number} [options.maxMemoryPages] - Maximum memory pages
   * @param {number} [options.stackSize] - Stack size per worker in bytes
   * @param {string} [options.workerUrl] - URL to worker.js (defaults to same directory)
   * @param {Object} [options.imports] - Additional WASM imports
   * @returns {Promise<ThreadPool>}
   */
  static async create(numWorkers, options = {}) {
    const pool = new ThreadPool();

    // Check for SharedArrayBuffer support
    if (typeof SharedArrayBuffer === 'undefined') {
      throw new Error(
        'SharedArrayBuffer not available. ' +
        'Ensure COOP/COEP headers are set: ' +
        'Cross-Origin-Opener-Policy: same-origin, ' +
        'Cross-Origin-Embedder-Policy: require-corp'
      );
    }

    const {
      wasmUrl,
      wasmBytes,
      initialMemoryPages = DEFAULT_MEMORY_PAGES,
      maxMemoryPages = MAX_MEMORY_PAGES,
      stackSize = DEFAULT_STACK_SIZE,
      workerUrl,
      imports = {}
    } = options;

    if (!wasmUrl && !wasmBytes) {
      throw new Error('Either wasmUrl or wasmBytes must be provided');
    }

    pool.numWorkers = numWorkers;

    // Create shared memory
    pool.memory = new WebAssembly.Memory({
      initial: initialMemoryPages,
      maximum: maxMemoryPages,
      shared: true
    });

    // Compile WASM module
    let wasmSource = wasmBytes;
    if (!wasmSource) {
      const response = await fetch(wasmUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch WASM: ${response.status} ${response.statusText}`);
      }
      wasmSource = await response.arrayBuffer();
    }

    pool.module = await WebAssembly.compile(wasmSource);

    // Create main thread instance (initializes memory)
    const importObject = pool._buildImports(imports);
    pool.instance = await WebAssembly.instantiate(pool.module, importObject);

    // Call _start or _initialize if present to set up memory
    if (pool.instance.exports._start) {
      pool.instance.exports._start();
    } else if (pool.instance.exports._initialize) {
      pool.instance.exports._initialize();
    }

    // Initialize the thread pool in WASM (allocates work queues, etc.)
    if (pool.instance.exports.threadpool_init) {
      pool.instance.exports.threadpool_init(numWorkers);
    }

    // Allocate stack space for workers from WASM heap
    const stackAllocations = [];
    if (pool.instance.exports.threadpool_alloc_stack) {
      for (let i = 0; i < numWorkers; i++) {
        const stackBase = pool.instance.exports.threadpool_alloc_stack(stackSize);
        if (stackBase === 0) {
          throw new Error(`Failed to allocate stack for worker ${i}`);
        }
        stackAllocations.push({ base: stackBase, size: stackSize });
      }
    } else {
      // Fallback: allocate from end of linear memory
      // This is less ideal but works for simple cases
      const memorySize = pool.memory.buffer.byteLength;
      const totalStackSize = stackSize * numWorkers;
      const stackRegionStart = memorySize - totalStackSize;
      for (let i = 0; i < numWorkers; i++) {
        stackAllocations.push({
          base: stackRegionStart + (i * stackSize),
          size: stackSize
        });
      }
    }

    // Create promise that resolves when all workers are ready
    pool.readyPromise = new Promise((resolve) => {
      pool.readyResolve = resolve;
    });

    // Resolve worker URL
    pool.workerScriptUrl = workerUrl || new URL('worker.js', import.meta.url).href;

    // Spawn workers
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(pool.workerScriptUrl, { type: 'module' });

      worker.onmessage = (event) => pool._handleWorkerMessage(event, i);
      worker.onerror = (event) => {
        console.error(`Worker ${i} error:`, event);
      };

      pool.workers.push(worker);

      // Send initialization data
      worker.postMessage({
        type: 'init',
        module: pool.module,
        memory: pool.memory,
        stackBase: stackAllocations[i].base,
        stackSize: stackAllocations[i].size,
        idx: i,
        imports: pool._serializableImports(imports)
      });
    }

    // Wait for all workers to be ready
    await pool.readyPromise;

    return pool;
  }

  /**
   * Build import object for WASM instantiation
   * @private
   */
  _buildImports(customImports) {
    const imports = {
      env: {
        memory: this.memory,
      },
      wasi_snapshot_preview1: {
        proc_exit: (code) => {
          if (code !== 0) {
            console.error(`WASM proc_exit with code ${code}`);
          }
        },
        fd_write: (fd, iovs, iovs_len, nwritten_ptr) => {
          // Basic stdout/stderr support for main thread
          const view = new DataView(this.memory.buffer);
          const memory8 = new Uint8Array(this.memory.buffer);
          let written = 0;

          for (let i = 0; i < iovs_len; i++) {
            const ptr = view.getUint32(iovs + i * 8, true);
            const len = view.getUint32(iovs + i * 8 + 4, true);
            const bytes = memory8.slice(ptr, ptr + len);
            const text = new TextDecoder().decode(bytes);

            if (fd === 1) {
              console.log(text);
            } else if (fd === 2) {
              console.error(text);
            }
            written += len;
          }

          view.setUint32(nwritten_ptr, written, true);
          return 0;
        },
        fd_read: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0,
        fd_fdstat_get: () => 0,
        fd_prestat_get: () => 8, // EBADF
        fd_prestat_dir_name: () => 8,
        environ_get: () => 0,
        environ_sizes_get: (count_ptr, size_ptr) => {
          const view = new DataView(this.memory.buffer);
          view.setUint32(count_ptr, 0, true);
          view.setUint32(size_ptr, 0, true);
          return 0;
        },
        args_get: () => 0,
        args_sizes_get: (argc_ptr, argv_buf_size_ptr) => {
          const view = new DataView(this.memory.buffer);
          view.setUint32(argc_ptr, 0, true);
          view.setUint32(argv_buf_size_ptr, 0, true);
          return 0;
        },
        clock_time_get: (clock_id, precision, time_ptr) => {
          const view = new DataView(this.memory.buffer);
          const time = BigInt(Math.floor(performance.now() * 1_000_000));
          view.setBigUint64(time_ptr, time, true);
          return 0;
        },
        random_get: (buf, len) => {
          const view = new Uint8Array(this.memory.buffer, buf, len);
          crypto.getRandomValues(view);
          return 0;
        },
        sched_yield: () => 0,
      },
    };

    // Merge custom imports
    for (const [namespace, funcs] of Object.entries(customImports)) {
      imports[namespace] = { ...imports[namespace], ...funcs };
    }

    return imports;
  }

  /**
   * Create serializable version of imports for workers
   * (Functions can't be sent via postMessage)
   * @private
   */
  _serializableImports(imports) {
    // Workers get minimal imports - just names to set up stubs
    // The actual imports are defined in worker.js
    return null;
  }

  /**
   * Handle messages from workers
   * @private
   */
  _handleWorkerMessage(event, workerIdx) {
    const { type, idx, error, stack } = event.data;

    switch (type) {
      case 'ready':
        this.readyCount++;
        if (this.readyCount === this.numWorkers) {
          this.readyResolve();
        }
        break;

      case 'error':
        console.error(`Worker ${idx} error:`, error);
        if (stack) console.error(stack);
        break;

      case 'exited':
        console.log(`Worker ${idx} exited`);
        break;

      case 'shutdown_ack':
        console.log(`Worker ${idx} shutdown acknowledged`);
        break;

      default:
        console.warn(`Unknown message from worker ${idx}:`, event.data);
    }
  }

  /**
   * Dispatch work to the thread pool
   *
   * This calls into the WASM module's dispatch function which:
   * 1. Sets up the work item in shared memory
   * 2. Wakes workers via Atomics.notify
   * 3. Optionally waits for completion
   *
   * @param {number} dataPtr - Pointer to data in WASM memory
   * @param {number} m - First dimension
   * @param {number} k - Second dimension
   * @param {number} n - Third dimension
   * @param {number} taskFnPtr - Function pointer for the task
   * @returns {number} Result from WASM dispatch function
   */
  dispatch(dataPtr, m, k, n, taskFnPtr) {
    if (this.isShutdown) {
      throw new Error('Thread pool has been shut down');
    }

    if (!this.instance.exports.threadpool_dispatch) {
      throw new Error('WASM module missing threadpool_dispatch export');
    }

    return this.instance.exports.threadpool_dispatch(dataPtr, m, k, n, taskFnPtr);
  }

  /**
   * Generic dispatch with arbitrary arguments
   *
   * @param {string} funcName - Name of the exported WASM function to call
   * @param  {...any} args - Arguments to pass to the function
   * @returns {any} Return value from the WASM function
   */
  call(funcName, ...args) {
    if (this.isShutdown) {
      throw new Error('Thread pool has been shut down');
    }

    const func = this.instance.exports[funcName];
    if (!func) {
      throw new Error(`WASM module missing ${funcName} export`);
    }

    return func(...args);
  }

  /**
   * Wake workers that are waiting on a futex
   *
   * @param {number} address - Address of the atomic to notify (must be 4-byte aligned)
   * @param {number} count - Number of workers to wake (Infinity for all)
   * @returns {number} Number of workers woken
   */
  notify(address, count = Infinity) {
    const view = new Int32Array(this.memory.buffer);
    return Atomics.notify(view, address >> 2, count);
  }

  /**
   * Get a typed view of WASM memory
   *
   * @param {string} type - One of: 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'f32', 'f64'
   * @returns {TypedArray} Typed array view of memory
   */
  getMemoryView(type) {
    const buffer = this.memory.buffer;
    switch (type) {
      case 'i8': return new Int8Array(buffer);
      case 'u8': return new Uint8Array(buffer);
      case 'i16': return new Int16Array(buffer);
      case 'u16': return new Uint16Array(buffer);
      case 'i32': return new Int32Array(buffer);
      case 'u32': return new Uint32Array(buffer);
      case 'f32': return new Float32Array(buffer);
      case 'f64': return new Float64Array(buffer);
      default: throw new Error(`Unknown type: ${type}`);
    }
  }

  /**
   * Allocate memory in WASM heap
   *
   * @param {number} size - Size in bytes
   * @returns {number} Pointer to allocated memory, or 0 on failure
   */
  alloc(size) {
    if (this.instance.exports.threadpool_alloc) {
      return this.instance.exports.threadpool_alloc(size);
    }
    throw new Error('WASM module missing threadpool_alloc export');
  }

  /**
   * Free memory in WASM heap
   *
   * @param {number} ptr - Pointer to free
   */
  free(ptr) {
    if (this.instance.exports.threadpool_free) {
      this.instance.exports.threadpool_free(ptr);
    } else {
      throw new Error('WASM module missing threadpool_free export');
    }
  }

  /**
   * Shut down the thread pool
   *
   * Signals all workers to exit and terminates them.
   */
  shutdown() {
    if (this.isShutdown) return;
    this.isShutdown = true;

    // Signal WASM side to shut down (sets atomic flag)
    if (this.instance.exports.threadpool_shutdown) {
      this.instance.exports.threadpool_shutdown();
    }

    // Wake all workers so they see the shutdown flag
    if (this.instance.exports.threadpool_get_futex_addr) {
      const futexAddr = this.instance.exports.threadpool_get_futex_addr();
      this.notify(futexAddr, this.numWorkers);
    }

    // Give workers a moment to exit gracefully, then terminate
    setTimeout(() => {
      for (const worker of this.workers) {
        worker.postMessage({ type: 'shutdown' });
      }

      setTimeout(() => {
        for (const worker of this.workers) {
          worker.terminate();
        }
        this.workers = [];
      }, 100);
    }, 50);
  }

  /**
   * Get number of workers in the pool
   * @returns {number}
   */
  get size() {
    return this.numWorkers;
  }
}

// Default export for convenience
export default ThreadPool;
