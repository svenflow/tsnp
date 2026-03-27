# WASM Thread Pool JavaScript Infrastructure

JavaScript infrastructure for spawning Web Workers that run WASM thread pool workers with shared memory.

## Files

- **worker.js** - Web Worker script that receives WASM module + shared memory and enters the worker loop
- **threadpool.js** - Main thread coordinator that creates workers and exposes dispatch API

## Requirements

### COOP/COEP Headers

SharedArrayBuffer requires these headers on the serving page:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Example with a simple HTTP server:

```javascript
// Node.js example
const http = require('http');
const fs = require('fs');
const path = require('path');

http.createServer((req, res) => {
  // Required headers for SharedArrayBuffer
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  // ... serve files
}).listen(8080);
```

### WASM Module Exports

The WASM module must export these functions:

```rust
// Required exports
#[no_mangle]
pub extern "C" fn threadpool_worker_entry(thread_idx: u32);

#[no_mangle]
pub extern "C" fn threadpool_dispatch(data: *mut u8, m: u32, k: u32, n: u32, task_fn: *const ()) -> i32;

// Optional but recommended
#[no_mangle]
pub extern "C" fn threadpool_init(num_workers: u32);

#[no_mangle]
pub extern "C" fn threadpool_shutdown();

#[no_mangle]
pub extern "C" fn threadpool_get_futex_addr() -> *const i32;

#[no_mangle]
pub extern "C" fn threadpool_alloc_stack(size: u32) -> *mut u8;

#[no_mangle]
pub extern "C" fn threadpool_alloc(size: u32) -> *mut u8;

#[no_mangle]
pub extern "C" fn threadpool_free(ptr: *mut u8);
```

### WASM Build Flags

Build the WASM module with atomics and shared memory:

```bash
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
  cargo build --target wasm32-unknown-unknown -Z build-std=std,panic_abort
```

Or in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+atomics,+bulk-memory,+mutable-globals"]

[unstable]
build-std = ["std", "panic_abort"]
```

## Usage

### Basic Example

```javascript
import { ThreadPool } from './threadpool.js';

// Create pool with 4 workers
const pool = await ThreadPool.create(4, {
  wasmUrl: './rumpy.wasm',
  stackSize: 1024 * 1024,  // 1MB per worker
});

// Allocate data in WASM memory
const dataPtr = pool.alloc(1024);
const view = pool.getMemoryView('f32');

// Fill data...
view[dataPtr / 4] = 1.0;
view[dataPtr / 4 + 1] = 2.0;
// ...

// Dispatch parallel work
// This wakes workers and waits for completion
const result = pool.dispatch(dataPtr, 64, 64, 64, taskFnPtr);

// Clean up
pool.free(dataPtr);
pool.shutdown();
```

### With Pre-loaded WASM

```javascript
const wasmBytes = await fetch('./rumpy.wasm').then(r => r.arrayBuffer());

const pool = await ThreadPool.create(navigator.hardwareConcurrency, {
  wasmBytes,
  initialMemoryPages: 512,  // 32MB
  maxMemoryPages: 8192,     // 512MB
});
```

### Calling Arbitrary WASM Functions

```javascript
// Generic function call
const result = pool.call('matmul', aPtr, bPtr, cPtr, m, k, n);

// Direct access to exports
const sumResult = pool.instance.exports.vector_sum(dataPtr, length);
```

### Memory Access

```javascript
// Get typed views
const f32View = pool.getMemoryView('f32');
const i32View = pool.getMemoryView('i32');

// Read/write data
f32View[ptr / 4] = 3.14;
const value = f32View[ptr / 4];

// Bulk copy
const data = new Float32Array([1, 2, 3, 4]);
f32View.set(data, ptr / 4);
```

### Manual Worker Wake (Advanced)

```javascript
// Get the futex address from WASM
const futexAddr = pool.instance.exports.threadpool_get_futex_addr();

// Wake specific number of workers
const woken = pool.notify(futexAddr, 2);  // Wake 2 workers
console.log(`Woke ${woken} workers`);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ThreadPool                                          │    │
│  │  - Creates SharedArrayBuffer memory                  │    │
│  │  - Compiles WASM module                              │    │
│  │  - Instantiates main WASM instance (_start runs)    │    │
│  │  - Spawns N workers                                  │    │
│  │  - Exposes dispatch() API                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ postMessage(module, memory)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Web Workers                             │
│  ┌─────────────┐  ┌─────────────┐      ┌─────────────┐     │
│  │  Worker 0   │  │  Worker 1   │ ···  │  Worker N   │     │
│  │             │  │             │      │             │     │
│  │ Instantiate │  │ Instantiate │      │ Instantiate │     │
│  │ (no _start) │  │ (no _start) │      │ (no _start) │     │
│  │             │  │             │      │             │     │
│  │ worker_     │  │ worker_     │      │ worker_     │     │
│  │ entry(idx)  │  │ entry(idx)  │      │ entry(idx)  │     │
│  │     │       │  │     │       │      │     │       │     │
│  │     ▼       │  │     ▼       │      │     ▼       │     │
│  │ Atomics.    │  │ Atomics.    │      │ Atomics.    │     │
│  │ wait()      │  │ wait()      │      │ wait()      │     │
│  └─────────────┘  └─────────────┘      └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Shared Memory
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 SharedArrayBuffer                            │
│  ┌──────────┬──────────┬──────────┬─────────────────────┐  │
│  │  WASM    │  Work    │  Data    │  Worker Stacks      │  │
│  │  Heap    │  Queue   │  Buffers │  (separate regions) │  │
│  └──────────┴──────────┴──────────┴─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Points

### Workers Don't Call _start

Workers receive a pre-compiled WASM module and shared memory. They instantiate the module with the existing memory but do NOT call `_start` or `_initialize`. This is critical because:

1. `_start` would reinitialize globals and heap, corrupting shared state
2. Memory is already set up by the main thread
3. Workers jump directly to `threadpool_worker_entry()`

### Separate Stack Space

Each worker needs its own stack for function calls. The main thread allocates stack regions from the WASM heap (via `threadpool_alloc_stack`) and passes the base address to each worker. The worker sets its `__stack_pointer` to the top of its region.

### Atomics for Synchronization

Workers use `Atomics.wait()` to sleep efficiently until work arrives. The main thread (or WASM dispatch function) uses `Atomics.notify()` to wake workers. This is much more efficient than busy-polling.

### Worker Entry Never Returns (Usually)

The `threadpool_worker_entry()` function contains an infinite loop:

```rust
pub extern "C" fn threadpool_worker_entry(thread_idx: u32) {
    loop {
        // Wait for work
        futex_wait(&WORK_QUEUE.futex, 0);

        // Check shutdown flag
        if SHUTDOWN.load(Ordering::Relaxed) {
            return;
        }

        // Process work item
        if let Some(task) = WORK_QUEUE.pop() {
            (task.func)(task.data, thread_idx);
        }
    }
}
```

## Troubleshooting

### SharedArrayBuffer is undefined

The page is not served with COOP/COEP headers. Check your server configuration.

### Workers report "memory access out of bounds"

- Stack size too small - increase `stackSize` option
- Memory not large enough - increase `initialMemoryPages`
- Bug in stack allocation - ensure workers have non-overlapping stack regions

### Workers hang and never become ready

- `threadpool_worker_entry` might be missing or crashing
- Check browser console for errors
- Ensure WASM was built with atomics support

### Performance is worse than single-threaded

- Work items too small - parallelize larger chunks
- Too much contention on shared data
- Memory bandwidth limited - consider data layout

## Browser Support

Requires browsers with:
- Web Workers
- WebAssembly
- SharedArrayBuffer (Chrome 67+, Firefox 79+, Safari 15.2+)
- Atomics (same versions)

Note: SharedArrayBuffer was disabled in most browsers in 2018 due to Spectre, then re-enabled with site isolation and COOP/COEP requirements.
