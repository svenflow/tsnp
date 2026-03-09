# pthreadpool-rs Design Document

## Overview

This document describes the architecture and design decisions for reimplementing pthreadpool in Rust. pthreadpool is a portable and efficient thread pool library originally written in C, designed for parallelizing computational workloads with minimal overhead.

**Source Repository**: https://github.com/Maratyszcza/pthreadpool

## Architecture Overview

### Core Design Philosophy

pthreadpool is optimized for:
1. **Low latency** - Minimal overhead for starting parallel work
2. **High throughput** - Efficient work distribution and load balancing
3. **Work stealing** - Automatic load balancing when work is unevenly distributed
4. **Wait-free synchronization** - Using atomic operations for work item distribution

### Structural Components

```
+------------------------+
|     pthreadpool        |
|------------------------|
| - threads_count        |  FXdiv divisor for fast modulo
| - active_threads       |  Atomic counter
| - command              |  Atomic command word
| - thread_function      |  Current parallelize function
| - task                 |  User callback
| - argument             |  User context
| - params               |  Union of all param structs
| - flags                |  Operation flags
| - execution_mutex      |  Serializes parallel calls
| - threads[]            |  Flexible array of thread_info
+------------------------+

+------------------------+
|     thread_info        |  (64-byte cache-line aligned)
|------------------------|
| - range_start          |  Atomic - start of owned range
| - range_end            |  Atomic - end of owned range
| - range_length         |  Atomic - items remaining
| - thread_number        |  Thread index (0..N-1)
| - threadpool           |  Back-pointer to pool
| - thread_object        |  pthread_t / HANDLE
+------------------------+
```

### Work Distribution Model

The key insight is that pthreadpool linearizes multi-dimensional loops into a 1D range and distributes work statically across threads, then uses work stealing for dynamic load balancing.

1. **Initial Distribution**: Work is split evenly among threads
   - Each thread gets `range/threads_count` items (with remainder distributed)
   - `range_start` initialized to beginning of thread's chunk
   - `range_end` initialized to end of thread's chunk
   - `range_length` initialized to chunk size

2. **Local Processing**: Each thread processes its own range from front
   ```
   while (decrement_fetch(range_length) < threshold) {
       task(argument, range_start++);
   }
   ```

3. **Work Stealing**: When local work exhausted, steal from other threads
   ```
   for each other_thread (in reverse order):
       while (decrement_fetch(other_thread.range_length) < threshold) {
           index = decrement_fetch(other_thread.range_end);
           task(argument, index);
       }
   ```

### Synchronization Strategy

pthreadpool uses multiple synchronization backends:

| Platform | Backend | Implementation |
|----------|---------|----------------|
| Linux    | futex   | `syscall(SYS_futex, ...)` |
| macOS    | GCD     | `dispatch_apply_f()` |
| Windows  | Event   | `CreateEvent/WaitForSingleObject` |
| Emscripten | futex | `emscripten_futex_wait()` |
| Fallback | condvar | `pthread_cond_wait()` |

#### Key Synchronization Points

1. **Command Notification** (main -> workers)
   - Main thread stores new command with release semantics
   - Workers spin-wait then fall back to futex/condvar

2. **Completion Notification** (workers -> main)
   - Workers atomically decrement `active_threads`
   - Last worker wakes main thread

3. **Work Item Claiming**
   - All work items claimed via atomic decrement
   - No locks needed for actual work distribution

### The "Fastpath" Optimization

On x86/x86-64, pthreadpool uses a "fastpath" that replaces compare-and-swap with atomic decrement:

```c
// Fastpath (x86): Uses underflow detection
const size_t range_threshold = -threads_count;
while (decrement_fetch_relaxed(&thread->range_length) < range_threshold) {
    task(argument, range_start++);
}

// Portable path: Uses CAS loop
while (try_decrement_relaxed(&thread->range_length)) {
    task(argument, range_start++);
}
```

The fastpath works because:
- `range_length` is initialized to a positive value
- Decrementing past 0 causes underflow to a large negative number
- Checking `< -threads_count` detects this reliably

## Key APIs to Implement

### Core APIs

```rust
/// Create a thread pool
pub fn create(threads_count: usize) -> Option<ThreadPool>;

/// Query thread count
pub fn get_threads_count(&self) -> usize;

/// Destroy thread pool
pub fn destroy(self);
```

### Parallelization APIs (Priority Order)

#### Essential (used by ONNX Runtime, PyTorch)

1. `parallelize_1d` - Basic 1D parallel loop
2. `parallelize_2d_tile_2d` - 2D loop with tiling (most important for GEMM)
3. `parallelize_1d_tile_1d` - 1D loop with tiling

#### Secondary

4. `parallelize_2d` - Basic 2D loop
5. `parallelize_2d_tile_1d` - 2D with 1D tiling
6. `parallelize_3d_tile_2d` - 3D with 2D tiling (for conv)
7. `parallelize_4d_tile_2d` - 4D with 2D tiling

### Task Function Signatures

```rust
// 1D
type Task1D = fn(context: *mut c_void, i: usize);
type Task1DTile1D = fn(context: *mut c_void, start: usize, tile: usize);

// 2D
type Task2D = fn(context: *mut c_void, i: usize, j: usize);
type Task2DTile2D = fn(context: *mut c_void,
    start_i: usize, start_j: usize,
    tile_i: usize, tile_j: usize);

// With thread index (for thread-local buffers)
type Task1DWithThread = fn(context: *mut c_void, thread_index: usize, i: usize);
```

### Flags

```rust
pub const FLAG_DISABLE_DENORMALS: u32 = 0x00000001;
pub const FLAG_YIELD_WORKERS: u32 = 0x00000002;
```

## Rust-Specific Design Decisions

### Memory Layout

```rust
#[repr(C, align(64))]  // Cache-line aligned
pub struct ThreadInfo {
    range_start: AtomicUsize,
    range_end: AtomicUsize,
    range_length: AtomicUsize,
    thread_number: usize,
    threadpool: *const ThreadPool,
    // Platform-specific thread handle
    #[cfg(not(target_arch = "wasm32"))]
    thread: Option<JoinHandle<()>>,
}

#[repr(C)]
pub struct ThreadPool {
    threads_count: usize,
    active_threads: AtomicUsize,
    command: AtomicU32,
    // ...
    threads: Box<[ThreadInfo]>,  // Dynamically sized
}
```

### Synchronization Primitives

```rust
// Use parking_lot for efficiency on all platforms
use parking_lot::{Mutex, Condvar};

// For WASM with threads, use:
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod wasm_sync {
    // Use std::sync::atomic with Ordering::SeqCst for SharedArrayBuffer
    // Use wasm_bindgen's Atomics::wait/notify for parking
}
```

### Task Closure Safety

The C API passes raw pointers. For Rust, we need to handle this safely:

```rust
// Low-level C-compatible API
pub unsafe fn parallelize_1d_raw(
    pool: &ThreadPool,
    task: Task1D,
    context: *mut c_void,
    range: usize,
    flags: u32,
);

// Safe Rust API
pub fn parallelize_1d<F, T>(
    pool: &ThreadPool,
    context: &T,
    range: usize,
    task: F,
) where F: Fn(&T, usize) + Sync;
```

### Fast Division (FXdiv equivalent)

The C code uses FXdiv for fast constant division. In Rust:

```rust
// Use the fastdiv crate or implement similar
struct FastDivisor {
    value: usize,
    multiplier: u64,
    shift: u32,
}

impl FastDivisor {
    fn divide(&self, dividend: usize) -> (usize, usize) {
        // quotient and remainder
    }
}
```

## WASM-Specific Considerations

### Threading Model

WASM has several threading configurations:

1. **No threads** (default): Use shim implementation (serial)
2. **SharedArrayBuffer + Atomics**: Full threading support
3. **wasm-bindgen-rayon**: Integration with existing ecosystem

### Feature Flags

```toml
[features]
default = []
wasm-threads = ["wasm-bindgen", "web-sys"]  # Enable WASM threading
rayon-compat = ["rayon"]  # Use rayon as backend
```

### WASM Shim (No Threads)

When `target_arch = "wasm32"` without atomics:

```rust
pub fn parallelize_1d<F>(task: F, range: usize)
where F: Fn(usize)
{
    for i in 0..range {
        task(i);
    }
}
```

### WASM with Threads

When WASM has threading support:

1. Use `std::thread::spawn` (requires `wasm32-unknown-unknown` + threads)
2. Use `Atomics.wait()` and `Atomics.notify()` via web-sys
3. Limit thread pool size (Emscripten default: 8 threads)

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
fn futex_wait(addr: &AtomicU32, expected: u32) {
    // Use web-sys Atomics.wait
    let array = js_sys::Int32Array::new_with_byte_offset_and_length(...);
    js_sys::Atomics::wait(&array, 0, expected as i32);
}
```

## Test Cases to Port

### Basic Functionality

1. `CreateAndDestroy` - Pool lifecycle
   - NullThreadPool
   - SingleThreadPool
   - MultiThreadPool

2. `Parallelize1D`
   - SingleThreadPoolCompletes
   - MultiThreadPoolCompletes
   - AllItemsInBounds
   - AllItemsProcessed
   - EachItemProcessedOnce
   - EachItemProcessedMultipleTimes
   - HighContention
   - **WorkStealing** (critical - verifies load balancing)

3. `Parallelize2DTile2D` (most important for ML)
   - Similar test suite to 1D
   - Verify tile bounds

### Performance Tests

1. **Latency benchmark** - Time to complete minimal work
2. **Throughput benchmark** - Maximum items/second
3. **Scaling test** - Performance vs thread count

### Edge Cases

1. Single item (range = 1)
2. Items fewer than threads
3. Empty range (range = 0)
4. Very large range
5. Uneven work distribution

## Implementation Phases

### Phase 1: Core Infrastructure
- ThreadPool struct with allocation
- Thread spawning and lifecycle
- Basic command/completion synchronization
- `parallelize_1d` with serial fallback

### Phase 2: Work Stealing
- Implement work distribution
- Implement atomic work claiming
- Implement work stealing loop
- Port fastpath optimization

### Phase 3: Multi-dimensional APIs
- `parallelize_2d_tile_2d`
- `parallelize_1d_tile_1d`
- FastDivisor for dimension mapping

### Phase 4: WASM Support
- Feature-gated compilation
- Shim for non-threaded WASM
- Threaded WASM with SharedArrayBuffer

### Phase 5: Optimization
- Cache-line alignment
- Memory ordering optimization
- Spin-wait tuning
- FPU denormal handling

## References

- pthreadpool source: https://github.com/Maratyszcza/pthreadpool
- FXdiv (fast division): https://github.com/Maratyszcza/FXdiv
- WASM threads proposal: https://github.com/WebAssembly/threads
- Rust atomics: https://doc.rust-lang.org/std/sync/atomic/

## Open Questions

1. **Rayon integration**: Should we provide a Rayon-based backend for better Rust ecosystem integration?

2. **Async support**: Should the API support async/await for better integration with async runtimes?

3. **Custom allocators**: Should we support custom allocators for the thread pool memory?

4. **Thread affinity**: Should we expose CPU affinity APIs for big.LITTLE architectures?

5. **Microarchitecture detection**: How much of cpuinfo functionality do we need to port?
