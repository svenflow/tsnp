# WASM Threading Models for High-Performance Compute

This document explains WASM threading options for CPU-intensive workloads like GEMM (General Matrix Multiplication).

## Overview

In the browser, "threading" is fundamentally different from native OS threading. There's no `fork()` or `pthread_create()`. Instead, we use **Web Workers** running in parallel, sharing memory via **SharedArrayBuffer**.

## Threading Models

### 1. wasm-bindgen-rayon (What We Currently Use)

**How it works:**
- Pool of Web Workers initialized at startup
- All workers share the main WASM `WebAssembly.Memory` (backed by SharedArrayBuffer)
- Uses atomic Wait/Notify for task queues and work-stealing
- `par_iter()` breaks work into tasks; idle workers steal from busy workers

**Overhead Characteristics:**
- **High Setup Cost**: Spawning workers and initializing WASM in each is slow
- **Scheduler Overhead**: Work-stealing logic costs cycles, even for simple regular loops

**Pros:**
- Drop-in compatibility with existing Rust code using Rayon
- Excellent load balancing for irregular workloads

**Cons:**
- Binary size bloat
- Work-stealing scheduler is overkill for predictable GEMM loops
- Complex module resolution for workerHelpers.js

**Best for:** Complex, irregular workloads. Less ideal for strict GEMM.

---

### 2. Raw SharedArrayBuffer + Atomics (Custom Thread Pool)

**How it works:**
- Main thread spawns N Web Workers
- Workers share WebAssembly.Memory
- Control Block in shared memory coordinates work
- Workers run infinite loop: `Atomics.wait()` → compute → sleep

**Implementation Pattern:**
```rust
// Control Block in Shared Memory
struct ThreadControl {
    state: AtomicU32,     // 0 = Sleep, 1 = Work
    task_id: AtomicU32,
    args_ptr: AtomicU32,
}

// Worker Loop (WASM)
loop {
    memory::atomic_wait(&control.state, 0);  // Wait for work
    let args = read_args(control.args_ptr);
    compute_gemm_slice(args);
    control.state.store(0, Release);
    memory::atomic_notify(&main_thread_flag);
}
```

**Overhead Characteristics:**
- **Lowest Latency**: Job dispatch is just memory write + notify (< 5 microseconds)
- **Zero Allocation**: Pre-allocate Control Block, no GC during compute

**Pros:** Maximum performance, total control over cache locality
**Cons:** High complexity, manual race condition handling, unsafe code

**Best for:** High-performance GEMM libraries where every cycle counts.

---

### 3. Web Workers with Separate WASM Instances

**How it works:**
- Each worker instantiates its own WASM module with its own memory
- Data must be copied between workers via `postMessage`

**Overhead Characteristics:**
- **Massive Overhead**: Copying large matrices (MB/GB) destroys performance
- **Memory Duplication**: 4 threads = 4x memory usage

**Pros:** No data races, no COOP/COEP headers needed
**Cons:** Copy overhead kills parallelism for math

**Best for:** Independent background tasks (image compression, audio processing). **NOT suitable for GEMM.**

---

### 4. pthreadpool Pattern (XNNPACK Style)

This is what TensorFlow.js uses via XNNPACK/Emscripten.

**How it works:**
- Uses Emscripten's pthread emulation
- Pool of workers on spin-wait loop
- Static range partitioning (Thread 1 = rows 0-10, Thread 2 = rows 11-20)
- Barrier synchronization at start/end

**Why it's good for GEMM:**
- **L2 Cache Awareness**: Threads pinned to specific data ranges
- **No Allocation**: Task definition is static struct updated in place
- **One-Shot Parallelization**: Perfect for outer loop of 3-nested GEMM loop

**Overhead:** Minimal - effectively optimized "Raw SharedArrayBuffer" wrapped in reusable C library.

**Best for:** Deep learning inference, heavy matrix multiplication.

---

### 5. Compilation Targets

#### `wasm32-unknown-unknown`
- Standard "Web" target
- No native threading instructions
- Threading via library methods (wasm-bindgen, JS glue)
- **Use for:** Browsers

#### `wasm32-wasi-threads`
- For standalone WASM runtimes (Wasmtime, WAMR)
- Assumes host provides `wasi-threads` implementation
- **Browsers do NOT support this** without heavy polyfill
- **Use for:** Server-side WASM

---

## The `memory.grow` Problem

### The Mechanism
WASM memory is linear. When you run out of heap, `memory.grow` asks the host to resize the buffer.

### The Multi-threading Problem
If Thread A triggers a grow:
1. Browser allocates new SharedArrayBuffer
2. Copies data from old to new
3. Thread A points to new buffer
4. **CRASH:** Thread B still references old buffer

### The Solution
**Never grow at runtime:**
1. Calculate maximum memory needed at startup
2. Create Memory with `initial: X, maximum: X`
3. Abort if allocation exceeds limit

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = [
  "-C", "link-arg=--import-memory",
  "-C", "link-arg=--max-memory=1073741824",  # 1GB Max
]
```

---

## Synchronization Primitives

### Atomics.wait / Atomics.notify (JS)
```javascript
// Wait until value changes from expected
Atomics.wait(int32Array, index, expectedValue);

// Wake up N waiters
Atomics.notify(int32Array, index, count);
```

### memory.atomic.wait / memory.atomic.notify (WASM)
Same semantics, but from WASM instructions. Maps to browser's Atomics API.

### Atomic Operations
```rust
use std::sync::atomic::{AtomicU32, Ordering};

// Compare-and-swap
val.compare_exchange(expected, new, Ordering::SeqCst, Ordering::Relaxed);

// Fetch-and-add (for task counter)
task_idx.fetch_add(1, Ordering::Relaxed);
```

---

## GEMM-Specific Considerations

### Thread Count
- M4 Pro: 10 performance cores + 4 efficiency cores
- Use **10 threads** (perf cores only) for compute-bound work
- E-cores are slower and will bottleneck the job

### Work Division Strategy
- **1D Row Partitioning**: Simple but cache-inefficient (all threads read entire B)
- **2D Tile Partitioning**: Better cache locality (each thread reads subset of B)
- **Static vs Dynamic**: GEMM is predictable; static partitioning avoids scheduler overhead

### Cache Aliasing (Power-of-2 Curse)
- 2048 width = 8KB stride = perfect cache set aliasing
- Solution: Pad matrices to non-power-of-2 (e.g., 2056)

### Memory Layout
- Row-major A, column-major packed B
- Pack B once, reuse across all M-tiles
- Sequential access in inner loop for prefetcher

---

---

## Current Bottlenecks in Our Rayon Approach

### 1. Rayon Work-Stealing Overhead

**Problem:** Work-stealing is optimal for irregular workloads (Quicksort, graph traversal). For GEMM, workload is perfectly predictable (M × N × K). The overhead of "thief" looking for work plus deque CAS operations causes jitter.

**Measured Impact:** ~300ns to 2µs per task, depending on contention.

**Why It Matters:** If your micro-kernel takes 10µs, you lose 5-20% to overhead.

**Sources of Overhead:**
- Deque contention (CAS instructions on task deque)
- Cold cache (stolen work has no locality)
- False sharing (updating "job completed" counter if not properly padded)

### 2. Cache Aliasing at Power-of-2 Widths

**Problem:** 2048 width = 8KB row stride. With 32KB L1 cache (8-way associative), addresses separated by 8KB map to same set. Accessing A, B, C simultaneously evicts each other constantly.

**Evidence:**
- 2047x2047: 3.75x parallel speedup
- 2048x2048: 1.00x (NO speedup!)
- 2049x2049: 4.21x parallel speedup

**The Math:** For n-way associative cache of size S, if stride = S/n (or any multiple), all rows alias.

### 3. Module Resolution Complexity

**Problem:** `workerHelpers.js` import chain is fragile. The `import('../../..')` must resolve correctly based on server configuration.

**Our Fix:** Had to modify server to route `/pkg/` → `/pkg/rumpy_wasm.js`.

### 4. Per-Task Closure Overhead

**Problem:** Rayon's `par_iter` creates closures and state machines. For tight loops (micro-kernels), we want raw function pointers and pre-calculated indices.

### 5. Memory Allocation Inside Parallel Loop

**Problem (Fixed):** Originally allocated `packed_b` inside `for_each`. Fixed with `for_each_init`, but there's still per-task overhead.

---

## Theoretical Minimum Sync Overhead

For a 2048x2048 GEMM with 10 threads using static partitioning:

**Mechanism:** 1 atomic store (main thread "Go") + 10 atomic loads (workers see "Go")

**Time Estimates:**
- **Spin-waiting:** < 100 nanoseconds for all threads to wake
- **Atomics.wait (OS sleep):** 10-50µs (wake-up latency is high)

**Conclusion:** Sub-microsecond dispatch is achievable with spin-locks on dedicated cores.

---

## How XNNPACK/pthreadpool Avoids These Issues

1. **No Stealing:** Strictly static partitioning for 1D/2D loops
2. **Command Primitive:** Barrier synchronization. Main thread sets up task, updates generation counter, workers wake on counter change
3. **Spin-then-Park:** Workers spin briefly (~1-2µs) expecting more work, then fall back to `futex`/`Atomics.wait`

This keeps latency low for back-to-back inference layers while saving battery during idle.

---

## Custom Thread Pool Design

### ⚠️ CRITICAL: Main Thread Cannot Use Atomics.wait

Browsers **throw TypeError** if you call `Atomics.wait()` on the Window/Main thread. This prevents UI freezing.

**Solutions:**
1. **Async/Promise:** Main thread polls (bad) or receives `postMessage` from last worker to resolve Promise
2. **Proxy Worker (Recommended):** "Driver Thread" runs in dedicated Web Worker. Main UI sends message to Driver, Driver uses `Atomics.wait` to sync Compute Workers.

### ⚠️ CRITICAL: Cannot Detect P-Cores vs E-Cores

`navigator.hardwareConcurrency` reports total logical cores. Browser cannot distinguish Performance from Efficiency cores, and you cannot pin threads.

**Solution:** Make thread count configurable. Use `navigator.hardwareConcurrency - 1` or `- 2` as heuristic (leaves room for browser tasks).

### The Control Structure (Cache-Line Aware)

```rust
#[repr(C, align(64))]  // Align to cache line (64 bytes)
struct ThreadPoolState {
    // Cache Line 0: Read-mostly (Workers poll this)
    job_id: AtomicU32,
    status: AtomicU32,  // 0=Idle, 1=Working, 2=Shutdown
    _pad0: [u8; 56],    // Fill to 64 bytes

    // Cache Line 1: High-contention Write (Workers decrement)
    workers_remaining: AtomicU32,
    _pad1: [u8; 60],    // Fill to 64 bytes - PREVENTS FALSE SHARING

    // Cache Line 2+: Read-only Kernel Args
    // Using u32 for WASM offsets (cleaner than raw pointers)
    ptr_a: u32,         // Offset into WASM memory
    ptr_b: u32,
    ptr_c: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_a: u32,      // Use strides to handle padding!
    stride_b: u32,
    stride_c: u32,
}
```

**Why padding matters:** When a worker does `fetch_sub` on `workers_remaining`, it invalidates the cache line. Without padding, `job_id` would share that line, causing cache misses for all workers just reading static data.

### The Worker Loop

```rust
fn worker_entry(id: usize, total_threads: usize, state: &ThreadPoolState) {
    let mut last_job_id = state.job_id.load(Ordering::SeqCst);

    loop {
        // 1. Hybrid Spin/Wait
        let mut spins = 0;
        let mut current_job_id;
        loop {
            current_job_id = state.job_id.load(Ordering::Relaxed);
            if current_job_id != last_job_id { break; }

            if spins < 10_000 {
                spins += 1;
                std::hint::spin_loop();
            } else {
                atomic_wait(&state.job_id, last_job_id);
            }
        }
        last_job_id = current_job_id;

        // 2. Check for shutdown
        if state.status.load(Ordering::Relaxed) == 2 { break; }

        // 3. Calculate Work Partition (Static Scheduling)
        let rows_per_thread = (state.m + total_threads - 1) / total_threads;
        let start_row = id * rows_per_thread;
        let end_row = (start_row + rows_per_thread).min(state.m);

        if start_row < end_row {
            unsafe {
                run_gemm_slice(state, start_row, end_row);
            }
        }

        // 4. Signal Completion
        let remaining = state.workers_remaining.fetch_sub(1, Ordering::SeqCst);
        if remaining == 1 {
            atomic_notify(&state.workers_remaining);
        }
    }
}
```

### Memory Barriers Needed

**Per GEMM Call:** 2 barriers
1. **Dispatch Barrier:** Main writes params → Memory Fence → Workers read params
2. **Completion Barrier:** Workers finish → Memory Fence → Main reads result

### Wait vs Spin Decision

**Recommendation:** Hybrid approach
- Spin for ~10,000 iterations (~1-2µs)
- Then `Atomics.wait`

**Rationale:** Waking from `Atomics.wait` relies on browser/OS scheduler (slow). But 100% spinning burns battery on mobile. Hybrid catches "next layer" in neural net quickly but sleeps during idle.

---

## Solving Cache Aliasing

### Option 1: Allocation-Time Padding (Recommended)

```rust
fn allocate_matrix(rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let mut stride = cols;

    // Avoid stride % 2048 == 0 (8KB boundary)
    if (stride * 4) % 8192 == 0 {
        stride += 16;  // Add 64 bytes of padding
    }

    let vec = vec![0.0; rows * stride];
    (vec, stride)
}
```

Pass `stride` (not `cols`) into ThreadPoolState for pointer arithmetic.

### Option 2: Runtime Packing

Better for performance but harder to code. Pack sub-panels into continuous memory buffers (L2/L3 sized) before computing. Solves aliasing AND TLB thrashing.

---

## Optimization Opportunities

### Pre-packing Weights (Matrix B)

For inference, pack Matrix B (weights) at load time:
- Arrange for SIMD kernel (e.g., float32x4 layout)
- Skip packing step in hot path

### Ring Buffer for Task Dispatch

**Overkill for GEMM.** A single Command Struct is enough because GEMM is blocking. Ring buffers are useful for pipelining (CPU layer 1, GPU layer 2).

### Persistent Kernel Threads

**Required.** Threads initialized once at startup, enter `while(true)` loop. Never spawn threads inside `gemm` function.

---

## JavaScript Integration (No wasm-bindgen-rayon)

Instead of implicit workers:

1. Create standard JS `Worker` pool manually
2. Send `WebAssembly.Module` and `WebAssembly.Memory` to workers
3. Call exported `init_worker(thread_id)` in each worker
4. Main thread calls `dispatch_gemm(...)`:
   - Write to shared memory
   - Use `Atomics.notify` (JS) or WASM atomics to wake workers

This removes dynamic scheduling overhead and gives total control over synchronization.

---

## Implementation Plan

### Phase 1: Measure Current Overhead
- Profile rayon dispatch with Chrome DevTools Performance tab
- Measure exact per-task overhead
- Quantify cache aliasing impact at power-of-2 sizes

### Phase 2: Build Custom Thread Pool
1. Implement ThreadPoolState struct in WASM with cache-line padding
2. Create minimal `worker.js` (don't rely on bundler magic)
3. Export specific entry point `init_compute_worker(thread_id)`
4. Implement hybrid spin/wait in worker loop
5. **Use Driver Thread pattern** - controller runs in Worker, not UI thread
6. Add stride-based padding to matrix allocator
7. Implement lifecycle management (terminate/sleep signal)

### Phase 3: Integrate with GEMM Kernel
- Wire up custom thread pool to optimized 6x8 micro-kernel
- **Start with 1D partitioning** (split M dimension - rows of A) for simplicity
- Only move to 2D if M is small and N is huge
- Handle edge cases (non-divisible matrix sizes)
- Add heuristic: if M*N*K is small, run single-threaded

### Phase 4: Benchmark and Iterate
- Compare vs rayon at all matrix sizes
- Test on various power-of-2 sizes (512, 1024, 2048, 4096)
- Test with padded sizes (2056, 4104) to verify aliasing fix
- Tune spin count for optimal latency/power tradeoff
- Profile wake-up latency vs compute time for small matrices

### Additional Considerations

**SIMD Alignment:**
- WASM SIMD v128 loads are faster if 16-byte aligned
- Stride padding (+16 floats = +64 bytes) maintains alignment
- Handle case where user-provided matrices aren't 16-byte aligned

**Module Loading in Workers:**
- Loading WASM module in workers without generated glue is tricky
- Export specific entry point from Rust code
- Write minimal manual worker.js that imports WASM directly

---

## References

- [WebAssembly Threads Proposal](https://github.com/WebAssembly/threads)
- [wasm-bindgen-rayon](https://github.com/GoogleChromeLabs/wasm-bindgen-rayon)
- [XNNPACK](https://github.com/google/XNNPACK)
- [pthreadpool](https://github.com/Maratyszcza/pthreadpool)
