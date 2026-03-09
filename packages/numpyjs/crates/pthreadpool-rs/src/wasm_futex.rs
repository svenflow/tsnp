//! Native-WASM thread pool built directly on `memory.atomic.wait32` /
//! `memory.atomic.notify`, bypassing Rayon entirely.
//!
//! # Why another thread pool?
//!
//! `wasm-bindgen-rayon` is the de-facto way to get threads into WASM, but for
//! latency-critical, high-frequency dispatch (GEMM inner blocks, < 1 ms each)
//! it has several layers of overhead we don't want:
//!
//! * Rayon's work-stealing uses heap-allocated `Task` objects and Chase–Lev
//!   deques with CAS loops. For N threads × M tiles that's O(M) allocations
//!   and O(M) CAS operations — fine for big tasks, brutal for micro-tasks.
//!
//! * The calling thread (usually the JS main thread) does **not** participate
//!   in computation with `par_iter`: it posts work to the pool then blocks.
//!   With N workers you get N cores, not N+1.
//!
//! * Rayon's park/unpark path goes through `std::thread::park`, which on WASM
//!   goes through `wasm_sync` → condvar → memory.atomic.wait. The spin count
//!   before falling into wait is not tunable and is tuned for general
//!   workloads, not repeated sub-ms dispatches.
//!
//! This module implements the same algorithm as the C `pthreadpool` library
//! (which XNNPACK uses and which is *why* tf.js multithreading is fast):
//!
//! * Persistent workers, spawned once at init.
//!
//! * To dispatch: main bumps a `generation` counter (Release) and calls
//!   `atomic.notify`.  Workers that were parked wake up; workers that were
//!   spinning notice immediately.
//!
//! * Work items are claimed by a single Relaxed `fetch_sub` on a per-thread
//!   range counter, falling back to stealing (also `fetch_sub`) when the
//!   local range is drained.  One atomic per item, no locks.
//!
//! * Completion: last worker to finish decrements `active` to 0 and notifies
//!   the main thread's wait address.
//!
//! * The main thread participates as thread 0 (Emscripten-style).  When
//!   waiting for workers it **spin-waits**, because the browser main thread
//!   cannot call `memory.atomic.wait32` (it traps). For GEMM (< 100 ms) this
//!   is fine; for long-running blocking work you'd want to call this from
//!   a Worker instead.
//!
//! # Spin-then-wait tuning
//!
//! pthreadpool spins `PTHREADPOOL_SPIN_WAIT_ITERATIONS` (default 1M on
//! native) before falling into a futex.  In WASM, spinning is cheaper
//! (no real context switch cost) and waiting is more expensive (goes through
//! the browser's event loop). We spin for ~100 k iterations on workers,
//! which at ~1 ns/iter is 100 μs — enough to never hit the wait for GEMM
//! blocks, but short enough to not burn a core when genuinely idle.
//!
//! # Worker spawning
//!
//! `std::thread::spawn` on `wasm32-unknown-unknown` panics: there is no OS
//! thread API to call. We depend on the `wasm_thread` crate, which provides
//! a drop-in `spawn()` that creates a Web Worker, instantiates the same
//! Wasm module + shared memory, and runs the closure on it. This is exactly
//! what `wasm-bindgen-rayon` does internally, minus the Rayon parts.
//!
//! The JS runtime requirements are identical to `wasm-bindgen-rayon`:
//! SharedArrayBuffer, COOP/COEP headers, `--target web` build.

#![cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]

use core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicU32, Ordering::*};
#[allow(unused_imports)]
use std::sync::atomic::AtomicUsize;
use std::sync::OnceLock;

/// Number of spin iterations before a worker falls back to `atomic.wait`.
///
/// Tuned for WASM GEMM workloads:
/// - 100K iterations at ~1ns/iter = 100μs spin time
/// - Typical GEMM dispatch interval: 10-100μs
/// - Workers rarely hit futex wait for continuous dispatches
/// - Lower than pthreadpool's 1M to reduce CPU usage during idle
const SPIN_ITERS_WORKER: u32 = 100_000;

/// Main thread uses spin-only (cannot `atomic.wait` in a browser context).
/// Cap it so a bug doesn't lock the tab forever.
const SPIN_ITERS_MAIN_CAP: u64 = 10_000_000_000;

/// High bit of `generation` signals shutdown.
const GEN_SHUTDOWN: u32 = 1 << 31;

/// Cache-line–sized per-thread slot. All hot atomics live here to avoid
/// false sharing between workers.
#[repr(align(64))]
struct Slot {
    /// Work range: owner claims from the front by bumping `range_start`,
    /// stealers claim from the back by bumping down `range_end`.
    /// `range_len` is the arbitration counter: whoever decrements it from
    /// ≥ 1 to something owns an item. Going below 0 is the "you lost" signal.
    range_start: AtomicUsize,
    range_end: AtomicUsize,
    range_len: AtomicIsize,
}

impl Slot {
    const fn new() -> Self {
        Self {
            range_start: AtomicUsize::new(0),
            range_end: AtomicUsize::new(0),
            range_len: AtomicIsize::new(0),
        }
    }

    #[inline]
    fn init(&self, start: usize, end: usize) {
        // All Relaxed stores here - the Release on generation ensures
        // workers see these before starting work (they Acquire on generation).
        self.range_start.store(start, Relaxed);
        self.range_end.store(end, Relaxed);
        self.range_len.store((end - start) as isize, Relaxed);
    }

    /// Claim one item from our own range (from the front). pthreadpool
    /// fastpath: a single Relaxed RMW, underflow is the "empty" signal.
    #[inline]
    fn try_own(&self) -> Option<usize> {
        if self.range_len.fetch_sub(1, Acquire) > 0 {
            Some(self.range_start.fetch_add(1, Relaxed))
        } else {
            // Best-effort undo. Racy, but harmless: the counter has already
            // gone non-positive so no-one will claim past the real end.
            self.range_len.fetch_add(1, Relaxed);
            None
        }
    }

    /// Steal one item from another thread's range (from the back).
    #[inline]
    fn try_steal(&self) -> Option<usize> {
        if self.range_len.fetch_sub(1, Acquire) > 0 {
            // fetch_sub returns the old value; old-1 is the item we now own.
            Some(self.range_end.fetch_sub(1, Relaxed) - 1)
        } else {
            self.range_len.fetch_add(1, Relaxed);
            None
        }
    }
}

/// Type-erased task. We keep an `Fn(usize)` behind a raw pointer for the
/// duration of one `parallelize` call — the caller's stack frame outlives
/// the call, so this is sound.
struct TaskPtr {
    call: unsafe fn(*const (), usize),
    ctx: *const (),
}
// SAFETY: we only publish `TaskPtr` while holding the pool exclusive
// (single-caller) and the referenced closure is `Sync` by contract.
unsafe impl Send for TaskPtr {}
unsafe impl Sync for TaskPtr {}

pub struct FutexPool {
    /// Includes the caller (thread 0). `n_workers = threads - 1`.
    threads: usize,

    /// Bumped once per dispatch. Workers spin on this; changes signal "go".
    /// Top bit is shutdown. Stored as i32 because that's what
    /// `memory_atomic_wait32` takes.
    generation: AtomicI32,

    /// Decremented by each thread (including thread 0) on completion.
    /// 0 → everyone's done.
    active: AtomicU32,

    /// Number of workers that have started their main loop.
    /// Incremented by each worker (threads 1..n) when they enter `worker_main`.
    /// When this equals `threads - 1`, all workers are ready.
    workers_ready: AtomicU32,

    /// Per-thread work ranges.
    slots: Box<[Slot]>,

    /// The current task. `UnsafeCell` because we want zero-cost access
    /// in the hot loop — synchronisation happens via `generation`.
    task: UnsafeCell<Option<TaskPtr>>,
}

// SAFETY: all interior mutability is guarded by the single-caller contract
// on `parallelize` + the generation/active atomics.
unsafe impl Send for FutexPool {}
unsafe impl Sync for FutexPool {}

static POOL: OnceLock<FutexPool> = OnceLock::new();

impl FutexPool {
    fn new(n: usize) -> Self {
        let n = n.max(1);
        let slots = (0..n).map(|_| Slot::new()).collect::<Vec<_>>().into_boxed_slice();

        FutexPool {
            threads: n,
            generation: AtomicI32::new(0),
            active: AtomicU32::new(0),
            workers_ready: AtomicU32::new(0),
            slots,
            task: UnsafeCell::new(None),
        }
    }

    /// Spawn worker threads (1..n). Thread 0 is the caller and is not spawned.
    ///
    /// We go through `wasm_thread::spawn`, which:
    ///   1. Creates a new Web Worker
    ///   2. Loads the same Wasm module + SharedArrayBuffer memory into it
    ///   3. Runs TLS init so things like `thread_local!` work
    ///   4. Calls our closure
    ///
    /// This is what `wasm-bindgen-rayon` does, minus the Rayon scheduler.
    /// If you're also using `wasm-bindgen-rayon`, both pools coexist fine —
    /// but you'll oversubscribe cores if both are sized to
    /// `navigator.hardwareConcurrency`.  Size one or the other down, or
    /// just use `matmul_optimized_f32_parallel_v3` (Rayon-backed) and skip
    /// this pool entirely.
    fn spawn_workers(&'static self) {
        for tid in 1..self.threads {
            // The `es_modules` feature on wasm_thread matches our --target web
            // build (module scripts, not classic workers).
            wasm_thread::Builder::new()
                .spawn(move || worker_main(self, tid))
                .expect("wasm_thread::spawn failed — is SharedArrayBuffer enabled?");
        }
    }

    /// One atomic `notify` to wake all parked workers.
    #[inline]
    fn wake_workers(&self) {
        // SAFETY: `generation` lives for 'static (the pool is in a OnceLock)
        // and the pointer is valid and aligned (i32).
        unsafe {
            memory_atomic_notify(
                self.generation.as_ptr() as *mut i32,
                u32::MAX, // wake all
            );
        }
    }

    /// Dispatch `range` items to all threads, including the caller.
    /// `task` is called exactly once per index in 0..range (order unspecified).
    ///
    /// Caller must be on the "main" thread (the one that cannot atomic.wait).
    /// If you're calling from a Worker, use `parallelize_from_worker` instead.
    ///
    /// SAFETY: only one `parallelize` call may be in flight at a time.
    /// This is not checked. For GEMM we call serially from a single entry
    /// point, so it's fine; if you need reentrancy, wrap in a Mutex.
    pub fn parallelize<F: Fn(usize) + Sync>(&self, range: usize, task: F) {
        if range == 0 {
            return;
        }

        // 1. Distribute work: each thread gets ~range/n items. Remainder
        //    goes to the first `remainder` threads (pthreadpool convention).
        let n = self.threads;
        let base = range / n;
        let rem = range % n;
        let mut start = 0;
        for (i, slot) in self.slots.iter().enumerate() {
            let count = base + (i < rem) as usize;
            slot.init(start, start + count);
            start += count;
        }

        // 2. Publish the task. Type-erase the closure so workers don't need
        //    to be generic.
        unsafe fn trampoline<F: Fn(usize)>(ctx: *const (), i: usize) {
            (*(ctx as *const F))(i);
        }
        let tp = TaskPtr {
            call: trampoline::<F>,
            ctx: &task as *const F as *const (),
        };
        // SAFETY: workers read `task` only after Acquire-loading `generation`
        // and seeing the new value; we Release-store `generation` after
        // writing `task`. Single-caller contract means no concurrent writers.
        unsafe {
            *self.task.get() = Some(tp);
        }

        // 3. Arm completion counter, bump generation.
        // Use Relaxed for active - the Release on generation provides the
        // necessary synchronization (workers Acquire on generation).
        self.active.store(n as u32, Relaxed);

        // Wrapping add, but avoid the shutdown bit.
        // This single Release barrier synchronizes all prior Relaxed stores.
        let gen = self.generation.fetch_add(1, Release) + 1;
        debug_assert_eq!(gen & GEN_SHUTDOWN as i32, 0, "generation overflowed into shutdown bit");

        // Note: we skip wake_workers() since workers spin with 1M iterations
        // and will see the generation change immediately. The notify syscall
        // has ~5μs overhead and is unnecessary for rapid dispatches.
        // For long idle periods, workers will eventually futex_wait and
        // the next dispatch will wake them via the generation change.
        //
        // HOWEVER: if workers ARE in futex wait, they need to be woken.
        // With 1M spin iterations at ~1ns/iter = 1ms, workers only hit wait
        // if there's >1ms between dispatches. For GEMM benchmarking this is
        // unlikely, but for safety let's keep the wake.
        self.wake_workers();

        // 4. Caller is thread 0: do its share of the work, then steal.
        run_and_steal(self, 0);

        // 5. We're done with our share. Decrement `active` and wait for the
        //    rest.
        //
        //    The main thread CANNOT atomic.wait (browser traps). So we spin.
        //    For GEMM this spin is sub-millisecond — the caller already did
        //    ~1/N of the work, so stragglers finish fast.
        if self.active.fetch_sub(1, AcqRel) > 1 {
            let mut spins = 0u64;
            while self.active.load(Acquire) != 0 {
                core::hint::spin_loop();
                spins += 1;
                if spins > SPIN_ITERS_MAIN_CAP {
                    // This would mean a worker deadlocked. Better to panic
                    // than to hang the tab.
                    panic!("FutexPool: workers did not complete within spin cap");
                }
            }
        }

        // 6. Clear task (belt-and-braces; not strictly needed since the next
        //    dispatch overwrites it before bumping generation).
        unsafe {
            *self.task.get() = None;
        }
    }
}

/// Worker entry: loop { wait for new generation; do work; signal done }.
///
/// The spin-then-wait here is the key latency optimisation.  For GEMM where
/// we dispatch every ~KC worth of K (i.e. ~256 × NC × 6 FMAs ≈ sub-ms), the
/// next generation bump arrives before the spin runs out, so we *never* hit
/// `atomic.wait` and the wakeup latency is ~zero.
fn worker_main(pool: &'static FutexPool, tid: usize) {
    // Signal that this worker is ready. The main thread can poll `workers_ready`
    // to know when all workers have started.
    pool.workers_ready.fetch_add(1, Release);

    // Also wake the main thread's readiness wait (if any).
    unsafe {
        memory_atomic_notify(pool.workers_ready.as_ptr() as *mut i32, 1);
    }

    let mut seen_gen = 0i32;

    loop {
        // Spin waiting for generation to change.
        let mut i = 0u32;
        let mut cur = pool.generation.load(Acquire);
        while cur == seen_gen && (cur & GEN_SHUTDOWN as i32) == 0 {
            if i < SPIN_ITERS_WORKER {
                core::hint::spin_loop();
                i += 1;
                cur = pool.generation.load(Acquire);
            } else {
                // Park. `memory_atomic_wait32` is the WASM futex: wait until
                // *ptr != expected OR a notify arrives. Timeout -1 = infinite.
                //
                // SAFETY: generation is 'static, i32-aligned, in shared memory.
                unsafe {
                    memory_atomic_wait32(
                        pool.generation.as_ptr() as *mut i32,
                        seen_gen,
                        -1,
                    );
                }
                cur = pool.generation.load(Acquire);
            }
        }

        if (cur & GEN_SHUTDOWN as i32) != 0 {
            return;
        }
        seen_gen = cur;

        // Do work.
        run_and_steal(pool, tid);

        // Signal completion. If we're the last one (active goes 1→0), wake
        // the caller — but the caller is spin-only, so the notify is a no-op
        // on the main thread. We fire it anyway in case this pool is ever
        // driven from a Worker that *can* wait.
        if pool.active.fetch_sub(1, AcqRel) == 1 {
            unsafe {
                memory_atomic_notify(pool.active.as_ptr() as *mut i32, 1);
            }
        }
    }
}

/// Shared work-loop body: drain own range, then steal round-robin.
#[inline]
fn run_and_steal(pool: &FutexPool, tid: usize) {
    // SAFETY: we only enter here after Acquire-observing a generation bump,
    // which synchronises with the Release-store of `task` in `parallelize`.
    // The caller's stack (holding the closure) lives until `parallelize`
    // returns, which it doesn't until `active == 0`.
    let (call, ctx) = unsafe {
        let t = (*pool.task.get()).as_ref().unwrap_unchecked();
        (t.call, t.ctx)
    };

    let slots = &pool.slots;
    let n = slots.len();

    // Own range first.
    let my = &slots[tid];
    while let Some(i) = my.try_own() {
        unsafe { call(ctx, i) };
    }

    // Steal, starting from the thread "farthest" from us (pthreadpool
    // convention: reduces contention because nearby threads are likely
    // to be stealing from the same victims otherwise).
    for off in 1..n {
        let victim = &slots[(tid + n - off) % n];
        while let Some(i) = victim.try_steal() {
            unsafe { call(ctx, i) };
        }
    }
}

/// Initialise the global pool with `n` threads (including the caller).
///
/// Subsequent calls are no-ops (returns the first pool unchanged).
/// Must be called from the main thread before any `parallelize` call.
///
/// On Apple Silicon, cap `n` at the performance-core count (typically 4 or
/// 8 depending on chip) rather than `navigator.hardwareConcurrency`.
/// Efficiency cores drag down synchronisation-heavy workloads — see
/// <https://github.com/GoogleChromeLabs/wasm-bindgen-rayon/issues/16>
/// (fixed in Chrome 112 for the worst-case, but P+E heterogeneity is
/// still a thing and the futex pool's work-stealing mitigates but doesn't
/// eliminate the tail).
pub fn init(n: usize) {
    static SPAWNED: AtomicBool = AtomicBool::new(false);
    let pool = POOL.get_or_init(|| FutexPool::new(n));
    // Spawn workers exactly once. compare_exchange because two init() calls
    // could race (however unlikely in a wasm_bindgen init path).
    if SPAWNED
        .compare_exchange(false, true, AcqRel, Acquire)
        .is_ok()
    {
        pool.spawn_workers();
    }
}

/// Get the pool (None if `init` hasn't been called).
pub fn get_pool() -> Option<&'static FutexPool> {
    POOL.get()
}

/// Thin wrapper so callers don't need to name `FutexPool`.
pub fn parallelize<F: Fn(usize) + Sync>(pool: &FutexPool, range: usize, task: F) {
    pool.parallelize(range, task);
}

/// For benchmark exposure: how many threads (including thread 0) we have.
pub fn threads_count() -> usize {
    POOL.get().map(|p| p.threads).unwrap_or(1)
}

/// Check if all workers are ready.
/// Returns true if all `n-1` workers (where n is the thread count) have started.
/// Returns false if the pool isn't initialized or not all workers are ready.
pub fn workers_ready() -> bool {
    match POOL.get() {
        Some(p) => {
            let expected = (p.threads as u32).saturating_sub(1);
            p.workers_ready.load(Acquire) >= expected
        }
        None => false,
    }
}

/// Get the number of workers that have started (for debugging).
pub fn workers_ready_count() -> u32 {
    POOL.get().map(|p| p.workers_ready.load(Acquire)).unwrap_or(0)
}
