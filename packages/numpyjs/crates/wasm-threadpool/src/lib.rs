//! High-performance WASM thread pool using static dispatch and barrier synchronization.
//!
//! This implementation uses:
//! - Static dispatch with barrier synchronization (not work-stealing)
//! - Spin-then-wait strategy: spin for ~5000 cycles, then Atomics.wait
//! - 64-byte cache-line aligned thread control structs
//! - Raw WASM atomics (memory_atomic_wait32/notify)
//! - Job ID epoch pattern with single AtomicU32 job counter

#![no_std]
#![cfg_attr(target_arch = "wasm32", feature(stdarch_wasm_atomic_wait))]

#[cfg(not(target_arch = "wasm32"))]
extern crate std;

mod worker;

// Panic handler for WASM (required for no_std)
#[cfg(all(target_arch = "wasm32", not(test)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In production WASM, we just trap
    core::arch::wasm32::unreachable()
}

use core::sync::atomic::{AtomicU32, AtomicPtr, Ordering};

pub use worker::{worker_entry, WorkerConfig};

/// Maximum number of worker threads supported
pub const MAX_WORKERS: usize = 16;

/// Number of spin cycles before falling back to Atomics.wait
pub const SPIN_CYCLES: u32 = 5000;

/// Worker status values
pub mod status {
    pub const IDLE: u32 = 0;
    pub const WORKING: u32 = 1;
    pub const DONE: u32 = 2;
    pub const SHUTDOWN: u32 = 3;
}

/// Signal values for worker threads
pub mod signal {
    pub const NONE: u32 = 0;
    pub const WAKE: u32 = 1;
    pub const SHUTDOWN: u32 = 2;
}

/// Per-thread control structure, aligned to 64-byte cache line to prevent false sharing.
#[repr(C, align(64))]
pub struct ThreadControl {
    /// Current status of this worker (IDLE, WORKING, DONE, SHUTDOWN)
    pub status: AtomicU32,
    /// Signal sent to this worker (NONE, WAKE, SHUTDOWN)
    pub signal: AtomicU32,
    /// The last job_id this worker has seen/completed
    pub last_job_id: AtomicU32,
    /// Padding to fill cache line
    _padding: [u32; 13],
}

impl ThreadControl {
    pub const fn new() -> Self {
        Self {
            status: AtomicU32::new(status::IDLE),
            signal: AtomicU32::new(signal::NONE),
            last_job_id: AtomicU32::new(0),
            _padding: [0; 13],
        }
    }
}

impl Default for ThreadControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Task function signature: (data_ptr, m, k, n, thread_idx, thread_count)
/// Using extern "C" for FFI safety
pub type TaskFn = unsafe extern "C" fn(*mut f32, usize, usize, usize, usize, usize);

/// The main thread pool structure.
///
/// Layout is optimized for cache efficiency:
/// - Frequently accessed fields first
/// - Per-worker control structs cache-line aligned
#[repr(C)]
pub struct ThreadPool {
    /// Number of active worker threads
    pub worker_count: AtomicU32,

    /// Current job ID (epoch counter) - incremented for each dispatch
    pub job_id: AtomicU32,

    /// Count of workers that have finished current job
    pub finished_count: AtomicU32,

    /// Pool is initialized and ready
    pub initialized: AtomicU32,

    /// Padding to separate metadata from task params
    _meta_padding: [u32; 12],

    // Task parameters (set before dispatch)
    /// Pointer to data buffer
    pub data_ptr: AtomicPtr<f32>,

    /// Matrix dimension M
    pub m: AtomicU32,

    /// Matrix dimension K
    pub k: AtomicU32,

    /// Matrix dimension N
    pub n: AtomicU32,

    /// Task function pointer
    pub task_fn: AtomicPtr<()>,

    /// Padding before workers array
    _task_padding: [u32; 10],

    /// Per-worker control structures (cache-line aligned)
    pub workers: [ThreadControl; MAX_WORKERS],
}

impl ThreadPool {
    /// Create a new thread pool (const for static initialization)
    pub const fn new() -> Self {
        Self {
            worker_count: AtomicU32::new(0),
            job_id: AtomicU32::new(0),
            finished_count: AtomicU32::new(0),
            initialized: AtomicU32::new(0),
            _meta_padding: [0; 12],
            data_ptr: AtomicPtr::new(core::ptr::null_mut()),
            m: AtomicU32::new(0),
            k: AtomicU32::new(0),
            n: AtomicU32::new(0),
            task_fn: AtomicPtr::new(core::ptr::null_mut()),
            _task_padding: [0; 10],
            workers: [
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
                ThreadControl::new(), ThreadControl::new(),
            ],
        }
    }

    /// Initialize the pool with a specific worker count
    pub fn init(&self, worker_count: usize) {
        let count = worker_count.min(MAX_WORKERS);
        self.worker_count.store(count as u32, Ordering::Release);
        self.job_id.store(0, Ordering::Release);
        self.finished_count.store(0, Ordering::Release);

        // Reset all worker states
        for i in 0..MAX_WORKERS {
            self.workers[i].status.store(status::IDLE, Ordering::Release);
            self.workers[i].signal.store(signal::NONE, Ordering::Release);
            self.workers[i].last_job_id.store(0, Ordering::Release);
        }

        self.initialized.store(1, Ordering::Release);
    }

    /// Dispatch work to all workers and wait for completion.
    ///
    /// This function:
    /// 1. Sets task parameters
    /// 2. Increments job_id (epoch)
    /// 3. Signals all workers to wake
    /// 4. Waits for all workers to complete
    pub fn dispatch(
        &self,
        data: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        task_fn: TaskFn,
    ) {
        let worker_count = self.worker_count.load(Ordering::Acquire) as usize;
        if worker_count == 0 {
            return;
        }

        // Set task parameters
        self.data_ptr.store(data, Ordering::Release);
        self.m.store(m as u32, Ordering::Release);
        self.k.store(k as u32, Ordering::Release);
        self.n.store(n as u32, Ordering::Release);
        self.task_fn.store(task_fn as *mut (), Ordering::Release);

        // Reset finished count
        self.finished_count.store(0, Ordering::Release);

        // Memory barrier before incrementing job_id
        core::sync::atomic::fence(Ordering::SeqCst);

        // Increment job_id to signal new work
        let new_job_id = self.job_id.fetch_add(1, Ordering::SeqCst) + 1;

        // Signal all workers
        for i in 0..worker_count {
            self.workers[i].signal.store(signal::WAKE, Ordering::Release);

            // Use platform-specific wake
            #[cfg(target_arch = "wasm32")]
            unsafe {
                notify_worker(&self.workers[i].signal);
            }
        }

        // Wait for all workers to complete
        self.wait_for_completion(worker_count, new_job_id);
    }

    /// Wait for all workers to finish the current job
    fn wait_for_completion(&self, worker_count: usize, _job_id: u32) {
        // Spin-then-wait on finished_count
        let mut spin_count = 0u32;

        loop {
            let finished = self.finished_count.load(Ordering::Acquire);
            if finished >= worker_count as u32 {
                return;
            }

            spin_count += 1;
            if spin_count < SPIN_CYCLES {
                // Spin
                core::hint::spin_loop();
            } else {
                // Wait on the finished_count
                #[cfg(target_arch = "wasm32")]
                unsafe {
                    wait_on_finished_count(&self.finished_count, finished);
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    std::thread::yield_now();
                }

                spin_count = 0;
            }
        }
    }

    /// Shutdown all workers
    pub fn shutdown(&self) {
        let worker_count = self.worker_count.load(Ordering::Acquire) as usize;

        for i in 0..worker_count {
            self.workers[i].signal.store(signal::SHUTDOWN, Ordering::Release);
            self.workers[i].status.store(status::SHUTDOWN, Ordering::Release);

            #[cfg(target_arch = "wasm32")]
            unsafe {
                notify_worker(&self.workers[i].signal);
            }
        }
    }

    /// Get the address of the pool for FFI
    pub fn as_ptr(&self) -> *const ThreadPool {
        self as *const ThreadPool
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

// WASM-specific atomic operations
#[cfg(target_arch = "wasm32")]
unsafe fn notify_worker(signal: &AtomicU32) {
    use core::arch::wasm32::memory_atomic_notify;
    let ptr = signal as *const AtomicU32 as *mut i32;
    memory_atomic_notify(ptr, 1);
}

#[cfg(target_arch = "wasm32")]
unsafe fn wait_on_finished_count(count: &AtomicU32, expected: u32) {
    use core::arch::wasm32::memory_atomic_wait32;
    let ptr = count as *const AtomicU32 as *mut i32;
    // Wait with a short timeout (1ms = 1_000_000 ns) to avoid deadlocks
    let _ = memory_atomic_wait32(ptr, expected as i32, 1_000_000);
}

// Global static pool instance for WASM
static POOL: ThreadPool = ThreadPool::new();

/// Get a reference to the global pool
pub fn global_pool() -> &'static ThreadPool {
    &POOL
}

// FFI exports for WASM
#[cfg(target_arch = "wasm32")]
mod ffi {
    use super::*;

    #[no_mangle]
    pub extern "C" fn threadpool_init(worker_count: u32) {
        global_pool().init(worker_count as usize);
    }

    #[no_mangle]
    pub extern "C" fn threadpool_get_ptr() -> *const ThreadPool {
        global_pool().as_ptr()
    }

    #[no_mangle]
    pub extern "C" fn threadpool_worker_entry(thread_idx: u32) {
        worker_entry(global_pool(), thread_idx as usize);
    }

    #[no_mangle]
    pub extern "C" fn threadpool_dispatch(
        data_ptr: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        task_fn: TaskFn,
    ) {
        global_pool().dispatch(
            data_ptr,
            m as usize,
            k as usize,
            n as usize,
            task_fn,
        );
    }

    #[no_mangle]
    pub extern "C" fn threadpool_shutdown() {
        global_pool().shutdown();
    }

    /// Get the job_id address for JavaScript to wait on
    #[no_mangle]
    pub extern "C" fn threadpool_get_job_id_ptr() -> *const AtomicU32 {
        &global_pool().job_id
    }

    /// Get the finished_count address for JavaScript to notify
    #[no_mangle]
    pub extern "C" fn threadpool_get_finished_count_ptr() -> *const AtomicU32 {
        &global_pool().finished_count
    }

    /// Get a worker's signal address for JavaScript to wait/notify
    #[no_mangle]
    pub extern "C" fn threadpool_get_worker_signal_ptr(thread_idx: u32) -> *const AtomicU32 {
        &global_pool().workers[thread_idx as usize].signal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_control_size() {
        // ThreadControl should be exactly 64 bytes (one cache line)
        assert_eq!(core::mem::size_of::<ThreadControl>(), 64);
    }

    #[test]
    fn test_thread_control_alignment() {
        // ThreadControl should be 64-byte aligned
        assert_eq!(core::mem::align_of::<ThreadControl>(), 64);
    }

    #[test]
    fn test_pool_init() {
        let pool = ThreadPool::new();
        pool.init(4);

        assert_eq!(pool.worker_count.load(Ordering::Relaxed), 4);
        assert_eq!(pool.job_id.load(Ordering::Relaxed), 0);
        assert_eq!(pool.initialized.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_pool_init_max_workers() {
        let pool = ThreadPool::new();
        pool.init(100); // Should clamp to MAX_WORKERS

        assert_eq!(pool.worker_count.load(Ordering::Relaxed), MAX_WORKERS as u32);
    }

    #[test]
    fn test_worker_states_initialized() {
        let pool = ThreadPool::new();
        pool.init(8);

        for i in 0..8 {
            assert_eq!(pool.workers[i].status.load(Ordering::Relaxed), status::IDLE);
            assert_eq!(pool.workers[i].signal.load(Ordering::Relaxed), signal::NONE);
        }
    }
}
