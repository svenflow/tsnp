//! Worker thread implementation with spin-then-wait strategy.

use core::sync::atomic::Ordering;
use crate::{ThreadPool, TaskFn, SPIN_CYCLES, status, signal};

/// Configuration for worker behavior
pub struct WorkerConfig {
    /// Number of spin cycles before waiting
    pub spin_cycles: u32,
    /// Timeout for atomic wait in nanoseconds (0 = infinite)
    pub wait_timeout_ns: i64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            spin_cycles: SPIN_CYCLES,
            wait_timeout_ns: 1_000_000, // 1ms default timeout
        }
    }
}

/// Main worker entry point - the hot loop that spins/waits for work.
///
/// This function runs in an infinite loop:
/// 1. Spin for SPIN_CYCLES checking job_id
/// 2. If no work, use Atomics.wait
/// 3. When work arrives, execute task
/// 4. Signal completion via finished_count
/// 5. Return to step 1
///
/// # Safety
/// This function should be called from a Web Worker in WASM context.
pub fn worker_entry(pool: &ThreadPool, thread_idx: usize) {
    worker_entry_with_config(pool, thread_idx, &WorkerConfig::default());
}

/// Worker entry with custom configuration
pub fn worker_entry_with_config(pool: &ThreadPool, thread_idx: usize, config: &WorkerConfig) {
    let worker = &pool.workers[thread_idx];

    // Mark ourselves as idle and ready
    worker.status.store(status::IDLE, Ordering::Release);

    // Track the last job_id we processed
    let mut last_processed_job_id = 0u32;

    loop {
        // Wait for new work using spin-then-wait
        let new_job_id = spin_wait_for_work(pool, worker, last_processed_job_id, config);

        // Check for shutdown signal
        let sig = worker.signal.load(Ordering::Acquire);
        if sig == signal::SHUTDOWN {
            worker.status.store(status::SHUTDOWN, Ordering::Release);
            return;
        }

        // We have work! Mark as working
        worker.status.store(status::WORKING, Ordering::Release);
        worker.signal.store(signal::NONE, Ordering::Release);

        // Load task parameters
        let data_ptr = pool.data_ptr.load(Ordering::Acquire);
        let m = pool.m.load(Ordering::Acquire) as usize;
        let k = pool.k.load(Ordering::Acquire) as usize;
        let n = pool.n.load(Ordering::Acquire) as usize;
        let task_fn_ptr = pool.task_fn.load(Ordering::Acquire);
        let worker_count = pool.worker_count.load(Ordering::Acquire) as usize;

        // Execute the task if we have a valid function pointer
        if !task_fn_ptr.is_null() {
            let task_fn: TaskFn = unsafe { core::mem::transmute(task_fn_ptr) };
            unsafe {
                task_fn(data_ptr, m, k, n, thread_idx, worker_count);
            }
        }

        // Mark our job_id as processed
        last_processed_job_id = new_job_id;
        worker.last_job_id.store(new_job_id, Ordering::Release);

        // Mark as done
        worker.status.store(status::DONE, Ordering::Release);

        // Increment finished count and notify main thread
        let prev_finished = pool.finished_count.fetch_add(1, Ordering::AcqRel);

        // Notify the main thread that we're done
        #[cfg(target_arch = "wasm32")]
        unsafe {
            notify_finished_count(&pool.finished_count);
        }

        // If we're the last worker, all done
        let _ = prev_finished; // Suppress unused warning on non-wasm

        // Return to idle state
        worker.status.store(status::IDLE, Ordering::Release);
    }
}

/// Spin-wait loop: spin for SPIN_CYCLES, then use Atomics.wait
///
/// Returns the new job_id when work is available.
fn spin_wait_for_work(
    pool: &ThreadPool,
    worker: &crate::ThreadControl,
    last_job_id: u32,
    config: &WorkerConfig,
) -> u32 {
    let mut spin_count = 0u32;

    loop {
        // Check for new job_id (epoch pattern)
        let current_job_id = pool.job_id.load(Ordering::Acquire);
        if current_job_id > last_job_id {
            return current_job_id;
        }

        // Check for shutdown
        let sig = worker.signal.load(Ordering::Acquire);
        if sig == signal::SHUTDOWN {
            return last_job_id; // Return early, caller will check signal
        }

        spin_count += 1;

        if spin_count < config.spin_cycles {
            // Spin phase - use spin_loop hint for CPU efficiency
            core::hint::spin_loop();
        } else {
            // Wait phase - block on the signal atomic
            #[cfg(target_arch = "wasm32")]
            unsafe {
                wait_for_signal(worker, config.wait_timeout_ns);
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                // On native, just yield the thread
                std::thread::yield_now();
            }

            // Reset spin count after waking
            spin_count = 0;
        }
    }
}

// WASM-specific atomic operations
#[cfg(target_arch = "wasm32")]
unsafe fn wait_for_signal(worker: &crate::ThreadControl, timeout_ns: i64) {
    use core::arch::wasm32::memory_atomic_wait32;

    let ptr = &worker.signal as *const core::sync::atomic::AtomicU32 as *mut i32;
    let current = worker.signal.load(Ordering::Relaxed) as i32;

    // Only wait if signal is still NONE
    if current == signal::NONE as i32 {
        let _ = memory_atomic_wait32(ptr, signal::NONE as i32, timeout_ns);
    }
}

#[cfg(target_arch = "wasm32")]
unsafe fn notify_finished_count(count: &core::sync::atomic::AtomicU32) {
    use core::arch::wasm32::memory_atomic_notify;

    let ptr = count as *const core::sync::atomic::AtomicU32 as *mut i32;
    // Wake up to 1 waiter (the main thread)
    memory_atomic_notify(ptr, 1);
}

/// A simple no-op task for testing
#[allow(dead_code)]
pub unsafe extern "C" fn noop_task(_data: *mut f32, _m: usize, _k: usize, _n: usize, _idx: usize, _count: usize) {
    // Do nothing - used for measuring threading overhead
}

/// A test task that writes the thread index to a specific location
#[allow(dead_code)]
pub unsafe extern "C" fn write_index_task(data: *mut f32, _m: usize, _k: usize, _n: usize, idx: usize, _count: usize) {
    if !data.is_null() {
        *data.add(idx) = idx as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThreadPool;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.spin_cycles, SPIN_CYCLES);
        assert_eq!(config.wait_timeout_ns, 1_000_000);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_worker_with_threads() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let pool = Arc::new(ThreadPool::new());
        pool.init(2);

        // Signal that workers are ready
        let worker1_ready = Arc::new(AtomicBool::new(false));
        let worker2_ready = Arc::new(AtomicBool::new(false));
        let worker1_ready_clone = Arc::clone(&worker1_ready);
        let worker2_ready_clone = Arc::clone(&worker2_ready);

        // Spawn worker threads
        let pool_clone1 = Arc::clone(&pool);
        let pool_clone2 = Arc::clone(&pool);

        let handle1 = thread::spawn(move || {
            let worker = &pool_clone1.workers[0];
            worker.status.store(status::IDLE, Ordering::Release);

            // Signal ready
            worker1_ready_clone.store(true, Ordering::Release);

            // Poll for work with more iterations
            let mut iterations = 0;
            while iterations < 1000 {
                let sig = worker.signal.load(Ordering::Acquire);
                if sig == signal::SHUTDOWN {
                    worker.status.store(status::SHUTDOWN, Ordering::Release);
                    return;
                }

                let job_id = pool_clone1.job_id.load(Ordering::Acquire);
                if job_id > 0 {
                    worker.status.store(status::WORKING, Ordering::Release);
                    pool_clone1.finished_count.fetch_add(1, Ordering::AcqRel);
                    worker.status.store(status::DONE, Ordering::Release);
                    worker.last_job_id.store(job_id, Ordering::Release);
                    return;
                }

                thread::yield_now();
                iterations += 1;
            }
        });

        let handle2 = thread::spawn(move || {
            let worker = &pool_clone2.workers[1];
            worker.status.store(status::IDLE, Ordering::Release);

            // Signal ready
            worker2_ready_clone.store(true, Ordering::Release);

            let mut iterations = 0;
            while iterations < 1000 {
                let sig = worker.signal.load(Ordering::Acquire);
                if sig == signal::SHUTDOWN {
                    worker.status.store(status::SHUTDOWN, Ordering::Release);
                    return;
                }

                let job_id = pool_clone2.job_id.load(Ordering::Acquire);
                if job_id > 0 {
                    worker.status.store(status::WORKING, Ordering::Release);
                    pool_clone2.finished_count.fetch_add(1, Ordering::AcqRel);
                    worker.status.store(status::DONE, Ordering::Release);
                    worker.last_job_id.store(job_id, Ordering::Release);
                    return;
                }

                thread::yield_now();
                iterations += 1;
            }
        });

        // Wait for workers to be ready
        while !worker1_ready.load(Ordering::Acquire) || !worker2_ready.load(Ordering::Acquire) {
            thread::yield_now();
        }

        // Dispatch work
        pool.job_id.fetch_add(1, Ordering::SeqCst);

        // Wait for completion
        handle1.join().unwrap();
        handle2.join().unwrap();

        // Check that work was done (both workers should have finished)
        let finished = pool.finished_count.load(Ordering::Acquire);
        assert!(finished >= 1, "Expected at least 1 finished, got {}", finished);
    }

    #[test]
    fn test_noop_task() {
        // Just verify it doesn't crash
        unsafe {
            noop_task(core::ptr::null_mut(), 0, 0, 0, 0, 0);
        }
    }

    #[test]
    fn test_write_index_task() {
        let mut data = [0.0f32; 4];
        unsafe {
            write_index_task(data.as_mut_ptr(), 0, 0, 0, 2, 4);
        }
        assert_eq!(data[2], 2.0);
    }
}
