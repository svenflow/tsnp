// Enable the unstable stdarch_wasm_atomic_wait feature for memory.atomic.wait32/notify
// intrinsics. These are the WASM equivalent of futex syscalls.
#![cfg_attr(
    all(target_arch = "wasm32", target_feature = "atomics", feature = "wasm-futex"),
    feature(stdarch_wasm_atomic_wait)
)]

//! pthreadpool-rs: A Rust port of pthreadpool
//!
//! This library provides efficient parallel execution of computational workloads
//! with minimal overhead. It is designed for use cases like matrix operations
//! where low latency and high throughput are critical.
//!
//! # Key Features
//!
//! - **Caller thread participation**: Thread 0 (the calling thread) participates
//!   in computation, avoiding the overhead of waking a sleeping thread.
//! - **Work stealing**: Threads that finish early steal work from slower threads.
//! - **Cache-line alignment**: Thread state is aligned to avoid false sharing.
//! - **Wait-free work distribution**: Uses atomic operations for work claiming.
//!
//! # WASM Support
//!
//! When compiled for WebAssembly with the `wasm-threads` feature, this library
//! uses `wasm-bindgen-rayon` for thread pool management. You must initialize
//! the thread pool from JavaScript before using parallel operations:
//!
//! ```javascript
//! import init, { initThreadPool } from './your_wasm_module.js';
//!
//! await init();
//! await initThreadPool(navigator.hardwareConcurrency);
//! ```
//!
//! # Example
//!
//! ```
//! use pthreadpool_rs::ThreadPool;
//!
//! let pool = ThreadPool::new(4);
//! let results = std::sync::atomic::AtomicUsize::new(0);
//!
//! pool.parallelize_1d(1000, |i| {
//!     // Process item i
//!     results.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//! });
//!
//! assert_eq!(results.load(std::sync::atomic::Ordering::SeqCst), 1000);
//! ```

// Native implementation using std::thread
#[cfg(not(all(target_arch = "wasm32", feature = "wasm-threads")))]
mod pool;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm-threads")))]
mod thread_info;

#[cfg(not(all(target_arch = "wasm32", feature = "wasm-threads")))]
pub use pool::ThreadPool;

// WASM implementation using wasm-bindgen-rayon
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
mod pool_wasm;

#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
pub use pool_wasm::ThreadPool;

// Re-export init_thread_pool for WASM consumers
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;

// Raw WASM futex pool — bypasses Rayon entirely for latency-critical dispatch.
// Uses `memory.atomic.wait32`/`notify` directly, spin-then-park on workers,
// spin-only on the main thread. See module docs for rationale.
//
// Feature-gated because it pulls in `wasm_thread` for Web Worker spawning
// (std::thread::spawn panics on wasm32-unknown-unknown — there's no OS thread
// API to call, you need JS glue to create Workers).
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub mod wasm_futex;

/// Flags for controlling parallelization behavior.
pub mod flags {
    /// Disable denormal floating-point numbers for performance.
    /// This sets the FTZ (flush-to-zero) and DAZ (denormals-are-zero) flags.
    pub const DISABLE_DENORMALS: u32 = 0x00000001;

    /// Yield worker threads when waiting for work instead of spinning.
    /// This reduces CPU usage but may increase latency.
    pub const YIELD_WORKERS: u32 = 0x00000002;
}

// Tests for native implementation (WASM tests are in pool_wasm.rs)
#[cfg(all(test, not(all(target_arch = "wasm32", feature = "wasm-threads"))))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_create_single_thread_pool() {
        let pool = ThreadPool::new(1);
        assert_eq!(pool.threads_count(), 1);
    }

    #[test]
    fn test_create_multi_thread_pool() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.threads_count(), 4);
    }

    #[test]
    fn test_create_default_pool() {
        let pool = ThreadPool::default();
        // Should use available parallelism
        assert!(pool.threads_count() >= 1);
    }

    #[test]
    fn test_parallelize_1d_single_item() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);

        pool.parallelize_1d(1, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_parallelize_1d_empty_range() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);

        pool.parallelize_1d(0, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_parallelize_1d_all_items_processed() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);

        pool.parallelize_1d(1000, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn test_parallelize_1d_each_item_processed_once() {
        let pool = ThreadPool::new(4);
        let items: Vec<AtomicUsize> = (0..100).map(|_| AtomicUsize::new(0)).collect();

        pool.parallelize_1d(100, |i| {
            items[i].fetch_add(1, Ordering::Relaxed);
        });

        // Each item should be processed exactly once
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.load(Ordering::SeqCst), 1, "Item {} was processed {} times", i, item.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn test_parallelize_1d_items_in_bounds() {
        let pool = ThreadPool::new(4);
        let range = 100;
        let max_seen = AtomicUsize::new(0);
        let out_of_bounds = AtomicUsize::new(0);

        pool.parallelize_1d(range, |i| {
            if i >= range {
                out_of_bounds.fetch_add(1, Ordering::Relaxed);
            }
            // Update max seen using CAS loop
            let mut current = max_seen.load(Ordering::Relaxed);
            loop {
                if i <= current {
                    break;
                }
                match max_seen.compare_exchange_weak(current, i, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(c) => current = c,
                }
            }
        });

        assert_eq!(out_of_bounds.load(Ordering::SeqCst), 0);
        assert_eq!(max_seen.load(Ordering::SeqCst), range - 1);
    }

    #[test]
    fn test_parallelize_1d_fewer_items_than_threads() {
        let pool = ThreadPool::new(8);
        let counter = AtomicUsize::new(0);

        pool.parallelize_1d(3, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_parallelize_1d_with_single_thread_pool() {
        let pool = ThreadPool::new(1);
        let items: Vec<AtomicUsize> = (0..50).map(|_| AtomicUsize::new(0)).collect();

        pool.parallelize_1d(50, |i| {
            items[i].fetch_add(1, Ordering::Relaxed);
        });

        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.load(Ordering::SeqCst), 1, "Item {} count: {}", i, item.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn test_parallelize_1d_large_range() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);

        pool.parallelize_1d(10_000, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 10_000);
    }

    #[test]
    fn test_multiple_parallel_calls() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);

        // First call
        pool.parallelize_1d(100, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 100);

        // Second call
        pool.parallelize_1d(200, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 300);

        // Third call
        pool.parallelize_1d(50, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 350);
    }

    #[test]
    fn test_parallelize_1d_with_context() {
        let pool = ThreadPool::new(4);

        struct Context {
            values: Vec<AtomicUsize>,
            multiplier: usize,
        }

        let ctx = Context {
            values: (0..100).map(|_| AtomicUsize::new(0)).collect(),
            multiplier: 5,
        };

        pool.parallelize_1d(100, |i| {
            ctx.values[i].store(i * ctx.multiplier, Ordering::Relaxed);
        });

        for i in 0..100 {
            assert_eq!(ctx.values[i].load(Ordering::SeqCst), i * 5);
        }
    }

    #[test]
    fn test_pool_reuse_stress() {
        let pool = ThreadPool::new(4);

        for iteration in 0..100 {
            let counter = AtomicUsize::new(0);
            let range = (iteration % 50) + 1;

            pool.parallelize_1d(range, |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });

            assert_eq!(counter.load(Ordering::SeqCst), range, "Iteration {}", iteration);
        }
    }

    #[test]
    fn test_high_contention_each_item_once() {
        // Use many threads on a small range to stress work stealing
        let pool = ThreadPool::new(8);
        let items: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();

        pool.parallelize_1d(16, |i| {
            items[i].fetch_add(1, Ordering::Relaxed);
        });

        for (i, item) in items.iter().enumerate() {
            assert_eq!(
                item.load(Ordering::SeqCst),
                1,
                "Item {} was processed {} times (expected 1)",
                i,
                item.load(Ordering::SeqCst)
            );
        }
    }

    #[test]
    fn test_work_stealing_correctness() {
        // Verify all items processed exactly once with uneven workloads
        for thread_count in [2, 3, 4, 5, 7, 8] {
            for range in [1, 2, 3, 5, 7, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129] {
                let pool = ThreadPool::new(thread_count);
                let items: Vec<AtomicUsize> = (0..range).map(|_| AtomicUsize::new(0)).collect();

                pool.parallelize_1d(range, |i| {
                    items[i].fetch_add(1, Ordering::Relaxed);
                });

                for (i, item) in items.iter().enumerate() {
                    assert_eq!(
                        item.load(Ordering::SeqCst),
                        1,
                        "threads={}, range={}: item {} processed {} times",
                        thread_count,
                        range,
                        i,
                        item.load(Ordering::SeqCst)
                    );
                }
            }
        }
    }

    #[test]
    fn test_sum_correctness() {
        // Verify that parallel sum matches sequential sum
        let pool = ThreadPool::new(4);
        let sum = AtomicUsize::new(0);
        let range = 1000;

        pool.parallelize_1d(range, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });

        let expected: usize = (0..range).sum();
        assert_eq!(sum.load(Ordering::SeqCst), expected);
    }

    #[test]
    fn test_concurrent_pool_usage() {
        use std::sync::Arc;

        // Create pool in Arc
        let pool = Arc::new(ThreadPool::new(4));

        // Run multiple sequential operations from the same pool
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            // Note: parallelize_1d blocks, so this is sequential
            pool.parallelize_1d(100, |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }

        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn test_drop_pool_with_pending_work() {
        // Ensure clean shutdown
        for _ in 0..10 {
            let pool = ThreadPool::new(4);
            pool.parallelize_1d(100, |_| {
                // Light work
            });
            // Pool dropped here - should not hang
        }
    }
}

#[cfg(all(test, not(all(target_arch = "wasm32", feature = "wasm-threads"))))]
mod bench_compare {
    use super::*;
    use std::time::Instant;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[test]
    fn bench_pthreadpool_vs_simple() {
        let pool = ThreadPool::new(4);
        let iterations = 1000;
        let range = 1000;
        
        // Warmup
        for _ in 0..10 {
            let counter = AtomicUsize::new(0);
            pool.parallelize_1d(range, |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
        
        // Benchmark pthreadpool-rs
        let start = Instant::now();
        for _ in 0..iterations {
            let counter = AtomicUsize::new(0);
            pool.parallelize_1d(range, |_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
        let pthreadpool_time = start.elapsed();
        
        println!("\n=== Benchmark: {} iterations of parallelize_1d({}) ===", iterations, range);
        println!("pthreadpool-rs: {:?} ({:.2}μs/call)", pthreadpool_time, pthreadpool_time.as_micros() as f64 / iterations as f64);
    }
}
