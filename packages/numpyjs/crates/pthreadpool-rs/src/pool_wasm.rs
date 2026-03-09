//! WASM thread pool implementation using wasm-bindgen-rayon.
//!
//! This module provides a ThreadPool implementation for WebAssembly that uses
//! rayon's thread pool (via wasm-bindgen-rayon) under the hood. The API is
//! identical to the native implementation.
//!
//! # Prerequisites
//!
//! Before using parallelization, you must initialize the thread pool from JavaScript:
//!
//! ```javascript
//! import init, { initThreadPool } from './your_wasm_module.js';
//!
//! await init();
//! await initThreadPool(navigator.hardwareConcurrency);
//! ```
//!
//! This requires:
//! - SharedArrayBuffer support (COOP/COEP headers)
//! - Web Workers

use rayon::prelude::*;

/// A thread pool for parallel computation (WASM implementation).
///
/// This implementation wraps rayon's thread pool which is managed by
/// wasm-bindgen-rayon. The pool is initialized via `init_thread_pool()`
/// from JavaScript before any parallel operations.
pub struct ThreadPool {
    /// Number of threads requested (may differ from actual rayon pool size).
    #[allow(dead_code)]
    threads_count: usize,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads.
    ///
    /// Note: In WASM, the actual thread count is determined by the
    /// `init_thread_pool()` call from JavaScript. This constructor
    /// stores the requested count for API compatibility but rayon
    /// will use whatever was configured at initialization.
    ///
    /// # Panics
    ///
    /// Panics if `threads_count` is 0.
    pub fn new(threads_count: usize) -> Self {
        assert!(threads_count > 0, "Thread pool must have at least 1 thread");
        ThreadPool { threads_count }
    }

    /// Get the number of threads in the pool.
    ///
    /// Returns the number of threads configured in rayon's global thread pool.
    pub fn threads_count(&self) -> usize {
        // Return the actual rayon thread count for accuracy
        rayon::current_num_threads()
    }

    /// Execute a parallel 1D loop.
    ///
    /// The task function is called once for each index in `0..range`.
    /// The calls may happen in any order and from any thread.
    ///
    /// # Arguments
    ///
    /// * `range` - The number of items to process (indices 0..range).
    /// * `task` - A function that processes a single index.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pthreadpool_rs::ThreadPool;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let pool = ThreadPool::new(4);
    /// let sum = AtomicUsize::new(0);
    ///
    /// pool.parallelize_1d(100, |i| {
    ///     sum.fetch_add(i, Ordering::Relaxed);
    /// });
    ///
    /// // Sum of 0..100 = 4950
    /// assert_eq!(sum.load(Ordering::SeqCst), 4950);
    /// ```
    pub fn parallelize_1d<F>(&self, range: usize, task: F)
    where
        F: Fn(usize) + Sync,
    {
        if range == 0 {
            return;
        }

        // Use rayon's parallel iterator
        (0..range).into_par_iter().for_each(|i| {
            task(i);
        });
    }

    /// Execute a parallel 1D loop with tile size hint.
    ///
    /// Similar to `parallelize_1d`, but allows specifying a tile size
    /// for better cache locality when processing blocks of work.
    ///
    /// # Arguments
    ///
    /// * `range` - The number of items to process (indices 0..range).
    /// * `tile_size` - Suggested number of items per work unit.
    /// * `task` - A function that processes a single index.
    pub fn parallelize_1d_tile<F>(&self, range: usize, tile_size: usize, task: F)
    where
        F: Fn(usize) + Sync,
    {
        if range == 0 {
            return;
        }

        // Use rayon's par_chunks equivalent for better work distribution
        let tile_size = tile_size.max(1);
        let num_tiles = (range + tile_size - 1) / tile_size;

        (0..num_tiles).into_par_iter().for_each(|tile_idx| {
            let start = tile_idx * tile_size;
            let end = (start + tile_size).min(range);
            for i in start..end {
                task(i);
            }
        });
    }

    /// Execute a parallel 2D loop.
    ///
    /// The task function is called once for each (i, j) pair where
    /// 0 <= i < range_i and 0 <= j < range_j.
    ///
    /// # Arguments
    ///
    /// * `range_i` - The size of the first dimension.
    /// * `range_j` - The size of the second dimension.
    /// * `task` - A function that processes indices (i, j).
    pub fn parallelize_2d<F>(&self, range_i: usize, range_j: usize, task: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if range_i == 0 || range_j == 0 {
            return;
        }

        let total = range_i * range_j;
        (0..total).into_par_iter().for_each(|idx| {
            let i = idx / range_j;
            let j = idx % range_j;
            task(i, j);
        });
    }
}

impl Default for ThreadPool {
    /// Create a thread pool using rayon's configured thread count.
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_create_pool() {
        let pool = ThreadPool::new(4);
        // In tests without init_thread_pool, rayon uses default (usually 1)
        assert!(pool.threads_count() >= 1);
    }

    #[test]
    fn test_parallelize_1d_empty() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);
        pool.parallelize_1d(0, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_parallelize_1d_correctness() {
        let pool = ThreadPool::new(4);
        let counter = AtomicUsize::new(0);
        let range = 1000;

        pool.parallelize_1d(range, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), range);
    }

    #[test]
    fn test_parallelize_1d_sum() {
        let pool = ThreadPool::new(4);
        let sum = AtomicUsize::new(0);
        let range = 100;

        pool.parallelize_1d(range, |i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });

        let expected: usize = (0..range).sum();
        assert_eq!(sum.load(Ordering::SeqCst), expected);
    }
}
