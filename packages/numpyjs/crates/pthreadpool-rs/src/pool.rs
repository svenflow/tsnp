//! Thread pool implementation.
//!
//! The pool manages a set of worker threads that can execute parallel workloads.
//! The calling thread (thread 0) participates in computation to avoid the
//! latency of waking a sleeping thread.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::thread_info::ThreadInfo;

/// A thread pool for parallel computation.
///
/// The pool maintains a set of worker threads that are kept alive between
/// parallel operations. When `parallelize_1d` is called, work is distributed
/// across all threads (including the calling thread).
pub struct ThreadPool {
    /// Shared state between the main thread and workers.
    inner: Arc<PoolInner>,

    /// Worker thread handles (not including thread 0, which is the caller).
    workers: Vec<JoinHandle<()>>,
}

/// Shared state for the thread pool.
///
/// This is wrapped in an Arc and shared between the main thread and all workers.
struct PoolInner {
    /// Number of threads in the pool (including thread 0).
    threads_count: usize,

    /// Generation counter that increments each time a new job starts.
    /// Workers compare against their last seen generation to know when new work is available.
    /// High bit (bit 63) is set when shutdown is requested.
    generation: AtomicU64,

    /// Number of threads that haven't finished the current operation.
    active_threads: AtomicUsize,

    /// Per-thread state (cache-line aligned).
    threads: Box<[ThreadInfo]>,

    /// The task to execute (stored as a raw function pointer).
    /// This is set before signaling workers and cleared after completion.
    task: Mutex<Option<TaskFn>>,

    /// Mutex for coordinating parallel operations.
    /// Only one parallel operation can be in flight at a time.
    execution_mutex: Mutex<()>,

    /// Condvar for signaling workers to start.
    worker_condvar: Condvar,

    /// Mutex paired with worker_condvar.
    worker_mutex: Mutex<()>,

    /// Condvar for signaling completion.
    completion_condvar: Condvar,

    /// Mutex paired with completion_condvar.
    completion_mutex: Mutex<()>,
}

const SHUTDOWN_BIT: u64 = 1 << 63;

/// Type-erased task function.
///
/// We store a raw function pointer and context to avoid generics in the shared state.
/// Safety: The task function and context must remain valid for the duration of
/// the parallel operation.
struct TaskFn {
    /// Raw function pointer: fn(*const (), usize)
    func: unsafe fn(*const (), usize),
    /// Context pointer passed to the function.
    context: *const (),
}

// Safety: TaskFn is only accessed while the execution_mutex is held,
// ensuring the context remains valid.
unsafe impl Send for TaskFn {}
unsafe impl Sync for TaskFn {}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads.
    ///
    /// The pool will have exactly `threads_count` threads, including the
    /// calling thread (thread 0). So `ThreadPool::new(4)` creates 3 worker
    /// threads plus uses the calling thread.
    ///
    /// # Panics
    ///
    /// Panics if `threads_count` is 0.
    pub fn new(threads_count: usize) -> Self {
        assert!(threads_count > 0, "Thread pool must have at least 1 thread");

        // Create per-thread state
        let threads: Box<[ThreadInfo]> = (0..threads_count)
            .map(ThreadInfo::new)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inner = Arc::new(PoolInner {
            threads_count,
            generation: AtomicU64::new(0),
            active_threads: AtomicUsize::new(0),
            threads,
            task: Mutex::new(None),
            execution_mutex: Mutex::new(()),
            worker_condvar: Condvar::new(),
            worker_mutex: Mutex::new(()),
            completion_condvar: Condvar::new(),
            completion_mutex: Mutex::new(()),
        });

        // Spawn worker threads (threads 1..threads_count)
        // Thread 0 is the calling thread
        let workers: Vec<JoinHandle<()>> = (1..threads_count)
            .map(|thread_idx| {
                let inner = Arc::clone(&inner);
                thread::spawn(move || {
                    worker_loop(inner, thread_idx);
                })
            })
            .collect();

        ThreadPool { inner, workers }
    }

    /// Get the number of threads in the pool.
    pub fn threads_count(&self) -> usize {
        self.inner.threads_count
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
    /// ```
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

        // Acquire the execution mutex to ensure only one parallel op at a time
        let _guard = self.inner.execution_mutex.lock().unwrap();

        // Type-erase the task
        // Safety: We hold the execution_mutex until the operation completes,
        // and the task closure lives on the stack of the calling function.
        let task_ptr = &task as *const F as *const ();

        unsafe fn call_task<F: Fn(usize)>(context: *const (), index: usize) {
            let task = &*(context as *const F);
            task(index);
        }

        *self.inner.task.lock().unwrap() = Some(TaskFn {
            func: call_task::<F>,
            context: task_ptr,
        });

        // Distribute work across threads
        self.distribute_work(range);

        // Set active threads count (all threads including thread 0)
        self.inner
            .active_threads
            .store(self.inner.threads_count, Ordering::Release);

        // Increment generation to signal new work
        {
            let _guard = self.inner.worker_mutex.lock().unwrap();
            self.inner.generation.fetch_add(1, Ordering::Release);
        }
        self.inner.worker_condvar.notify_all();

        // Thread 0 (caller) participates in computation
        self.execute_thread_work(0);

        // Mark thread 0 as done
        let remaining = self.inner.active_threads.fetch_sub(1, Ordering::AcqRel);
        if remaining > 1 {
            // Wait for other threads to complete
            let guard = self.inner.completion_mutex.lock().unwrap();
            let _guard = self
                .inner
                .completion_condvar
                .wait_while(guard, |_| {
                    self.inner.active_threads.load(Ordering::Acquire) > 0
                })
                .unwrap();
        }

        // Clear the task
        *self.inner.task.lock().unwrap() = None;
    }

    /// Distribute work evenly across all threads.
    fn distribute_work(&self, range: usize) {
        let threads_count = self.inner.threads_count;
        let items_per_thread = range / threads_count;
        let remainder = range % threads_count;

        let mut start = 0;
        for i in 0..threads_count {
            // First `remainder` threads get one extra item
            let extra = if i < remainder { 1 } else { 0 };
            let count = items_per_thread + extra;
            let end = start + count;

            self.inner.threads[i].init_range(start, end);
            start = end;
        }
    }

    /// Execute work for a specific thread, including work stealing.
    fn execute_thread_work(&self, thread_idx: usize) {
        let task_guard = self.inner.task.lock().unwrap();
        let task = task_guard.as_ref().unwrap();
        let func = task.func;
        let context = task.context;
        drop(task_guard);

        let my_thread = &self.inner.threads[thread_idx];

        // Process own work
        while let Some(index) = my_thread.try_claim_own() {
            unsafe {
                func(context, index);
            }
        }

        // Work stealing: try to steal from other threads
        // Go through threads in reverse order (furthest first)
        let threads_count = self.inner.threads_count;
        for offset in 1..threads_count {
            // Steal from thread (thread_idx + offset) % threads_count
            // But iterate in reverse for better cache behavior
            let steal_idx = (thread_idx + threads_count - offset) % threads_count;
            let steal_thread = &self.inner.threads[steal_idx];

            while let Some(index) = steal_thread.try_steal() {
                unsafe {
                    func(context, index);
                }
            }
        }
    }
}

impl Default for ThreadPool {
    /// Create a thread pool with the number of threads equal to available parallelism.
    fn default() -> Self {
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::new(threads)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal workers to shut down by setting the shutdown bit
        {
            let _guard = self.inner.worker_mutex.lock().unwrap();
            self.inner.generation.fetch_or(SHUTDOWN_BIT, Ordering::Release);
        }
        self.inner.worker_condvar.notify_all();

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Worker thread main loop.
///
/// Workers wait for the generation counter to change and then execute work.
fn worker_loop(inner: Arc<PoolInner>, thread_idx: usize) {
    let mut last_generation = 0u64;

    loop {
        // Wait for generation to change
        let current_gen = {
            let guard = inner.worker_mutex.lock().unwrap();
            let guard = inner
                .worker_condvar
                .wait_while(guard, |_| {
                    let gen = inner.generation.load(Ordering::Acquire);
                    gen == last_generation && (gen & SHUTDOWN_BIT) == 0
                })
                .unwrap();
            drop(guard);
            inner.generation.load(Ordering::Acquire)
        };

        // Check for shutdown
        if (current_gen & SHUTDOWN_BIT) != 0 {
            break;
        }

        // Update our last seen generation
        last_generation = current_gen;

        // Execute work for this thread
        execute_worker_work(&inner, thread_idx);

        // Mark this thread as done
        let remaining = inner.active_threads.fetch_sub(1, Ordering::AcqRel);
        if remaining == 1 {
            // This was the last thread, wake the main thread
            let _guard = inner.completion_mutex.lock().unwrap();
            inner.completion_condvar.notify_one();
        }
    }
}

/// Execute work for a worker thread.
fn execute_worker_work(inner: &PoolInner, thread_idx: usize) {
    let task_guard = inner.task.lock().unwrap();
    let task = task_guard.as_ref().unwrap();
    let func = task.func;
    let context = task.context;
    drop(task_guard);

    let my_thread = &inner.threads[thread_idx];

    // Process own work
    while let Some(index) = my_thread.try_claim_own() {
        unsafe {
            func(context, index);
        }
    }

    // Work stealing
    let threads_count = inner.threads_count;
    for offset in 1..threads_count {
        let steal_idx = (thread_idx + threads_count - offset) % threads_count;
        let steal_thread = &inner.threads[steal_idx];

        while let Some(index) = steal_thread.try_steal() {
            unsafe {
                func(context, index);
            }
        }
    }
}
