//! Thread state management with cache-line alignment.
//!
//! Each thread in the pool has its own `ThreadInfo` structure that tracks
//! the work range assigned to that thread. The structure is cache-line
//! aligned (64 bytes) to prevent false sharing between threads.

use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// Size of a cache line on most modern CPUs.
#[allow(dead_code)]
pub const CACHE_LINE_SIZE: usize = 64;

/// Per-thread state for work distribution.
///
/// Each thread has an owned range of work items `[range_start, range_end)`.
/// The thread processes items from `range_start` upward, while work stealers
/// take items from `range_end` downward.
///
/// The structure is aligned to 64 bytes (cache line size) to prevent
/// false sharing between threads.
#[repr(C, align(64))]
pub struct ThreadInfo {
    /// Start of this thread's work range (inclusive).
    /// The owning thread increments this as it processes items.
    pub range_start: AtomicUsize,

    /// End of this thread's work range (exclusive).
    /// Work stealers decrement this to claim items.
    pub range_end: AtomicUsize,

    /// Number of items remaining in this thread's range.
    /// Used for quick checking if work is available.
    /// Can go negative when multiple threads are stealing.
    pub range_length: AtomicIsize,

    /// This thread's index in the pool (0..threads_count).
    /// Thread 0 is always the calling thread.
    pub thread_number: usize,

    // Padding to ensure 64-byte alignment
    _padding: [u8; 64 - 32 - std::mem::size_of::<usize>()],
}

// Compile-time assertion that ThreadInfo is exactly 64 bytes
const _: () = assert!(std::mem::size_of::<ThreadInfo>() == 64);
const _: () = assert!(std::mem::align_of::<ThreadInfo>() == 64);

impl ThreadInfo {
    /// Create a new ThreadInfo for the given thread number.
    pub fn new(thread_number: usize) -> Self {
        Self {
            range_start: AtomicUsize::new(0),
            range_end: AtomicUsize::new(0),
            range_length: AtomicIsize::new(0),
            thread_number,
            _padding: [0; 64 - 32 - std::mem::size_of::<usize>()],
        }
    }

    /// Initialize the work range for this thread.
    ///
    /// Called before starting parallel work to assign this thread's
    /// portion of the total work range.
    #[inline]
    pub fn init_range(&self, start: usize, end: usize) {
        let length = end.saturating_sub(start);
        self.range_start.store(start, Ordering::Relaxed);
        self.range_end.store(end, Ordering::Relaxed);
        self.range_length.store(length as isize, Ordering::Release);
    }

    /// Try to claim an item from this thread's own range.
    ///
    /// Returns `Some(index)` if an item was claimed, `None` if the range is exhausted.
    /// The owning thread claims from the front (incrementing range_start).
    #[inline]
    pub fn try_claim_own(&self) -> Option<usize> {
        // Use the "fastpath" optimization: decrement range_length first,
        // then load range_start. This avoids CAS loops for the common case.
        //
        // The threshold check handles underflow: if range_length goes negative,
        // we've exhausted our range (or someone stole all our work).
        let remaining = self.range_length.fetch_sub(1, Ordering::Acquire);

        if remaining > 0 {
            // We successfully claimed an item
            let index = self.range_start.fetch_add(1, Ordering::Relaxed);
            Some(index)
        } else {
            // Range exhausted - restore the counter (best effort, may race)
            self.range_length.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Try to steal an item from this thread's range.
    ///
    /// Returns `Some(index)` if an item was stolen, `None` if no work available.
    /// Stealers take from the back (decrementing range_end).
    #[inline]
    pub fn try_steal(&self) -> Option<usize> {
        // Same fastpath: decrement range_length first
        let remaining = self.range_length.fetch_sub(1, Ordering::Acquire);

        if remaining > 0 {
            // We successfully claimed an item - take from the end
            let index = self.range_end.fetch_sub(1, Ordering::Relaxed);
            // The index we get is the NEW end, but we want the item AT the old end - 1
            // Since fetch_sub returns the value BEFORE subtraction, index is the old end.
            // We want the item at (old_end - 1), which is (index - 1).
            Some(index - 1)
        } else {
            // Range exhausted - restore the counter
            self.range_length.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Check if this thread has work remaining (without claiming).
    #[inline]
    #[allow(dead_code)]
    pub fn has_work(&self) -> bool {
        self.range_length.load(Ordering::Relaxed) > 0
    }
}

impl Default for ThreadInfo {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_info_size_and_alignment() {
        assert_eq!(std::mem::size_of::<ThreadInfo>(), CACHE_LINE_SIZE);
        assert_eq!(std::mem::align_of::<ThreadInfo>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_init_range() {
        let info = ThreadInfo::new(0);
        info.init_range(10, 20);

        assert_eq!(info.range_start.load(Ordering::SeqCst), 10);
        assert_eq!(info.range_end.load(Ordering::SeqCst), 20);
        assert_eq!(info.range_length.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_try_claim_own() {
        let info = ThreadInfo::new(0);
        info.init_range(0, 5);

        assert_eq!(info.try_claim_own(), Some(0));
        assert_eq!(info.try_claim_own(), Some(1));
        assert_eq!(info.try_claim_own(), Some(2));
        assert_eq!(info.try_claim_own(), Some(3));
        assert_eq!(info.try_claim_own(), Some(4));
        assert_eq!(info.try_claim_own(), None);
        assert_eq!(info.try_claim_own(), None);
    }

    #[test]
    fn test_try_steal() {
        let info = ThreadInfo::new(0);
        info.init_range(0, 5);

        // Stealing takes from the end
        assert_eq!(info.try_steal(), Some(4));
        assert_eq!(info.try_steal(), Some(3));
        assert_eq!(info.try_steal(), Some(2));
        assert_eq!(info.try_steal(), Some(1));
        assert_eq!(info.try_steal(), Some(0));
        assert_eq!(info.try_steal(), None);
    }

    #[test]
    fn test_has_work() {
        let info = ThreadInfo::new(0);
        info.init_range(0, 2);

        assert!(info.has_work());
        info.try_claim_own();
        assert!(info.has_work());
        info.try_claim_own();
        assert!(!info.has_work());
    }

    #[test]
    fn test_empty_range() {
        let info = ThreadInfo::new(0);
        info.init_range(5, 5);

        assert!(!info.has_work());
        assert_eq!(info.try_claim_own(), None);
        assert_eq!(info.try_steal(), None);
    }
}
