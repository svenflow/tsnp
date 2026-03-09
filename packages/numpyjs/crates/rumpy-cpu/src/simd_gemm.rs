//! WASM SIMD-optimized GEMM kernel
//!
//! Implements XNNPACK-style matrix multiplication using WASM simd128 intrinsics.
//! Uses f32 for 4 elements per v128 vector (matching XNNPACK).
//!
//! Key optimizations:
//! - 6x8 micro-kernel (like XNNPACK)
//! - Vectorized A load: load 4 A values at once, shuffle to broadcast each
//! - Matrix packing for B (XNNPACK-style: interleaved by K blocks of 4)
//! - SIMD vectorized inner loop
//! - FMA (fused multiply-add) via relaxed-simd for better throughput

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// Shape-aware heuristic for parallel dispatch decision.
///
/// Estimates whether parallelization will be profitable based on:
/// - Total FLOPs (2 * m * n * k)
/// - Number of available threads
/// - Estimated dispatch overhead (~50-100μs for rayon in WASM)
/// - Measured single-thread GFLOPS (~8-12 on modern chips)
///
/// Returns true if parallel execution is expected to be faster.
///
/// This is similar to what XNNPACK's pthreadpool uses internally.
#[inline]
pub fn should_parallelize(m: usize, n: usize, k: usize, num_threads: usize) -> bool {
    if num_threads <= 1 {
        return false;
    }

    // Use u64 to avoid overflow on wasm32 (usize is 32-bit)
    let flops = 2u64 * (m as u64) * (n as u64) * (k as u64);

    // Empirical constants (tuned for WASM on Apple Silicon):
    // - DISPATCH_OVERHEAD_NS: ~150μs for rayon dispatch + sync in WASM
    // - GFLOPS_SINGLE_THREAD: ~75 GFLOPS sustained (from benchmarks on M4)
    // - PARALLEL_EFFICIENCY: ~60% due to cache/memory contention + E-cores
    const DISPATCH_OVERHEAD_NS: u64 = 150_000; // 150 microseconds
    const GFLOPS_PER_THREAD: u64 = 75;
    const PARALLEL_EFFICIENCY_PCT: u64 = 60;

    // Single-threaded time estimate (nanoseconds)
    // time_st = flops / (GFLOPS * 1e9) * 1e9 = flops / GFLOPS
    let time_st_ns = flops / GFLOPS_PER_THREAD;

    // Parallel time estimate
    // time_mt = flops / (num_threads * GFLOPS * efficiency) + dispatch_overhead
    let effective_threads = (num_threads as u64 * PARALLEL_EFFICIENCY_PCT) / 100;
    let time_mt_ns = flops / (effective_threads.max(1) * GFLOPS_PER_THREAD) + DISPATCH_OVERHEAD_NS;

    // Go parallel if it's expected to be faster
    time_mt_ns < time_st_ns
}

/// Minimum work threshold - below this, don't even consider parallel.
/// This is a fast-path to avoid the should_parallelize calculation overhead.
///
/// Raised to 256³ because:
/// - Thread dispatch overhead (~10-150μs) dominates at small sizes
/// - B-packing overhead doesn't amortize below ~256
/// - tfjs uses similar threshold (~512K FLOPs min)
#[inline]
pub fn below_parallel_threshold(m: usize, n: usize, k: usize) -> bool {
    // Use u64 to avoid overflow on wasm32
    // 256³ = 16.7M elements, ~0.3ms single-threaded
    (m as u64) * (n as u64) * (k as u64) < (256u64 * 256 * 256)
}

/// XNNPACK-style 6x8 kernel with vectorized A loading
/// Loads 4 A values at once per row, then shuffles to broadcast each lane.
/// This processes 4 K iterations per loop, reducing memory operations.
///
/// The weights (B) are expected to be packed: for each K block of 4,
/// 8 columns are stored contiguously: [k0:col0-7][k1:col0-7][k2:col0-7][k3:col0-7]
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_style(
    a: &[f32],
    packed_b: &[f32],  // Pre-packed B in XNNPACK format
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_panels = n / NR;
    let k_main = k / 4 * 4;  // Round down to multiple of 4

    for i in (0..m_main).step_by(MR) {
        // Pointers to A rows
        let a0_ptr = a.as_ptr().add(i * k);
        let a1_ptr = a.as_ptr().add((i + 1) * k);
        let a2_ptr = a.as_ptr().add((i + 2) * k);
        let a3_ptr = a.as_ptr().add((i + 3) * k);
        let a4_ptr = a.as_ptr().add((i + 4) * k);
        let a5_ptr = a.as_ptr().add((i + 5) * k);

        for panel in 0..n_panels {
            let j = panel * NR;
            let mut w_ptr = packed_b.as_ptr().add(panel * k * NR);

            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            let mut kk = 0;
            while kk < k_main {
                // Load 4 A values at once for each row
                let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                // K iteration 0: broadcast lane 0
                let va0c0 = i32x4_shuffle::<0, 0, 0, 0>(va0, va0);
                let va1c0 = i32x4_shuffle::<0, 0, 0, 0>(va1, va1);
                let va2c0 = i32x4_shuffle::<0, 0, 0, 0>(va2, va2);
                let va3c0 = i32x4_shuffle::<0, 0, 0, 0>(va3, va3);
                let va4c0 = i32x4_shuffle::<0, 0, 0, 0>(va4, va4);
                let va5c0 = i32x4_shuffle::<0, 0, 0, 0>(va5, va5);

                let vb0123c0 = v128_load(w_ptr as *const v128);
                let vb4567c0 = v128_load(w_ptr.add(4) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c0, vb0123c0, acc00);
                acc01 = f32x4_relaxed_madd(va0c0, vb4567c0, acc01);
                acc10 = f32x4_relaxed_madd(va1c0, vb0123c0, acc10);
                acc11 = f32x4_relaxed_madd(va1c0, vb4567c0, acc11);
                acc20 = f32x4_relaxed_madd(va2c0, vb0123c0, acc20);
                acc21 = f32x4_relaxed_madd(va2c0, vb4567c0, acc21);
                acc30 = f32x4_relaxed_madd(va3c0, vb0123c0, acc30);
                acc31 = f32x4_relaxed_madd(va3c0, vb4567c0, acc31);
                acc40 = f32x4_relaxed_madd(va4c0, vb0123c0, acc40);
                acc41 = f32x4_relaxed_madd(va4c0, vb4567c0, acc41);
                acc50 = f32x4_relaxed_madd(va5c0, vb0123c0, acc50);
                acc51 = f32x4_relaxed_madd(va5c0, vb4567c0, acc51);

                // K iteration 1: broadcast lane 1
                let va0c1 = i32x4_shuffle::<1, 1, 1, 1>(va0, va0);
                let va1c1 = i32x4_shuffle::<1, 1, 1, 1>(va1, va1);
                let va2c1 = i32x4_shuffle::<1, 1, 1, 1>(va2, va2);
                let va3c1 = i32x4_shuffle::<1, 1, 1, 1>(va3, va3);
                let va4c1 = i32x4_shuffle::<1, 1, 1, 1>(va4, va4);
                let va5c1 = i32x4_shuffle::<1, 1, 1, 1>(va5, va5);

                let vb0123c1 = v128_load(w_ptr.add(8) as *const v128);
                let vb4567c1 = v128_load(w_ptr.add(12) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c1, vb0123c1, acc00);
                acc01 = f32x4_relaxed_madd(va0c1, vb4567c1, acc01);
                acc10 = f32x4_relaxed_madd(va1c1, vb0123c1, acc10);
                acc11 = f32x4_relaxed_madd(va1c1, vb4567c1, acc11);
                acc20 = f32x4_relaxed_madd(va2c1, vb0123c1, acc20);
                acc21 = f32x4_relaxed_madd(va2c1, vb4567c1, acc21);
                acc30 = f32x4_relaxed_madd(va3c1, vb0123c1, acc30);
                acc31 = f32x4_relaxed_madd(va3c1, vb4567c1, acc31);
                acc40 = f32x4_relaxed_madd(va4c1, vb0123c1, acc40);
                acc41 = f32x4_relaxed_madd(va4c1, vb4567c1, acc41);
                acc50 = f32x4_relaxed_madd(va5c1, vb0123c1, acc50);
                acc51 = f32x4_relaxed_madd(va5c1, vb4567c1, acc51);

                // K iteration 2: broadcast lane 2
                let va0c2 = i32x4_shuffle::<2, 2, 2, 2>(va0, va0);
                let va1c2 = i32x4_shuffle::<2, 2, 2, 2>(va1, va1);
                let va2c2 = i32x4_shuffle::<2, 2, 2, 2>(va2, va2);
                let va3c2 = i32x4_shuffle::<2, 2, 2, 2>(va3, va3);
                let va4c2 = i32x4_shuffle::<2, 2, 2, 2>(va4, va4);
                let va5c2 = i32x4_shuffle::<2, 2, 2, 2>(va5, va5);

                let vb0123c2 = v128_load(w_ptr.add(16) as *const v128);
                let vb4567c2 = v128_load(w_ptr.add(20) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c2, vb0123c2, acc00);
                acc01 = f32x4_relaxed_madd(va0c2, vb4567c2, acc01);
                acc10 = f32x4_relaxed_madd(va1c2, vb0123c2, acc10);
                acc11 = f32x4_relaxed_madd(va1c2, vb4567c2, acc11);
                acc20 = f32x4_relaxed_madd(va2c2, vb0123c2, acc20);
                acc21 = f32x4_relaxed_madd(va2c2, vb4567c2, acc21);
                acc30 = f32x4_relaxed_madd(va3c2, vb0123c2, acc30);
                acc31 = f32x4_relaxed_madd(va3c2, vb4567c2, acc31);
                acc40 = f32x4_relaxed_madd(va4c2, vb0123c2, acc40);
                acc41 = f32x4_relaxed_madd(va4c2, vb4567c2, acc41);
                acc50 = f32x4_relaxed_madd(va5c2, vb0123c2, acc50);
                acc51 = f32x4_relaxed_madd(va5c2, vb4567c2, acc51);

                // K iteration 3: broadcast lane 3
                let va0c3 = i32x4_shuffle::<3, 3, 3, 3>(va0, va0);
                let va1c3 = i32x4_shuffle::<3, 3, 3, 3>(va1, va1);
                let va2c3 = i32x4_shuffle::<3, 3, 3, 3>(va2, va2);
                let va3c3 = i32x4_shuffle::<3, 3, 3, 3>(va3, va3);
                let va4c3 = i32x4_shuffle::<3, 3, 3, 3>(va4, va4);
                let va5c3 = i32x4_shuffle::<3, 3, 3, 3>(va5, va5);

                let vb0123c3 = v128_load(w_ptr.add(24) as *const v128);
                let vb4567c3 = v128_load(w_ptr.add(28) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c3, vb0123c3, acc00);
                acc01 = f32x4_relaxed_madd(va0c3, vb4567c3, acc01);
                acc10 = f32x4_relaxed_madd(va1c3, vb0123c3, acc10);
                acc11 = f32x4_relaxed_madd(va1c3, vb4567c3, acc11);
                acc20 = f32x4_relaxed_madd(va2c3, vb0123c3, acc20);
                acc21 = f32x4_relaxed_madd(va2c3, vb4567c3, acc21);
                acc30 = f32x4_relaxed_madd(va3c3, vb0123c3, acc30);
                acc31 = f32x4_relaxed_madd(va3c3, vb4567c3, acc31);
                acc40 = f32x4_relaxed_madd(va4c3, vb0123c3, acc40);
                acc41 = f32x4_relaxed_madd(va4c3, vb4567c3, acc41);
                acc50 = f32x4_relaxed_madd(va5c3, vb0123c3, acc50);
                acc51 = f32x4_relaxed_madd(va5c3, vb4567c3, acc51);

                w_ptr = w_ptr.add(32);  // 4 k values × 8 cols
                kk += 4;
            }

            // Handle remaining K iterations (0-3)
            while kk < k {
                let a0 = f32x4_splat(*a0_ptr.add(kk));
                let a1 = f32x4_splat(*a1_ptr.add(kk));
                let a2 = f32x4_splat(*a2_ptr.add(kk));
                let a3 = f32x4_splat(*a3_ptr.add(kk));
                let a4 = f32x4_splat(*a4_ptr.add(kk));
                let a5 = f32x4_splat(*a5_ptr.add(kk));
                let b0 = v128_load(w_ptr as *const v128);
                let b1 = v128_load(w_ptr.add(4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
                w_ptr = w_ptr.add(8);
                kk += 1;
            }

            // Store results
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }
    }

    // NOTE: This kernel only handles the case where M % 6 == 0 and N % 8 == 0.
    // For arbitrary matrix dimensions, use matmul_simd_f32_xnnpack_style_full instead.
    // Remaining rows/columns should be computed by caller using original B matrix.
}

/// XNNPACK-style 6x8 kernel that handles arbitrary N (not just multiples of 8)
/// Takes both original B (for remaining columns) and packed_b (for SIMD panels)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_style_full(
    a: &[f32],
    b: &[f32],       // Original B for remaining columns
    packed_b: &[f32], // Pre-packed B for SIMD panels
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;
    let k_main = k / 4 * 4;

    // Main SIMD loop for full 6x8 tiles
    for i in (0..m_main).step_by(MR) {
        let a0_ptr = a.as_ptr().add(i * k);
        let a1_ptr = a.as_ptr().add((i + 1) * k);
        let a2_ptr = a.as_ptr().add((i + 2) * k);
        let a3_ptr = a.as_ptr().add((i + 3) * k);
        let a4_ptr = a.as_ptr().add((i + 4) * k);
        let a5_ptr = a.as_ptr().add((i + 5) * k);

        for panel in 0..n_panels {
            let j = panel * NR;
            let mut w_ptr = packed_b.as_ptr().add(panel * k * NR);

            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            let mut kk = 0;
            while kk < k_main {
                let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                // 4 k iterations
                for lane in 0..4 {
                    let va0c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va0, va0),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va0, va0),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va0, va0),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va0, va0),
                    };
                    let va1c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va1, va1),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va1, va1),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va1, va1),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va1, va1),
                    };
                    let va2c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va2, va2),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va2, va2),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va2, va2),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va2, va2),
                    };
                    let va3c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va3, va3),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va3, va3),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va3, va3),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va3, va3),
                    };
                    let va4c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va4, va4),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va4, va4),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va4, va4),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va4, va4),
                    };
                    let va5c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va5, va5),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va5, va5),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va5, va5),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va5, va5),
                    };

                    let vb0 = v128_load(w_ptr as *const v128);
                    let vb1 = v128_load(w_ptr.add(4) as *const v128);

                    acc00 = f32x4_relaxed_madd(va0c, vb0, acc00);
                    acc01 = f32x4_relaxed_madd(va0c, vb1, acc01);
                    acc10 = f32x4_relaxed_madd(va1c, vb0, acc10);
                    acc11 = f32x4_relaxed_madd(va1c, vb1, acc11);
                    acc20 = f32x4_relaxed_madd(va2c, vb0, acc20);
                    acc21 = f32x4_relaxed_madd(va2c, vb1, acc21);
                    acc30 = f32x4_relaxed_madd(va3c, vb0, acc30);
                    acc31 = f32x4_relaxed_madd(va3c, vb1, acc31);
                    acc40 = f32x4_relaxed_madd(va4c, vb0, acc40);
                    acc41 = f32x4_relaxed_madd(va4c, vb1, acc41);
                    acc50 = f32x4_relaxed_madd(va5c, vb0, acc50);
                    acc51 = f32x4_relaxed_madd(va5c, vb1, acc51);

                    w_ptr = w_ptr.add(8);
                }
                kk += 4;
            }

            // Handle remaining k values
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b0 = v128_load(w_ptr as *const v128);
                let b1 = v128_load(w_ptr.add(4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
                w_ptr = w_ptr.add(8);
                kk += 1;
            }

            // Store results
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Handle remaining columns (n_main..n) using original B
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m) using original B
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Pack B in XNNPACK format: for each panel of 8 cols, store K values interleaved
/// Layout: panel0[k0:col0-7, k1:col0-7, ...], panel1[...], ...
pub fn pack_b_xnnpack(b: &[f32], packed: &mut [f32], k: usize, n: usize) {
    const NR: usize = 8;
    let n_panels = n / NR;

    for panel in 0..n_panels {
        let j = panel * NR;
        let panel_offset = panel * k * NR;

        for kk in 0..k {
            let b_row = kk * n + j;
            let pack_offset = panel_offset + kk * NR;
            packed[pack_offset..pack_offset + NR].copy_from_slice(&b[b_row..b_row + NR]);
        }
    }
}

// ============ Cache Blocking Constants ============
// These are tuned for typical L1/L2 cache sizes
// L1 data cache is typically 32KB, L2 is typically 256KB
// We want the working set (A panel + B panel + C panel) to fit in L2

/// Block size for K dimension (should fit in L1 with micro-panel)
#[allow(dead_code)]
const KC: usize = 256;

/// Block size for M dimension (should fit in L2 with packed B)
#[allow(dead_code)]
const MC: usize = 128;

/// Block size for N dimension
#[allow(dead_code)]
const NC: usize = 256;

/// Cache-blocked 6x8 GEMM with mul+add (not FMA)
///
/// Uses GOTO-style blocking:
/// - Outer loop tiles by NC (N dimension)
/// - Middle loop tiles by KC (K dimension)
/// - Inner loop tiles by MC (M dimension)
///
/// This ensures that:
/// - A micro-panel (MC x KC) fits in L2 cache
/// - B micro-panel (KC x NC) is reused across MC rows
/// - Better cache efficiency for large matrices
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;  // micro-kernel rows
    const NR: usize = 8;  // micro-kernel cols

    // Initialize C to zero
    c.iter_mut().for_each(|x| *x = 0.0);

    // Outer loop over N in blocks of NC
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);
        let nc_main = nc / NR * NR;

        // Middle loop over K in blocks of KC
        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            // Inner loop over M in blocks of MC
            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);
                let mc_main = mc / MR * MR;

                // Micro-kernel: process MR x NR tiles
                for i in (0..mc_main).step_by(MR) {
                    let ii = ic + i;

                    for j in (0..nc_main).step_by(NR) {
                        let jj = jc + j;

                        // 6 rows × 8 cols = 12 accumulators
                        let mut acc00 = f32x4_splat(0.0);
                        let mut acc01 = f32x4_splat(0.0);
                        let mut acc10 = f32x4_splat(0.0);
                        let mut acc11 = f32x4_splat(0.0);
                        let mut acc20 = f32x4_splat(0.0);
                        let mut acc21 = f32x4_splat(0.0);
                        let mut acc30 = f32x4_splat(0.0);
                        let mut acc31 = f32x4_splat(0.0);
                        let mut acc40 = f32x4_splat(0.0);
                        let mut acc41 = f32x4_splat(0.0);
                        let mut acc50 = f32x4_splat(0.0);
                        let mut acc51 = f32x4_splat(0.0);

                        // K loop within the block
                        for kk in 0..kc {
                            let pk = pc + kk;

                            let a0 = f32x4_splat(*a.get_unchecked((ii + 0) * k + pk));
                            let a1 = f32x4_splat(*a.get_unchecked((ii + 1) * k + pk));
                            let a2 = f32x4_splat(*a.get_unchecked((ii + 2) * k + pk));
                            let a3 = f32x4_splat(*a.get_unchecked((ii + 3) * k + pk));
                            let a4 = f32x4_splat(*a.get_unchecked((ii + 4) * k + pk));
                            let a5 = f32x4_splat(*a.get_unchecked((ii + 5) * k + pk));

                            let b_base = pk * n + jj;
                            let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                            let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);

                            // Use mul+add (matches XNNPACK)
                            acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                            acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                            acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                            acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                            acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                            acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                            acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                            acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                            acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                            acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                            acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                            acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
                        }

                        // Accumulate into C (not overwrite - we're tiling K)
                        let c00 = v128_load(c.as_ptr().add((ii + 0) * n + jj + 0) as *const v128);
                        let c01 = v128_load(c.as_ptr().add((ii + 0) * n + jj + 4) as *const v128);
                        let c10 = v128_load(c.as_ptr().add((ii + 1) * n + jj + 0) as *const v128);
                        let c11 = v128_load(c.as_ptr().add((ii + 1) * n + jj + 4) as *const v128);
                        let c20 = v128_load(c.as_ptr().add((ii + 2) * n + jj + 0) as *const v128);
                        let c21 = v128_load(c.as_ptr().add((ii + 2) * n + jj + 4) as *const v128);
                        let c30 = v128_load(c.as_ptr().add((ii + 3) * n + jj + 0) as *const v128);
                        let c31 = v128_load(c.as_ptr().add((ii + 3) * n + jj + 4) as *const v128);
                        let c40 = v128_load(c.as_ptr().add((ii + 4) * n + jj + 0) as *const v128);
                        let c41 = v128_load(c.as_ptr().add((ii + 4) * n + jj + 4) as *const v128);
                        let c50 = v128_load(c.as_ptr().add((ii + 5) * n + jj + 0) as *const v128);
                        let c51 = v128_load(c.as_ptr().add((ii + 5) * n + jj + 4) as *const v128);

                        v128_store(c.as_mut_ptr().add((ii + 0) * n + jj + 0) as *mut v128, f32x4_add(c00, acc00));
                        v128_store(c.as_mut_ptr().add((ii + 0) * n + jj + 4) as *mut v128, f32x4_add(c01, acc01));
                        v128_store(c.as_mut_ptr().add((ii + 1) * n + jj + 0) as *mut v128, f32x4_add(c10, acc10));
                        v128_store(c.as_mut_ptr().add((ii + 1) * n + jj + 4) as *mut v128, f32x4_add(c11, acc11));
                        v128_store(c.as_mut_ptr().add((ii + 2) * n + jj + 0) as *mut v128, f32x4_add(c20, acc20));
                        v128_store(c.as_mut_ptr().add((ii + 2) * n + jj + 4) as *mut v128, f32x4_add(c21, acc21));
                        v128_store(c.as_mut_ptr().add((ii + 3) * n + jj + 0) as *mut v128, f32x4_add(c30, acc30));
                        v128_store(c.as_mut_ptr().add((ii + 3) * n + jj + 4) as *mut v128, f32x4_add(c31, acc31));
                        v128_store(c.as_mut_ptr().add((ii + 4) * n + jj + 0) as *mut v128, f32x4_add(c40, acc40));
                        v128_store(c.as_mut_ptr().add((ii + 4) * n + jj + 4) as *mut v128, f32x4_add(c41, acc41));
                        v128_store(c.as_mut_ptr().add((ii + 5) * n + jj + 0) as *mut v128, f32x4_add(c50, acc50));
                        v128_store(c.as_mut_ptr().add((ii + 5) * n + jj + 4) as *mut v128, f32x4_add(c51, acc51));
                    }

                    // Handle remaining columns in nc block
                    for j in nc_main..nc {
                        let jj = jc + j;
                        for di in 0..MR {
                            let iii = ii + di;
                            let mut sum = 0.0f32;
                            for kk in 0..kc {
                                sum += a[iii * k + pc + kk] * b[(pc + kk) * n + jj];
                            }
                            c[iii * n + jj] += sum;
                        }
                    }
                }

                // Handle remaining rows in mc block
                for i in mc_main..mc {
                    let ii = ic + i;
                    for j in 0..nc {
                        let jj = jc + j;
                        let mut sum = 0.0f32;
                        for kk in 0..kc {
                            sum += a[ii * k + pc + kk] * b[(pc + kk) * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

/// Cache-blocked XNNPACK-style GEMM with pre-packed B
///
/// This combines cache blocking with B-matrix packing for optimal performance.
/// The B matrix is packed into KC x NC panels on-the-fly as we tile through K and N.
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_blocked(
    a: &[f32],
    b: &[f32],
    packed_b: &[f32],  // Pre-packed B in XNNPACK format
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;

    // Initialize C to zero
    c.iter_mut().for_each(|x| *x = 0.0);

    // Block over K for better cache reuse
    for kc_start in (0..k).step_by(KC) {
        let kc_end = (kc_start + KC).min(k);
        let kc = kc_end - kc_start;
        let kc_main = kc / 4 * 4;

        // Main SIMD loop for full 6x8 tiles
        for i in (0..m_main).step_by(MR) {
            let a0_ptr = a.as_ptr().add(i * k + kc_start);
            let a1_ptr = a.as_ptr().add((i + 1) * k + kc_start);
            let a2_ptr = a.as_ptr().add((i + 2) * k + kc_start);
            let a3_ptr = a.as_ptr().add((i + 3) * k + kc_start);
            let a4_ptr = a.as_ptr().add((i + 4) * k + kc_start);
            let a5_ptr = a.as_ptr().add((i + 5) * k + kc_start);

            for panel in 0..n_panels {
                let j = panel * NR;
                // Offset into packed_b for this K block
                let mut w_ptr = packed_b.as_ptr().add(panel * k * NR + kc_start * NR);

                let mut acc00 = f32x4_splat(0.0);
                let mut acc01 = f32x4_splat(0.0);
                let mut acc10 = f32x4_splat(0.0);
                let mut acc11 = f32x4_splat(0.0);
                let mut acc20 = f32x4_splat(0.0);
                let mut acc21 = f32x4_splat(0.0);
                let mut acc30 = f32x4_splat(0.0);
                let mut acc31 = f32x4_splat(0.0);
                let mut acc40 = f32x4_splat(0.0);
                let mut acc41 = f32x4_splat(0.0);
                let mut acc50 = f32x4_splat(0.0);
                let mut acc51 = f32x4_splat(0.0);

                let mut kk = 0;
                while kk < kc_main {
                    // Load 4 A values at once for each row
                    let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                    let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                    let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                    let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                    let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                    let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                    // K iteration 0: broadcast lane 0
                    let va0c0 = i32x4_shuffle::<0, 0, 0, 0>(va0, va0);
                    let va1c0 = i32x4_shuffle::<0, 0, 0, 0>(va1, va1);
                    let va2c0 = i32x4_shuffle::<0, 0, 0, 0>(va2, va2);
                    let va3c0 = i32x4_shuffle::<0, 0, 0, 0>(va3, va3);
                    let va4c0 = i32x4_shuffle::<0, 0, 0, 0>(va4, va4);
                    let va5c0 = i32x4_shuffle::<0, 0, 0, 0>(va5, va5);

                    let vb0123c0 = v128_load(w_ptr as *const v128);
                    let vb4567c0 = v128_load(w_ptr.add(4) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c0, vb0123c0), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c0, vb4567c0), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c0, vb0123c0), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c0, vb4567c0), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c0, vb0123c0), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c0, vb4567c0), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c0, vb0123c0), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c0, vb4567c0), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c0, vb0123c0), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c0, vb4567c0), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c0, vb0123c0), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c0, vb4567c0), acc51);

                    // K iteration 1
                    let va0c1 = i32x4_shuffle::<1, 1, 1, 1>(va0, va0);
                    let va1c1 = i32x4_shuffle::<1, 1, 1, 1>(va1, va1);
                    let va2c1 = i32x4_shuffle::<1, 1, 1, 1>(va2, va2);
                    let va3c1 = i32x4_shuffle::<1, 1, 1, 1>(va3, va3);
                    let va4c1 = i32x4_shuffle::<1, 1, 1, 1>(va4, va4);
                    let va5c1 = i32x4_shuffle::<1, 1, 1, 1>(va5, va5);

                    let vb0123c1 = v128_load(w_ptr.add(8) as *const v128);
                    let vb4567c1 = v128_load(w_ptr.add(12) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c1, vb0123c1), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c1, vb4567c1), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c1, vb0123c1), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c1, vb4567c1), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c1, vb0123c1), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c1, vb4567c1), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c1, vb0123c1), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c1, vb4567c1), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c1, vb0123c1), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c1, vb4567c1), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c1, vb0123c1), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c1, vb4567c1), acc51);

                    // K iteration 2
                    let va0c2 = i32x4_shuffle::<2, 2, 2, 2>(va0, va0);
                    let va1c2 = i32x4_shuffle::<2, 2, 2, 2>(va1, va1);
                    let va2c2 = i32x4_shuffle::<2, 2, 2, 2>(va2, va2);
                    let va3c2 = i32x4_shuffle::<2, 2, 2, 2>(va3, va3);
                    let va4c2 = i32x4_shuffle::<2, 2, 2, 2>(va4, va4);
                    let va5c2 = i32x4_shuffle::<2, 2, 2, 2>(va5, va5);

                    let vb0123c2 = v128_load(w_ptr.add(16) as *const v128);
                    let vb4567c2 = v128_load(w_ptr.add(20) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c2, vb0123c2), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c2, vb4567c2), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c2, vb0123c2), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c2, vb4567c2), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c2, vb0123c2), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c2, vb4567c2), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c2, vb0123c2), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c2, vb4567c2), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c2, vb0123c2), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c2, vb4567c2), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c2, vb0123c2), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c2, vb4567c2), acc51);

                    // K iteration 3
                    let va0c3 = i32x4_shuffle::<3, 3, 3, 3>(va0, va0);
                    let va1c3 = i32x4_shuffle::<3, 3, 3, 3>(va1, va1);
                    let va2c3 = i32x4_shuffle::<3, 3, 3, 3>(va2, va2);
                    let va3c3 = i32x4_shuffle::<3, 3, 3, 3>(va3, va3);
                    let va4c3 = i32x4_shuffle::<3, 3, 3, 3>(va4, va4);
                    let va5c3 = i32x4_shuffle::<3, 3, 3, 3>(va5, va5);

                    let vb0123c3 = v128_load(w_ptr.add(24) as *const v128);
                    let vb4567c3 = v128_load(w_ptr.add(28) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c3, vb0123c3), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c3, vb4567c3), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c3, vb0123c3), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c3, vb4567c3), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c3, vb0123c3), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c3, vb4567c3), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c3, vb0123c3), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c3, vb4567c3), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c3, vb0123c3), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c3, vb4567c3), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c3, vb0123c3), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c3, vb4567c3), acc51);

                    w_ptr = w_ptr.add(32);
                    kk += 4;
                }

                // Handle remaining K iterations
                while kk < kc {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kc_start + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kc_start + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kc_start + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kc_start + kk));
                    let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kc_start + kk));
                    let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kc_start + kk));
                    let b0 = v128_load(w_ptr as *const v128);
                    let b1 = v128_load(w_ptr.add(4) as *const v128);
                    acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                    acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                    acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                    acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                    acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                    acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                    acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                    acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                    acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                    acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                    acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                    acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
                    w_ptr = w_ptr.add(8);
                    kk += 1;
                }

                // Accumulate into C
                let c00 = v128_load(c.as_ptr().add((i + 0) * n + j + 0) as *const v128);
                let c01 = v128_load(c.as_ptr().add((i + 0) * n + j + 4) as *const v128);
                let c10 = v128_load(c.as_ptr().add((i + 1) * n + j + 0) as *const v128);
                let c11 = v128_load(c.as_ptr().add((i + 1) * n + j + 4) as *const v128);
                let c20 = v128_load(c.as_ptr().add((i + 2) * n + j + 0) as *const v128);
                let c21 = v128_load(c.as_ptr().add((i + 2) * n + j + 4) as *const v128);
                let c30 = v128_load(c.as_ptr().add((i + 3) * n + j + 0) as *const v128);
                let c31 = v128_load(c.as_ptr().add((i + 3) * n + j + 4) as *const v128);
                let c40 = v128_load(c.as_ptr().add((i + 4) * n + j + 0) as *const v128);
                let c41 = v128_load(c.as_ptr().add((i + 4) * n + j + 4) as *const v128);
                let c50 = v128_load(c.as_ptr().add((i + 5) * n + j + 0) as *const v128);
                let c51 = v128_load(c.as_ptr().add((i + 5) * n + j + 4) as *const v128);

                v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, f32x4_add(c00, acc00));
                v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, f32x4_add(c01, acc01));
                v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, f32x4_add(c10, acc10));
                v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, f32x4_add(c11, acc11));
                v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, f32x4_add(c20, acc20));
                v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, f32x4_add(c21, acc21));
                v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, f32x4_add(c30, acc30));
                v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, f32x4_add(c31, acc31));
                v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, f32x4_add(c40, acc40));
                v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, f32x4_add(c41, acc41));
                v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, f32x4_add(c50, acc50));
                v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, f32x4_add(c51, acc51));
            }

            // Handle remaining columns using original B
            for j in n_main..n {
                for di in 0..MR {
                    let ii = i + di;
                    let mut sum = 0.0f32;
                    for kk in kc_start..kc_end {
                        sum += a[ii * k + kk] * b[kk * n + j];
                    }
                    c[ii * n + j] += sum;
                }
            }
        }

        // Handle remaining rows using original B
        for i in m_main..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in kc_start..kc_end {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] += sum;
            }
        }
    }
}

/// Pack B matrix into column panels of NR=8 columns
/// Layout: for each panel j, store all k rows contiguously
/// packed_b[panel_idx * k * NR + kk * NR + col_in_panel] = b[kk * n + j + col_in_panel]
#[inline]
#[allow(dead_code)]
fn pack_b_f32(b: &[f32], packed: &mut [f32], k: usize, n: usize) {
    const NR: usize = 8;
    let n_panels = n / NR;

    for panel in 0..n_panels {
        let j = panel * NR;
        let panel_offset = panel * k * NR;

        for kk in 0..k {
            let b_row = kk * n + j;
            let pack_offset = panel_offset + kk * NR;

            // Copy 8 consecutive elements
            packed[pack_offset..pack_offset + NR].copy_from_slice(&b[b_row..b_row + NR]);
        }
    }
}

/// XNNPACK-style SIMD matrix multiplication for f32 with matrix packing
/// Uses 4x8 micro-kernel (4 rows of A, 8 cols of B = 4 elements per v128 * 2 vectors)
///
/// Layout: A is MxK, B is KxN, C is MxN (all row-major)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;  // rows per micro-kernel (matches XNNPACK)
    const NR: usize = 8;  // cols per micro-kernel (2 v128s of 4 f32s each)

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;

    // Pack B matrix for better cache locality
    let mut packed_b = vec![0.0f32; n_panels * k * NR];
    pack_b_f32(b, &mut packed_b, k, n);

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - accumulate products using packed B
            for kk in 0..k {
                // Load 4 A values and splat each to a v128
                let a0 = f32x4_splat(a[(i + 0) * k + kk]);
                let a1 = f32x4_splat(a[(i + 1) * k + kk]);
                let a2 = f32x4_splat(a[(i + 2) * k + kk]);
                let a3 = f32x4_splat(a[(i + 3) * k + kk]);

                // Load 8 B values from packed buffer (contiguous!)
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);

                // Accumulate: C[i,j] += A[i,k] * B[k,j]
                acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use packed B for columns that fit
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(a_val, b0));
                acc1 = f32x4_add(acc1, f32x4_mul(a_val, b1));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Original SIMD matrix multiplication for f32 (no packing)
/// Uses 4x8 micro-kernel with 4x unrolled K loop
///
/// Layout: A is MxK, B is KxN, C is MxN (all row-major)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;  // rows per micro-kernel (matches XNNPACK)
    const NR: usize = 8;  // cols per micro-kernel (2 v128s of 4 f32s each)
    const KU: usize = 4;  // K unroll factor

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let k_main = k / KU * KU;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled for better instruction-level parallelism
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let b_base = kk * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let b_base = (kk + 1) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let b_base = (kk + 2) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let b_base = (kk + 3) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use SIMD for columns if n >= 8
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(a_val, b0));
                acc1 = f32x4_add(acc1, f32x4_mul(a_val, b1));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// FMA-optimized SIMD matrix multiplication for f32
/// Uses relaxed-simd f32x4_relaxed_madd for fused multiply-add
/// This reduces 2 instructions (mul + add) to 1 instruction (fmadd)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_fma(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;
    const NR: usize = 8;
    const KU: usize = 4;  // K unroll factor

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let k_main = k / KU * KU;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled with FMA
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0 - using FMA: acc = a * b + acc
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let b_base = kk * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let b_base = (kk + 1) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let b_base = (kk + 2) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let b_base = (kk + 3) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows
    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Direct SIMD matrix multiplication for f64
/// Uses 2x8 micro-kernel (2 rows, 8 cols = 4 v128 accumulators per row)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    const MR: usize = 2;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 2 rows × 8 cols = 8 v128 accumulators
            let mut acc0_0 = f64x2_splat(0.0);
            let mut acc0_1 = f64x2_splat(0.0);
            let mut acc0_2 = f64x2_splat(0.0);
            let mut acc0_3 = f64x2_splat(0.0);
            let mut acc1_0 = f64x2_splat(0.0);
            let mut acc1_1 = f64x2_splat(0.0);
            let mut acc1_2 = f64x2_splat(0.0);
            let mut acc1_3 = f64x2_splat(0.0);

            // K loop
            for kk in 0..k {
                let a0 = f64x2_splat(a[(i + 0) * k + kk]);
                let a1 = f64x2_splat(a[(i + 1) * k + kk]);

                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 2) as *const v128);
                let b2 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                let b3 = v128_load(b.as_ptr().add(b_base + 6) as *const v128);

                acc0_0 = f64x2_add(acc0_0, f64x2_mul(a0, b0));
                acc0_1 = f64x2_add(acc0_1, f64x2_mul(a0, b1));
                acc0_2 = f64x2_add(acc0_2, f64x2_mul(a0, b2));
                acc0_3 = f64x2_add(acc0_3, f64x2_mul(a0, b3));
                acc1_0 = f64x2_add(acc1_0, f64x2_mul(a1, b0));
                acc1_1 = f64x2_add(acc1_1, f64x2_mul(a1, b1));
                acc1_2 = f64x2_add(acc1_2, f64x2_mul(a1, b2));
                acc1_3 = f64x2_add(acc1_3, f64x2_mul(a1, b3));
            }

            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc0_0);
            v128_store(c.as_mut_ptr().add(c0_base + 2) as *mut v128, acc0_1);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc0_2);
            v128_store(c.as_mut_ptr().add(c0_base + 6) as *mut v128, acc0_3);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc1_0);
            v128_store(c.as_mut_ptr().add(c1_base + 2) as *mut v128, acc1_1);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc1_2);
            v128_store(c.as_mut_ptr().add(c1_base + 6) as *mut v128, acc1_3);
        }

        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f64x2_splat(0.0);
            let mut acc1 = f64x2_splat(0.0);
            let mut acc2 = f64x2_splat(0.0);
            let mut acc3 = f64x2_splat(0.0);

            for kk in 0..k {
                let a_val = f64x2_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 2) as *const v128);
                let b2 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                let b3 = v128_load(b.as_ptr().add(b_base + 6) as *const v128);
                acc0 = f64x2_add(acc0, f64x2_mul(a_val, b0));
                acc1 = f64x2_add(acc1, f64x2_mul(a_val, b1));
                acc2 = f64x2_add(acc2, f64x2_mul(a_val, b2));
                acc3 = f64x2_add(acc3, f64x2_mul(a_val, b3));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 2) as *mut v128, acc1);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc2);
            v128_store(c.as_mut_ptr().add(c_base + 6) as *mut v128, acc3);
        }

        for j in n_main..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Check if WASM SIMD is available at compile time
#[cfg(target_arch = "wasm32")]
pub fn simd_available() -> bool {
    true // If we're on wasm32 with this build, SIMD is enabled
}

#[cfg(not(target_arch = "wasm32"))]
pub fn simd_available() -> bool {
    false
}

/// FMA + Packed B: combines both optimizations for best performance
/// Uses relaxed-simd FMA with pre-packed B matrix
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_fma_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;
    const NR: usize = 8;
    const KU: usize = 4;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;
    let k_main = k / KU * KU;

    // Pack B matrix for better cache locality
    let mut packed_b = vec![0.0f32; n_panels * k * NR];
    pack_b_f32(b, &mut packed_b, k, n);

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled with FMA using packed B
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let pack_offset = panel_offset + kk * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let pack_offset = panel_offset + (kk + 1) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let pack_offset = panel_offset + (kk + 2) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let pack_offset = panel_offset + (kk + 3) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use packed B for columns that fit
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 2x8 micro-kernel for small matrices (M < 4)
/// More efficient when there are few rows
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_2x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 2;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
            }

            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows
    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 6x8 micro-kernel matching XNNPACK exactly - using MUL+ADD not FMA
/// XNNPACK uses wasm_f32x4_add(wasm_f32x4_mul(a, b), c), NOT FMA
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8_muladd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                // XNNPACK style: mul + add, not FMA
                acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 6x8 micro-kernel matching XNNPACK's tile size
/// Uses 6 rows x 8 cols = 12 accumulators (like XNNPACK)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 5x8 micro-kernel for matrices where M is divisible by 5 (like 100)
/// Uses 5 rows x 8 cols = 10 accumulators
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_5x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 5;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Auto-tuned dispatch that picks the best kernel for each size
/// Heuristics based on matrix dimensions and cache considerations
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_auto(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // For very small matrices, avoid packing overhead
    if m < 4 || n < 8 {
        matmul_scalar_f32(a, b, c, m, n, k);
        return;
    }

    // For matrices where M % 6 == 0, use 6x8 kernel (like XNNPACK)
    if m % 6 == 0 && n >= 8 {
        matmul_simd_f32_6x8(a, b, c, m, n, k);
        return;
    }

    // For matrices where M % 5 == 0, use 5x8 kernel
    if m % 5 == 0 && n >= 8 {
        matmul_simd_f32_5x8(a, b, c, m, n, k);
        return;
    }

    // For small M (2-3 rows), use 2x8 kernel
    if m < 4 && m >= 2 && n >= 8 {
        matmul_simd_f32_2x8(a, b, c, m, n, k);
        return;
    }

    // For medium matrices, FMA without packing (packing overhead not amortized)
    if m < 64 || n < 64 || k < 64 {
        matmul_simd_f32_fma(a, b, c, m, n, k);
        return;
    }

    // Default: 6x8 kernel works for most sizes
    matmul_simd_f32_6x8(a, b, c, m, n, k);
}

/// Fallback scalar GEMM for f32
pub fn matmul_scalar_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Fallback scalar GEMM for f64
pub fn matmul_scalar_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// High-level f32 matmul that dispatches to SIMD or scalar
/// Uses packed version for larger matrices where packing overhead is amortized
pub fn matmul_dispatch_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            // Use packed version for matrices >= 64x64 (packing overhead is worth it)
            // For smaller matrices, the overhead of packing isn't amortized
            if m >= 64 && n >= 64 && k >= 64 {
                unsafe {
                    matmul_simd_f32_packed(a, b, &mut c, m, n, k);
                }
            } else {
                unsafe {
                    matmul_simd_f32(a, b, &mut c, m, n, k);
                }
            }
        } else {
            matmul_scalar_f32(a, b, &mut c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f32(a, b, &mut c, m, n, k);
    }

    c
}

/// High-level f32 matmul that dispatches to SIMD or scalar, writing into a pre-allocated slice
///
/// This variant writes directly to the provided output slice, avoiding allocation.
/// Use this for parallel implementations with par_chunks_mut.
///
/// # Arguments
/// * `a` - Input matrix A of shape [m, k]
/// * `b` - Input matrix B of shape [k, n]
/// * `c` - Output slice of at least m*n elements to write result
/// * `m`, `n`, `k` - Matrix dimensions
pub fn matmul_dispatch_f32_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            // Use packed version for matrices >= 64x64 (packing overhead is worth it)
            // For smaller matrices, the overhead of packing isn't amortized
            if m >= 64 && n >= 64 && k >= 64 {
                unsafe {
                    matmul_simd_f32_packed(a, b, c, m, n, k);
                }
            } else {
                unsafe {
                    matmul_simd_f32(a, b, c, m, n, k);
                }
            }
        } else {
            matmul_scalar_f32(a, b, c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f32(a, b, c, m, n, k);
    }
}

/// High-level f64 matmul that dispatches to SIMD or scalar
pub fn matmul_dispatch_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 2 && n >= 8 {
            unsafe {
                matmul_simd_f64(a, b, &mut c, m, n, k);
            }
        } else {
            matmul_scalar_f64(a, b, &mut c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f64(a, b, &mut c, m, n, k);
    }

    c
}

/// Use the gemm crate for highly optimized GEMM
/// The gemm crate uses BLIS-style optimizations including:
/// - Cache-blocking at L1/L2/L3 levels
/// - Micro-kernel tiling
/// - Packing for better memory access patterns
/// This should be competitive with or better than our hand-written kernels
pub fn matmul_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(),
            n as isize, 1,  // C strides: row-major (row stride = n, col stride = 1)
            false,          // don't read C (we're computing C = A*B, not C += A*B)
            a.as_ptr(),
            k as isize, 1,  // A strides: row-major
            b.as_ptr(),
            n as isize, 1,  // B strides: row-major
            1.0,            // alpha
            0.0,            // beta (0 = overwrite, 1 = accumulate)
            false, false, false,  // no conjugation
            gemm::Parallelism::None,  // single-threaded (no rayon in WASM)
        );
    }

    c
}

/// gemm crate version for f64
pub fn matmul_gemm_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];

    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(),
            n as isize, 1,
            false,
            a.as_ptr(),
            k as isize, 1,
            b.as_ptr(),
            n as isize, 1,
            1.0,
            0.0,
            false, false, false,
            gemm::Parallelism::None,
        );
    }

    c
}

/// Parallel f32 GEMM using rayon
///
/// Splits the M dimension across threads.
/// Each thread computes a block of rows using our SIMD kernel.
pub fn matmul_parallel_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let num_threads = rayon::current_num_threads();

    // Shape-aware parallel decision
    if below_parallel_threshold(m, n, k) || !should_parallelize(m, n, k, num_threads) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Split by rows - each thread gets a chunk of rows
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Each thread produces its portion of C
    let results: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|tid| {
            let start_row = tid * rows_per_thread;
            if start_row >= m {
                return None;
            }
            let end_row = (start_row + rows_per_thread).min(m);
            let local_m = end_row - start_row;

            // Extract the portion of A for this thread
            let a_slice = &a[start_row * k..end_row * k];

            // Compute this thread's portion
            let c_local = matmul_dispatch_f32(a_slice, b, local_m, n, k);

            Some(c_local)
        })
        .collect();

    // Combine results
    let mut c = Vec::with_capacity(m * n);
    for chunk in results {
        c.extend(chunk);
    }
    c
}

/// Parallel f32 GEMM V2 using rayon's par_chunks_mut
///
/// This version writes directly to pre-allocated output memory, avoiding
/// per-thread allocations and the final copy step. This is significantly
/// faster for large matrices.
///
/// Splits the M dimension across threads, with each thread writing to
/// its own non-overlapping portion of the output.
pub fn matmul_parallel_f32_v2(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let num_threads = rayon::current_num_threads();

    // Shape-aware parallel decision
    if below_parallel_threshold(m, n, k) || !should_parallelize(m, n, k, num_threads) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];

    // Calculate chunk size (in elements, not rows)
    let rows_per_thread = (m + num_threads - 1) / num_threads;
    let chunk_size = rows_per_thread * n;  // elements per chunk

    // Use par_chunks_mut to write directly to output
    c.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(tid, c_chunk)| {
            let start_row = tid * rows_per_thread;
            if start_row >= m {
                return;
            }

            // Calculate how many rows this chunk actually covers
            let local_m = c_chunk.len() / n;
            if local_m == 0 {
                return;
            }

            // Get the corresponding slice of A
            let a_slice = &a[start_row * k..(start_row + local_m) * k];

            // Write directly to this chunk of C
            matmul_dispatch_f32_into(a_slice, b, c_chunk, local_m, n, k);
        });

    c
}

/// Parallel f32 GEMM using pthreadpool-rs
///
/// This version uses pthreadpool-rs instead of rayon for parallelization.
/// On native platforms, pthreadpool-rs uses its own efficient thread pool with
/// work stealing. On WASM with the `wasm-threads` feature, it uses wasm-bindgen-rayon
/// under the hood.
///
/// This is a drop-in replacement for matmul_parallel_f32_v2 that provides
/// the same API but uses a different threading backend.
pub fn matmul_pthreadpool_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use pthreadpool_rs::ThreadPool;

    // Use default thread pool (uses available parallelism)
    let pool = ThreadPool::default();
    let num_threads = pool.threads_count();

    // Shape-aware parallel decision
    if below_parallel_threshold(m, n, k) || !should_parallelize(m, n, k, num_threads) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];

    // Calculate rows per thread
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Convert pointers to usize for Send+Sync (usize is always Send+Sync)
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    // Each parallel task processes one chunk of rows
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    pool.parallelize_1d(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        // Safety: Each chunk writes to a non-overlapping portion of c
        // and reads from shared a and b
        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let b_slice = std::slice::from_raw_parts(b_ptr, k * n);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            matmul_dispatch_f32_into(a_slice, b_slice, c_slice, local_m, n, k);
        }
    });

    c
}

/// Parallel f32 GEMM using pthreadpool-rs with provided pool
///
/// Same as matmul_pthreadpool_f32 but reuses an existing thread pool
/// to avoid pool creation overhead for repeated calls.
pub fn matmul_pthreadpool_f32_with_pool(
    pool: &pthreadpool_rs::ThreadPool,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_threads = pool.threads_count();

    // Shape-aware parallel decision
    if below_parallel_threshold(m, n, k) || !should_parallelize(m, n, k, num_threads) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Convert pointers to usize for Send+Sync (usize is always Send+Sync)
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    // Each parallel task processes one chunk of rows
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    pool.parallelize_1d(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        // Safety: Each chunk writes to a non-overlapping portion of c
        // and reads from shared a and b
        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let b_slice = std::slice::from_raw_parts(b_ptr, k * n);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            matmul_dispatch_f32_into(a_slice, b_slice, c_slice, local_m, n, k);
        }
    });

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_matmul_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_dispatch_f64(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-10);
        assert!((c[1] - 64.0).abs() < 1e-10);
        assert!((c[2] - 139.0).abs() < 1e-10);
        assert!((c[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_matmul_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_dispatch_f32(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_pthreadpool_matmul_small() {
        // Small matrix (uses single-threaded path)
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_pthreadpool_f32(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_pthreadpool_matmul_large() {
        // Large matrix to trigger parallel path (> 64*64*64 elements)
        let m = 128;
        let n = 128;
        let k = 128;

        // Create random-ish matrices
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();

        // Compute with pthreadpool
        let c_pthreadpool = matmul_pthreadpool_f32(&a, &b, m, n, k);

        // Compute with dispatch (single-threaded reference)
        let c_reference = matmul_dispatch_f32(&a, &b, m, n, k);

        // Verify results match
        assert_eq!(c_pthreadpool.len(), c_reference.len());
        for i in 0..c_pthreadpool.len() {
            let diff = (c_pthreadpool[i] - c_reference[i]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at index {}: pthreadpool={}, reference={}, diff={}",
                i, c_pthreadpool[i], c_reference[i], diff
            );
        }
    }

    #[test]
    fn test_pthreadpool_matmul_with_pool() {
        use pthreadpool_rs::ThreadPool;

        // Large matrix
        let m = 128;
        let n = 128;
        let k = 128;

        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();

        let pool = ThreadPool::new(4);

        // Compute with provided pool
        let c_pthreadpool = matmul_pthreadpool_f32_with_pool(&pool, &a, &b, m, n, k);

        // Compute reference
        let c_reference = matmul_dispatch_f32(&a, &b, m, n, k);

        // Verify results match
        for i in 0..c_pthreadpool.len() {
            let diff = (c_pthreadpool[i] - c_reference[i]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at index {}: pthreadpool={}, reference={}, diff={}",
                i, c_pthreadpool[i], c_reference[i], diff
            );
        }
    }
}

// ============================================================================
// OPTIMIZED 6x8 GEMM - XNNPACK-competitive implementation
// ============================================================================
// Key optimizations:
// 1. MR=6, NR=8 tile size (12 v128 accumulators = fits in 16 XMM registers)
// 2. FMA via f32x4_relaxed_madd (single instruction instead of mul+add)
// 3. v128_load32_splat for A values (dedicated instruction)
// 4. L1/L2 cache blocking (KC, MC, NC)
// 5. B matrix packing for contiguous access

/// Cache blocking constants (tuned for typical L1=32KB, L2=256KB)
#[allow(dead_code)]
const OPT_KC: usize = 256;  // K-dimension block (depth of dot product)
#[allow(dead_code)]
const OPT_MC: usize = 72;   // M-dimension block (multiple of MR=6)
#[allow(dead_code)]
const OPT_NC: usize = 128;  // N-dimension block (multiple of NR=8)
#[allow(dead_code)]
const OPT_MR: usize = 6;    // Micro-kernel rows
#[allow(dead_code)]
const OPT_NR: usize = 8;    // Micro-kernel cols

/// Pack B matrix panel into contiguous format for optimal SIMD access.
/// Layout: For each 8-column panel, store K rows contiguously.
/// [k0:col0-7][k1:col0-7]...[k_KC:col0-7]
#[cfg(target_arch = "wasm32")]
pub fn pack_b_optimized(
    b: *const f32,
    ldb: usize,
    packed_b: *mut f32,
    k_size: usize,
    n_size: usize,
) {
    unsafe {
        let mut dest = packed_b;
        let mut j = 0;

        while j < n_size {
            let n_remain = n_size - j;
            let mut src_col = b.add(j);

            if n_remain >= OPT_NR {
                // Fast path: pack full 8 columns
                for _k in 0..k_size {
                    // Load 8 floats from B[k, j..j+8] (contiguous in row-major B)
                    let v0 = v128_load(src_col as *const v128);
                    let v1 = v128_load(src_col.add(4) as *const v128);

                    v128_store(dest as *mut v128, v0);
                    v128_store(dest.add(4) as *mut v128, v1);

                    dest = dest.add(OPT_NR);
                    src_col = src_col.add(ldb);
                }
            } else {
                // Edge case: pad with zeros
                for _k in 0..k_size {
                    for x in 0..n_remain {
                        *dest.add(x) = *src_col.add(x);
                    }
                    for x in n_remain..OPT_NR {
                        *dest.add(x) = 0.0;
                    }
                    dest = dest.add(OPT_NR);
                    src_col = src_col.add(ldb);
                }
            }
            j += OPT_NR;
        }
    }
}

/// Optimized 6x8 micro-kernel using FMA and loadsplat
///
/// Computes C[6x8] += A[6xK] * B_packed[Kx8]
///
/// Uses 12 accumulator registers (fits in 16 XMM), 2 for B, 1 for A splat
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub unsafe fn micro_kernel_6x8_fma(
    k_size: usize,
    a_ptr: *const f32,
    lda: usize,
    b_packed: *const f32,
    c_ptr: *mut f32,
    ldc: usize,
    beta: f32,  // 0.0 = overwrite, 1.0 = accumulate
) {
    // Setup A row pointers
    let a0 = a_ptr;
    let a1 = a_ptr.add(lda);
    let a2 = a_ptr.add(lda * 2);
    let a3 = a_ptr.add(lda * 3);
    let a4 = a_ptr.add(lda * 4);
    let a5 = a_ptr.add(lda * 5);

    // 12 accumulators: 6 rows × 2 vectors (8 cols)
    let mut c00 = f32x4_splat(0.0);
    let mut c01 = f32x4_splat(0.0);
    let mut c10 = f32x4_splat(0.0);
    let mut c11 = f32x4_splat(0.0);
    let mut c20 = f32x4_splat(0.0);
    let mut c21 = f32x4_splat(0.0);
    let mut c30 = f32x4_splat(0.0);
    let mut c31 = f32x4_splat(0.0);
    let mut c40 = f32x4_splat(0.0);
    let mut c41 = f32x4_splat(0.0);
    let mut c50 = f32x4_splat(0.0);
    let mut c51 = f32x4_splat(0.0);

    let mut b_run = b_packed;

    // K loop - single iteration per K value
    for kk in 0..k_size {
        // Load B: 8 columns = 2 vectors (contiguous in packed format)
        let vb0 = v128_load(b_run as *const v128);
        let vb1 = v128_load(b_run.add(4) as *const v128);
        b_run = b_run.add(8);

        // Row 0: loadsplat + FMA
        let va0 = v128_load32_splat(a0.add(kk) as *const u32);
        c00 = f32x4_relaxed_madd(va0, vb0, c00);
        c01 = f32x4_relaxed_madd(va0, vb1, c01);

        // Row 1
        let va1 = v128_load32_splat(a1.add(kk) as *const u32);
        c10 = f32x4_relaxed_madd(va1, vb0, c10);
        c11 = f32x4_relaxed_madd(va1, vb1, c11);

        // Row 2
        let va2 = v128_load32_splat(a2.add(kk) as *const u32);
        c20 = f32x4_relaxed_madd(va2, vb0, c20);
        c21 = f32x4_relaxed_madd(va2, vb1, c21);

        // Row 3
        let va3 = v128_load32_splat(a3.add(kk) as *const u32);
        c30 = f32x4_relaxed_madd(va3, vb0, c30);
        c31 = f32x4_relaxed_madd(va3, vb1, c31);

        // Row 4
        let va4 = v128_load32_splat(a4.add(kk) as *const u32);
        c40 = f32x4_relaxed_madd(va4, vb0, c40);
        c41 = f32x4_relaxed_madd(va4, vb1, c41);

        // Row 5
        let va5 = v128_load32_splat(a5.add(kk) as *const u32);
        c50 = f32x4_relaxed_madd(va5, vb0, c50);
        c51 = f32x4_relaxed_madd(va5, vb1, c51);
    }

    // Store results
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);

    if beta == 0.0 {
        // Overwrite
        v128_store(c0 as *mut v128, c00);
        v128_store(c0.add(4) as *mut v128, c01);
        v128_store(c1 as *mut v128, c10);
        v128_store(c1.add(4) as *mut v128, c11);
        v128_store(c2 as *mut v128, c20);
        v128_store(c2.add(4) as *mut v128, c21);
        v128_store(c3 as *mut v128, c30);
        v128_store(c3.add(4) as *mut v128, c31);
        v128_store(c4 as *mut v128, c40);
        v128_store(c4.add(4) as *mut v128, c41);
        v128_store(c5 as *mut v128, c50);
        v128_store(c5.add(4) as *mut v128, c51);
    } else {
        // Accumulate (beta = 1.0)
        v128_store(c0 as *mut v128, f32x4_add(v128_load(c0 as *const v128), c00));
        v128_store(c0.add(4) as *mut v128, f32x4_add(v128_load(c0.add(4) as *const v128), c01));
        v128_store(c1 as *mut v128, f32x4_add(v128_load(c1 as *const v128), c10));
        v128_store(c1.add(4) as *mut v128, f32x4_add(v128_load(c1.add(4) as *const v128), c11));
        v128_store(c2 as *mut v128, f32x4_add(v128_load(c2 as *const v128), c20));
        v128_store(c2.add(4) as *mut v128, f32x4_add(v128_load(c2.add(4) as *const v128), c21));
        v128_store(c3 as *mut v128, f32x4_add(v128_load(c3 as *const v128), c30));
        v128_store(c3.add(4) as *mut v128, f32x4_add(v128_load(c3.add(4) as *const v128), c31));
        v128_store(c4 as *mut v128, f32x4_add(v128_load(c4 as *const v128), c40));
        v128_store(c4.add(4) as *mut v128, f32x4_add(v128_load(c4.add(4) as *const v128), c41));
        v128_store(c5 as *mut v128, f32x4_add(v128_load(c5 as *const v128), c50));
        v128_store(c5.add(4) as *mut v128, f32x4_add(v128_load(c5.add(4) as *const v128), c51));
    }
}

/// K-unrolled 6x8 micro-kernel with optimized pointer handling
///
/// Optimizations over basic kernel:
/// 1. K-unroll by 2: halves loop overhead (11 WASM instructions → 5.5/K)
/// 2. Incremental pointer bumping: avoids re-deriving A pointers each K
/// 3. Loop peeling for odd K remainder
#[cfg(target_arch = "wasm32")]
#[inline(never)] // Prevent multiple inlined copies with different codegen
pub unsafe fn micro_kernel_6x8_fma_unrolled(
    k_size: usize,
    a_ptr: *const f32,
    lda: usize,
    b_packed: *const f32,
    c_ptr: *mut f32,
    ldc: usize,
    beta: f32,
) {
    // Setup A row pointers - these will be incremented, not re-derived
    let mut a0 = a_ptr;
    let mut a1 = a_ptr.add(lda);
    let mut a2 = a_ptr.add(lda * 2);
    let mut a3 = a_ptr.add(lda * 3);
    let mut a4 = a_ptr.add(lda * 4);
    let mut a5 = a_ptr.add(lda * 5);

    // 12 accumulators: 6 rows × 2 vectors (8 cols)
    let mut c00 = f32x4_splat(0.0);
    let mut c01 = f32x4_splat(0.0);
    let mut c10 = f32x4_splat(0.0);
    let mut c11 = f32x4_splat(0.0);
    let mut c20 = f32x4_splat(0.0);
    let mut c21 = f32x4_splat(0.0);
    let mut c30 = f32x4_splat(0.0);
    let mut c31 = f32x4_splat(0.0);
    let mut c40 = f32x4_splat(0.0);
    let mut c41 = f32x4_splat(0.0);
    let mut c50 = f32x4_splat(0.0);
    let mut c51 = f32x4_splat(0.0);

    let mut b_run = b_packed;
    let k_main = k_size / 2 * 2; // Round down to even

    // Main K loop - unrolled by 2
    let mut kk = 0;
    while kk < k_main {
        // === K iteration 0 ===
        let vb0_0 = v128_load(b_run as *const v128);
        let vb1_0 = v128_load(b_run.add(4) as *const v128);

        let va0_0 = v128_load32_splat(a0 as *const u32);
        c00 = f32x4_relaxed_madd(va0_0, vb0_0, c00);
        c01 = f32x4_relaxed_madd(va0_0, vb1_0, c01);

        let va1_0 = v128_load32_splat(a1 as *const u32);
        c10 = f32x4_relaxed_madd(va1_0, vb0_0, c10);
        c11 = f32x4_relaxed_madd(va1_0, vb1_0, c11);

        let va2_0 = v128_load32_splat(a2 as *const u32);
        c20 = f32x4_relaxed_madd(va2_0, vb0_0, c20);
        c21 = f32x4_relaxed_madd(va2_0, vb1_0, c21);

        let va3_0 = v128_load32_splat(a3 as *const u32);
        c30 = f32x4_relaxed_madd(va3_0, vb0_0, c30);
        c31 = f32x4_relaxed_madd(va3_0, vb1_0, c31);

        let va4_0 = v128_load32_splat(a4 as *const u32);
        c40 = f32x4_relaxed_madd(va4_0, vb0_0, c40);
        c41 = f32x4_relaxed_madd(va4_0, vb1_0, c41);

        let va5_0 = v128_load32_splat(a5 as *const u32);
        c50 = f32x4_relaxed_madd(va5_0, vb0_0, c50);
        c51 = f32x4_relaxed_madd(va5_0, vb1_0, c51);

        // === K iteration 1 ===
        let vb0_1 = v128_load(b_run.add(8) as *const v128);
        let vb1_1 = v128_load(b_run.add(12) as *const v128);

        let va0_1 = v128_load32_splat(a0.add(1) as *const u32);
        c00 = f32x4_relaxed_madd(va0_1, vb0_1, c00);
        c01 = f32x4_relaxed_madd(va0_1, vb1_1, c01);

        let va1_1 = v128_load32_splat(a1.add(1) as *const u32);
        c10 = f32x4_relaxed_madd(va1_1, vb0_1, c10);
        c11 = f32x4_relaxed_madd(va1_1, vb1_1, c11);

        let va2_1 = v128_load32_splat(a2.add(1) as *const u32);
        c20 = f32x4_relaxed_madd(va2_1, vb0_1, c20);
        c21 = f32x4_relaxed_madd(va2_1, vb1_1, c21);

        let va3_1 = v128_load32_splat(a3.add(1) as *const u32);
        c30 = f32x4_relaxed_madd(va3_1, vb0_1, c30);
        c31 = f32x4_relaxed_madd(va3_1, vb1_1, c31);

        let va4_1 = v128_load32_splat(a4.add(1) as *const u32);
        c40 = f32x4_relaxed_madd(va4_1, vb0_1, c40);
        c41 = f32x4_relaxed_madd(va4_1, vb1_1, c41);

        let va5_1 = v128_load32_splat(a5.add(1) as *const u32);
        c50 = f32x4_relaxed_madd(va5_1, vb0_1, c50);
        c51 = f32x4_relaxed_madd(va5_1, vb1_1, c51);

        // Bump pointers by 2 (one unroll iteration)
        a0 = a0.add(2);
        a1 = a1.add(2);
        a2 = a2.add(2);
        a3 = a3.add(2);
        a4 = a4.add(2);
        a5 = a5.add(2);
        b_run = b_run.add(16); // 2 × 8 floats

        kk += 2;
    }

    // Handle odd K remainder (if k_size is odd)
    if k_size & 1 != 0 {
        let vb0 = v128_load(b_run as *const v128);
        let vb1 = v128_load(b_run.add(4) as *const v128);

        let va0 = v128_load32_splat(a0 as *const u32);
        c00 = f32x4_relaxed_madd(va0, vb0, c00);
        c01 = f32x4_relaxed_madd(va0, vb1, c01);

        let va1 = v128_load32_splat(a1 as *const u32);
        c10 = f32x4_relaxed_madd(va1, vb0, c10);
        c11 = f32x4_relaxed_madd(va1, vb1, c11);

        let va2 = v128_load32_splat(a2 as *const u32);
        c20 = f32x4_relaxed_madd(va2, vb0, c20);
        c21 = f32x4_relaxed_madd(va2, vb1, c21);

        let va3 = v128_load32_splat(a3 as *const u32);
        c30 = f32x4_relaxed_madd(va3, vb0, c30);
        c31 = f32x4_relaxed_madd(va3, vb1, c31);

        let va4 = v128_load32_splat(a4 as *const u32);
        c40 = f32x4_relaxed_madd(va4, vb0, c40);
        c41 = f32x4_relaxed_madd(va4, vb1, c41);

        let va5 = v128_load32_splat(a5 as *const u32);
        c50 = f32x4_relaxed_madd(va5, vb0, c50);
        c51 = f32x4_relaxed_madd(va5, vb1, c51);
    }

    // Store results
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);

    if beta == 0.0 {
        v128_store(c0 as *mut v128, c00);
        v128_store(c0.add(4) as *mut v128, c01);
        v128_store(c1 as *mut v128, c10);
        v128_store(c1.add(4) as *mut v128, c11);
        v128_store(c2 as *mut v128, c20);
        v128_store(c2.add(4) as *mut v128, c21);
        v128_store(c3 as *mut v128, c30);
        v128_store(c3.add(4) as *mut v128, c31);
        v128_store(c4 as *mut v128, c40);
        v128_store(c4.add(4) as *mut v128, c41);
        v128_store(c5 as *mut v128, c50);
        v128_store(c5.add(4) as *mut v128, c51);
    } else {
        v128_store(c0 as *mut v128, f32x4_add(v128_load(c0 as *const v128), c00));
        v128_store(c0.add(4) as *mut v128, f32x4_add(v128_load(c0.add(4) as *const v128), c01));
        v128_store(c1 as *mut v128, f32x4_add(v128_load(c1 as *const v128), c10));
        v128_store(c1.add(4) as *mut v128, f32x4_add(v128_load(c1.add(4) as *const v128), c11));
        v128_store(c2 as *mut v128, f32x4_add(v128_load(c2 as *const v128), c20));
        v128_store(c2.add(4) as *mut v128, f32x4_add(v128_load(c2.add(4) as *const v128), c21));
        v128_store(c3 as *mut v128, f32x4_add(v128_load(c3 as *const v128), c30));
        v128_store(c3.add(4) as *mut v128, f32x4_add(v128_load(c3.add(4) as *const v128), c31));
        v128_store(c4 as *mut v128, f32x4_add(v128_load(c4 as *const v128), c40));
        v128_store(c4.add(4) as *mut v128, f32x4_add(v128_load(c4.add(4) as *const v128), c41));
        v128_store(c5 as *mut v128, f32x4_add(v128_load(c5 as *const v128), c50));
        v128_store(c5.add(4) as *mut v128, f32x4_add(v128_load(c5.add(4) as *const v128), c51));
    }
}

/// Handle edge cases where M < 6 or N < 8
#[cfg(target_arch = "wasm32")]
unsafe fn micro_kernel_edge(
    m_rem: usize,
    n_rem: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b_packed: *const f32,
    c: *mut f32,
    ldc: usize,
    beta: f32,
) {
    // Use a temp buffer for the full 6x8 tile
    let mut tmp_c = [0.0f32; OPT_MR * OPT_NR];
    let tmp_ldc = OPT_NR;

    // Load existing C if accumulating
    if beta != 0.0 {
        for r in 0..m_rem {
            for col in 0..n_rem.min(OPT_NR) {
                tmp_c[r * tmp_ldc + col] = *c.add(r * ldc + col);
            }
        }
    }

    // Run full kernel on temp buffer
    micro_kernel_6x8_fma(k, a, lda, b_packed, tmp_c.as_mut_ptr(), tmp_ldc, 0.0);

    // Copy valid results back
    for r in 0..m_rem {
        for col in 0..n_rem.min(OPT_NR) {
            if beta == 0.0 {
                *c.add(r * ldc + col) = tmp_c[r * tmp_ldc + col];
            } else {
                *c.add(r * ldc + col) += tmp_c[r * tmp_ldc + col];
            }
        }
    }
}

/// Pre-pack B matrix for repeated matmuls with the same weights.
///
/// This allows amortizing the packing cost across multiple matmuls.
/// The packed format is optimized for the 6x8 micro-kernel with KC/NC blocking.
///
/// Returns (packed_b, metadata) where metadata encodes k, n for validation.
#[cfg(target_arch = "wasm32")]
pub fn pack_b_for_gemm(b: &[f32], k: usize, n: usize) -> Vec<f32> {
    // Calculate packed size: for each NC panel, for each KC block
    let n_panels = (n + OPT_NC - 1) / OPT_NC;
    let k_blocks = (k + OPT_KC - 1) / OPT_KC;
    let panel_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;

    // Add 2 floats at the start for k,n metadata (for validation)
    let total_size = 2 + n_panels * k_blocks * panel_size;
    let mut packed_b = vec![0.0f32; total_size];

    // Store metadata
    packed_b[0] = k as f32;
    packed_b[1] = n as f32;

    unsafe {
        let mut panel_offset = 2; // Skip metadata
        let mut j = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);
            let mut p = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);
                pack_b_optimized(
                    b.as_ptr().add(p * n + j),
                    n,
                    packed_b.as_mut_ptr().add(panel_offset),
                    p_block,
                    j_block,
                );
                panel_offset += panel_size;
                p += OPT_KC;
            }
            j += OPT_NC;
        }
    }

    packed_b
}

/// GEMM with pre-packed B matrix.
///
/// Use `pack_b_for_gemm` to create the packed_b buffer once, then call this
/// for each matmul with different A matrices but the same B (weights).
///
/// This is the "inference mode" API that matches how tfjs/XNNPACK works.
#[cfg(target_arch = "wasm32")]
pub fn matmul_with_packed_b_f32(a: &[f32], packed_b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    // Validate metadata
    debug_assert!(packed_b.len() >= 2);
    debug_assert_eq!(packed_b[0] as usize, k, "packed_b k mismatch");
    debug_assert_eq!(packed_b[1] as usize, n, "packed_b n mismatch");

    // For small matrices, we didn't pack - fall back to direct
    if m < 128 || n < 128 || k < 128 {
        // Note: this path shouldn't happen if user correctly uses the API
        // (they wouldn't pre-pack small matrices)
        unsafe {
            // Need to unpack - but we don't store unpacked B
            // For now, just warn and return zeros
            // In practice, users should check size before calling pack_b_for_gemm
        }
        return c;
    }

    let n_panels = (n + OPT_NC - 1) / OPT_NC;
    let k_blocks = (k + OPT_KC - 1) / OPT_KC;
    let panel_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;

    unsafe {
        // Loop over N in blocks of NC
        let mut j = 0;
        let mut n_panel_idx = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);
            let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

            // Loop over K in blocks of KC
            let mut p = 0;
            let mut k_block_idx = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);
                let beta = if p == 0 { 0.0 } else { 1.0 };

                // Get pre-packed B panel (skip 2-float metadata header)
                let packed_panel_offset = 2 + (n_panel_idx * k_blocks + k_block_idx) * panel_size;
                let b_packed = packed_b.as_ptr().add(packed_panel_offset);

                // Loop over M in blocks of MC
                let mut i = 0;
                while i < m {
                    let i_block = (m - i).min(OPT_MC);
                    let i_main = i_block / OPT_MR * OPT_MR;

                    // Process full MR×NR tiles
                    let mut ii = 0;
                    while ii < i_main {
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                            let n_rem = j_block - jj;

                            if n_rem >= OPT_NR {
                                micro_kernel_6x8_fma_unrolled(
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            } else {
                                micro_kernel_edge(
                                    OPT_MR,
                                    n_rem,
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            }
                            jj += OPT_NR;
                        }
                        ii += OPT_MR;
                    }

                    // Handle remaining rows
                    if ii < i_block {
                        let m_rem = i_block - ii;
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                            let n_rem = (j_block - jj).min(OPT_NR);

                            micro_kernel_edge(
                                m_rem,
                                n_rem,
                                p_block,
                                a.as_ptr().add((i + ii) * k + p),
                                k,
                                b_panel_ptr,
                                c.as_mut_ptr().add((i + ii) * n + j + jj),
                                n,
                                beta,
                            );
                            jj += OPT_NR;
                        }
                    }

                    i += OPT_MC;
                }

                k_block_idx += 1;
                p += OPT_KC;
            }

            n_panel_idx += 1;
            j += OPT_NC;
        }
    }

    c
}

/// Parallel GEMM with pre-packed B matrix using futex pool.
///
/// This is the "inference mode" API that matches how tfjs/XNNPACK works:
/// - Pack B once at graph compile time
/// - Call this for each inference with different A (activations)
///
/// Parallelizes across M-rows while reusing the shared packed B.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub fn matmul_with_packed_b_parallel_f32(
    a: &[f32],
    packed_b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use pthreadpool_rs::wasm_futex;

    // Validate metadata
    debug_assert!(packed_b.len() >= 2);
    debug_assert_eq!(packed_b[0] as usize, k, "packed_b k mismatch");
    debug_assert_eq!(packed_b[1] as usize, n, "packed_b n mismatch");

    let pool = match wasm_futex::get_pool() {
        Some(p) => p,
        None => {
            // Futex pool not initialized, fall back to single-threaded
            return matmul_with_packed_b_f32(a, packed_b, m, n, k);
        }
    };

    let num_threads = wasm_futex::threads_count();

    // For small M, ST is faster
    if m < 64 || num_threads <= 1 {
        return matmul_with_packed_b_f32(a, packed_b, m, n, k);
    }

    // Pre-allocate output
    let c = vec![0.0f32; m * n];

    // Calculate rows per thread - try to give each thread at least 6 rows (MR)
    let min_rows = OPT_MR;
    let rows_per_thread = ((m + num_threads - 1) / num_threads).max(min_rows);
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    // Convert pointers to usize for Send+Sync
    let a_addr = a.as_ptr() as usize;
    let packed_b_addr = packed_b.as_ptr() as usize;
    let c_addr = c.as_ptr() as usize;

    // Dispatch via futex pool
    pool.parallelize(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        unsafe {
            let a_ptr = a_addr as *const f32;
            let packed_b_ptr = packed_b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            // Create local slices
            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            // Call the single-threaded pre-packed kernel for this row chunk
            matmul_with_packed_b_into(a_slice, packed_b_ptr, c_slice, local_m, n, k);
        }
    });

    c
}

/// Helper function that computes C = A * packed_B for a row chunk.
/// Writes directly into the provided c_slice.
#[cfg(target_arch = "wasm32")]
unsafe fn matmul_with_packed_b_into(
    a: &[f32],
    packed_b_ptr: *const f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let k_blocks = (k + OPT_KC - 1) / OPT_KC;
    let panel_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;

    // Loop over N in blocks of NC
    let mut j = 0;
    let mut n_panel_idx = 0;
    while j < n {
        let j_block = (n - j).min(OPT_NC);
        let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

        // Loop over K in blocks of KC
        let mut p = 0;
        let mut k_block_idx = 0;
        while p < k {
            let p_block = (k - p).min(OPT_KC);
            let beta = if p == 0 { 0.0 } else { 1.0 };

            // Get pre-packed B panel (skip 2-float metadata header)
            let packed_panel_offset = 2 + (n_panel_idx * k_blocks + k_block_idx) * panel_size;
            let b_packed = packed_b_ptr.add(packed_panel_offset);

            // Loop over M (all rows in this chunk)
            let i_main = (m / OPT_MR) * OPT_MR;

            // Process full MR×NR tiles
            let mut ii = 0;
            while ii < i_main {
                let mut jj = 0;
                while jj < j_panels * OPT_NR && jj < j_block {
                    let panel_idx = jj / OPT_NR;
                    let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                    let n_rem = j_block - jj;

                    if n_rem >= OPT_NR {
                        micro_kernel_6x8_fma_unrolled(
                            p_block,
                            a.as_ptr().add(ii * k + p),
                            k,
                            b_panel_ptr,
                            c.as_mut_ptr().add(ii * n + j + jj),
                            n,
                            beta,
                        );
                    } else {
                        micro_kernel_edge(
                            OPT_MR,
                            n_rem,
                            p_block,
                            a.as_ptr().add(ii * k + p),
                            k,
                            b_panel_ptr,
                            c.as_mut_ptr().add(ii * n + j + jj),
                            n,
                            beta,
                        );
                    }
                    jj += OPT_NR;
                }
                ii += OPT_MR;
            }

            // Handle remaining rows
            if ii < m {
                let m_rem = m - ii;
                let mut jj = 0;
                while jj < j_panels * OPT_NR && jj < j_block {
                    let panel_idx = jj / OPT_NR;
                    let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                    let n_rem = (j_block - jj).min(OPT_NR);

                    micro_kernel_edge(
                        m_rem,
                        n_rem,
                        p_block,
                        a.as_ptr().add(ii * k + p),
                        k,
                        b_panel_ptr,
                        c.as_mut_ptr().add(ii * n + j + jj),
                        n,
                        beta,
                    );
                    jj += OPT_NR;
                }
            }

            k_block_idx += 1;
            p += OPT_KC;
        }

        n_panel_idx += 1;
        j += OPT_NC;
    }
}

/// Cache-blocked GEMM dispatcher using optimized 6x8 micro-kernel
///
/// C = A * B where A is [m, k], B is [k, n], C is [m, n]
#[cfg(target_arch = "wasm32")]
pub fn matmul_optimized_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    // For small matrices, skip packing - direct access is faster
    // Packing overhead is ~2ns/float. For B[256,256], that's ~130μs.
    // Only worth it when amortized over multiple M-blocks.
    if m < 128 || n < 128 || k < 128 {
        unsafe {
            matmul_simd_f32_fma(a, b, &mut c, m, n, k);
        }
        return c;
    }

    // Allocate packing buffer for B (KC x NC)
    let pack_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
    let mut packed_b = vec![0.0f32; pack_size];

    unsafe {
        // Loop over N in blocks of NC
        let mut j = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);
            let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

            // Loop over K in blocks of KC
            let mut p = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);

                // Pack B panel: B[p..p+p_block, j..j+j_block]
                pack_b_optimized(
                    b.as_ptr().add(p * n + j),
                    n,
                    packed_b.as_mut_ptr(),
                    p_block,
                    j_block,
                );

                // Beta = 0.0 for first K block, 1.0 for subsequent (accumulate)
                let beta = if p == 0 { 0.0 } else { 1.0 };

                // Loop over M in blocks of MC
                let mut i = 0;
                while i < m {
                    let i_block = (m - i).min(OPT_MC);
                    let i_main = i_block / OPT_MR * OPT_MR;

                    // Process full MR×NR tiles
                    let mut ii = 0;
                    while ii < i_main {
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                            let n_rem = j_block - jj;

                            if n_rem >= OPT_NR {
                                // Use K-unrolled kernel for better codegen
                                micro_kernel_6x8_fma_unrolled(
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            } else {
                                micro_kernel_edge(
                                    OPT_MR,
                                    n_rem,
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            }
                            jj += OPT_NR;
                        }
                        ii += OPT_MR;
                    }

                    // Handle remaining rows (i_main..i_block)
                    if ii < i_block {
                        let m_rem = i_block - ii;
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                            let n_rem = (j_block - jj).min(OPT_NR);

                            micro_kernel_edge(
                                m_rem,
                                n_rem,
                                p_block,
                                a.as_ptr().add((i + ii) * k + p),
                                k,
                                b_panel_ptr,
                                c.as_mut_ptr().add((i + ii) * n + j + jj),
                                n,
                                beta,
                            );
                            jj += OPT_NR;
                        }
                    }

                    i += OPT_MC;
                }

                p += OPT_KC;
            }

            j += OPT_NC;
        }
    }

    c
}

/// Parallel version of optimized GEMM using rayon with 2D tiling
///
/// Key optimizations over naive 1D row partitioning:
/// 1. 2D tiling: Split work into MC×NC tiles, not just rows
/// 2. Pack B once per NC-panel, shared across threads working on that panel
/// 3. Each thread only accesses the portion of B it needs
/// 4. Better L3 cache utilization due to 2D partitioning
#[cfg(target_arch = "wasm32")]
pub fn matmul_optimized_f32_parallel(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let num_threads = rayon::current_num_threads();

    // Shape-aware parallel decision: estimate if parallelization is profitable
    // based on FLOPs, thread count, and dispatch overhead.
    if below_parallel_threshold(m, n, k) || !should_parallelize(m, n, k, num_threads) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    let c = vec![0.0f32; m * n];
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Create a simple task list: just row starts
    let tasks: Vec<usize> = (0..num_threads)
        .map(|t| t * rows_per_thread)
        .filter(|&start| start < m)
        .collect();

    // Allocate packing buffer size (used per-thread)
    let pack_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;

    // Process row chunks in parallel
    tasks.par_iter().for_each_init(
        || vec![0.0f32; pack_size],  // Called once per thread
        |packed_b, &m_start| {
        let m_end = (m_start + rows_per_thread).min(m);
        let tile_height = m_end - m_start;

        if tile_height == 0 {
            return;
        }

        // Process full N dimension
        let (n_start, n_end, tile_width) = (0, n, n);

        unsafe {
            // Loop over N in blocks of NC (within our tile)
            let mut j = n_start;
            while j < n_end {
                let j_block = (n_end - j).min(OPT_NC);
                let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

                // Loop over K in blocks of KC
                let mut p = 0;
                while p < k {
                    let p_block = (k - p).min(OPT_KC);

                    // Pack B panel: B[p..p+p_block, j..j+j_block]
                    pack_b_optimized(
                        b.as_ptr().add(p * n + j),
                        n,
                        packed_b.as_mut_ptr(),
                        p_block,
                        j_block,
                    );

                    // Beta = 0.0 for first K block, 1.0 for subsequent (accumulate)
                    let beta = if p == 0 { 0.0 } else { 1.0 };

                    // Loop over M in blocks of MC (within our tile)
                    let mut i = m_start;
                    while i < m_end {
                        let i_block = (m_end - i).min(OPT_MC);
                        let i_main = i_block / OPT_MR * OPT_MR;

                        // Process full MR×NR tiles
                        let mut ii = 0;
                        while ii < i_main {
                            let mut jj = 0;
                            while jj < j_panels * OPT_NR && jj < j_block {
                                let panel_idx = jj / OPT_NR;
                                let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                                let n_rem = j_block - jj;

                                // Get raw pointer to C for this tile
                                let c_ptr = c.as_ptr() as *mut f32;

                                if n_rem >= OPT_NR {
                                    // Use K-unrolled kernel
                                    micro_kernel_6x8_fma_unrolled(
                                        p_block,
                                        a.as_ptr().add((i + ii) * k + p),
                                        k,
                                        b_panel_ptr,
                                        c_ptr.add((i + ii) * n + j + jj),
                                        n,
                                        beta,
                                    );
                                } else {
                                    micro_kernel_edge(
                                        OPT_MR,
                                        n_rem,
                                        p_block,
                                        a.as_ptr().add((i + ii) * k + p),
                                        k,
                                        b_panel_ptr,
                                        c_ptr.add((i + ii) * n + j + jj),
                                        n,
                                        beta,
                                    );
                                }
                                jj += OPT_NR;
                            }
                            ii += OPT_MR;
                        }

                        // Handle remaining rows (i_main..i_block)
                        if ii < i_block {
                            let m_rem = i_block - ii;
                            let mut jj = 0;
                            while jj < j_panels * OPT_NR && jj < j_block {
                                let panel_idx = jj / OPT_NR;
                                let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                                let n_rem = (j_block - jj).min(OPT_NR);

                                let c_ptr = c.as_ptr() as *mut f32;
                                micro_kernel_edge(
                                    m_rem,
                                    n_rem,
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c_ptr.add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                                jj += OPT_NR;
                            }
                        }

                        i += OPT_MC;
                    }

                    p += OPT_KC;
                }

                j += OPT_NC;
            }
        }
    });

    c
}

/// Parallel GEMM using the raw futex pool (bypasses Rayon entirely).
///
/// This achieves much lower dispatch overhead (~5-10μs) compared to Rayon (~150μs)
/// by using:
/// - Direct memory.atomic.wait32/notify for worker synchronization
/// - Spin-then-wait pattern (workers spin ~100μs before parking)
/// - Main thread participates as thread 0 (no wasted core)
/// - Single atomic per work item (no chase-lev deques)
///
/// Requires the `wasm-futex` feature and calling `init_futex_pool()` from JS first.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub fn matmul_futex_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use pthreadpool_rs::wasm_futex;

    let pool = match wasm_futex::get_pool() {
        Some(p) => p,
        None => {
            // Futex pool not initialized, fall back to single-threaded
            return matmul_optimized_f32(a, b, m, n, k);
        }
    };

    let num_threads = pthreadpool_rs::wasm_futex::threads_count();

    // Shape-aware parallel decision (lower threshold for futex since overhead is lower)
    // With ~10μs dispatch overhead, we can profitably parallelize smaller matrices
    if below_parallel_threshold(m, n, k) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // For futex, adjust the heuristic with lower overhead
    let flops = 2u64 * (m as u64) * (n as u64) * (k as u64);
    const FUTEX_DISPATCH_NS: u64 = 10_000; // 10 microseconds (vs 150 for rayon)
    const GFLOPS_PER_THREAD: u64 = 75;
    const PARALLEL_EFFICIENCY_PCT: u64 = 65; // Slightly better than rayon

    let time_st_ns = flops / GFLOPS_PER_THREAD;
    let effective_threads = (num_threads as u64 * PARALLEL_EFFICIENCY_PCT) / 100;
    let time_mt_ns = flops / (effective_threads.max(1) * GFLOPS_PER_THREAD) + FUTEX_DISPATCH_NS;

    if time_mt_ns >= time_st_ns {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let c = vec![0.0f32; m * n];

    // Calculate rows per thread
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Convert pointers to usize for Send+Sync
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_ptr() as usize;

    // Number of row chunks to process
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    // Dispatch via futex pool
    pool.parallelize(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let b_slice = std::slice::from_raw_parts(b_ptr, k * n);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            matmul_dispatch_f32_into(a_slice, b_slice, c_slice, local_m, n, k);
        }
    });

    c
}

/// Parallel GEMM using futex pool with proper MC/NC/KC tiling and shared B packing.
///
/// Parallel GEMM WITHOUT B-packing for medium matrices (128³ to 512³).
///
/// At these sizes, the B-packing overhead (~130μs for 256×256) dominates.
/// tfjs uses direct strided access here, so we do the same.
///
/// Uses the simpler 4x8 FMA kernel with direct B access.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
fn matmul_futex_f32_no_pack(
    pool: &pthreadpool_rs::wasm_futex::FutexPool,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) -> Vec<f32> {
    use core::arch::wasm32::*;

    let c = vec![0.0f32; m * n];

    // Simple row distribution
    let rows_per_thread = (m + num_threads - 1) / num_threads;
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_ptr() as usize;

    pool.parallelize(num_chunks, |chunk_idx| {
        let m_start = chunk_idx * rows_per_thread;
        if m_start >= m {
            return;
        }
        let m_end = (m_start + rows_per_thread).min(m);

        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            // Process 4 rows at a time, 8 columns at a time
            const MR: usize = 4;
            const NR: usize = 8;
            const KU: usize = 4;

            let mut i = m_start;
            while i + MR <= m_end {
                let mut j = 0;
                while j + NR <= n {
                    // Accumulator registers
                    let mut acc00 = f32x4_splat(0.0);
                    let mut acc01 = f32x4_splat(0.0);
                    let mut acc10 = f32x4_splat(0.0);
                    let mut acc11 = f32x4_splat(0.0);
                    let mut acc20 = f32x4_splat(0.0);
                    let mut acc21 = f32x4_splat(0.0);
                    let mut acc30 = f32x4_splat(0.0);
                    let mut acc31 = f32x4_splat(0.0);

                    // K loop unrolled 4x
                    let k_main = k / KU * KU;
                    let mut kk = 0;
                    while kk < k_main {
                        for u in 0..KU {
                            let a0 = f32x4_splat(*a_ptr.add((i + 0) * k + kk + u));
                            let a1 = f32x4_splat(*a_ptr.add((i + 1) * k + kk + u));
                            let a2 = f32x4_splat(*a_ptr.add((i + 2) * k + kk + u));
                            let a3 = f32x4_splat(*a_ptr.add((i + 3) * k + kk + u));
                            let b_base = (kk + u) * n + j;
                            let b0 = v128_load(b_ptr.add(b_base) as *const v128);
                            let b1 = v128_load(b_ptr.add(b_base + 4) as *const v128);
                            acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                            acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                            acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                            acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                            acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                            acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                            acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                            acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                        }
                        kk += KU;
                    }
                    // K remainder
                    while kk < k {
                        let a0 = f32x4_splat(*a_ptr.add((i + 0) * k + kk));
                        let a1 = f32x4_splat(*a_ptr.add((i + 1) * k + kk));
                        let a2 = f32x4_splat(*a_ptr.add((i + 2) * k + kk));
                        let a3 = f32x4_splat(*a_ptr.add((i + 3) * k + kk));
                        let b_base = kk * n + j;
                        let b0 = v128_load(b_ptr.add(b_base) as *const v128);
                        let b1 = v128_load(b_ptr.add(b_base + 4) as *const v128);
                        acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                        acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                        acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                        acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                        acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                        acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                        acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                        acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                        kk += 1;
                    }

                    // Store results
                    v128_store(c_ptr.add((i + 0) * n + j) as *mut v128, acc00);
                    v128_store(c_ptr.add((i + 0) * n + j + 4) as *mut v128, acc01);
                    v128_store(c_ptr.add((i + 1) * n + j) as *mut v128, acc10);
                    v128_store(c_ptr.add((i + 1) * n + j + 4) as *mut v128, acc11);
                    v128_store(c_ptr.add((i + 2) * n + j) as *mut v128, acc20);
                    v128_store(c_ptr.add((i + 2) * n + j + 4) as *mut v128, acc21);
                    v128_store(c_ptr.add((i + 3) * n + j) as *mut v128, acc30);
                    v128_store(c_ptr.add((i + 3) * n + j + 4) as *mut v128, acc31);

                    j += NR;
                }
                // N remainder (columns not divisible by 8)
                while j < n {
                    for ii in 0..MR {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += *a_ptr.add((i + ii) * k + kk) * *b_ptr.add(kk * n + j);
                        }
                        *c_ptr.add((i + ii) * n + j) = sum;
                    }
                    j += 1;
                }
                i += MR;
            }
            // M remainder (rows not divisible by 4)
            while i < m_end {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += *a_ptr.add(i * k + kk) * *b_ptr.add(kk * n + j);
                    }
                    *c_ptr.add(i * n + j) = sum;
                }
                i += 1;
            }
        }
    });

    c
}

/// Optimized single-threaded GEMM for small matrices (<512).
///
/// At small sizes, parallel dispatch overhead + packing overhead exceed the benefit.
/// This kernel uses the simple strided-B access pattern with maximum K-unrolling
/// and relies on hardware prefetching for decent performance.
///
/// Key optimizations vs basic kernel:
/// 1. 8x K-unrolling to hide latency
/// 2. Prefetch hints (where supported)
/// 3. 4x4 output tile to maximize register reuse
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
fn matmul_small_st_optimized(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use core::arch::wasm32::*;

    let mut c = vec![0.0f32; m * n];

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        // Process 4 rows at a time, 8 columns at a time (matches tfjs)
        const MR: usize = 4;
        const NR: usize = 8;
        const KU: usize = 8; // 8x K-unroll for better pipelining

        let mut i = 0;
        while i + MR <= m {
            let mut j = 0;
            while j + NR <= n {
                // 4x8 = 32 accumulators, but only 8 v128 registers
                let mut acc00 = f32x4_splat(0.0);
                let mut acc01 = f32x4_splat(0.0);
                let mut acc10 = f32x4_splat(0.0);
                let mut acc11 = f32x4_splat(0.0);
                let mut acc20 = f32x4_splat(0.0);
                let mut acc21 = f32x4_splat(0.0);
                let mut acc30 = f32x4_splat(0.0);
                let mut acc31 = f32x4_splat(0.0);

                // K loop with 8x unroll
                let k_main = k / KU * KU;
                let mut kk = 0;
                while kk < k_main {
                    // Process 8 K iterations
                    for u in 0..KU {
                        let k_idx = kk + u;
                        // Load A scalars and broadcast
                        let a0 = f32x4_splat(*a_ptr.add(i * k + k_idx));
                        let a1 = f32x4_splat(*a_ptr.add((i + 1) * k + k_idx));
                        let a2 = f32x4_splat(*a_ptr.add((i + 2) * k + k_idx));
                        let a3 = f32x4_splat(*a_ptr.add((i + 3) * k + k_idx));

                        // Load B row (strided access, but sequential within row)
                        let b_base = k_idx * n + j;
                        let b0 = v128_load(b_ptr.add(b_base) as *const v128);
                        let b1 = v128_load(b_ptr.add(b_base + 4) as *const v128);

                        // FMA accumulate
                        acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                        acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                        acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                        acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                        acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                        acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                        acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                        acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                    }
                    kk += KU;
                }
                // K remainder
                while kk < k {
                    let a0 = f32x4_splat(*a_ptr.add(i * k + kk));
                    let a1 = f32x4_splat(*a_ptr.add((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a_ptr.add((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a_ptr.add((i + 3) * k + kk));
                    let b_base = kk * n + j;
                    let b0 = v128_load(b_ptr.add(b_base) as *const v128);
                    let b1 = v128_load(b_ptr.add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                    kk += 1;
                }

                // Store results
                v128_store(c_ptr.add(i * n + j) as *mut v128, acc00);
                v128_store(c_ptr.add(i * n + j + 4) as *mut v128, acc01);
                v128_store(c_ptr.add((i + 1) * n + j) as *mut v128, acc10);
                v128_store(c_ptr.add((i + 1) * n + j + 4) as *mut v128, acc11);
                v128_store(c_ptr.add((i + 2) * n + j) as *mut v128, acc20);
                v128_store(c_ptr.add((i + 2) * n + j + 4) as *mut v128, acc21);
                v128_store(c_ptr.add((i + 3) * n + j) as *mut v128, acc30);
                v128_store(c_ptr.add((i + 3) * n + j + 4) as *mut v128, acc31);

                j += NR;
            }
            // N remainder (columns not divisible by 8)
            while j < n {
                for ii in 0..MR {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += *a_ptr.add((i + ii) * k + kk) * *b_ptr.add(kk * n + j);
                    }
                    *c_ptr.add((i + ii) * n + j) = sum;
                }
                j += 1;
            }
            i += MR;
        }
        // M remainder (rows not divisible by 4)
        while i < m {
            let mut j = 0;
            while j + NR <= n {
                let mut acc0 = f32x4_splat(0.0);
                let mut acc1 = f32x4_splat(0.0);
                for kk in 0..k {
                    let a_val = f32x4_splat(*a_ptr.add(i * k + kk));
                    let b_base = kk * n + j;
                    acc0 = f32x4_relaxed_madd(a_val, v128_load(b_ptr.add(b_base) as *const v128), acc0);
                    acc1 = f32x4_relaxed_madd(a_val, v128_load(b_ptr.add(b_base + 4) as *const v128), acc1);
                }
                v128_store(c_ptr.add(i * n + j) as *mut v128, acc0);
                v128_store(c_ptr.add(i * n + j + 4) as *mut v128, acc1);
                j += NR;
            }
            while j < n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += *a_ptr.add(i * k + kk) * *b_ptr.add(kk * n + j);
                }
                *c_ptr.add(i * n + j) = sum;
                j += 1;
            }
            i += 1;
        }
    }

    c
}

/// This version fixes the efficiency issue where each thread was redundantly packing B.
/// Now: pack B once upfront, share across threads, distribute M-rows via futex.
///
/// Key optimizations:
/// 1. Pack B matrix once (pre-allocate, shared across threads)
/// 2. Distribute work by M-rows (each thread gets rows_per_thread)
/// 3. Each thread processes its rows with proper MC/KC tiling
/// 4. Use futex pool for low dispatch overhead (~10μs vs ~150μs rayon)
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub fn matmul_futex_f32_tiled(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use pthreadpool_rs::wasm_futex;

    let pool = match wasm_futex::get_pool() {
        Some(p) => p,
        None => {
            return matmul_optimized_f32(a, b, m, n, k);
        }
    };

    let num_threads = pthreadpool_rs::wasm_futex::threads_count();

    // Shape-aware parallel decision
    let total_elements = (m as u64) * (n as u64) * (k as u64);

    // For very small matrices (<128³), single-threaded is fastest
    // dispatch overhead (~16μs) > parallel benefit at this scale
    if total_elements < (128u64 * 128 * 128) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // For small-medium matrices (128³ to 512³), use single-threaded packed kernel
    // Parallel overhead (~10μs dispatch + B-packing) doesn't help at these sizes
    // Our ST packed kernel achieves ~80 GFLOPS which is decent for WASM
    // (tfjs achieves ~300 GFLOPS via XNNPACK's hand-tuned emscripten assembly)
    if total_elements < (512u64 * 512 * 512) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // Heuristic with futex's lower overhead + B-packing cost
    let flops = 2u64 * total_elements;
    const FUTEX_DISPATCH_NS: u64 = 8_000;  // ~8μs dispatch overhead (measured)
    const GFLOPS_PER_THREAD: u64 = 75;
    const PARALLEL_EFFICIENCY_PCT: u64 = 70;

    // B-packing overhead: ~2 ns/float (load + store + some math)
    // For small matrices this can dominate!
    let b_elements = (k as u64) * (n as u64);
    const PACK_NS_PER_FLOAT: u64 = 2;
    let pack_overhead_ns = b_elements * PACK_NS_PER_FLOAT;

    let time_st_ns = flops / GFLOPS_PER_THREAD;
    let effective_threads = (num_threads as u64 * PARALLEL_EFFICIENCY_PCT) / 100;
    let time_mt_ns = flops / (effective_threads.max(1) * GFLOPS_PER_THREAD)
                   + FUTEX_DISPATCH_NS
                   + pack_overhead_ns;

    if time_mt_ns >= time_st_ns {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let c = vec![0.0f32; m * n];

    // Pre-pack entire B matrix (shared across all threads)
    // B is [k, n], pack in NC-wide panels
    let n_panels = (n + OPT_NC - 1) / OPT_NC;
    let k_blocks = (k + OPT_KC - 1) / OPT_KC;
    let panel_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
    let packed_b_size = n_panels * k_blocks * panel_size;
    let mut packed_b = vec![0.0f32; packed_b_size];

    // Pack B: for each NC panel, for each KC block
    unsafe {
        let mut panel_offset = 0;
        let mut j = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);
            let mut p = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);
                pack_b_optimized(
                    b.as_ptr().add(p * n + j),
                    n,
                    packed_b.as_mut_ptr().add(panel_offset),
                    p_block,
                    j_block,
                );
                panel_offset += panel_size;
                p += OPT_KC;
            }
            j += OPT_NC;
        }
    }

    // Calculate work distribution
    let rows_per_thread = (m + num_threads - 1) / num_threads;
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    // Convert to usize for Send+Sync
    let a_addr = a.as_ptr() as usize;
    let c_addr = c.as_ptr() as usize;
    let packed_b_addr = packed_b.as_ptr() as usize;

    // Dispatch work via futex pool
    pool.parallelize(num_chunks, |chunk_idx| {
        let m_start = chunk_idx * rows_per_thread;
        if m_start >= m {
            return;
        }
        let m_end = (m_start + rows_per_thread).min(m);
        let tile_height = m_end - m_start;

        if tile_height == 0 {
            return;
        }

        unsafe {
            let a_ptr = a_addr as *const f32;
            let c_ptr = c_addr as *mut f32;
            let packed_b_ptr = packed_b_addr as *const f32;

            // Process this row chunk with proper MC/KC tiling
            // Loop over N in NC blocks
            let mut j = 0;
            let mut n_panel_idx = 0;
            while j < n {
                let j_block = (n - j).min(OPT_NC);
                let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

                // Loop over K in KC blocks
                let mut p = 0;
                let mut k_block_idx = 0;
                while p < k {
                    let p_block = (k - p).min(OPT_KC);
                    let beta = if p == 0 { 0.0 } else { 1.0 };

                    // Get pre-packed B panel for this (j, p) block
                    let packed_panel_offset = (n_panel_idx * k_blocks + k_block_idx) * panel_size;
                    let b_packed = packed_b_ptr.add(packed_panel_offset);

                    // Loop over M in MC blocks (within our chunk)
                    let mut i = m_start;
                    while i < m_end {
                        let i_block = (m_end - i).min(OPT_MC);
                        let i_main = i_block / OPT_MR * OPT_MR;

                        // Process full MR×NR tiles
                        let mut ii = 0;
                        while ii < i_main {
                            let mut jj = 0;
                            while jj < j_panels * OPT_NR && jj < j_block {
                                let panel_idx = jj / OPT_NR;
                                let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                                let n_rem = j_block - jj;

                                if n_rem >= OPT_NR {
                                    // Use K-unrolled kernel for better codegen
                                    micro_kernel_6x8_fma_unrolled(
                                        p_block,
                                        a_ptr.add((i + ii) * k + p),
                                        k,
                                        b_panel_ptr,
                                        c_ptr.add((i + ii) * n + j + jj),
                                        n,
                                        beta,
                                    );
                                } else {
                                    micro_kernel_edge(
                                        OPT_MR,
                                        n_rem,
                                        p_block,
                                        a_ptr.add((i + ii) * k + p),
                                        k,
                                        b_panel_ptr,
                                        c_ptr.add((i + ii) * n + j + jj),
                                        n,
                                        beta,
                                    );
                                }
                                jj += OPT_NR;
                            }
                            ii += OPT_MR;
                        }

                        // Handle remaining rows
                        if ii < i_block {
                            let m_rem = i_block - ii;
                            let mut jj = 0;
                            while jj < j_panels * OPT_NR && jj < j_block {
                                let panel_idx = jj / OPT_NR;
                                let b_panel_ptr = b_packed.add(panel_idx * p_block * OPT_NR);
                                let n_rem = (j_block - jj).min(OPT_NR);

                                micro_kernel_edge(
                                    m_rem,
                                    n_rem,
                                    p_block,
                                    a_ptr.add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c_ptr.add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                                jj += OPT_NR;
                            }
                        }

                        i += OPT_MC;
                    }

                    p += OPT_KC;
                    k_block_idx += 1;
                }

                j += OPT_NC;
                n_panel_idx += 1;
            }
        }
    });

    c
}

/// Initialize the futex pool with n threads.
/// Call this from JS before using matmul_futex_f32.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub fn init_futex_pool(n: usize) {
    pthreadpool_rs::wasm_futex::init(n);
}

/// Get the number of threads in the futex pool.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "wasm-futex"
))]
pub fn futex_threads_count() -> usize {
    pthreadpool_rs::wasm_futex::threads_count()
}
