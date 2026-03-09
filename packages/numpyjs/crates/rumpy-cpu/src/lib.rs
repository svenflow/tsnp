//! CPU Backend for RumPy
//!
//! Uses ndarray for array operations and faer for linear algebra.
//! On WASM targets with simd128, uses hand-optimized SIMD GEMM kernels.

// memory_atomic_wait32/notify are still unstable as of 2025-12 nightly.
// We use them in the v4 parallel GEMM kernel for spin-then-park workers
// (the pthreadpool dispatch model, hosted inside Rayon's Web Workers).
#![cfg_attr(
    all(target_arch = "wasm32", target_feature = "atomics"),
    feature(stdarch_wasm_atomic_wait)
)]

mod array;
mod broadcast;
mod compare;
mod creation;
mod linalg;
mod manipulation;
mod math;
mod random;
pub mod simd_gemm;
mod sort;
mod stats;

pub use array::CpuArray;

use rumpy_core::Backend;

/// CPU backend using ndarray + faer
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name() -> &'static str {
        "cpu"
    }

    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[cfg(target_feature = "avx2")]
    fn has_simd() -> bool {
        true
    }

    #[cfg(not(target_feature = "avx2"))]
    fn has_simd() -> bool {
        false
    }
}

// Re-export the array type
pub type Array = CpuArray;
