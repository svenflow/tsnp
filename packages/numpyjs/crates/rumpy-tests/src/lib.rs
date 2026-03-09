//! Shared test suite for RumPy backends
//!
//! This crate contains parameterized tests that run against any backend.
//! Tests are designed to match NumPy behavior exactly.

pub mod creation;
pub mod linalg;
pub mod math;
pub mod stats;

/// Test utilities
pub mod utils {
    use rumpy_core::Array;

    /// Check if two f64 values are approximately equal
    pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() {
            return a.signum() == b.signum();
        }
        (a - b).abs() < tol
    }

    /// Check if two arrays are approximately equal
    pub fn arrays_approx_eq<A: Array>(a: &A, b: &A, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        let a_data = a.as_f64_slice();
        let b_data = b.as_f64_slice();
        a_data
            .iter()
            .zip(b_data.iter())
            .all(|(&x, &y)| approx_eq(x, y, tol))
    }

    /// Default tolerance for floating point comparisons
    pub const DEFAULT_TOL: f64 = 1e-10;

    /// Relaxed tolerance for operations with accumulated error
    pub const RELAXED_TOL: f64 = 1e-6;
}

/// Macro to generate parameterized tests for different backends
#[macro_export]
macro_rules! backend_tests {
    ($backend:ty, $test_name:ident, $body:block) => {
        #[test]
        fn $test_name() {
            type B = $backend;
            $body
        }
    };
}
