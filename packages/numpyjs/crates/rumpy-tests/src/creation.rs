//! Creation function tests - NumPy compatible
//!
//! These tests verify that array creation functions match NumPy behavior.

#[cfg(test)]
mod tests {
    use crate::utils::*;
    use rumpy_core::{ops::CreationOps, Array};
    use rumpy_cpu::CpuBackend;

    // ============ zeros ============

    #[test]
    fn test_zeros_1d() {
        let arr = CpuBackend::zeros(vec![5]);
        assert_eq!(arr.shape(), &[5]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zeros_2d() {
        let arr = CpuBackend::zeros(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zeros_3d() {
        let arr = CpuBackend::zeros(vec![2, 3, 4]);
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.size(), 24);
    }

    #[test]
    fn test_zeros_empty() {
        let arr = CpuBackend::zeros(vec![0]);
        assert_eq!(arr.size(), 0);
    }

    // ============ ones ============

    #[test]
    fn test_ones_1d() {
        let arr = CpuBackend::ones(vec![5]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_ones_2d() {
        let arr = CpuBackend::ones(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 1.0));
    }

    // ============ full ============

    #[test]
    fn test_full() {
        let arr = CpuBackend::full(vec![3, 3], 7.5);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 7.5));
    }

    #[test]
    fn test_full_negative() {
        let arr = CpuBackend::full(vec![2, 2], -3.15);
        assert!(arr
            .as_f64_slice()
            .iter()
            .all(|&x| approx_eq(x, -3.15, DEFAULT_TOL)));
    }

    // ============ arange ============

    #[test]
    fn test_arange_basic() {
        let arr = CpuBackend::arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let arr = CpuBackend::arange(0.0, 10.0, 2.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_arange_float_step() {
        let arr = CpuBackend::arange(0.0, 1.0, 0.25).unwrap();
        let expected = [0.0, 0.25, 0.5, 0.75];
        let data = arr.as_f64_slice();
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, DEFAULT_TOL));
        }
    }

    #[test]
    fn test_arange_negative_step() {
        let arr = CpuBackend::arange(5.0, 0.0, -1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_arange_zero_step_error() {
        let result = CpuBackend::arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_arange_empty_range() {
        let arr = CpuBackend::arange(5.0, 0.0, 1.0).unwrap();
        assert_eq!(arr.size(), 0);
    }

    // ============ linspace ============

    #[test]
    fn test_linspace_basic() {
        let arr = CpuBackend::linspace(0.0, 1.0, 5);
        let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
        let data = arr.as_f64_slice();
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b, DEFAULT_TOL));
        }
    }

    #[test]
    fn test_linspace_single() {
        let arr = CpuBackend::linspace(5.0, 5.0, 1);
        assert_eq!(arr.as_f64_slice(), vec![5.0]);
    }

    #[test]
    fn test_linspace_two() {
        let arr = CpuBackend::linspace(0.0, 10.0, 2);
        assert_eq!(arr.as_f64_slice(), vec![0.0, 10.0]);
    }

    #[test]
    fn test_linspace_empty() {
        let arr = CpuBackend::linspace(0.0, 1.0, 0);
        assert_eq!(arr.size(), 0);
    }

    #[test]
    fn test_linspace_negative() {
        let arr = CpuBackend::linspace(-5.0, 5.0, 11);
        let data = arr.as_f64_slice();
        assert!(approx_eq(data[0], -5.0, DEFAULT_TOL));
        assert!(approx_eq(data[5], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[10], 5.0, DEFAULT_TOL));
    }

    // ============ eye ============

    #[test]
    fn test_eye_3x3() {
        let arr = CpuBackend::eye(3);
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        // Check diagonal is 1, off-diagonal is 0
        assert_eq!(data[0], 1.0); // [0,0]
        assert_eq!(data[1], 0.0); // [0,1]
        assert_eq!(data[2], 0.0); // [0,2]
        assert_eq!(data[3], 0.0); // [1,0]
        assert_eq!(data[4], 1.0); // [1,1]
        assert_eq!(data[5], 0.0); // [1,2]
        assert_eq!(data[6], 0.0); // [2,0]
        assert_eq!(data[7], 0.0); // [2,1]
        assert_eq!(data[8], 1.0); // [2,2]
    }

    #[test]
    fn test_eye_1x1() {
        let arr = CpuBackend::eye(1);
        assert_eq!(arr.as_f64_slice(), vec![1.0]);
    }

    // ============ diag ============

    #[test]
    fn test_diag_create_from_vector() {
        let vec = rumpy_cpu::CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let arr = CpuBackend::diag(&vec, 0).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 2.0);
        assert_eq!(data[8], 3.0);
    }

    #[test]
    fn test_diag_extract_from_matrix() {
        let mat = rumpy_cpu::CpuArray::from_f64_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let diag = CpuBackend::diag(&mat, 0).unwrap();
        assert_eq!(diag.as_f64_slice(), vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diag_upper_diagonal() {
        let mat = rumpy_cpu::CpuArray::from_f64_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let diag = CpuBackend::diag(&mat, 1).unwrap();
        assert_eq!(diag.as_f64_slice(), vec![2.0, 6.0]);
    }

    #[test]
    fn test_diag_lower_diagonal() {
        let mat = rumpy_cpu::CpuArray::from_f64_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let diag = CpuBackend::diag(&mat, -1).unwrap();
        assert_eq!(diag.as_f64_slice(), vec![4.0, 8.0]);
    }
}
