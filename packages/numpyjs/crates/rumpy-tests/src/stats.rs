//! Statistics tests - NumPy compatible

#[cfg(test)]
mod tests {
    use crate::utils::*;
    use rumpy_core::{ops::StatsOps, Array};
    use rumpy_cpu::{CpuArray, CpuBackend};

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    // ============ sum ============

    #[test]
    fn test_sum_1d() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::sum(&a), 15.0);
    }

    #[test]
    fn test_sum_2d() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(CpuBackend::sum(&a), 21.0);
    }

    #[test]
    fn test_sum_empty() {
        let a = arr(vec![]);
        assert_eq!(CpuBackend::sum(&a), 0.0);
    }

    // ============ prod ============

    #[test]
    fn test_prod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(CpuBackend::prod(&a), 24.0);
    }

    #[test]
    fn test_prod_with_zero() {
        let a = arr(vec![1.0, 2.0, 0.0, 4.0]);
        assert_eq!(CpuBackend::prod(&a), 0.0);
    }

    // ============ mean ============

    #[test]
    fn test_mean() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::mean(&a), 3.0);
    }

    #[test]
    fn test_mean_single() {
        let a = arr(vec![42.0]);
        assert_eq!(CpuBackend::mean(&a), 42.0);
    }

    #[test]
    fn test_mean_empty() {
        let a = arr(vec![]);
        assert!(CpuBackend::mean(&a).is_nan());
    }

    // ============ var ============

    #[test]
    fn test_var() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // Variance = E[X^2] - E[X]^2 = (1+4+9+16+25)/5 - 9 = 11 - 9 = 2
        assert!(approx_eq(CpuBackend::var(&a), 2.0, DEFAULT_TOL));
    }

    #[test]
    fn test_var_constant() {
        let a = arr(vec![5.0, 5.0, 5.0, 5.0]);
        assert!(approx_eq(CpuBackend::var(&a), 0.0, DEFAULT_TOL));
    }

    // ============ std ============

    #[test]
    fn test_std() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(approx_eq(CpuBackend::std(&a), 2.0_f64.sqrt(), DEFAULT_TOL));
    }

    // ============ min/max ============

    #[test]
    fn test_min() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::min(&a).unwrap(), 1.0);
    }

    #[test]
    fn test_max() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::max(&a).unwrap(), 9.0);
    }

    #[test]
    fn test_min_negative() {
        let a = arr(vec![-5.0, -2.0, -10.0, -1.0]);
        assert_eq!(CpuBackend::min(&a).unwrap(), -10.0);
    }

    #[test]
    fn test_min_max_empty() {
        // NumPy raises ValueError for empty arrays
        let a = arr(vec![]);
        assert!(CpuBackend::min(&a).is_err());
        assert!(CpuBackend::max(&a).is_err());
    }

    // ============ argmin/argmax ============

    #[test]
    fn test_argmin() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        assert_eq!(CpuBackend::argmin(&a).unwrap(), 1);
    }

    #[test]
    fn test_argmax() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::argmax(&a).unwrap(), 5);
    }

    #[test]
    fn test_argmin_argmax_empty() {
        // NumPy raises ValueError for empty arrays
        let a = arr(vec![]);
        assert!(CpuBackend::argmin(&a).is_err());
        assert!(CpuBackend::argmax(&a).is_err());
    }

    // ============ sum_axis ============

    #[test]
    fn test_sum_axis_0() {
        // [[1, 2, 3], [4, 5, 6]]
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::sum_axis(&a, 0).unwrap();
        assert_eq!(result.as_f64_slice(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis_1() {
        // [[1, 2, 3], [4, 5, 6]]
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::sum_axis(&a, 1).unwrap();
        assert_eq!(result.as_f64_slice(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_axis_invalid() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(CpuBackend::sum_axis(&a, 2).is_err());
    }

    // ============ mean_axis ============

    #[test]
    fn test_mean_axis_0() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::mean_axis(&a, 0).unwrap();
        assert_eq!(result.as_f64_slice(), vec![2.5, 3.5, 4.5]);
    }

    // ============ cumsum ============

    #[test]
    fn test_cumsum() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumsum(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_single() {
        let a = arr(vec![5.0]);
        let result = CpuBackend::cumsum(&a);
        assert_eq!(result.as_f64_slice(), vec![5.0]);
    }

    // ============ cumprod ============

    #[test]
    fn test_cumprod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumprod(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 2.0, 6.0, 24.0]);
    }

    // ============ all/any ============

    #[test]
    fn test_all_true() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        assert!(CpuBackend::all(&a));
    }

    #[test]
    fn test_all_false() {
        let a = arr(vec![1.0, 0.0, 3.0]);
        assert!(!CpuBackend::all(&a));
    }

    #[test]
    fn test_any_true() {
        let a = arr(vec![0.0, 0.0, 1.0]);
        assert!(CpuBackend::any(&a));
    }

    #[test]
    fn test_any_false() {
        let a = arr(vec![0.0, 0.0, 0.0]);
        assert!(!CpuBackend::any(&a));
    }

    #[test]
    fn test_all_empty() {
        let a = arr(vec![]);
        // NumPy: np.all([]) = True
        assert!(CpuBackend::all(&a));
    }

    #[test]
    fn test_any_empty() {
        let a = arr(vec![]);
        // NumPy: np.any([]) = False
        assert!(!CpuBackend::any(&a));
    }
}
