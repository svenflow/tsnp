//! Math function tests - NumPy compatible

#[cfg(test)]
mod tests {
    use crate::utils::*;
    use rumpy_core::{ops::MathOps, Array};
    use rumpy_cpu::{CpuArray, CpuBackend};
    use std::f64::consts::PI;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    // ============ Trigonometric ============

    #[test]
    fn test_sin() {
        let a = arr(vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI]);
        let result = CpuBackend::sin(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 0.5, DEFAULT_TOL));
        assert!(approx_eq(data[2], 2.0_f64.sqrt() / 2.0, DEFAULT_TOL));
        assert!(approx_eq(data[3], 3.0_f64.sqrt() / 2.0, DEFAULT_TOL));
        assert!(approx_eq(data[4], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[5], 0.0, DEFAULT_TOL));
    }

    #[test]
    fn test_cos() {
        let a = arr(vec![0.0, PI / 3.0, PI / 2.0, PI]);
        let result = CpuBackend::cos(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 0.5, DEFAULT_TOL));
        assert!(approx_eq(data[2], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[3], -1.0, DEFAULT_TOL));
    }

    #[test]
    fn test_tan() {
        let a = arr(vec![0.0, PI / 4.0]);
        let result = CpuBackend::tan(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 1.0, DEFAULT_TOL));
    }

    #[test]
    fn test_arcsin() {
        let a = arr(vec![0.0, 0.5, 1.0]);
        let result = CpuBackend::arcsin(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], PI / 6.0, DEFAULT_TOL));
        assert!(approx_eq(data[2], PI / 2.0, DEFAULT_TOL));
    }

    // ============ Hyperbolic ============

    #[test]
    fn test_sinh_cosh() {
        let a = arr(vec![0.0, 1.0, 2.0]);
        let sinh = CpuBackend::sinh(&a);
        let cosh = CpuBackend::cosh(&a);

        // sinh(0) = 0, cosh(0) = 1
        assert!(approx_eq(sinh.as_f64_slice()[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(cosh.as_f64_slice()[0], 1.0, DEFAULT_TOL));

        // Identity: cosh^2 - sinh^2 = 1
        for i in 0..3 {
            let s = sinh.as_f64_slice()[i];
            let c = cosh.as_f64_slice()[i];
            assert!(approx_eq(c * c - s * s, 1.0, DEFAULT_TOL));
        }
    }

    #[test]
    fn test_tanh() {
        let a = arr(vec![0.0, 1.0, -1.0, 10.0, -10.0]);
        let result = CpuBackend::tanh(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(data[1] > 0.0 && data[1] < 1.0);
        assert!(data[2] < 0.0 && data[2] > -1.0);
        assert!(approx_eq(data[3], 1.0, RELAXED_TOL)); // tanh saturates
        assert!(approx_eq(data[4], -1.0, RELAXED_TOL));
    }

    // ============ Exponential and Logarithmic ============

    #[test]
    fn test_exp() {
        let a = arr(vec![0.0, 1.0, 2.0, -1.0]);
        let result = CpuBackend::exp(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], std::f64::consts::E, DEFAULT_TOL));
        assert!(approx_eq(
            data[2],
            std::f64::consts::E * std::f64::consts::E,
            DEFAULT_TOL
        ));
        assert!(approx_eq(data[3], 1.0 / std::f64::consts::E, DEFAULT_TOL));
    }

    #[test]
    fn test_log() {
        let a = arr(vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ]);
        let result = CpuBackend::log(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[2], 2.0, DEFAULT_TOL));
    }

    #[test]
    fn test_log_negative() {
        let a = arr(vec![-1.0]);
        let result = CpuBackend::log(&a);
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn test_exp_log_inverse() {
        let a = arr(vec![0.5, 1.0, 2.0, 5.0, 10.0]);
        let exp_a = CpuBackend::exp(&a);
        let log_exp_a = CpuBackend::log(&exp_a);

        for (x, y) in a.as_f64_slice().iter().zip(log_exp_a.as_f64_slice().iter()) {
            assert!(approx_eq(*x, *y, DEFAULT_TOL));
        }
    }

    #[test]
    fn test_log2() {
        let a = arr(vec![1.0, 2.0, 4.0, 8.0]);
        let result = CpuBackend::log2(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[2], 2.0, DEFAULT_TOL));
        assert!(approx_eq(data[3], 3.0, DEFAULT_TOL));
    }

    #[test]
    fn test_log10() {
        let a = arr(vec![1.0, 10.0, 100.0, 1000.0]);
        let result = CpuBackend::log10(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[2], 2.0, DEFAULT_TOL));
        assert!(approx_eq(data[3], 3.0, DEFAULT_TOL));
    }

    // ============ Power and Roots ============

    #[test]
    fn test_sqrt() {
        let a = arr(vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        let result = CpuBackend::sqrt(&a);

        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = arr(vec![-1.0]);
        let result = CpuBackend::sqrt(&a);
        assert!(result.as_f64_slice()[0].is_nan());
    }

    #[test]
    fn test_cbrt() {
        let a = arr(vec![0.0, 1.0, 8.0, 27.0, -8.0]);
        let result = CpuBackend::cbrt(&a);
        let data = result.as_f64_slice();

        assert!(approx_eq(data[0], 0.0, DEFAULT_TOL));
        assert!(approx_eq(data[1], 1.0, DEFAULT_TOL));
        assert!(approx_eq(data[2], 2.0, DEFAULT_TOL));
        assert!(approx_eq(data[3], 3.0, DEFAULT_TOL));
        assert!(approx_eq(data[4], -2.0, DEFAULT_TOL));
    }

    #[test]
    fn test_square() {
        let a = arr(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let result = CpuBackend::square(&a);

        assert_eq!(
            result.as_f64_slice(),
            vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]
        );
    }

    // ============ Rounding ============

    #[test]
    fn test_floor() {
        let a = arr(vec![-2.7, -0.5, 0.0, 0.5, 2.7]);
        let result = CpuBackend::floor(&a);

        assert_eq!(result.as_f64_slice(), vec![-3.0, -1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_ceil() {
        let a = arr(vec![-2.7, -0.5, 0.0, 0.5, 2.7]);
        let result = CpuBackend::ceil(&a);

        assert_eq!(result.as_f64_slice(), vec![-2.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_round() {
        let a = arr(vec![-2.7, -0.5, 0.0, 0.5, 2.7]);
        let result = CpuBackend::round(&a);
        let data = result.as_f64_slice();

        assert_eq!(data[0], -3.0);
        // Note: Rust's round uses "round half away from zero"
        assert_eq!(data[2], 0.0);
        assert_eq!(data[4], 3.0);
    }

    // ============ Other Unary ============

    #[test]
    fn test_abs() {
        let a = arr(vec![-5.0, -2.5, 0.0, 2.5, 5.0]);
        let result = CpuBackend::abs(&a);

        assert_eq!(result.as_f64_slice(), vec![5.0, 2.5, 0.0, 2.5, 5.0]);
    }

    #[test]
    fn test_sign() {
        let a = arr(vec![-5.0, -0.5, 0.0, 0.5, 5.0]);
        let result = CpuBackend::sign(&a);

        assert_eq!(result.as_f64_slice(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_neg() {
        let a = arr(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = CpuBackend::neg(&a);

        assert_eq!(result.as_f64_slice(), vec![2.0, 1.0, 0.0, -1.0, -2.0]);
    }

    #[test]
    fn test_reciprocal() {
        let a = arr(vec![1.0, 2.0, 4.0, 0.5]);
        let result = CpuBackend::reciprocal(&a);

        assert_eq!(result.as_f64_slice(), vec![1.0, 0.5, 0.25, 2.0]);
    }

    // ============ Binary Operations ============

    #[test]
    fn test_add() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![4.0, 5.0, 6.0]);
        let result = CpuBackend::add(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = arr(vec![5.0, 7.0, 9.0]);
        let b = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::sub(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::mul(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_div() {
        let a = arr(vec![4.0, 9.0, 16.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::div(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pow() {
        let a = arr(vec![2.0, 3.0, 4.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::pow(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_maximum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::maximum(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![2.0, 5.0, 4.0]);
    }

    #[test]
    fn test_minimum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::minimum(&a, &b).unwrap();

        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 3.0]);
    }

    // ============ Scalar Operations ============

    #[test]
    fn test_add_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::add_scalar(&a, 10.0);

        assert_eq!(result.as_f64_slice(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::mul_scalar(&a, 2.0);

        assert_eq!(result.as_f64_slice(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_pow_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::pow_scalar(&a, 2.0);

        assert_eq!(result.as_f64_slice(), vec![1.0, 4.0, 9.0, 16.0]);
    }

    // ============ Clip ============

    #[test]
    fn test_clip() {
        let a = arr(vec![-5.0, 0.0, 5.0, 10.0, 15.0]);
        let result = CpuBackend::clip(&a, 0.0, 10.0);

        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    // ============ Shape Mismatch ============

    #[test]
    fn test_binary_shape_mismatch() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 2.0]);

        assert!(CpuBackend::add(&a, &b).is_err());
        assert!(CpuBackend::sub(&a, &b).is_err());
        assert!(CpuBackend::mul(&a, &b).is_err());
        assert!(CpuBackend::div(&a, &b).is_err());
    }
}
