//! Element-wise math operations for CPU backend

use crate::broadcast::broadcast_binary_op;
use crate::{CpuArray, CpuBackend};
use rumpy_core::{ops::MathOps, Result};

macro_rules! impl_unary_op {
    ($name:ident, $op:expr) => {
        fn $name(arr: &CpuArray) -> CpuArray {
            CpuArray::from_ndarray(arr.as_ndarray().mapv($op))
        }
    };
}

macro_rules! impl_binary_op_broadcast {
    ($name:ident, $op:tt) => {
        fn $name(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
            let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| x $op y)?;
            Ok(CpuArray::from_ndarray(result))
        }
    };
}

macro_rules! impl_scalar_op {
    ($name:ident, $op:tt) => {
        fn $name(arr: &CpuArray, scalar: f64) -> CpuArray {
            CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x $op scalar))
        }
    };
}

impl MathOps for CpuBackend {
    type Array = CpuArray;

    // Trigonometric
    impl_unary_op!(sin, |x: f64| x.sin());
    impl_unary_op!(cos, |x: f64| x.cos());
    impl_unary_op!(tan, |x: f64| x.tan());
    impl_unary_op!(arcsin, |x: f64| x.asin());
    impl_unary_op!(arccos, |x: f64| x.acos());
    impl_unary_op!(arctan, |x: f64| x.atan());

    // Hyperbolic
    impl_unary_op!(sinh, |x: f64| x.sinh());
    impl_unary_op!(cosh, |x: f64| x.cosh());
    impl_unary_op!(tanh, |x: f64| x.tanh());

    // Exponential and logarithmic
    impl_unary_op!(exp, |x: f64| x.exp());
    impl_unary_op!(exp2, |x: f64| x.exp2());
    impl_unary_op!(expm1, |x: f64| x.exp_m1());  // exp(x) - 1, numerically stable
    impl_unary_op!(log, |x: f64| x.ln());
    impl_unary_op!(log2, |x: f64| x.log2());
    impl_unary_op!(log10, |x: f64| x.log10());
    impl_unary_op!(log1p, |x: f64| x.ln_1p());  // log(1 + x), numerically stable

    // Power and roots
    impl_unary_op!(sqrt, |x: f64| x.sqrt());
    impl_unary_op!(cbrt, |x: f64| x.cbrt());
    impl_unary_op!(square, |x: f64| x * x);

    // Rounding
    impl_unary_op!(floor, |x: f64| x.floor());
    impl_unary_op!(ceil, |x: f64| x.ceil());
    // NumPy uses round-half-to-even (banker's rounding) by default for .round()
    // but with 0 decimals np.round uses round-half-away-from-zero for consistency
    // with Python's round(). We match NumPy behavior here.
    fn round(arr: &CpuArray) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            // Round half away from zero (NumPy behavior)
            if x.is_nan() {
                f64::NAN
            } else {
                let frac = x.fract().abs();
                if (frac - 0.5).abs() < 1e-15 {
                    // Exactly 0.5: round away from zero
                    if x > 0.0 { x.ceil() } else { x.floor() }
                } else {
                    x.round()
                }
            }
        }))
    }

    fn rint(arr: &CpuArray) -> CpuArray {
        // Round to nearest even (banker's rounding) - NumPy rint behavior
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            if x.is_nan() {
                f64::NAN
            } else {
                let frac = x.fract().abs();
                if (frac - 0.5).abs() < 1e-15 {
                    // Exactly 0.5: round to nearest even
                    let floor_val = x.floor();
                    let ceil_val = x.ceil();
                    if (floor_val as i64) % 2 == 0 {
                        floor_val
                    } else {
                        ceil_val
                    }
                } else {
                    x.round()
                }
            }
        }))
    }

    // Other unary
    impl_unary_op!(abs, |x: f64| x.abs());
    impl_unary_op!(neg, |x: f64| -x);
    impl_unary_op!(reciprocal, |x: f64| 1.0 / x);
    impl_unary_op!(deg2rad, |x: f64| x.to_radians());
    impl_unary_op!(rad2deg, |x: f64| x.to_degrees());

    fn sign(arr: &CpuArray) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            if x.is_nan() {
                f64::NAN  // Propagate NaN (NumPy behavior)
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                // Handle -0.0 and 0.0 - NumPy returns -0.0 for sign(-0.0)
                if x.is_sign_negative() { -0.0 } else { 0.0 }
            }
        }))
    }

    // Binary operations with broadcasting support
    impl_binary_op_broadcast!(add, +);
    impl_binary_op_broadcast!(sub, -);
    impl_binary_op_broadcast!(mul, *);
    impl_binary_op_broadcast!(div, /);

    fn pow(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| x.powf(y))?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn maximum(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // NumPy np.maximum behavior: propagate NaN (if either is NaN, result is NaN)
        // Note: np.fmax ignores NaN and returns non-NaN value
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if x.is_nan() || y.is_nan() { f64::NAN }
            else { x.max(y) }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn minimum(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // NumPy np.minimum behavior: propagate NaN (if either is NaN, result is NaN)
        // Note: np.fmin ignores NaN and returns non-NaN value
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if x.is_nan() || y.is_nan() { f64::NAN }
            else { x.min(y) }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn fmax(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // NumPy np.fmax behavior: ignores NaN, returns non-NaN value if one exists
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if x.is_nan() && y.is_nan() { f64::NAN }
            else if x.is_nan() { y }
            else if y.is_nan() { x }
            else { x.max(y) }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn fmin(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // NumPy np.fmin behavior: ignores NaN, returns non-NaN value if one exists
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if x.is_nan() && y.is_nan() { f64::NAN }
            else if x.is_nan() { y }
            else if y.is_nan() { x }
            else { x.min(y) }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn arctan2(y: &CpuArray, x: &CpuArray) -> Result<CpuArray> {
        // Two-argument arctangent: returns angle in radians between positive x-axis and point (x, y)
        let result = broadcast_binary_op(y.as_ndarray(), x.as_ndarray(), |y_val, x_val| {
            y_val.atan2(x_val)
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn mod_op(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // Python/NumPy modulo: result has same sign as divisor
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if y == 0.0 {
                f64::NAN
            } else {
                let rem = x % y;
                if rem != 0.0 && rem.signum() != y.signum() {
                    rem + y
                } else {
                    rem
                }
            }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn fmod(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // C-style fmod: result has same sign as dividend
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            x % y  // Rust's % is fmod behavior
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn remainder(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // IEEE 754 remainder: result is x - y * round(x/y)
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            if y == 0.0 {
                f64::NAN
            } else {
                // IEEE remainder: round to nearest even
                let quotient = x / y;
                let rounded = if (quotient.fract().abs() - 0.5).abs() < 1e-15 {
                    // Exactly 0.5: round to nearest even
                    let floor_val = quotient.floor();
                    let ceil_val = quotient.ceil();
                    if (floor_val as i64) % 2 == 0 { floor_val } else { ceil_val }
                } else {
                    quotient.round()
                };
                x - y * rounded
            }
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    fn hypot(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // Hypotenuse: sqrt(x^2 + y^2) without overflow
        let result = broadcast_binary_op(a.as_ndarray(), b.as_ndarray(), |x, y| {
            x.hypot(y)
        })?;
        Ok(CpuArray::from_ndarray(result))
    }

    // Scalar operations
    impl_scalar_op!(add_scalar, +);
    impl_scalar_op!(sub_scalar, -);
    impl_scalar_op!(mul_scalar, *);
    impl_scalar_op!(div_scalar, /);

    fn pow_scalar(arr: &CpuArray, scalar: f64) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x.powf(scalar)))
    }

    fn clip(arr: &CpuArray, min: f64, max: f64) -> CpuArray {
        // NumPy behavior: if min or max is NaN, result is NaN
        // Also warn/handle when min > max (NumPy uses min as the result)
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            if x.is_nan() || min.is_nan() || max.is_nan() {
                f64::NAN
            } else if min > max {
                // NumPy behavior: when min > max, result is min
                min
            } else {
                x.clamp(min, max)
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumpy_core::Array;
    use std::f64::consts::PI;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_sin() {
        let a = arr(vec![0.0, PI / 2.0, PI]);
        let result = CpuBackend::sin(&a);
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 0.0));
        assert!(approx_eq(data[1], 1.0));
        assert!(approx_eq(data[2], 0.0));
    }

    #[test]
    fn test_cos() {
        let a = arr(vec![0.0, PI / 2.0, PI]);
        let result = CpuBackend::cos(&a);
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], -1.0));
    }

    #[test]
    fn test_exp_log() {
        let a = arr(vec![0.0, 1.0, 2.0]);
        let exp_a = CpuBackend::exp(&a);
        let log_exp_a = CpuBackend::log(&exp_a);
        for (x, y) in a.as_f64_slice().iter().zip(log_exp_a.as_f64_slice().iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_sqrt() {
        let a = arr(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
        let result = CpuBackend::sqrt(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs() {
        let a = arr(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = CpuBackend::abs(&a);
        assert_eq!(result.as_f64_slice(), vec![2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sign() {
        let a = arr(vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
        let result = CpuBackend::sign(&a);
        assert_eq!(result.as_f64_slice(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sign_negative_zero() {
        // NumPy: sign(-0.0) returns -0.0
        let a = arr(vec![-0.0, 0.0]);
        let result = CpuBackend::sign(&a);
        let data = result.as_f64_slice();
        // Check that sign(-0.0) is -0.0 (negative zero)
        assert!(data[0].is_sign_negative());
        assert!(data[0] == 0.0); // -0.0 == 0.0 is true
        // Check that sign(0.0) is 0.0 (positive zero)
        assert!(!data[1].is_sign_negative());
        assert!(data[1] == 0.0);
    }

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
    fn test_add_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::add_scalar(&a, 10.0);
        assert_eq!(result.as_f64_slice(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_clip() {
        let a = arr(vec![-5.0, 0.0, 5.0, 10.0, 15.0]);
        let result = CpuBackend::clip(&a, 0.0, 10.0);
        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clip_nan_bounds() {
        // NumPy: if min or max is NaN, result is NaN
        let a = arr(vec![1.0, 5.0, 10.0]);

        // NaN min
        let result = CpuBackend::clip(&a, f64::NAN, 10.0);
        assert!(result.as_f64_slice().iter().all(|x| x.is_nan()));

        // NaN max
        let result = CpuBackend::clip(&a, 0.0, f64::NAN);
        assert!(result.as_f64_slice().iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_clip_min_greater_than_max() {
        // NumPy behavior when min > max: result is min
        let a = arr(vec![1.0, 5.0, 10.0]);
        let result = CpuBackend::clip(&a, 8.0, 2.0); // min=8, max=2
        // All values should be 8.0 (min)
        assert_eq!(result.as_f64_slice(), vec![8.0, 8.0, 8.0]);
    }

    #[test]
    fn test_maximum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::maximum(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![2.0, 5.0, 4.0]);
    }

    #[test]
    fn test_maximum_nan_propagates() {
        // NumPy np.maximum propagates NaN - if either is NaN, result is NaN
        let a = arr(vec![1.0, f64::NAN, 3.0]);
        let b = arr(vec![2.0, 5.0, f64::NAN]);
        let result = CpuBackend::maximum(&a, &b).unwrap();
        let data = result.as_f64_slice();
        assert_eq!(data[0], 2.0);
        assert!(data[1].is_nan()); // NaN propagates
        assert!(data[2].is_nan()); // NaN propagates
    }

    #[test]
    fn test_minimum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::minimum(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 3.0]);
    }

    #[test]
    fn test_minimum_nan_propagates() {
        // NumPy np.minimum propagates NaN - if either is NaN, result is NaN
        let a = arr(vec![1.0, f64::NAN, 3.0]);
        let b = arr(vec![2.0, 5.0, f64::NAN]);
        let result = CpuBackend::minimum(&a, &b).unwrap();
        let data = result.as_f64_slice();
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_nan()); // NaN propagates
        assert!(data[2].is_nan()); // NaN propagates
    }

    #[test]
    fn test_round_half_away_from_zero() {
        // NumPy rounds 0.5 away from zero, not banker's rounding
        let a = arr(vec![0.5, 1.5, 2.5, -0.5, -1.5, -2.5]);
        let result = CpuBackend::round(&a);
        // 0.5 -> 1, 1.5 -> 2, 2.5 -> 3, -0.5 -> -1, -1.5 -> -2, -2.5 -> -3
        assert_eq!(result.as_f64_slice(), vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        // Incompatible shapes that can't be broadcast
        let a = arr(vec![1.0, 2.0, 3.0]); // shape [3]
        let b = arr(vec![1.0, 2.0]); // shape [2]
        let result = CpuBackend::add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_add() {
        // (2, 3) + (3,) should broadcast
        let a = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = CpuArray::from_f64_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let result = CpuBackend::add(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_f64_slice(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn test_broadcast_mul_col_row() {
        // (2, 1) * (1, 3) should broadcast to (2, 3)
        let col = CpuArray::from_f64_vec(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let row = CpuArray::from_f64_vec(vec![1.0, 10.0, 100.0], vec![1, 3]).unwrap();
        let result = CpuBackend::mul(&col, &row).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // [[2*1, 2*10, 2*100], [3*1, 3*10, 3*100]]
        assert_eq!(
            result.as_f64_slice(),
            vec![2.0, 20.0, 200.0, 3.0, 30.0, 300.0]
        );
    }

    #[test]
    fn test_expm1_numerical_stability() {
        // expm1(x) = exp(x) - 1, stable near zero
        let a = arr(vec![1e-15, 0.0, 1.0]);
        let result = CpuBackend::expm1(&a);
        let data = result.as_f64_slice();
        // For very small x, expm1(x) ≈ x
        assert!(approx_eq(data[0], 1e-15));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], std::f64::consts::E - 1.0));
    }

    #[test]
    fn test_log1p_numerical_stability() {
        // log1p(x) = log(1 + x), stable near zero
        let a = arr(vec![1e-15, 0.0, 1.0]);
        let result = CpuBackend::log1p(&a);
        let data = result.as_f64_slice();
        // For very small x, log1p(x) ≈ x
        assert!(approx_eq(data[0], 1e-15));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], 2.0_f64.ln()));
    }

    #[test]
    fn test_rint_bankers_rounding() {
        // rint uses banker's rounding (round half to even)
        let a = arr(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]);
        let result = CpuBackend::rint(&a);
        // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4
        // -0.5 -> 0, -1.5 -> -2, -2.5 -> -2, -3.5 -> -4
        assert_eq!(result.as_f64_slice(), vec![0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0, -4.0]);
    }

    #[test]
    fn test_deg2rad_rad2deg() {
        let degrees = arr(vec![0.0, 90.0, 180.0, 360.0]);
        let radians = CpuBackend::deg2rad(&degrees);
        let data = radians.as_f64_slice();
        assert!(approx_eq(data[0], 0.0));
        assert!(approx_eq(data[1], PI / 2.0));
        assert!(approx_eq(data[2], PI));
        assert!(approx_eq(data[3], 2.0 * PI));

        // Round trip
        let back_to_deg = CpuBackend::rad2deg(&radians);
        let data = back_to_deg.as_f64_slice();
        assert!(approx_eq(data[0], 0.0));
        assert!(approx_eq(data[1], 90.0));
        assert!(approx_eq(data[2], 180.0));
        assert!(approx_eq(data[3], 360.0));
    }

    #[test]
    fn test_fmax_fmin_ignore_nan() {
        // fmax/fmin ignore NaN (return non-NaN if one exists)
        let a = arr(vec![1.0, f64::NAN, 3.0]);
        let b = arr(vec![2.0, 5.0, f64::NAN]);

        let fmax_result = CpuBackend::fmax(&a, &b).unwrap();
        let fmax_data = fmax_result.as_f64_slice();
        assert_eq!(fmax_data[0], 2.0);
        assert_eq!(fmax_data[1], 5.0);  // NaN ignored, returns 5.0
        assert_eq!(fmax_data[2], 3.0);  // NaN ignored, returns 3.0

        let fmin_result = CpuBackend::fmin(&a, &b).unwrap();
        let fmin_data = fmin_result.as_f64_slice();
        assert_eq!(fmin_data[0], 1.0);
        assert_eq!(fmin_data[1], 5.0);  // NaN ignored, returns 5.0
        assert_eq!(fmin_data[2], 3.0);  // NaN ignored, returns 3.0
    }

    #[test]
    fn test_arctan2() {
        let y = arr(vec![0.0, 1.0, 0.0, -1.0]);
        let x = arr(vec![1.0, 0.0, -1.0, 0.0]);
        let result = CpuBackend::arctan2(&y, &x).unwrap();
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 0.0));           // (1, 0)
        assert!(approx_eq(data[1], PI / 2.0));     // (0, 1)
        assert!(approx_eq(data[2], PI));           // (-1, 0)
        assert!(approx_eq(data[3], -PI / 2.0));    // (0, -1)
    }

    #[test]
    fn test_hypot() {
        let a = arr(vec![3.0, 5.0, 0.0]);
        let b = arr(vec![4.0, 12.0, 1.0]);
        let result = CpuBackend::hypot(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![5.0, 13.0, 1.0]);
    }

    #[test]
    fn test_mod_op() {
        // Python-style modulo: result has same sign as divisor
        let a = arr(vec![7.0, -7.0, 7.0, -7.0]);
        let b = arr(vec![3.0, 3.0, -3.0, -3.0]);
        let result = CpuBackend::mod_op(&a, &b).unwrap();
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));   // 7 mod 3 = 1
        assert!(approx_eq(data[1], 2.0));   // -7 mod 3 = 2 (Python)
        assert!(approx_eq(data[2], -2.0));  // 7 mod -3 = -2 (Python)
        assert!(approx_eq(data[3], -1.0));  // -7 mod -3 = -1 (Python)
    }

    #[test]
    fn test_fmod() {
        // C-style fmod: result has same sign as dividend
        let a = arr(vec![7.0, -7.0, 7.0, -7.0]);
        let b = arr(vec![3.0, 3.0, -3.0, -3.0]);
        let result = CpuBackend::fmod(&a, &b).unwrap();
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));   // 7 fmod 3 = 1
        assert!(approx_eq(data[1], -1.0));  // -7 fmod 3 = -1
        assert!(approx_eq(data[2], 1.0));   // 7 fmod -3 = 1
        assert!(approx_eq(data[3], -1.0));  // -7 fmod -3 = -1
    }
}
