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
    impl_unary_op!(log, |x: f64| x.ln());
    impl_unary_op!(log2, |x: f64| x.log2());
    impl_unary_op!(log10, |x: f64| x.log10());

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

    // Other unary
    impl_unary_op!(abs, |x: f64| x.abs());
    impl_unary_op!(neg, |x: f64| -x);
    impl_unary_op!(reciprocal, |x: f64| 1.0 / x);

    fn sign(arr: &CpuArray) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            if x.is_nan() {
                f64::NAN  // Propagate NaN (NumPy behavior)
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
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

    // Scalar operations
    impl_scalar_op!(add_scalar, +);
    impl_scalar_op!(sub_scalar, -);
    impl_scalar_op!(mul_scalar, *);
    impl_scalar_op!(div_scalar, /);

    fn pow_scalar(arr: &CpuArray, scalar: f64) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x.powf(scalar)))
    }

    fn clip(arr: &CpuArray, min: f64, max: f64) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x.clamp(min, max)))
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
}
