//! Comparison operations for CPU backend

use crate::broadcast::broadcast_compare_op;
use crate::{CpuArray, CpuBackend};
use rumpy_core::{ops::CompareOps, Array, Result};

macro_rules! impl_compare_op_broadcast {
    ($name:ident, $op:tt) => {
        fn $name(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
            let result = broadcast_compare_op(a.as_ndarray(), b.as_ndarray(), |x, y| x $op y)?;
            Ok(CpuArray::from_ndarray(result))
        }
    };
}

macro_rules! impl_compare_scalar_op {
    ($name:ident, $op:tt) => {
        fn $name(arr: &CpuArray, scalar: f64) -> CpuArray {
            let result: Vec<f64> = arr
                .as_f64_slice()
                .iter()
                .map(|&x| if x $op scalar { 1.0 } else { 0.0 })
                .collect();
            CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
        }
    };
}

impl CompareOps for CpuBackend {
    type Array = CpuArray;

    // Comparison operations with broadcasting support
    impl_compare_op_broadcast!(eq, ==);
    impl_compare_op_broadcast!(ne, !=);
    impl_compare_op_broadcast!(lt, <);
    impl_compare_op_broadcast!(le, <=);
    impl_compare_op_broadcast!(gt, >);
    impl_compare_op_broadcast!(ge, >=);

    impl_compare_scalar_op!(eq_scalar, ==);
    impl_compare_scalar_op!(ne_scalar, !=);
    impl_compare_scalar_op!(lt_scalar, <);
    impl_compare_scalar_op!(le_scalar, <=);
    impl_compare_scalar_op!(gt_scalar, >);
    impl_compare_scalar_op!(ge_scalar, >=);

    fn isnan(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }

    fn isinf(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_infinite() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }

    fn isfinite(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_finite() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumpy_core::Array;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_eq() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 5.0, 3.0]);
        let result = CpuBackend::eq(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ne() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 5.0, 3.0]);
        let result = CpuBackend::ne(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_lt() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::lt(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_le() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::le(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::gt(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ge() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::ge(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_eq_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0, 2.0]);
        let result = CpuBackend::eq_scalar(&a, 2.0);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_lt_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::lt_scalar(&a, 3.0);
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_isnan() {
        let a = arr(vec![1.0, f64::NAN, 3.0, f64::NAN]);
        let result = CpuBackend::isnan(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isinf() {
        let a = arr(vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY]);
        let result = CpuBackend::isinf(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isfinite() {
        let a = arr(vec![1.0, f64::INFINITY, f64::NAN, 4.0]);
        let result = CpuBackend::isfinite(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        // Incompatible shapes that can't be broadcast
        let a = arr(vec![1.0, 2.0, 3.0]); // shape [3]
        let b = arr(vec![1.0, 2.0]); // shape [2]
        let result = CpuBackend::eq(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_compare() {
        // (2, 3) vs (3,) should broadcast
        let a = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = CpuArray::from_f64_vec(vec![2.0, 2.0, 2.0], vec![3]).unwrap();
        let result = CpuBackend::gt(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: [1>2, 2>2, 3>2] = [0, 0, 1]
        // Row 1: [4>2, 5>2, 6>2] = [1, 1, 1]
        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        // IEEE 754: -0.0 == 0.0 is true
        let a = arr(vec![-0.0, 0.0, -0.0]);
        let b = arr(vec![0.0, -0.0, -0.0]);
        let result = CpuBackend::eq(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, 1.0]);
    }
}
