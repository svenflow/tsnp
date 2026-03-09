//! NumPy-style broadcasting utilities
//!
//! Implements broadcasting rules for element-wise operations between arrays
//! of different shapes using ndarray's efficient zero-cost views.
//!
//! Broadcasting Rules (from NumPy):
//! 1. If arrays have different number of dimensions, prepend 1s to the smaller shape
//! 2. Arrays are compatible if for each dimension:
//!    - Dimensions are equal, OR
//!    - One of them is 1
//! 3. Result dimension = max(dim1, dim2) for each axis

use ndarray::{ArrayD, IxDyn};
use rumpy_core::{Result, RumpyError};

/// Compute the broadcast shape of two input shapes.
///
/// Returns the resulting shape after broadcasting, or an error if shapes are incompatible.
///
/// # Examples
///
/// - `[3, 1] + [1, 4] → [3, 4]`
/// - `[3, 4] + [4] → [3, 4]`
/// - `[3, 4] + [2, 4] → Error (incompatible)`
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let ndim_a = shape_a.len();
    let ndim_b = shape_b.len();
    let ndim_out = ndim_a.max(ndim_b);

    let mut result = vec![0; ndim_out];

    // Iterate from right to left (trailing dimensions)
    for i in 0..ndim_out {
        // Get dimension from right, defaulting to 1 if not present
        let dim_a = if i < ndim_a {
            shape_a[ndim_a - 1 - i]
        } else {
            1
        };
        let dim_b = if i < ndim_b {
            shape_b[ndim_b - 1 - i]
        } else {
            1
        };

        // Check compatibility: valid if equal, or one is 1
        if dim_a == dim_b || dim_a == 1 || dim_b == 1 {
            result[ndim_out - 1 - i] = dim_a.max(dim_b);
        } else {
            return Err(RumpyError::IncompatibleShapes(
                shape_a.to_vec(),
                shape_b.to_vec(),
            ));
        }
    }

    Ok(result)
}

/// Broadcast an ndarray to a target shape.
///
/// Uses ndarray's efficient broadcasting views internally.
/// The target shape must be compatible with the input shape according to broadcasting rules.
pub fn broadcast_to(arr: &ArrayD<f64>, target_shape: &[usize]) -> Result<ArrayD<f64>> {
    // Fast path: already the right shape
    if arr.shape() == target_shape {
        return Ok(arr.clone());
    }

    // Use ndarray's built-in broadcasting which creates a zero-cost view
    // .to_owned() converts the view into a new, contiguous array
    arr.broadcast(IxDyn(target_shape))
        .map(|view| view.to_owned())
        .ok_or_else(|| {
            RumpyError::InvalidShape(format!(
                "Cannot broadcast shape {:?} to {:?}",
                arr.shape(),
                target_shape
            ))
        })
}

/// Apply a binary operation with broadcasting.
///
/// This uses ndarray's zero-cost broadcasting views for maximum efficiency.
/// No intermediate arrays are allocated - the computation happens directly
/// into the result array in a single pass.
pub fn broadcast_binary_op<F>(a: &ArrayD<f64>, b: &ArrayD<f64>, op: F) -> Result<ArrayD<f64>>
where
    F: Fn(f64, f64) -> f64,
{
    let shape_a = a.shape();
    let shape_b = b.shape();

    // Fast path: same shape, no broadcasting needed
    if shape_a == shape_b {
        let result = ndarray::Zip::from(a).and(b).map_collect(|&x, &y| op(x, y));
        return Ok(result);
    }

    // Compute the broadcast output shape
    let output_shape = broadcast_shapes(shape_a, shape_b)?;
    let output_dim = IxDyn(&output_shape);

    // Create lightweight, zero-cost broadcasting views
    // .unwrap() is safe because we've already validated shapes with broadcast_shapes
    let a_view = a.broadcast(output_dim.clone()).unwrap();
    let b_view = b.broadcast(output_dim).unwrap();

    // Zip the views to produce the final result directly, with no intermediate allocations
    let result = ndarray::Zip::from(a_view)
        .and(b_view)
        .map_collect(|&x, &y| op(x, y));

    Ok(result)
}

/// Apply a binary comparison operation with broadcasting.
///
/// Returns an array of 1.0 (true) or 0.0 (false).
pub fn broadcast_compare_op<F>(a: &ArrayD<f64>, b: &ArrayD<f64>, op: F) -> Result<ArrayD<f64>>
where
    F: Fn(f64, f64) -> bool,
{
    broadcast_binary_op(a, b, |x, y| if op(x, y) { 1.0 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        // Scalar broadcasts to any shape
        let result = broadcast_shapes(&[], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        let result = broadcast_shapes(&[3, 4], &[]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_trailing() {
        // (3, 4) + (4,) => (3, 4)
        let result = broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        // (4,) + (3, 4) => (3, 4)
        let result = broadcast_shapes(&[4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_ones() {
        // (3, 1) + (1, 4) => (3, 4)
        let result = broadcast_shapes(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        // (1, 3, 1) + (2, 1, 4) => (2, 3, 4)
        let result = broadcast_shapes(&[1, 3, 1], &[2, 1, 4]).unwrap();
        assert_eq!(result, vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // (3, 4) + (2, 4) => Error
        let result = broadcast_shapes(&[3, 4], &[2, 4]);
        assert!(result.is_err());

        // (3,) + (4,) => Error
        let result = broadcast_shapes(&[3], &[4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_same_shape() {
        let arr =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = broadcast_to(&arr, &[2, 3]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result, arr);
    }

    #[test]
    fn test_broadcast_to_expand_rows() {
        // (1, 3) -> (2, 3)
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let result = broadcast_to(&arr, &[2, 3]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_broadcast_to_expand_cols() {
        // (2, 1) -> (2, 3)
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 2.0]).unwrap();
        let result = broadcast_to(&arr, &[2, 3]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
    }

    #[test]
    fn test_broadcast_to_add_dims() {
        // (3,) -> (2, 3)
        let arr = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let result = broadcast_to(&arr, &[2, 3]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_broadcast_binary_op_same_shape() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            .unwrap();

        let result = broadcast_binary_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );
    }

    #[test]
    fn test_broadcast_binary_op_row_plus_matrix() {
        // (3,) + (2, 3) => (2, 3)
        let row = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let matrix =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();

        let result = broadcast_binary_op(&row, &matrix, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![11.0, 22.0, 33.0, 41.0, 52.0, 63.0]
        );
    }

    #[test]
    fn test_broadcast_binary_op_col_plus_row() {
        // (2, 1) + (1, 3) => (2, 3)
        let col = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 2.0]).unwrap();
        let row = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![10.0, 20.0, 30.0]).unwrap();

        let result = broadcast_binary_op(&col, &row, |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
        );
    }

    #[test]
    fn test_broadcast_compare_op() {
        let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 2.0, 2.0]).unwrap();

        let result = broadcast_compare_op(&a, &b, |x, y| x > y).unwrap();
        assert_eq!(
            result.iter().cloned().collect::<Vec<_>>(),
            vec![0.0, 0.0, 1.0]
        );
    }
}
