//! Array creation operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{Array2, ArrayD, IxDyn};
use rumpy_core::{ops::CreationOps, Array, Result, RumpyError};

impl CreationOps for CpuBackend {
    type Array = CpuArray;

    fn zeros(shape: Vec<usize>) -> CpuArray {
        CpuArray::zeros(shape)
    }

    fn ones(shape: Vec<usize>) -> CpuArray {
        CpuArray::ones(shape)
    }

    fn full(shape: Vec<usize>, value: f64) -> CpuArray {
        CpuArray::full(shape, value)
    }

    fn arange(start: f64, stop: f64, step: f64) -> Result<CpuArray> {
        if step == 0.0 {
            return Err(RumpyError::InvalidArgument(
                "Step cannot be zero".to_string(),
            ));
        }

        if (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Ok(CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0]))));
        }

        // Use floor instead of ceil for NumPy-compatible element count
        // NumPy uses: n = ceil((stop - start) / step) but then excludes values >= stop
        // We compute exact count to avoid floating-point edge cases
        let n = {
            let diff = stop - start;
            let count = (diff / step).abs();
            // Add small epsilon to handle floating-point precision
            let count_floor = count.floor();
            if (count - count_floor).abs() < 1e-10 {
                count_floor as usize
            } else {
                count.ceil() as usize
            }
        };

        let values: Vec<f64> = (0..n)
            .map(|i| start + (i as f64) * step)
            .take_while(|&v| if step > 0.0 { v < stop } else { v > stop })
            .collect();

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap(),
        ))
    }

    fn linspace(start: f64, stop: f64, num: usize) -> CpuArray {
        if num == 0 {
            return CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0])));
        }
        if num == 1 {
            return CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap(),
            );
        }

        let step = (stop - start) / (num - 1) as f64;
        let mut values: Vec<f64> = (0..num).map(|i| start + (i as f64) * step).collect();

        // Ensure last value is exactly stop (NumPy behavior)
        if let Some(last) = values.last_mut() {
            *last = stop;
        }

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[num]), values).unwrap())
    }

    fn logspace(start: f64, stop: f64, num: usize, base: f64) -> CpuArray {
        // Generate log-spaced values: base^linspace(start, stop, num)
        if num == 0 {
            return CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0])));
        }
        if num == 1 {
            return CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![base.powf(start)]).unwrap(),
            );
        }

        let step = (stop - start) / (num - 1) as f64;
        let mut values: Vec<f64> = (0..num)
            .map(|i| base.powf(start + (i as f64) * step))
            .collect();

        // Ensure last value is exactly base^stop
        if let Some(last) = values.last_mut() {
            *last = base.powf(stop);
        }

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[num]), values).unwrap())
    }

    fn geomspace(start: f64, stop: f64, num: usize) -> Result<CpuArray> {
        // Generate geometrically-spaced values
        if start == 0.0 || stop == 0.0 {
            return Err(RumpyError::InvalidArgument(
                "Geometric sequence cannot include zero".to_string(),
            ));
        }
        if (start < 0.0) != (stop < 0.0) {
            return Err(RumpyError::InvalidArgument(
                "Cannot compute geometric sequence of numbers with opposite signs".to_string(),
            ));
        }

        if num == 0 {
            return Ok(CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0]))));
        }
        if num == 1 {
            return Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap(),
            ));
        }

        let sign = start.signum();
        let log_start = start.abs().ln();
        let log_stop = stop.abs().ln();
        let step = (log_stop - log_start) / (num - 1) as f64;

        let mut values: Vec<f64> = (0..num)
            .map(|i| sign * (log_start + (i as f64) * step).exp())
            .collect();

        // Ensure endpoints are exact
        values[0] = start;
        if let Some(last) = values.last_mut() {
            *last = stop;
        }

        Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[num]), values).unwrap()))
    }

    fn eye(n: usize) -> CpuArray {
        let mut arr = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = 1.0;
        }
        CpuArray::from_ndarray(arr.into_dyn())
    }

    fn diag(arr: &CpuArray, k: i32) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();

        if shape.len() == 1 {
            // Create diagonal matrix from 1D array
            let n = shape[0];
            let size = n + k.unsigned_abs() as usize;
            let mut result = Array2::<f64>::zeros((size, size));

            for i in 0..n {
                let row = if k >= 0 { i } else { i + (-k) as usize };
                let col = if k >= 0 { i + k as usize } else { i };
                if row < size && col < size {
                    result[[row, col]] = data[IxDyn(&[i])];
                }
            }

            Ok(CpuArray::from_ndarray(result.into_dyn()))
        } else if shape.len() == 2 {
            // Extract diagonal from 2D array
            let (m, n) = (shape[0], shape[1]);
            let start_row = if k >= 0 { 0 } else { (-k) as usize };
            let start_col = if k >= 0 { k as usize } else { 0 };

            let diag_len = std::cmp::min(m.saturating_sub(start_row), n.saturating_sub(start_col));

            let values: Vec<f64> = (0..diag_len)
                .map(|i| data[IxDyn(&[start_row + i, start_col + i])])
                .collect();

            Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[diag_len]), values).unwrap(),
            ))
        } else {
            Err(RumpyError::InvalidArgument(
                "diag requires 1D or 2D array".to_string(),
            ))
        }
    }

    fn meshgrid(arrays: &[&CpuArray], indexing: &str) -> Result<Vec<CpuArray>> {
        // Create coordinate matrices from coordinate vectors
        // indexing: "xy" (Cartesian) or "ij" (matrix indexing)
        if arrays.is_empty() {
            return Ok(Vec::new());
        }

        // Validate all arrays are 1D
        for arr in arrays {
            if arr.ndim() != 1 {
                return Err(RumpyError::InvalidArgument(
                    "meshgrid requires 1D arrays".to_string(),
                ));
            }
        }

        let n = arrays.len();

        // Build output shape
        let output_shape: Vec<usize> = if indexing == "xy" && n >= 2 {
            // For "xy", swap first two dimensions
            let mut shape: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
            shape.swap(0, 1);
            shape
        } else {
            arrays.iter().map(|a| a.shape()[0]).collect()
        };

        let total_size: usize = output_shape.iter().product();

        let mut result = Vec::with_capacity(n);

        for (arr_idx, arr) in arrays.iter().enumerate() {
            let data = arr.as_f64_slice();
            let mut output_data = Vec::with_capacity(total_size);

            // Determine which dimension this array broadcasts along
            let broadcast_dim = if indexing == "xy" && n >= 2 {
                if arr_idx == 0 { 1 }
                else if arr_idx == 1 { 0 }
                else { arr_idx }
            } else {
                arr_idx
            };

            // Fill output array
            for flat_idx in 0..total_size {
                // Decompose flat index to coordinates
                let mut idx = flat_idx;
                let mut coords: Vec<usize> = vec![0; n];
                for d in (0..n).rev() {
                    coords[d] = idx % output_shape[d];
                    idx /= output_shape[d];
                }

                // Get value from input array using the broadcast dimension
                output_data.push(data[coords[broadcast_dim]]);
            }

            result.push(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&output_shape), output_data).unwrap(),
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumpy_core::Array;

    #[test]
    fn test_zeros() {
        let arr = CpuBackend::zeros(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let arr = CpuBackend::ones(vec![2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let arr = CpuBackend::full(vec![2, 2], 5.0);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_arange() {
        let arr = CpuBackend::arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let arr = CpuBackend::arange(0.0, 10.0, 2.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_arange_negative_step() {
        let arr = CpuBackend::arange(5.0, 0.0, -1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_arange_zero_step() {
        let result = CpuBackend::arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace() {
        let arr = CpuBackend::linspace(0.0, 1.0, 5);
        let data = arr.as_f64_slice();
        assert_eq!(data.len(), 5);
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
        assert!((data[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let arr = CpuBackend::eye(3);
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        assert_eq!(data[0], 1.0); // [0,0]
        assert_eq!(data[1], 0.0); // [0,1]
        assert_eq!(data[4], 1.0); // [1,1]
        assert_eq!(data[8], 1.0); // [2,2]
    }

    #[test]
    fn test_diag_create() {
        let vec = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let arr = CpuBackend::diag(&vec, 0).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 2.0);
        assert_eq!(data[8], 3.0);
    }

    #[test]
    fn test_diag_extract() {
        let mat = CpuArray::from_f64_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let diag = CpuBackend::diag(&mat, 0).unwrap();
        assert_eq!(diag.as_f64_slice(), vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_logspace() {
        // logspace(0, 2, 3) = [10^0, 10^1, 10^2] = [1, 10, 100]
        let arr = CpuBackend::logspace(0.0, 2.0, 3, 10.0);
        assert_eq!(arr.shape(), &[3]);
        let data = arr.as_f64_slice();
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 10.0).abs() < 1e-10);
        assert!((data[2] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_logspace_base2() {
        // logspace(0, 3, 4, base=2) = [2^0, 2^1, 2^2, 2^3] = [1, 2, 4, 8]
        let arr = CpuBackend::logspace(0.0, 3.0, 4, 2.0);
        let data = arr.as_f64_slice();
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 2.0).abs() < 1e-10);
        assert!((data[2] - 4.0).abs() < 1e-10);
        assert!((data[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_geomspace() {
        // geomspace(1, 1000, 4) = [1, 10, 100, 1000]
        let arr = CpuBackend::geomspace(1.0, 1000.0, 4).unwrap();
        let data = arr.as_f64_slice();
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 10.0).abs() < 1e-6);  // geometric mean
        assert!((data[2] - 100.0).abs() < 1e-6);
        assert!((data[3] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_geomspace_negative() {
        // geomspace(-1, -1000, 4) = [-1, -10, -100, -1000]
        let arr = CpuBackend::geomspace(-1.0, -1000.0, 4).unwrap();
        let data = arr.as_f64_slice();
        assert!((data[0] - (-1.0)).abs() < 1e-10);
        assert!((data[1] - (-10.0)).abs() < 1e-6);
        assert!((data[2] - (-100.0)).abs() < 1e-6);
        assert!((data[3] - (-1000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_geomspace_invalid_zero() {
        // Cannot include zero
        assert!(CpuBackend::geomspace(0.0, 100.0, 5).is_err());
    }

    #[test]
    fn test_geomspace_invalid_opposite_signs() {
        // Cannot span positive and negative
        assert!(CpuBackend::geomspace(-1.0, 100.0, 5).is_err());
    }

    #[test]
    fn test_meshgrid_basic() {
        let x = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let y = CpuArray::from_f64_vec(vec![4.0, 5.0], vec![2]).unwrap();

        let result = CpuBackend::meshgrid(&[&x, &y], "xy").unwrap();
        assert_eq!(result.len(), 2);

        // With "xy" indexing, X varies along columns, Y varies along rows
        // X grid: [[1,2,3], [1,2,3]]
        // Y grid: [[4,4,4], [5,5,5]]
        let xg = &result[0];
        let yg = &result[1];
        assert_eq!(xg.shape(), &[2, 3]);
        assert_eq!(yg.shape(), &[2, 3]);
        assert_eq!(xg.as_f64_slice(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(yg.as_f64_slice(), vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
    }
}
