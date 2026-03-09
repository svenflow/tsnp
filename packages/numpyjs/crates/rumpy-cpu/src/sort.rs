//! Sorting and searching operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, IxDyn};
use rumpy_core::{ops::SortOps, Array, Result, RumpyError};
use std::cmp::Ordering;

/// NaN-safe comparison for sorting: NaN values sort to the end (NumPy behavior)
fn nan_safe_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // Handle NaN: NaN is "greater" than all values, so sorts to end
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => unreachable!(), // partial_cmp only returns None for NaN
        }
    })
}

impl SortOps for CpuBackend {
    type Array = CpuArray;

    fn sort(arr: &CpuArray, axis: Option<usize>) -> Result<CpuArray> {
        let shape = arr.shape();
        let ndim = shape.len();

        match axis {
            None => {
                // Sort flattened array - NumPy returns 1D array when axis=None
                let mut data = arr.as_f64_slice();
                data.sort_by(nan_safe_cmp);
                Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()))
            }
            Some(ax) => {
                if ax >= ndim {
                    return Err(RumpyError::InvalidAxis {
                        axis: ax,
                        ndim,
                    });
                }

                // Sort along specified axis
                let data = arr.as_ndarray();
                let mut result = data.to_owned();
                let axis_len = shape[ax];

                // Calculate stride info
                let outer_size: usize = shape[..ax].iter().product();
                let inner_size: usize = shape[ax + 1..].iter().product();

                // For each "lane" along the axis
                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        // Extract values along this lane
                        let mut lane: Vec<f64> = (0..axis_len)
                            .map(|i| {
                                let mut coords = vec![0usize; ndim];
                                // Compute coordinates
                                let mut tmp = outer;
                                for d in (0..ax).rev() {
                                    coords[d] = tmp % shape[d];
                                    tmp /= shape[d];
                                }
                                coords[ax] = i;
                                let mut tmp = inner;
                                for d in (ax + 1..ndim).rev() {
                                    coords[d] = tmp % shape[d];
                                    tmp /= shape[d];
                                }
                                // Compute flat index
                                let mut flat = 0;
                                let mut mult = 1;
                                for d in (0..ndim).rev() {
                                    flat += coords[d] * mult;
                                    mult *= shape[d];
                                }
                                data.as_slice().unwrap()[flat]
                            })
                            .collect();

                        // Sort the lane
                        lane.sort_by(nan_safe_cmp);

                        // Write back
                        for (i, &val) in lane.iter().enumerate() {
                            let mut coords = vec![0usize; ndim];
                            let mut tmp = outer;
                            for d in (0..ax).rev() {
                                coords[d] = tmp % shape[d];
                                tmp /= shape[d];
                            }
                            coords[ax] = i;
                            let mut tmp = inner;
                            for d in (ax + 1..ndim).rev() {
                                coords[d] = tmp % shape[d];
                                tmp /= shape[d];
                            }
                            let mut flat = 0;
                            let mut mult = 1;
                            for d in (0..ndim).rev() {
                                flat += coords[d] * mult;
                                mult *= shape[d];
                            }
                            result.as_slice_mut().unwrap()[flat] = val;
                        }
                    }
                }

                Ok(CpuArray::from_ndarray(result))
            }
        }
    }

    fn argsort(arr: &CpuArray, axis: Option<usize>) -> Result<CpuArray> {
        let shape = arr.shape();
        let ndim = shape.len();

        match axis {
            None => {
                // Argsort flattened array
                let data = arr.as_f64_slice();
                let mut indices: Vec<usize> = (0..data.len()).collect();
                indices.sort_by(|&a, &b| nan_safe_cmp(&data[a], &data[b]));

                let result: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
                Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap()))
            }
            Some(ax) => {
                if ax >= ndim {
                    return Err(RumpyError::InvalidAxis {
                        axis: ax,
                        ndim,
                    });
                }

                // Argsort along specified axis
                let data = arr.as_ndarray();
                let mut result = ArrayD::<f64>::zeros(IxDyn(shape));
                let axis_len = shape[ax];

                let outer_size: usize = shape[..ax].iter().product();
                let inner_size: usize = shape[ax + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        // Extract values along this lane
                        let lane: Vec<f64> = (0..axis_len)
                            .map(|i| {
                                let mut coords = vec![0usize; ndim];
                                let mut tmp = outer;
                                for d in (0..ax).rev() {
                                    coords[d] = tmp % shape[d];
                                    tmp /= shape[d];
                                }
                                coords[ax] = i;
                                let mut tmp = inner;
                                for d in (ax + 1..ndim).rev() {
                                    coords[d] = tmp % shape[d];
                                    tmp /= shape[d];
                                }
                                let mut flat = 0;
                                let mut mult = 1;
                                for d in (0..ndim).rev() {
                                    flat += coords[d] * mult;
                                    mult *= shape[d];
                                }
                                data.as_slice().unwrap()[flat]
                            })
                            .collect();

                        // Get sorted indices
                        let mut indices: Vec<usize> = (0..axis_len).collect();
                        indices.sort_by(|&a, &b| nan_safe_cmp(&lane[a], &lane[b]));

                        // Write back indices
                        for (i, &idx) in indices.iter().enumerate() {
                            let mut coords = vec![0usize; ndim];
                            let mut tmp = outer;
                            for d in (0..ax).rev() {
                                coords[d] = tmp % shape[d];
                                tmp /= shape[d];
                            }
                            coords[ax] = i;
                            let mut tmp = inner;
                            for d in (ax + 1..ndim).rev() {
                                coords[d] = tmp % shape[d];
                                tmp /= shape[d];
                            }
                            let mut flat = 0;
                            let mut mult = 1;
                            for d in (0..ndim).rev() {
                                flat += coords[d] * mult;
                                mult *= shape[d];
                            }
                            result.as_slice_mut().unwrap()[flat] = idx as f64;
                        }
                    }
                }

                Ok(CpuArray::from_ndarray(result))
            }
        }
    }

    fn searchsorted(arr: &CpuArray, values: &CpuArray, side: Option<&str>) -> CpuArray {
        let data = arr.as_f64_slice();
        let vals = values.as_f64_slice();
        let use_right = side.map_or(false, |s| s == "right");

        let result: Vec<f64> = vals
            .iter()
            .map(|&v| {
                // Binary search for insertion point with NaN-safe comparison
                // NaN values in the search array sort to the end
                let idx = data.binary_search_by(|probe| nan_safe_cmp(probe, &v));
                match idx {
                    Ok(i) => {
                        if use_right {
                            // For 'right', find rightmost position after all equal elements
                            let mut pos = i;
                            while pos < data.len() && nan_safe_cmp(&data[pos], &v) == Ordering::Equal {
                                pos += 1;
                            }
                            pos as f64
                        } else {
                            // For 'left', find leftmost position before all equal elements
                            let mut pos = i;
                            while pos > 0 && nan_safe_cmp(&data[pos - 1], &v) == Ordering::Equal {
                                pos -= 1;
                            }
                            pos as f64
                        }
                    }
                    Err(i) => i as f64,
                }
            })
            .collect();

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap())
    }

    fn unique(arr: &CpuArray) -> CpuArray {
        let mut data = arr.as_f64_slice();
        data.sort_by(nan_safe_cmp);
        // dedup removes consecutive duplicates, but NaN != NaN so we need special handling
        // Keep only one NaN at the end
        let had_nan = data.iter().any(|x| x.is_nan());
        data.retain(|x| !x.is_nan());
        data.dedup();
        if had_nan {
            data.push(f64::NAN);
        }

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
    }

    fn nonzero(arr: &CpuArray) -> Vec<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();
        let ndim = shape.len();

        // Collect indices where value != 0
        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];

        for (flat_idx, &val) in data.iter().enumerate() {
            if val != 0.0 {
                // Convert flat index to multi-dimensional indices
                let mut idx = flat_idx;
                let mut coords = vec![0usize; ndim];
                for d in (0..ndim).rev() {
                    coords[d] = idx % shape[d];
                    idx /= shape[d];
                }
                // Push coordinates in the correct order
                for (d, &coord) in coords.iter().enumerate() {
                    indices[d].push(coord);
                }
            }
        }

        // Convert to CpuArrays
        indices
            .into_iter()
            .map(|idx| {
                let data: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
                CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
            })
            .collect()
    }

    fn digitize(arr: &CpuArray, bins: &CpuArray, right: bool) -> CpuArray {
        // Return indices of bins to which each value belongs
        let data = arr.as_f64_slice();
        let bin_edges = bins.as_f64_slice();

        let result: Vec<f64> = data
            .iter()
            .map(|&val| {
                // Binary search for bin index
                let idx = if right {
                    // right=True: bins[i-1] < x <= bins[i]
                    bin_edges.partition_point(|&b| b < val)
                } else {
                    // right=False (default): bins[i-1] <= x < bins[i]
                    bin_edges.partition_point(|&b| b <= val)
                };
                idx as f64
            })
            .collect();

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(arr.shape()), result).unwrap())
    }

    fn histogram(arr: &CpuArray, bins: usize) -> Result<(CpuArray, CpuArray)> {
        let data = arr.as_f64_slice();

        if data.is_empty() {
            let hist = CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[bins]), vec![0.0; bins]).unwrap());
            let edges = CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[bins + 1]), vec![0.0; bins + 1]).unwrap());
            return Ok((hist, edges));
        }

        // Find min and max
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Create bin edges
        let bin_width = if (max_val - min_val).abs() < 1e-15 {
            1.0 // Prevent division by zero for constant arrays
        } else {
            (max_val - min_val) / bins as f64
        };

        let edges: Vec<f64> = (0..=bins)
            .map(|i| min_val + (i as f64) * bin_width)
            .collect();

        // Count values in each bin
        let mut counts = vec![0.0; bins];
        for &val in &data {
            let bin_idx = if (val - max_val).abs() < 1e-15 {
                bins - 1 // Include right edge in last bin
            } else {
                ((val - min_val) / bin_width).floor() as usize
            };
            if bin_idx < bins {
                counts[bin_idx] += 1.0;
            }
        }

        let hist = CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[bins]), counts).unwrap());
        let edges_arr = CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[bins + 1]), edges).unwrap());

        Ok((hist, edges_arr))
    }

    fn where_cond(condition: &CpuArray, x: &CpuArray, y: &CpuArray) -> Result<CpuArray> {
        use crate::broadcast::{broadcast_shapes, broadcast_to};

        // First compute the broadcast shape of all three arrays
        let cond_shape = condition.shape();
        let x_shape = x.shape();
        let y_shape = y.shape();

        let temp_shape = broadcast_shapes(cond_shape, x_shape)?;
        let output_shape = broadcast_shapes(&temp_shape, y_shape)?;

        // Broadcast all arrays to output shape
        let cond_bc = broadcast_to(condition.as_ndarray(), &output_shape)?;
        let x_bc = broadcast_to(x.as_ndarray(), &output_shape)?;
        let y_bc = broadcast_to(y.as_ndarray(), &output_shape)?;

        // Apply condition
        let result: Vec<f64> = cond_bc
            .iter()
            .zip(x_bc.iter())
            .zip(y_bc.iter())
            .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
            .collect();

        Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&output_shape), result).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_sort() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let sorted = CpuBackend::sort(&a, None).unwrap();
        assert_eq!(
            sorted.as_f64_slice(),
            vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
        );
    }

    #[test]
    fn test_sort_axis_none_returns_1d() {
        // NumPy sort(axis=None) returns flattened 1D array, not original shape
        let a = CpuArray::from_f64_vec(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let sorted = CpuBackend::sort(&a, None).unwrap();
        assert_eq!(sorted.shape(), &[4]); // Should be 1D, not [2, 2]
        assert_eq!(sorted.as_f64_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sort_invalid_axis() {
        let a = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = CpuBackend::sort(&a, Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_argsort() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        let indices = CpuBackend::argsort(&a, None).unwrap();
        // Values at indices 1, 3 are smallest (1.0), then 0 (3.0), then 2 (4.0), then 4 (5.0)
        assert_eq!(indices.as_f64_slice(), vec![1.0, 3.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_argsort_invalid_axis() {
        let a = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = CpuBackend::argsort(&a, Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_searchsorted() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let values = arr(vec![2.5, 0.0, 5.0, 6.0]);
        let indices = CpuBackend::searchsorted(&a, &values, None);
        // 2.5 -> index 2, 0.0 -> index 0, 5.0 -> index 4, 6.0 -> index 5
        assert_eq!(indices.as_f64_slice(), vec![2.0, 0.0, 4.0, 5.0]);
    }

    #[test]
    fn test_searchsorted_right() {
        // Array: [1.0, 2.0, 2.0, 3.0, 4.0] (indices 0-4)
        let a = arr(vec![1.0, 2.0, 2.0, 3.0, 4.0]);
        let values = arr(vec![2.0]);

        // Left side (default): returns first position where value could be inserted
        // For 2.0, the first position is index 1 (before the first 2.0)
        let left = CpuBackend::searchsorted(&a, &values, Some("left"));
        assert_eq!(left.as_f64_slice(), vec![1.0]);

        // Right side: returns position after last equal element
        // For 2.0, after all 2.0s means index 3 (before 3.0)
        let right = CpuBackend::searchsorted(&a, &values, Some("right"));
        assert_eq!(right.as_f64_slice(), vec![3.0]);
    }

    #[test]
    fn test_unique() {
        let a = arr(vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let unique = CpuBackend::unique(&a);
        assert_eq!(unique.as_f64_slice(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nonzero_1d() {
        let a = arr(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
        let indices = CpuBackend::nonzero(&a);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].as_f64_slice(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_nonzero_2d() {
        let a = CpuArray::from_f64_vec(vec![0.0, 1.0, 2.0, 0.0, 3.0, 0.0], vec![2, 3]).unwrap();
        let indices = CpuBackend::nonzero(&a);
        assert_eq!(indices.len(), 2);
        // Non-zero at (0,1), (0,2), (1,1)
        assert_eq!(indices[0].as_f64_slice(), vec![0.0, 0.0, 1.0]); // row indices
        assert_eq!(indices[1].as_f64_slice(), vec![1.0, 2.0, 1.0]); // col indices
    }
}
