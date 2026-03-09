//! Statistical operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, Axis, IxDyn};
use rumpy_core::{ops::StatsOps, Array, Result, RumpyError};

impl StatsOps for CpuBackend {
    type Array = CpuArray;

    fn sum(arr: &CpuArray) -> f64 {
        arr.as_ndarray().sum()
    }

    fn prod(arr: &CpuArray) -> f64 {
        arr.as_ndarray().product()
    }

    fn mean(arr: &CpuArray) -> f64 {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return f64::NAN;
        }
        data.sum() / data.len() as f64
    }

    fn var(arr: &CpuArray) -> f64 {
        Self::var_ddof(arr, 0)
    }

    fn std(arr: &CpuArray) -> f64 {
        Self::std_ddof(arr, 0)
    }

    fn var_ddof(arr: &CpuArray, ddof: usize) -> f64 {
        let data = arr.as_ndarray();
        let n = data.len();
        if n == 0 || n <= ddof {
            return f64::NAN;
        }
        let mean = data.sum() / n as f64;
        data.mapv(|x| (x - mean).powi(2)).sum() / (n - ddof) as f64
    }

    fn std_ddof(arr: &CpuArray, ddof: usize) -> f64 {
        Self::var_ddof(arr, ddof).sqrt()
    }

    fn min(arr: &CpuArray) -> Result<f64> {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("minimum"));
        }
        Ok(data.iter()
            .cloned()
            .fold(f64::INFINITY, f64::min))
    }

    fn max(arr: &CpuArray) -> Result<f64> {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("maximum"));
        }
        Ok(data.iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max))
    }

    fn argmin(arr: &CpuArray) -> Result<usize> {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("argmin"));
        }
        // Handle NaN: treat NaN as greater than all values (NumPy behavior)
        Ok(data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or_else(|| {
                    if a.is_nan() { std::cmp::Ordering::Greater }
                    else { std::cmp::Ordering::Less }
                })
            })
            .map(|(i, _)| i)
            .unwrap())
    }

    fn argmax(arr: &CpuArray) -> Result<usize> {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("argmax"));
        }
        // Handle NaN: treat NaN as less than all values (NumPy behavior)
        Ok(data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or_else(|| {
                    if a.is_nan() { std::cmp::Ordering::Less }
                    else { std::cmp::Ordering::Greater }
                })
            })
            .map(|(i, _)| i)
            .unwrap())
    }

    fn sum_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        Ok(CpuArray::from_ndarray(data.sum_axis(Axis(axis))))
    }

    fn mean_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        match data.mean_axis(Axis(axis)) {
            Some(result) => Ok(CpuArray::from_ndarray(result)),
            None => {
                let mut new_shape = data.shape().to_vec();
                new_shape.remove(axis);
                if new_shape.is_empty() {
                    new_shape.push(1);
                }
                let nan_data = vec![f64::NAN; new_shape.iter().product()];
                Ok(CpuArray::from_ndarray(
                    ArrayD::from_shape_vec(IxDyn(&new_shape), nan_data).unwrap(),
                ))
            }
        }
    }

    fn min_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        let shape = data.shape();
        let axis_len = shape[axis];
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        // Initialize with INFINITY and track if we've seen NaN
        let mut result = vec![f64::INFINITY; result_size];
        let mut has_nan = vec![false; result_size];

        // Calculate outer and inner loop sizes for the axis reduction
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let result_idx = outer * inner_size + inner;

                for ax_idx in 0..axis_len {
                    // Compute flat index in original array
                    let flat_idx = outer * (axis_len * inner_size) + ax_idx * inner_size + inner;
                    let val = data.as_slice().unwrap()[flat_idx];

                    // NaN propagation: if any value is NaN, result is NaN
                    if val.is_nan() {
                        has_nan[result_idx] = true;
                    } else {
                        result[result_idx] = result[result_idx].min(val);
                    }
                }
            }
        }

        // Apply NaN propagation
        for i in 0..result_size {
            if has_nan[i] {
                result[i] = f64::NAN;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn max_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let axis_len = shape[axis];
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        // Initialize with NEG_INFINITY and track if we've seen NaN
        let mut result = vec![f64::NEG_INFINITY; result_size];
        let mut has_nan = vec![false; result_size];

        // Calculate outer and inner loop sizes for the axis reduction
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let result_idx = outer * inner_size + inner;

                for ax_idx in 0..axis_len {
                    // Compute flat index in original array
                    let flat_idx = outer * (axis_len * inner_size) + ax_idx * inner_size + inner;
                    let val = data.as_slice().unwrap()[flat_idx];

                    // NaN propagation: if any value is NaN, result is NaN
                    if val.is_nan() {
                        has_nan[result_idx] = true;
                    } else {
                        result[result_idx] = result[result_idx].max(val);
                    }
                }
            }
        }

        // Apply NaN propagation
        for i in 0..result_size {
            if has_nan[i] {
                result[i] = f64::NAN;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn prod_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![1.0; result_size];

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            result[result_idx] *= val;
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn var_axis(arr: &CpuArray, axis: usize, ddof: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let mean_arr = Self::mean_axis(arr, axis)?;
        let mean_data = mean_arr.as_ndarray();
        let shape = data.shape();
        let axis_len = shape[axis];

        if axis_len <= ddof {
            let nan_data = vec![f64::NAN; mean_data.len()];
            return Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(mean_data.shape()), nan_data).unwrap(),
            ));
        }

        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size];

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            let mean = mean_data.as_slice().unwrap()[result_idx];
            result[result_idx] += (val - mean).powi(2);
        }

        // Divide by (n - ddof)
        for r in &mut result {
            *r /= (axis_len - ddof) as f64;
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn std_axis(arr: &CpuArray, axis: usize, ddof: usize) -> Result<CpuArray> {
        let var_arr = Self::var_axis(arr, axis, ddof)?;
        let data = var_arr.as_ndarray();
        let result = data.mapv(|x| x.sqrt());
        Ok(CpuArray::from_ndarray(result))
    }

    fn argmin_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let _axis_len = shape[axis];
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size];
        let mut min_vals = vec![f64::INFINITY; result_size];

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut axis_coord = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d == axis {
                    axis_coord = coord;
                } else {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            // NaN handling: NaN is treated as greater than all values
            if !val.is_nan() && val < min_vals[result_idx] {
                min_vals[result_idx] = val;
                result[result_idx] = axis_coord as f64;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn argmax_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size];
        let mut max_vals = vec![f64::NEG_INFINITY; result_size];

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut axis_coord = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d == axis {
                    axis_coord = coord;
                } else {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            // NaN handling: NaN is treated as less than all values
            if !val.is_nan() && val > max_vals[result_idx] {
                max_vals[result_idx] = val;
                result[result_idx] = axis_coord as f64;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn cumsum(arr: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0;
        for &x in &data {
            acc += x;
            result.push(acc);
        }
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), result).unwrap())
    }

    fn cumprod(arr: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 1.0;
        for &x in &data {
            acc *= x;
            result.push(acc);
        }
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), result).unwrap())
    }

    fn cumsum_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();
        let ndim = shape.len();

        if axis >= ndim {
            return Err(RumpyError::InvalidAxis { axis, ndim });
        }

        let mut result = data.to_owned();
        let axis_len = shape[axis];
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = 0.0;
                for ax_idx in 0..axis_len {
                    let mut coords = vec![0usize; ndim];
                    let mut tmp = outer;
                    for d in (0..axis).rev() {
                        coords[d] = tmp % shape[d];
                        tmp /= shape[d];
                    }
                    coords[axis] = ax_idx;
                    let mut tmp = inner;
                    for d in (axis + 1..ndim).rev() {
                        coords[d] = tmp % shape[d];
                        tmp /= shape[d];
                    }
                    let mut flat = 0;
                    let mut mult = 1;
                    for d in (0..ndim).rev() {
                        flat += coords[d] * mult;
                        mult *= shape[d];
                    }
                    acc += data.as_slice().unwrap()[flat];
                    result.as_slice_mut().unwrap()[flat] = acc;
                }
            }
        }

        Ok(CpuArray::from_ndarray(result))
    }

    fn cumprod_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();
        let ndim = shape.len();

        if axis >= ndim {
            return Err(RumpyError::InvalidAxis { axis, ndim });
        }

        let mut result = data.to_owned();
        let axis_len = shape[axis];
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = 1.0;
                for ax_idx in 0..axis_len {
                    let mut coords = vec![0usize; ndim];
                    let mut tmp = outer;
                    for d in (0..axis).rev() {
                        coords[d] = tmp % shape[d];
                        tmp /= shape[d];
                    }
                    coords[axis] = ax_idx;
                    let mut tmp = inner;
                    for d in (axis + 1..ndim).rev() {
                        coords[d] = tmp % shape[d];
                        tmp /= shape[d];
                    }
                    let mut flat = 0;
                    let mut mult = 1;
                    for d in (0..ndim).rev() {
                        flat += coords[d] * mult;
                        mult *= shape[d];
                    }
                    acc *= data.as_slice().unwrap()[flat];
                    result.as_slice_mut().unwrap()[flat] = acc;
                }
            }
        }

        Ok(CpuArray::from_ndarray(result))
    }

    fn all(arr: &CpuArray) -> bool {
        arr.as_ndarray().iter().all(|&x| x != 0.0)
    }

    fn any(arr: &CpuArray) -> bool {
        arr.as_ndarray().iter().any(|&x| x != 0.0)
    }

    fn all_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![1.0; result_size]; // Start with "all true"

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            if val == 0.0 {
                result[result_idx] = 0.0;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn any_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size]; // Start with "none true"

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            if val != 0.0 {
                result[result_idx] = 1.0;
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    // NaN-ignoring functions
    fn nansum(arr: &CpuArray) -> f64 {
        arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .sum()
    }

    fn nanmean(arr: &CpuArray) -> f64 {
        let data = arr.as_ndarray();
        let (sum, count) = data.iter()
            .filter(|x| !x.is_nan())
            .fold((0.0, 0usize), |(s, c), &x| (s + x, c + 1));
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }

    fn nanvar(arr: &CpuArray, ddof: usize) -> f64 {
        let mean = Self::nanmean(arr);
        if mean.is_nan() {
            return f64::NAN;
        }
        let data = arr.as_ndarray();
        let (sum_sq, count) = data.iter()
            .filter(|x| !x.is_nan())
            .fold((0.0, 0usize), |(s, c), &x| (s + (x - mean).powi(2), c + 1));
        if count <= ddof {
            f64::NAN
        } else {
            sum_sq / (count - ddof) as f64
        }
    }

    fn nanstd(arr: &CpuArray, ddof: usize) -> f64 {
        Self::nanvar(arr, ddof).sqrt()
    }

    fn nanmin(arr: &CpuArray) -> f64 {
        // NumPy returns NaN for empty or all-NaN arrays
        let filtered: Vec<f64> = arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();
        if filtered.is_empty() {
            f64::NAN
        } else {
            filtered.into_iter().fold(f64::INFINITY, f64::min)
        }
    }

    fn nanmax(arr: &CpuArray) -> f64 {
        // NumPy returns NaN for empty or all-NaN arrays
        let filtered: Vec<f64> = arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();
        if filtered.is_empty() {
            f64::NAN
        } else {
            filtered.into_iter().fold(f64::NEG_INFINITY, f64::max)
        }
    }

    fn diff(arr: &CpuArray, n: usize, axis: Option<usize>) -> Result<CpuArray> {
        if n == 0 {
            return Ok(arr.clone());
        }

        let data = arr.as_ndarray();
        let shape = data.shape();
        let ndim = shape.len();

        match axis {
            None => {
                // Flatten and compute diff
                let flat: Vec<f64> = data.iter().cloned().collect();
                if flat.len() <= n {
                    return Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()));
                }
                let mut result = flat;
                for _ in 0..n {
                    let new_len = result.len() - 1;
                    result = (0..new_len).map(|i| result[i + 1] - result[i]).collect();
                }
                Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap()))
            }
            Some(ax) => {
                if ax >= ndim {
                    return Err(RumpyError::InvalidAxis { axis: ax, ndim });
                }

                let axis_len = shape[ax];
                if axis_len <= n {
                    let mut new_shape = shape.to_vec();
                    new_shape[ax] = 0;
                    return Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&new_shape), vec![]).unwrap()));
                }

                let mut current = data.to_owned();
                for _ in 0..n {
                    let cur_shape = current.shape().to_vec();
                    let cur_axis_len = cur_shape[ax];
                    let mut new_shape = cur_shape.clone();
                    new_shape[ax] = cur_axis_len - 1;

                    let outer_size: usize = cur_shape[..ax].iter().product();
                    let inner_size: usize = cur_shape[ax + 1..].iter().product();
                    let mut result_data = Vec::with_capacity(new_shape.iter().product());

                    for outer in 0..outer_size {
                        for inner in 0..inner_size {
                            for i in 0..(cur_axis_len - 1) {
                                let flat_idx_cur = outer * (cur_axis_len * inner_size) + i * inner_size + inner;
                                let flat_idx_next = outer * (cur_axis_len * inner_size) + (i + 1) * inner_size + inner;
                                result_data.push(current.as_slice().unwrap()[flat_idx_next] - current.as_slice().unwrap()[flat_idx_cur]);
                            }
                        }
                    }

                    current = ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap();
                }

                Ok(CpuArray::from_ndarray(current))
            }
        }
    }

    fn percentile(arr: &CpuArray, q: f64) -> Result<f64> {
        // q is in range 0-100
        Self::quantile(arr, q / 100.0)
    }

    fn quantile(arr: &CpuArray, q: f64) -> Result<f64> {
        // q is in range 0-1
        if !(0.0..=1.0).contains(&q) {
            return Err(RumpyError::InvalidArgument(format!("Quantile must be between 0 and 1, got {}", q)));
        }

        let data = arr.as_ndarray();
        if data.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("quantile"));
        }

        let mut sorted: Vec<f64> = data.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n == 1 {
            return Ok(sorted[0]);
        }

        // Linear interpolation method (NumPy default)
        let pos = q * (n - 1) as f64;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;

        if idx >= n - 1 {
            Ok(sorted[n - 1])
        } else {
            Ok(sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac)
        }
    }

    fn median(arr: &CpuArray) -> Result<f64> {
        Self::quantile(arr, 0.5)
    }
}

// Additional NaN-aware stats methods (not part of trait but used by WASM bindings)
impl CpuBackend {
    /// Index of minimum value, ignoring NaN
    pub fn nanargmin(arr: &CpuArray) -> Result<usize> {
        let data = arr.as_ndarray();
        let mut min_val = f64::INFINITY;
        let mut min_idx = None;
        for (i, &x) in data.iter().enumerate() {
            if !x.is_nan() && x < min_val {
                min_val = x;
                min_idx = Some(i);
            }
        }
        min_idx.ok_or_else(|| RumpyError::EmptyArrayReduction("nanargmin"))
    }

    /// Index of maximum value, ignoring NaN
    pub fn nanargmax(arr: &CpuArray) -> Result<usize> {
        let data = arr.as_ndarray();
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = None;
        for (i, &x) in data.iter().enumerate() {
            if !x.is_nan() && x > max_val {
                max_val = x;
                max_idx = Some(i);
            }
        }
        max_idx.ok_or_else(|| RumpyError::EmptyArrayReduction("nanargmax"))
    }

    /// Product ignoring NaN values
    pub fn nanprod(arr: &CpuArray) -> f64 {
        arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .product()
    }

    /// Median ignoring NaN values
    pub fn nanmedian(arr: &CpuArray) -> Result<f64> {
        let mut filtered: Vec<f64> = arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();
        if filtered.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("nanmedian"));
        }
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = filtered.len();
        if n % 2 == 0 {
            Ok((filtered[n / 2 - 1] + filtered[n / 2]) / 2.0)
        } else {
            Ok(filtered[n / 2])
        }
    }

    /// Percentile ignoring NaN values (q in range 0-100)
    pub fn nanpercentile(arr: &CpuArray, q: f64) -> Result<f64> {
        if !(0.0..=100.0).contains(&q) {
            return Err(RumpyError::InvalidArgument(
                format!("Percentile must be in range [0, 100], got {}", q)
            ));
        }
        let mut filtered: Vec<f64> = arr.as_ndarray()
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();
        if filtered.is_empty() {
            return Err(RumpyError::EmptyArrayReduction("nanpercentile"));
        }
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = filtered.len();
        let idx = (q / 100.0) * (n - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        if lower == upper {
            Ok(filtered[lower])
        } else {
            let frac = idx - lower as f64;
            Ok(filtered[lower] * (1.0 - frac) + filtered[upper] * frac)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_sum() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::sum(&a), 15.0);
    }

    #[test]
    fn test_prod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(CpuBackend::prod(&a), 24.0);
    }

    #[test]
    fn test_mean() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::mean(&a), 3.0);
    }

    #[test]
    fn test_var() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(approx_eq(CpuBackend::var(&a), 2.0));
    }

    #[test]
    fn test_std() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(approx_eq(CpuBackend::std(&a), 2.0_f64.sqrt()));
    }

    #[test]
    fn test_min_max() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::min(&a).unwrap(), 1.0);
        assert_eq!(CpuBackend::max(&a).unwrap(), 9.0);
    }

    #[test]
    fn test_min_max_empty() {
        // NumPy raises ValueError for empty arrays
        let empty = CpuArray::from_f64_vec(vec![], vec![0]).unwrap();
        assert!(CpuBackend::min(&empty).is_err());
        assert!(CpuBackend::max(&empty).is_err());
    }

    #[test]
    fn test_argmin_argmax() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::argmin(&a).unwrap(), 1);
        assert_eq!(CpuBackend::argmax(&a).unwrap(), 5);
    }

    #[test]
    fn test_argmin_argmax_empty() {
        // NumPy raises ValueError for empty arrays
        let empty = CpuArray::from_f64_vec(vec![], vec![0]).unwrap();
        assert!(CpuBackend::argmin(&empty).is_err());
        assert!(CpuBackend::argmax(&empty).is_err());
    }

    #[test]
    fn test_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]]
        let m = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        // Sum along axis 0 (rows) -> [5, 7, 9]
        let result = CpuBackend::sum_axis(&m, 0).unwrap();
        assert_eq!(result.as_f64_slice(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1 (cols) -> [6, 15]
        let result = CpuBackend::sum_axis(&m, 1).unwrap();
        assert_eq!(result.as_f64_slice(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_min_axis_nan_propagation() {
        // Test that NaN propagates in min_axis (NumPy behavior)
        // Matrix: [[1, NaN, 3], [4, 5, 6]]
        // Along axis 0 (reduce rows):
        // - col 0: min(1, 4) = 1
        // - col 1: min(NaN, 5) = NaN (NaN propagates)
        // - col 2: min(3, 6) = 3
        let m = mat(vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::min_axis(&m, 0).unwrap();
        let data = result.as_f64_slice();
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_nan()); // NaN propagates
        assert_eq!(data[2], 3.0);
    }

    #[test]
    fn test_max_axis_nan_propagation() {
        // Matrix: [[1, NaN, 3], [4, 5, 6]]
        // Along axis 0 (reduce rows):
        // - col 0: max(1, 4) = 4
        // - col 1: max(NaN, 5) = NaN (NaN propagates)
        // - col 2: max(3, 6) = 6
        let m = mat(vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::max_axis(&m, 0).unwrap();
        let data = result.as_f64_slice();
        assert_eq!(data[0], 4.0);
        assert!(data[1].is_nan()); // NaN propagates
        assert_eq!(data[2], 6.0);
    }

    #[test]
    fn test_prod_axis() {
        let m = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::prod_axis(&m, 0).unwrap();
        assert_eq!(result.as_f64_slice(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_var_axis() {
        let m = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = CpuBackend::var_axis(&m, 0, 0).unwrap();
        // var([1,3]) = 1.0, var([2,4]) = 1.0
        assert!(approx_eq(result.as_f64_slice()[0], 1.0));
        assert!(approx_eq(result.as_f64_slice()[1], 1.0));
    }

    #[test]
    fn test_argmin_axis() {
        let m = mat(vec![3.0, 1.0, 2.0, 4.0], 2, 2);
        let result = CpuBackend::argmin_axis(&m, 0).unwrap();
        // argmin along axis 0: col 0 has [3,2], min at row 1; col 1 has [1,4], min at row 0
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0]);
    }

    #[test]
    fn test_argmax_axis() {
        let m = mat(vec![3.0, 1.0, 2.0, 4.0], 2, 2);
        let result = CpuBackend::argmax_axis(&m, 0).unwrap();
        // argmax along axis 0: col 0 has [3,2], max at row 0; col 1 has [1,4], max at row 1
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_cumsum() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumsum(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumprod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumprod(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cumsum_axis() {
        let m = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = CpuBackend::cumsum_axis(&m, 1).unwrap();
        // cumsum along axis 1 (cols): [[1,3,6], [4,9,15]]
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_all_any() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        assert!(CpuBackend::all(&a));
        assert!(CpuBackend::any(&a));

        let b = arr(vec![0.0, 0.0, 0.0]);
        assert!(!CpuBackend::all(&b));
        assert!(!CpuBackend::any(&b));

        let c = arr(vec![1.0, 0.0, 1.0]);
        assert!(!CpuBackend::all(&c));
        assert!(CpuBackend::any(&c));
    }

    #[test]
    fn test_all_axis() {
        let m = mat(vec![1.0, 0.0, 1.0, 1.0], 2, 2);
        let result = CpuBackend::all_axis(&m, 0).unwrap();
        // all along axis 0: col 0 = [1,1] -> 1, col 1 = [0,1] -> 0
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0]);
    }

    #[test]
    fn test_any_axis() {
        let m = mat(vec![0.0, 0.0, 1.0, 0.0], 2, 2);
        let result = CpuBackend::any_axis(&m, 0).unwrap();
        // any along axis 0: col 0 = [0,1] -> 1, col 1 = [0,0] -> 0
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0]);
    }

    #[test]
    fn test_nansum() {
        let a = arr(vec![1.0, f64::NAN, 3.0, 4.0]);
        assert_eq!(CpuBackend::nansum(&a), 8.0);
    }

    #[test]
    fn test_nanmean() {
        let a = arr(vec![1.0, f64::NAN, 3.0, 4.0]);
        assert!(approx_eq(CpuBackend::nanmean(&a), 8.0 / 3.0));
    }

    #[test]
    fn test_nanmin_nanmax() {
        let a = arr(vec![3.0, f64::NAN, 1.0, 5.0]);
        assert_eq!(CpuBackend::nanmin(&a), 1.0);
        assert_eq!(CpuBackend::nanmax(&a), 5.0);
    }

    #[test]
    fn test_nanmin_all_nan() {
        // NumPy returns NaN (with warning) for all-NaN arrays
        let a = arr(vec![f64::NAN, f64::NAN, f64::NAN]);
        assert!(CpuBackend::nanmin(&a).is_nan());
        assert!(CpuBackend::nanmax(&a).is_nan());
    }

    #[test]
    fn test_nanmin_empty() {
        // NumPy returns NaN for empty arrays
        let a = CpuArray::from_f64_vec(vec![], vec![0]).unwrap();
        assert!(CpuBackend::nanmin(&a).is_nan());
        assert!(CpuBackend::nanmax(&a).is_nan());
    }

    #[test]
    fn test_diff_1d() {
        let a = arr(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
        let result = CpuBackend::diff(&a, 1, None).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 2.0, 3.0, -7.0]);
    }

    #[test]
    fn test_diff_n2() {
        let a = arr(vec![1.0, 2.0, 4.0, 7.0, 0.0]);
        let result = CpuBackend::diff(&a, 2, None).unwrap();
        // First diff: [1, 2, 3, -7], second diff: [1, 1, -10]
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, -10.0]);
    }

    #[test]
    fn test_percentile() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // 50th percentile = median = 3.0
        let p50 = CpuBackend::percentile(&a, 50.0).unwrap();
        assert!(approx_eq(p50, 3.0));
        // 0th percentile = min = 1.0
        let p0 = CpuBackend::percentile(&a, 0.0).unwrap();
        assert!(approx_eq(p0, 1.0));
        // 100th percentile = max = 5.0
        let p100 = CpuBackend::percentile(&a, 100.0).unwrap();
        assert!(approx_eq(p100, 5.0));
    }

    #[test]
    fn test_quantile() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let q50 = CpuBackend::quantile(&a, 0.5).unwrap();
        assert!(approx_eq(q50, 3.0));
    }

    #[test]
    fn test_median() {
        let a = arr(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        let m = CpuBackend::median(&a).unwrap();
        assert!(approx_eq(m, 3.0));

        // Even number of elements: interpolation
        let b = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = CpuBackend::median(&b).unwrap();
        assert!(approx_eq(m2, 2.5));
    }

    #[test]
    fn test_median_empty() {
        let a = CpuArray::from_f64_vec(vec![], vec![0]).unwrap();
        assert!(CpuBackend::median(&a).is_err());
    }
}
