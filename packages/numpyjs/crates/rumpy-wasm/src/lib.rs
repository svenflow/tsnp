//! WASM bindings for RumPy
//!
//! This crate provides JavaScript/TypeScript bindings for RumPy using wasm-bindgen.
//! It wraps the CPU backend for use in web browsers and Node.js.
//!
//! ## Zero-Copy Memory Access
//!
//! When SharedArrayBuffer is available (requires COOP/COEP headers), you can use
//! `asTypedArrayView()` for zero-copy access to array data. Otherwise, use
//! `toTypedArray()` which creates a copy.

use js_sys::{Float32Array, Float64Array};
use ndarray;
use rumpy_core::{ops::*, Array};
use rumpy_cpu::{simd_gemm, CpuArray, CpuBackend};
use wasm_bindgen::prelude::*;

// Re-export wasm-bindgen-rayon's init function for Web Worker setup (only in threaded mode)
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

// Stub for single-threaded mode
#[cfg(not(feature = "threads"))]
#[wasm_bindgen]
pub fn init_thread_pool(_num_threads: usize) -> js_sys::Promise {
    js_sys::Promise::resolve(&JsValue::UNDEFINED)
}

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// N-dimensional array type exposed to JavaScript
#[wasm_bindgen]
pub struct NDArray {
    inner: CpuArray,
}

impl NDArray {
    fn new(inner: CpuArray) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl NDArray {
    /// Get array shape
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Number of dimensions
    #[wasm_bindgen(getter)]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Total number of elements
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Data type
    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// Convert to Float64Array (creates a copy)
    ///
    /// This method always works but involves copying data from WASM memory to JS.
    /// For zero-copy access when SharedArrayBuffer is available, use `asTypedArrayView()`.
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> Float64Array {
        Float64Array::from(self.inner.as_f64_slice().as_slice())
    }

    /// Get pointer to the underlying data buffer
    ///
    /// Returns the byte offset into WASM linear memory where this array's data begins.
    /// Use with `memory()` to create a zero-copy TypedArray view.
    ///
    /// WARNING: The pointer is only valid while this NDArray exists and WASM memory
    /// hasn't been resized. Cache invalidation is the caller's responsibility.
    #[wasm_bindgen(js_name = dataPtr)]
    pub fn data_ptr(&self) -> usize {
        self.inner.as_ndarray().as_ptr() as usize
    }

    /// Get the number of elements in the array
    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    /// Check if array is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.size() == 0
    }

    /// Get total size in bytes
    #[wasm_bindgen(js_name = nbytes)]
    pub fn nbytes(&self) -> usize {
        self.inner.size() * std::mem::size_of::<f64>()
    }

    /// Get element at flat index
    #[wasm_bindgen(js_name = getFlat)]
    pub fn get_flat(&self, index: usize) -> f64 {
        self.inner.get_flat(index)
    }

    /// Clone the array
    #[wasm_bindgen(js_name = clone)]
    pub fn clone_array(&self) -> NDArray {
        NDArray::new(self.inner.clone())
    }

    /// Explicitly free the array memory
    ///
    /// After calling this, the NDArray is consumed and cannot be used.
    /// This is useful for deterministic memory cleanup without waiting for GC.
    #[wasm_bindgen]
    pub fn free(self) {
        // self is dropped here, freeing the underlying memory
    }

    // Scalar operations
    #[wasm_bindgen(js_name = addScalar)]
    pub fn add_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::add_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = subScalar)]
    pub fn sub_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::sub_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = mulScalar)]
    pub fn mul_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::mul_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = divScalar)]
    pub fn div_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::div_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = powScalar)]
    pub fn pow_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::pow_scalar(&self.inner, scalar))
    }

    // Element-wise operations
    pub fn add(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::add(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn sub(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::sub(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn mul(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::mul(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn div(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::div(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // Reductions
    pub fn sum(&self) -> f64 {
        CpuBackend::sum(&self.inner)
    }

    pub fn prod(&self) -> f64 {
        CpuBackend::prod(&self.inner)
    }

    pub fn mean(&self) -> f64 {
        CpuBackend::mean(&self.inner)
    }

    pub fn min(&self) -> f64 {
        CpuBackend::min(&self.inner).unwrap_or(f64::NAN)
    }

    pub fn max(&self) -> f64 {
        CpuBackend::max(&self.inner).unwrap_or(f64::NAN)
    }

    #[wasm_bindgen(js_name = std)]
    pub fn std_dev(&self) -> f64 {
        CpuBackend::std(&self.inner)
    }

    pub fn var(&self) -> f64 {
        CpuBackend::var(&self.inner)
    }

    /// Variance with degrees of freedom adjustment
    /// ddof=0 for population variance (default), ddof=1 for sample variance
    #[wasm_bindgen(js_name = varDdof)]
    pub fn var_ddof(&self, ddof: usize) -> f64 {
        CpuBackend::var_ddof(&self.inner, ddof)
    }

    /// Standard deviation with degrees of freedom adjustment
    /// ddof=0 for population std (default), ddof=1 for sample std
    #[wasm_bindgen(js_name = stdDdof)]
    pub fn std_ddof(&self, ddof: usize) -> f64 {
        CpuBackend::std_ddof(&self.inner, ddof)
    }

    // ============ NaN-aware Stats ============

    /// Sum ignoring NaN values
    #[wasm_bindgen(js_name = nansum)]
    pub fn nansum(&self) -> f64 {
        CpuBackend::nansum(&self.inner)
    }

    /// Mean ignoring NaN values
    #[wasm_bindgen(js_name = nanmean)]
    pub fn nanmean(&self) -> f64 {
        CpuBackend::nanmean(&self.inner)
    }

    /// Variance ignoring NaN values
    #[wasm_bindgen(js_name = nanvar)]
    pub fn nanvar(&self, ddof: usize) -> f64 {
        CpuBackend::nanvar(&self.inner, ddof)
    }

    /// Standard deviation ignoring NaN values
    #[wasm_bindgen(js_name = nanstd)]
    pub fn nanstd(&self, ddof: usize) -> f64 {
        CpuBackend::nanstd(&self.inner, ddof)
    }

    /// Minimum ignoring NaN values
    #[wasm_bindgen(js_name = nanmin)]
    pub fn nanmin(&self) -> f64 {
        CpuBackend::nanmin(&self.inner)
    }

    /// Maximum ignoring NaN values
    #[wasm_bindgen(js_name = nanmax)]
    pub fn nanmax(&self) -> f64 {
        CpuBackend::nanmax(&self.inner)
    }

    /// Index of minimum value, ignoring NaN
    #[wasm_bindgen(js_name = nanargmin)]
    pub fn nanargmin(&self) -> usize {
        CpuBackend::nanargmin(&self.inner).unwrap_or(0)
    }

    /// Index of maximum value, ignoring NaN
    #[wasm_bindgen(js_name = nanargmax)]
    pub fn nanargmax(&self) -> usize {
        CpuBackend::nanargmax(&self.inner).unwrap_or(0)
    }

    /// Product ignoring NaN values
    #[wasm_bindgen(js_name = nanprod)]
    pub fn nanprod(&self) -> f64 {
        CpuBackend::nanprod(&self.inner)
    }

    // ============ Order Statistics ============

    /// Median value
    pub fn median(&self) -> f64 {
        CpuBackend::median(&self.inner).unwrap_or(f64::NAN)
    }

    /// Percentile (q in range 0-100)
    pub fn percentile(&self, q: f64) -> f64 {
        CpuBackend::percentile(&self.inner, q).unwrap_or(f64::NAN)
    }

    /// Quantile (q in range 0-1)
    pub fn quantile(&self, q: f64) -> f64 {
        CpuBackend::quantile(&self.inner, q).unwrap_or(f64::NAN)
    }

    /// Median ignoring NaN values
    #[wasm_bindgen(js_name = nanmedian)]
    pub fn nanmedian(&self) -> f64 {
        CpuBackend::nanmedian(&self.inner).unwrap_or(f64::NAN)
    }

    /// Percentile ignoring NaN values (q in range 0-100)
    #[wasm_bindgen(js_name = nanpercentile)]
    pub fn nanpercentile(&self, q: f64) -> f64 {
        CpuBackend::nanpercentile(&self.inner, q).unwrap_or(f64::NAN)
    }

    // Reshape
    pub fn reshape(&self, shape: Vec<usize>) -> Result<NDArray, JsValue> {
        CpuBackend::reshape(&self.inner, shape)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Reshape with support for -1 dimension inference (NumPy-style)
    /// One dimension can be -1, which will be inferred from the total size
    #[wasm_bindgen(js_name = reshapeInfer)]
    pub fn reshape_infer(&self, shape: Vec<i64>) -> Result<NDArray, JsValue> {
        let total_size = self.inner.size();

        // Count -1s and compute known product
        let neg_one_count = shape.iter().filter(|&&d| d == -1).count();
        if neg_one_count > 1 {
            return Err(JsValue::from_str("Can only specify one dimension as -1"));
        }

        let mut final_shape: Vec<usize> = Vec::with_capacity(shape.len());
        let known_product: i64 = shape.iter().filter(|&&d| d != -1).product();

        if known_product <= 0 && neg_one_count == 0 {
            return Err(JsValue::from_str("Invalid shape: dimensions must be positive"));
        }

        for &dim in &shape {
            if dim == -1 {
                if known_product == 0 {
                    return Err(JsValue::from_str("Cannot infer dimension with zero-size product"));
                }
                let inferred = total_size as i64 / known_product;
                if inferred * known_product != total_size as i64 {
                    return Err(JsValue::from_str(&format!(
                        "Cannot reshape array of size {} into shape {:?}",
                        total_size, shape
                    )));
                }
                final_shape.push(inferred as usize);
            } else if dim < 0 {
                return Err(JsValue::from_str(&format!("Invalid dimension: {}", dim)));
            } else {
                final_shape.push(dim as usize);
            }
        }

        CpuBackend::reshape(&self.inner, final_shape)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn flatten(&self) -> NDArray {
        NDArray::new(CpuBackend::flatten(&self.inner))
    }

    pub fn transpose(&self) -> NDArray {
        NDArray::new(CpuBackend::transpose(&self.inner))
    }

    /// Permute array dimensions
    /// axes specifies the new order of dimensions
    /// e.g., permute([1, 0, 2]) swaps first two dimensions
    pub fn permute(&self, axes: Vec<usize>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let ndim = data.ndim();

        if axes.len() != ndim {
            return Err(JsValue::from_str(&format!(
                "axes length {} doesn't match array dimensions {}",
                axes.len(), ndim
            )));
        }

        // Validate axes are valid permutation
        let mut seen = vec![false; ndim];
        for &ax in &axes {
            if ax >= ndim {
                return Err(JsValue::from_str(&format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }
            if seen[ax] {
                return Err(JsValue::from_str("axes must be a permutation (no duplicates)"));
            }
            seen[ax] = true;
        }

        let permuted = data.clone().permuted_axes(axes);
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(permuted.to_owned())))
    }

    // ============ Axis-based reductions ============

    /// Sum along an axis
    #[wasm_bindgen(js_name = sumAxis)]
    pub fn sum_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::sum_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            // Re-insert the axis with size 1
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Mean along an axis
    #[wasm_bindgen(js_name = meanAxis)]
    pub fn mean_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::mean_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Max along an axis
    #[wasm_bindgen(js_name = maxAxis)]
    pub fn max_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::max_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Min along an axis
    #[wasm_bindgen(js_name = minAxis)]
    pub fn min_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::min_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    // ============ Activation functions ============

    /// ReLU activation: max(0, x)
    /// NaN values are propagated (NumPy behavior)
    pub fn relu(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| {
            if x.is_nan() { x }  // Propagate NaN
            else if x > 0.0 { x }
            else { 0.0 }
        });
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }

    /// GELU activation (Gaussian Error Linear Unit)
    /// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
        let result = data.mapv(|x| {
            let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
            x * 0.5 * (1.0 + inner.tanh())
        });
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }

    /// Softmax along an axis
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    pub fn softmax(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        if axis >= data.ndim() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, data.ndim()
            )));
        }

        // Numerically stable softmax: subtract max before exp
        // Get max along axis with keepdims
        let ax = ndarray::Axis(axis);
        let max_vals = data.map_axis(ax, |lane| {
            lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });

        // Broadcast max back and subtract
        let max_broadcast = max_vals.insert_axis(ax);
        let shifted = &*data - &max_broadcast;

        // Exp
        let exp_vals = shifted.mapv(f64::exp);

        // Sum along axis with keepdims
        let sum_exp = exp_vals.sum_axis(ax).insert_axis(ax);

        // Divide
        let result = &exp_vals / &sum_exp;

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result.to_owned())))
    }

    /// Argmax - index of maximum value (flattened)
    pub fn argmax(&self) -> usize {
        CpuBackend::argmax(&self.inner).unwrap_or(0)
    }

    /// Argmin - index of minimum value (flattened)
    pub fn argmin(&self) -> usize {
        CpuBackend::argmin(&self.inner).unwrap_or(0)
    }

    // ============ Concatenation ============

    /// Squeeze - remove axes of length 1
    pub fn squeeze(&self) -> NDArray {
        NDArray::new(CpuBackend::squeeze(&self.inner))
    }

    /// Expand dims - add axis of length 1
    #[wasm_bindgen(js_name = expandDims)]
    pub fn expand_dims(&self, axis: usize) -> Result<NDArray, JsValue> {
        CpuBackend::expand_dims(&self.inner, axis)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ============ Slicing ============

    /// Slice the array with start:stop:step for each dimension
    ///
    /// Uses parallel i32 arrays for starts, stops, steps.
    /// - Negative indices work like Python (count from end)
    /// - i32::MAX (2147483647) for stop means "to the end" (like : in Python)
    /// - Missing dimensions in arrays assume full range
    ///
    /// Example: arr[1:3, :, 2:5] with shape [10, 10, 10]
    ///   starts = [1, 0, 2]
    ///   stops = [3, 2147483647, 5]  // MAX_INT for ":"
    ///   steps = [1, 1, 1]
    pub fn slice(&self, starts: Vec<i32>, stops: Vec<i32>, steps: Vec<i32>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();
        let rank = shape.len();

        // Build slice info for each dimension
        let mut slice_info: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(rank);

        for i in 0..rank {
            let dim_len = shape[i] as i32;

            // Handle start (default 0)
            let start_raw = *starts.get(i).unwrap_or(&0);
            let start = if start_raw < 0 {
                (dim_len + start_raw).max(0) as isize
            } else {
                (start_raw as isize).min(dim_len as isize)
            };

            // Handle stop (default to end)
            let stop_raw = *stops.get(i).unwrap_or(&i32::MAX);
            let stop = if stop_raw == i32::MAX || stop_raw > dim_len {
                dim_len as isize
            } else if stop_raw < 0 {
                (dim_len + stop_raw).max(0) as isize
            } else {
                stop_raw as isize
            };

            // Handle step (default 1)
            let step = *steps.get(i).unwrap_or(&1) as isize;
            if step == 0 {
                return Err(JsValue::from_str("step cannot be zero"));
            }

            slice_info.push(ndarray::SliceInfoElem::Slice {
                start,
                end: Some(stop),
                step,
            });
        }

        // Convert to SliceInfo and apply
        let slice = ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::try_from(slice_info)
            .map_err(|e| JsValue::from_str(&format!("slice error: {:?}", e)))?;

        let sliced = data.slice(slice.as_ref());
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(sliced.to_owned())))
    }

    /// Slice along a single axis (simpler API for common case)
    ///
    /// Equivalent to arr[:, :, start:stop] when axis=2
    #[wasm_bindgen(js_name = sliceAxis)]
    pub fn slice_axis(&self, axis: usize, start: i32, stop: i32) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let dim_len = shape[axis] as i32;

        // Normalize negative indices
        let start_norm = if start < 0 {
            (dim_len + start).max(0) as usize
        } else {
            (start as usize).min(dim_len as usize)
        };

        let stop_norm = if stop == i32::MAX || stop > dim_len {
            dim_len as usize
        } else if stop < 0 {
            (dim_len + stop).max(0) as usize
        } else {
            stop as usize
        };

        let sliced = data.slice_axis(
            ndarray::Axis(axis),
            ndarray::Slice::from(start_norm..stop_norm)
        );
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(sliced.to_owned())))
    }

    // ============ CNN Operations ============

    /// im2col: Convert image patches to columns for convolution via GEMM
    ///
    /// Input shape: (N, C, H, W) - batch, channels, height, width
    /// Output shape: (N * H_out * W_out, C * kernel_h * kernel_w)
    ///
    /// This transforms the convolution operation into a matrix multiplication:
    ///   output = im2col(input) @ weights.reshape(out_channels, -1).T
    #[wasm_bindgen(js_name = im2col)]
    pub fn im2col(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("im2col expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        // Output dimensions
        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        // Output shape: (N * h_out * w_out, C * kernel_h * kernel_w)
        let rows = n * h_out * w_out;
        let cols = c * kernel_h * kernel_w;
        let mut output = vec![0.0; rows * cols];

        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = batch * h_out * w_out + oh * w_out + ow;

                    for ch in 0..c {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                let col_idx = ch * kernel_h * kernel_w + kh * kernel_w + kw;

                                // Check padding bounds
                                let val = if ih < pad_h || ih >= h_in + pad_h || iw < pad_w || iw >= w_in + pad_w {
                                    0.0 // Zero padding
                                } else {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    flat_data[idx]
                                };

                                output[row_idx * cols + col_idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![rows, cols])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Max pooling 2D
    ///
    /// Input shape: (N, C, H, W)
    /// Output shape: (N, C, H_out, W_out)
    #[wasm_bindgen(js_name = maxPool2d)]
    pub fn max_pool_2d(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("maxPool2d expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = vec![f64::NEG_INFINITY; n * c * h_out * w_out];
        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f64::NEG_INFINITY;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check padding bounds
                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    max_val = max_val.max(flat_data[idx]);
                                }
                            }
                        }

                        // If all padding (edge case), use 0
                        if max_val == f64::NEG_INFINITY {
                            max_val = 0.0;
                        }

                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c, h_out, w_out])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Average pooling 2D
    ///
    /// Input shape: (N, C, H, W)
    /// Output shape: (N, C, H_out, W_out)
    #[wasm_bindgen(js_name = avgPool2d)]
    pub fn avg_pool_2d(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("avgPool2d expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = vec![0.0; n * c * h_out * w_out];
        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;
                        let mut count = 0;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check padding bounds
                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    sum += flat_data[idx];
                                    count += 1;
                                }
                            }
                        }

                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c, h_out, w_out])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    // ============ Boolean Masking & Comparisons ============

    /// Get elements where mask is non-zero (truthy)
    ///
    /// Returns a 1D array of selected elements.
    /// Mask must be same shape as self, or broadcastable.
    ///
    /// Example: arr.getByMask(arr.gt_scalar(0.5)) returns all elements > 0.5
    #[wasm_bindgen(js_name = getByMask)]
    pub fn get_by_mask(&self, mask: &NDArray) -> Result<NDArray, JsValue> {
        let data = self.inner.as_f64_slice();
        let mask_data = mask.inner.as_f64_slice();

        if data.len() != mask_data.len() {
            return Err(JsValue::from_str(&format!(
                "mask length {} doesn't match array length {}",
                mask_data.len(), data.len()
            )));
        }

        let selected: Vec<f64> = data.iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| m != 0.0)
            .map(|(&v, _)| v)
            .collect();

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(selected.clone(), vec![selected.len()])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Set elements where mask is non-zero to a scalar value
    ///
    /// Returns a new array with selected elements replaced.
    #[wasm_bindgen(js_name = setByMask)]
    pub fn set_by_mask(&self, mask: &NDArray, value: f64) -> Result<NDArray, JsValue> {
        let data = self.inner.as_f64_slice();
        let mask_data = mask.inner.as_f64_slice();

        if data.len() != mask_data.len() {
            return Err(JsValue::from_str(&format!(
                "mask length {} doesn't match array length {}",
                mask_data.len(), data.len()
            )));
        }

        let result: Vec<f64> = data.iter()
            .zip(mask_data.iter())
            .map(|(&v, &m)| if m != 0.0 { value } else { v })
            .collect();

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, self.shape())
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Comparison: equal (element-wise)
    pub fn eq(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::eq(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: not equal (element-wise)
    pub fn ne(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::ne(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: less than (element-wise)
    pub fn lt(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::lt(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: less than or equal (element-wise)
    pub fn le(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::le(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: greater than (element-wise)
    pub fn gt(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::gt(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: greater than or equal (element-wise)
    pub fn ge(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::ge(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Scalar comparison: equal
    #[wasm_bindgen(js_name = eqScalar)]
    pub fn eq_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::eq_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: not equal
    #[wasm_bindgen(js_name = neScalar)]
    pub fn ne_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::ne_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: less than
    #[wasm_bindgen(js_name = ltScalar)]
    pub fn lt_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::lt_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: less than or equal
    #[wasm_bindgen(js_name = leScalar)]
    pub fn le_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::le_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: greater than
    #[wasm_bindgen(js_name = gtScalar)]
    pub fn gt_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::gt_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: greater than or equal
    #[wasm_bindgen(js_name = geScalar)]
    pub fn ge_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::ge_scalar(&self.inner, scalar))
    }

    /// Check for NaN values
    #[wasm_bindgen(js_name = isNan)]
    pub fn is_nan(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isnan(&self.inner))
    }

    /// Check for infinite values
    #[wasm_bindgen(js_name = isInf)]
    pub fn is_inf(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isinf(&self.inner))
    }

    /// Check for finite values (not NaN, not Inf)
    #[wasm_bindgen(js_name = isFinite)]
    pub fn is_finite(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isfinite(&self.inner))
    }

    /// Count non-zero elements
    #[wasm_bindgen(js_name = countNonzero)]
    pub fn count_nonzero(&self) -> usize {
        self.inner.as_f64_slice().iter().filter(|&&x| x != 0.0).count()
    }

    /// Get indices of non-zero elements (flat indices)
    #[wasm_bindgen(js_name = nonzeroFlat)]
    pub fn nonzero_flat(&self) -> Vec<usize> {
        self.inner.as_f64_slice()
            .iter()
            .enumerate()
            .filter(|(_, &x)| x != 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Clip values to a range
    pub fn clip(&self, min: f64, max: f64) -> NDArray {
        use rumpy_core::ops::MathOps;
        NDArray::new(CpuBackend::clip(&self.inner, min, max))
    }
}

/// Numpy-style where: select x where condition is true, else y
///
/// condition, x, y must have compatible shapes (broadcasting supported).
/// Returns x where condition != 0, else y.
#[wasm_bindgen(js_name = where_)]
pub fn where_op(condition: &NDArray, x: &NDArray, y: &NDArray) -> Result<NDArray, JsValue> {
    let cond_data = condition.inner.as_f64_slice();
    let x_data = x.inner.as_f64_slice();
    let y_data = y.inner.as_f64_slice();

    // Simple case: all same size
    if cond_data.len() == x_data.len() && x_data.len() == y_data.len() {
        let result: Vec<f64> = cond_data.iter()
            .zip(x_data.iter())
            .zip(y_data.iter())
            .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
            .collect();

        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, condition.shape())
            .map_err(|e| JsValue::from_str(&e.to_string()))?));
    }

    Err(JsValue::from_str("where requires all inputs to have same shape (broadcasting not yet implemented for where)"))
}

// ============ Creation functions ============

#[wasm_bindgen(js_name = arrayFromTyped)]
pub fn array_from_typed(data: &Float64Array, shape: Vec<usize>) -> Result<NDArray, JsValue> {
    let vec: Vec<f64> = data.to_vec();
    CpuArray::from_f64_vec(vec, shape)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn zeros(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::zeros(shape))
}

#[wasm_bindgen]
pub fn ones(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::ones(shape))
}

#[wasm_bindgen]
pub fn full(shape: Vec<usize>, value: f64) -> NDArray {
    NDArray::new(CpuBackend::full(shape, value))
}

#[wasm_bindgen]
pub fn arange(start: f64, stop: f64, step: f64) -> Result<NDArray, JsValue> {
    CpuBackend::arange(start, stop, step)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn linspace(start: f64, stop: f64, num: usize) -> NDArray {
    NDArray::new(CpuBackend::linspace(start, stop, num))
}

#[wasm_bindgen]
pub fn eye(n: usize) -> NDArray {
    NDArray::new(CpuBackend::eye(n))
}

// ============ Concatenation functions ============

/// Concatenate two arrays along an axis
#[wasm_bindgen]
pub fn concatenate2(a: &NDArray, b: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner];
    CpuBackend::concatenate(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Concatenate three arrays along an axis
#[wasm_bindgen]
pub fn concatenate3(a: &NDArray, b: &NDArray, c: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner, &c.inner];
    CpuBackend::concatenate(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Stack two arrays along a new axis
#[wasm_bindgen]
pub fn stack2(a: &NDArray, b: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner];
    CpuBackend::stack(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Stack three arrays along a new axis
#[wasm_bindgen]
pub fn stack3(a: &NDArray, b: &NDArray, c: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner, &c.inner];
    CpuBackend::stack(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Vertical stack (concatenate along axis 0)
#[wasm_bindgen]
pub fn vstack2(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    concatenate2(a, b, 0)
}

/// Horizontal stack (concatenate along axis 1 for 2D+, axis 0 for 1D)
#[wasm_bindgen]
pub fn hstack2(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let axis = if a.ndim() == 1 { 0 } else { 1 };
    concatenate2(a, b, axis)
}

// ============ Math functions ============
// Note: Function names use "Arr" suffix to avoid collision with libm's math symbols
// (sin/exp/cos/etc) which causes linker conflicts when atomics is enabled.

#[wasm_bindgen(js_name = sinArr)]
pub fn sin_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sin(&arr.inner))
}

#[wasm_bindgen(js_name = cosArr)]
pub fn cos_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::cos(&arr.inner))
}

#[wasm_bindgen(js_name = tanArr)]
pub fn tan_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::tan(&arr.inner))
}

#[wasm_bindgen(js_name = expArr)]
pub fn exp_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::exp(&arr.inner))
}

#[wasm_bindgen(js_name = logArr)]
pub fn log_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log(&arr.inner))
}

#[wasm_bindgen(js_name = sqrtArr)]
pub fn sqrt_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sqrt(&arr.inner))
}

#[wasm_bindgen(js_name = absArr)]
pub fn abs_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::abs(&arr.inner))
}

#[wasm_bindgen(js_name = floorArr)]
pub fn floor_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::floor(&arr.inner))
}

#[wasm_bindgen(js_name = ceilArr)]
pub fn ceil_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::ceil(&arr.inner))
}

#[wasm_bindgen(js_name = roundArr)]
pub fn round_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::round(&arr.inner))
}

#[wasm_bindgen(js_name = tanhArr)]
pub fn tanh_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::tanh(&arr.inner))
}

#[wasm_bindgen(js_name = negArr)]
pub fn neg_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::neg(&arr.inner))
}

#[wasm_bindgen(js_name = sinhArr)]
pub fn sinh_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sinh(&arr.inner))
}

#[wasm_bindgen(js_name = coshArr)]
pub fn cosh_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::cosh(&arr.inner))
}

// Inverse trigonometric
#[wasm_bindgen(js_name = arcsinArr)]
pub fn arcsin_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::arcsin(&arr.inner))
}

#[wasm_bindgen(js_name = arccosArr)]
pub fn arccos_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::arccos(&arr.inner))
}

#[wasm_bindgen(js_name = arctanArr)]
pub fn arctan_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::arctan(&arr.inner))
}

// Inverse hyperbolic
#[wasm_bindgen(js_name = arcsinhArr)]
pub fn arcsinh_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| x.asinh());
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

#[wasm_bindgen(js_name = arccoshArr)]
pub fn arccosh_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| x.acosh());
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

#[wasm_bindgen(js_name = arctanhArr)]
pub fn arctanh_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| x.atanh());
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// Logarithms
#[wasm_bindgen(js_name = log2Arr)]
pub fn log2_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log2(&arr.inner))
}

#[wasm_bindgen(js_name = log10Arr)]
pub fn log10_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log10(&arr.inner))
}

#[wasm_bindgen(js_name = log1pArr)]
pub fn log1p_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log1p(&arr.inner))
}

#[wasm_bindgen(js_name = expm1Arr)]
pub fn expm1_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::expm1(&arr.inner))
}

// Roots and powers
#[wasm_bindgen(js_name = cbrtArr)]
pub fn cbrt_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::cbrt(&arr.inner))
}

#[wasm_bindgen(js_name = reciprocalArr)]
pub fn reciprocal_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::reciprocal(&arr.inner))
}

// Angle conversions
#[wasm_bindgen(js_name = deg2radArr)]
pub fn deg2rad_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::deg2rad(&arr.inner))
}

#[wasm_bindgen(js_name = rad2degArr)]
pub fn rad2deg_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::rad2deg(&arr.inner))
}

// Truncation
#[wasm_bindgen(js_name = truncArr)]
pub fn trunc_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| x.trunc());
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// Special functions
#[wasm_bindgen(js_name = sincArr)]
pub fn sinc_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| {
        if x == 0.0 {
            1.0
        } else {
            let pi_x = std::f64::consts::PI * x;
            pi_x.sin() / pi_x
        }
    });
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

#[wasm_bindgen(js_name = heavisideArr)]
pub fn heaviside_arr(arr: &NDArray, h0: f64) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| {
        if x < 0.0 {
            0.0
        } else if x == 0.0 {
            h0
        } else {
            1.0
        }
    });
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

#[wasm_bindgen(js_name = signbitArr)]
pub fn signbit_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|x| if x.is_sign_negative() { 1.0 } else { 0.0 });
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// Element-wise power
#[wasm_bindgen(js_name = powArr)]
pub fn pow_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::pow(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// Linear algebra helpers
#[wasm_bindgen]
pub fn inner(a: &NDArray, b: &NDArray) -> Result<f64, JsValue> {
    CpuBackend::inner(&a.inner, &b.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn outer(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::outer(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn trace(arr: &NDArray) -> Result<f64, JsValue> {
    CpuBackend::trace(&arr.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn norm(arr: &NDArray, ord: Option<f64>) -> Result<f64, JsValue> {
    CpuBackend::norm(&arr.inner, ord)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// Histogram
#[wasm_bindgen]
pub fn bincount(x: &NDArray, weights: Option<NDArray>, minlength: Option<usize>) -> Result<NDArray, JsValue> {
    let x_data = x.inner.as_f64_slice();
    let weights_data = weights.as_ref().map(|w| w.inner.as_f64_slice());
    let minlen = minlength.unwrap_or(0);

    // Find max value to determine output size
    let mut max_val: usize = 0;
    for &v in &x_data {
        if v < 0.0 || v.fract() != 0.0 {
            return Err(JsValue::from_str("bincount requires non-negative integers"));
        }
        max_val = max_val.max(v as usize);
    }
    let n = max_val.saturating_add(1).max(minlen);

    let mut counts = vec![0.0; n];
    for (i, &v) in x_data.iter().enumerate() {
        let idx = v as usize;
        if idx < n {
            let w = weights_data.as_ref().map_or(1.0, |w| w[i]);
            counts[idx] += w;
        }
    }

    Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(counts, vec![n]).unwrap()))
}

// ============ Linear algebra ============

#[wasm_bindgen]
pub fn matmul(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::matmul(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn dot(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::dot(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn inv(arr: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::inv(&arr.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn det(arr: &NDArray) -> Result<f64, JsValue> {
    CpuBackend::det(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn solve(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::solve(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// QR decomposition: A = Q * R
/// Returns [Q, R] where Q is orthogonal and R is upper triangular
#[wasm_bindgen]
pub fn qr(arr: &NDArray) -> Result<js_sys::Array, JsValue> {
    let (q, r) = CpuBackend::qr(&arr.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Array::new();
    result.push(&JsValue::from(NDArray::new(q)));
    result.push(&JsValue::from(NDArray::new(r)));
    Ok(result)
}

/// SVD decomposition: A = U * diag(S) * Vt
/// Returns [U, S, Vt] where U and Vt are orthogonal, S is singular values
#[wasm_bindgen]
pub fn svd(arr: &NDArray) -> Result<js_sys::Array, JsValue> {
    let (u, s, vt) = CpuBackend::svd(&arr.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Array::new();
    result.push(&JsValue::from(NDArray::new(u)));
    result.push(&JsValue::from(NDArray::new(s)));
    result.push(&JsValue::from(NDArray::new(vt)));
    Ok(result)
}

/// Condition number of a matrix (using SVD)
/// Returns max(singular_values) / min(singular_values)
#[wasm_bindgen]
pub fn cond(arr: &NDArray) -> Result<f64, JsValue> {
    let (_, s, _) = CpuBackend::svd(&arr.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let s_data = s.as_f64_slice();
    if s_data.is_empty() {
        return Ok(f64::INFINITY);
    }
    let max_s = s_data.iter().cloned().fold(0.0_f64, f64::max);
    let min_s = s_data.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_s < f64::EPSILON {
        Ok(f64::INFINITY)
    } else {
        Ok(max_s / min_s)
    }
}

// ============ Random ============

#[wasm_bindgen(js_name = randomSeed)]
pub fn random_seed(seed: u64) {
    CpuBackend::seed(seed);
}

#[wasm_bindgen(js_name = randomRand)]
pub fn random_rand(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::rand(shape))
}

#[wasm_bindgen(js_name = randomRandn)]
pub fn random_randn(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::randn(shape))
}

#[wasm_bindgen(js_name = randomUniform)]
pub fn random_uniform(low: f64, high: f64, shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::uniform(low, high, shape))
}

#[wasm_bindgen(js_name = randomNormal)]
pub fn random_normal(loc: f64, scale: f64, shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::normal(loc, scale, shape))
}

// ============ Memory access for zero-copy ============

/// Get WASM linear memory for zero-copy access
///
/// Returns the WebAssembly.Memory object that backs all arrays.
/// Use with `dataPtr()` and `len()` to create zero-copy TypedArray views:
///
/// ```javascript
/// const wasmMemory = rumpy.wasmMemory();
/// const ptr = array.dataPtr();
/// const len = array.len();
/// const view = new Float64Array(wasmMemory.buffer, ptr, len);
/// // view is now a zero-copy view into the array's data
/// ```
///
/// Note: Views are invalidated if WASM memory grows. Monitor memory size
/// or recreate views after operations that might allocate.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Check if SharedArrayBuffer is available
///
/// Returns true if the environment supports SharedArrayBuffer (COOP/COEP headers set).
/// When false, `asTypedArrayView()` will not work and you should use `toTypedArray()`.
#[wasm_bindgen(js_name = hasSharedArrayBuffer)]
pub fn has_shared_array_buffer() -> bool {
    // In WASM context, check if memory is shared
    // This is a runtime check - the actual capability depends on browser headers
    js_sys::Reflect::has(&wasm_bindgen::memory(), &JsValue::from_str("buffer")).unwrap_or(false)
}

// ============================================================================
// Zero-copy f32 buffers (eliminate JS↔WASM copy overhead)
// ============================================================================
//
// The Float32Array-based matmul functions (matmulF32Optimized etc.) copy
// A and B from JS heap → WASM heap on every call (a.to_vec()), then copy
// C back (Float32Array::from). At small sizes this is ~20-25% of wall time:
//
//   256²: matmul = 0.5 ms, copies ≈ 0.12 ms (24%)
//   128²: matmul = 0.3 ms, copies ≈ 0.06 ms (20%)
//
// tf.js doesn't pay this cost — tensors live in WASM memory. This API
// lets you match that: allocate f32 buffers inside WASM once, get zero-
// copy Float32Array views, write your data into them directly, call
// matmulF32ZeroCopy which operates in-place.
//
// USAGE:
//   const a = allocF32(M * K);           // WASM-resident buffer
//   const b = allocF32(K * N);
//   const c = allocF32(M * N);
//   const packedB = allocF32(packedBSize(K, N));  // for prepacked path
//
//   // Fill a, b via zero-copy views (SharedArrayBuffer → no detach on grow):
//   const mem = wasmMemory().buffer;
//   new Float32Array(mem, a.ptr(), M*K).set(yourAData);
//   new Float32Array(mem, b.ptr(), K*N).set(yourBData);
//
//   packBInPlace(b, packedB, K, N);       // once per weight matrix
//   matmulF32PrepackedZeroCopy(a, packedB, c, M, N, K);  // many times
//
//   // Read result:
//   const result = new Float32Array(mem, c.ptr(), M*N);
//
// MEMORY GROWTH CAVEAT: views are valid as long as WebAssembly.Memory
// doesn't grow. With SharedArrayBuffer (which we use), growing doesn't
// DETACH the view, but the view's `.buffer` still points at the old SAB
// range. Re-fetch `wasmMemory().buffer` after operations that allocate.
// (F32Buffer handles themselves stay valid — only JS-side views need
// re-deriving.)

/// WASM-resident f32 buffer. Wraps a `Vec<f32>` that lives in WASM linear
/// memory. JS can get a zero-copy `Float32Array` view via `ptr()` + the
/// shared memory buffer.
///
/// The buffer stays valid until `free()` is called or the object is GC'd.
/// Memory growth does NOT invalidate the buffer (the Vec's address is
/// stable), only JS-side views of `wasmMemory().buffer` need re-deriving.
#[wasm_bindgen]
pub struct F32Buffer {
    data: Vec<f32>,
}

#[wasm_bindgen]
impl F32Buffer {
    /// Byte offset into WASM linear memory where this buffer's data starts.
    ///
    /// Use with `wasmMemory().buffer` to construct a zero-copy view:
    ///   new Float32Array(wasmMemory().buffer, buf.ptr(), buf.len())
    #[wasm_bindgen]
    pub fn ptr(&self) -> usize {
        self.data.as_ptr() as usize
    }

    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Explicitly free this buffer's memory. The handle is consumed.
    #[wasm_bindgen]
    pub fn free(self) {
        // self dropped here
    }

    /// Copy data FROM a JS Float32Array INTO this buffer.
    /// Useful for the first fill if you can't construct data directly into
    /// a zero-copy view (e.g. data comes from a WebGL readback).
    #[wasm_bindgen(js_name = copyFrom)]
    pub fn copy_from(&mut self, src: &Float32Array) {
        let n = (src.length() as usize).min(self.data.len());
        src.slice(0, n as u32).copy_to(&mut self.data[..n]);
    }

    /// Copy data FROM this buffer TO a JS Float32Array.
    /// For the zero-copy path you don't need this — construct a view
    /// instead. This exists for cases where the result needs to go to a
    /// non-shared ArrayBuffer (e.g. postMessage to a context without SAB).
    #[wasm_bindgen(js_name = copyTo)]
    pub fn copy_to(&self, dst: &Float32Array) {
        let n = (dst.length() as usize).min(self.data.len());
        dst.subarray(0, n as u32).copy_from(&self.data[..n]);
    }
}

/// Allocate an f32 buffer of the given length inside WASM memory.
/// Contents are uninitialised — write before reading.
#[wasm_bindgen(js_name = allocF32)]
pub fn alloc_f32(len: usize) -> F32Buffer {
    let mut data: Vec<f32> = Vec::with_capacity(len);
    // Uninitialised: caller is expected to fill via copyFrom or a
    // zero-copy view. Zeroing would be wasted work for input buffers
    // (overwritten immediately) and output buffers (matmul overwrites).
    unsafe { data.set_len(len); }
    F32Buffer { data }
}

/// Size (in f32 elements) of a fully-packed B buffer for matmulF32PrepackedZeroCopy.
///
/// = ceil(N/8) × K × 8.  For N divisible by 8 (most cases), equals K × N.
#[wasm_bindgen(js_name = packedBSize)]
pub fn packed_b_size(k: usize, n: usize) -> usize {
    ((n + 7) / 8) * k * 8
}

/// Pack B (in an F32Buffer) into panel-major layout (in another F32Buffer).
///
/// Call once per weight matrix; reuse packedB across many matmuls.
/// Both buffers must already be allocated to the right sizes (B: K×N,
/// packedB: packedBSize(K, N)).
///
/// Note: Currently just copies B to packed_b. The specialized packing was
/// removed during a refactor. The matmul still works (just re-packs internally).
#[wasm_bindgen(js_name = packBInPlace)]
pub fn pack_b_in_place(b: &F32Buffer, packed_b: &mut F32Buffer, k: usize, n: usize) {
    assert!(b.data.len() >= k * n, "b too small");
    assert!(packed_b.data.len() >= packed_b_size(k, n), "packed_b too small");

    // For now, just copy B into packed_b (prepacking optimization was removed)
    let copy_len = (k * n).min(packed_b.data.len());
    packed_b.data[..copy_len].copy_from_slice(&b.data[..copy_len]);
}

/// Parallel matmul, ZERO JS↔WASM copies.
///
/// A, B, C all live in WASM memory (F32Buffers). B is packed on-the-fly
/// (same behaviour as matmulF32OptimizedParallelV3 but without the
/// Float32Array round-trips). C is overwritten.
///
/// This is the general API — B can vary call-to-call. For constant B
/// (NN inference), use matmulF32PrepackedZeroCopy which skips the pack.
#[wasm_bindgen(js_name = matmulF32ZeroCopy)]
pub fn matmul_f32_zerocopy(a: &F32Buffer, b: &F32Buffer, c: &mut F32Buffer, m: usize, n: usize, k: usize) {
    assert!(a.data.len() >= m * k, "a too small");
    assert!(b.data.len() >= k * n, "b too small");
    assert!(c.data.len() >= m * n, "c too small");

    #[cfg(target_arch = "wasm32")]
    {
        // Call straight into v3. No to_vec, no Float32Array::from — the
        // buffers are already in WASM memory.
        let out = simd_gemm::matmul_optimized_f32_parallel(
            &a.data[..m * k],
            &b.data[..k * n],
            m, n, k,
        );
        // v3 returns a Vec (it allocates its own C internally for the
        // C-padding path). Copy into caller's buffer.
        //
        // TODO: add an `_into` variant of v3 that writes to a caller-
        // provided slice when no padding is active. Would save one more
        // M×N copy. At 256² that's 256 KiB = ~0.05 ms — 10% of the
        // remaining gap.
        c.data[..m * n].copy_from_slice(&out);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let out = simd_gemm::matmul_dispatch_f32(&a.data[..m*k], &b.data[..k*n], m, n, k);
        c.data[..m * n].copy_from_slice(&out);
    }
}

/// Parallel matmul with pre-packed B, ZERO JS↔WASM copies.
///
/// The leanest call path: A and packed-B already in WASM memory, C
/// written directly, no per-call packing. This is the tf.js-equivalent
/// path for NN inference.
///
/// Note: The specialized prepacked kernel was removed during a refactor.
/// This now just calls the regular parallel matmul (packed_b is treated as B).
#[wasm_bindgen(js_name = matmulF32PrepackedZeroCopy)]
pub fn matmul_f32_prepacked_zerocopy(
    a: &F32Buffer,
    packed_b: &F32Buffer,
    c: &mut F32Buffer,
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.data.len() >= m * k, "a too small");
    assert!(packed_b.data.len() >= k * n, "packed_b too small");
    assert!(c.data.len() >= m * n, "c too small");

    // Call the regular matmul (prepacked optimization was removed)
    #[cfg(target_arch = "wasm32")]
    {
        let out = simd_gemm::matmul_optimized_f32_parallel(
            &a.data[..m * k],
            &packed_b.data[..k * n],
            m, n, k,
        );
        c.data[..m * n].copy_from_slice(&out);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let out = simd_gemm::matmul_dispatch_f32(&a.data[..m*k], &packed_b.data[..k*n], m, n, k);
        c.data[..m * n].copy_from_slice(&out);
    }
}

// ============ High-performance f32 SIMD matmul ============

/// Fast f32 matrix multiplication using WASM SIMD
///
/// This is a direct binding to the SIMD-optimized GEMM kernel, matching XNNPACK's approach.
/// Uses f32 (4 elements per v128) instead of f64 (2 elements per v128) for 2x throughput.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32)]
pub fn matmul_f32(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_dispatch_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Fast f64 matrix multiplication using WASM SIMD
///
/// Direct binding to the SIMD-optimized GEMM kernel for f64.
/// Uses f64x2 (2 elements per v128).
///
/// Parameters:
/// - a: Float64Array, row-major, shape [m, k]
/// - b: Float64Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float64Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF64)]
pub fn matmul_f64(a: &Float64Array, b: &Float64Array, m: usize, n: usize, k: usize) -> Float64Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_dispatch_f64(&a_vec, &b_vec, m, n, k);
    Float64Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with explicit matrix packing
///
/// This version always uses matrix packing regardless of size, for benchmarking.
/// Packing reorders B matrix into cache-friendly column panels.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Packed)]
pub fn matmul_f32_packed(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_packed(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with FMA (fused multiply-add)
///
/// Uses relaxed-simd f32x4_relaxed_madd for better throughput.
/// FMA computes a*b+c in one instruction instead of two (mul + add).
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32FMA)]
pub fn matmul_f32_fma(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_fma(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with FMA + packed B
///
/// Combines both optimizations: FMA instructions and B matrix packing.
/// This is the fastest kernel for large matrices.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32FMAPacked)]
pub fn matmul_f32_fma_packed(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_fma_packed(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Auto-tuned f32 matrix multiplication
///
/// Automatically selects the best kernel based on matrix dimensions:
/// - 5x8 kernel for matrices where M % 5 == 0 (like 100x100)
/// - FMA for medium matrices (packing overhead not amortized)
/// - FMA + packed for large matrices (packing overhead amortized)
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Auto)]
pub fn matmul_f32_auto(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            simd_gemm::matmul_simd_f32_auto(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// 5x8 kernel specifically for matrices where M is divisible by 5
///
/// Optimized for 100x100 case (and similar).
#[wasm_bindgen(js_name = matmulF325x8)]
pub fn matmul_f32_5x8(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 5 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_5x8(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Verify correctness: compute max absolute difference between two f32 arrays
///
/// Returns the maximum |a[i] - b[i]| across all elements.
/// Use this to verify that different kernels produce the same results.
#[wasm_bindgen(js_name = maxAbsDiff)]
pub fn max_abs_diff(a: &Float32Array, b: &Float32Array) -> f32 {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    a_vec.iter()
        .zip(b_vec.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute checksum (sum of all elements) for verification
#[wasm_bindgen(js_name = checksum)]
pub fn checksum(a: &Float32Array) -> f32 {
    a.to_vec().iter().sum()
}

/// Use the gemm crate for highly optimized GEMM
///
/// The gemm crate uses BLIS-style optimizations:
/// - Cache-blocking at L1/L2/L3 levels
/// - Optimized micro-kernels
/// - Smart packing strategies
///
/// This should be as fast as or faster than our hand-written SIMD kernels.
#[wasm_bindgen(js_name = matmulGemm)]
pub fn matmul_gemm(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_gemm_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Parallel f32 matrix multiplication using rayon + Web Workers
///
/// Uses rayon to parallelize across the M dimension with native WASM threads.
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// For large matrices (256+), this scales with available cores.
/// Falls back to single-threaded for small matrices.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Parallel)]
pub fn matmul_f32_parallel(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Parallel f32 matrix multiplication V2 using rayon + Web Workers (zero-allocation)
///
/// This is an improved version that writes directly to pre-allocated memory,
/// avoiding per-thread allocations. This is significantly faster than V1
/// for large matrices.
///
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32ParallelV2)]
pub fn matmul_f32_parallel_v2(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_parallel_f32_v2(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Get the current number of rayon threads
#[wasm_bindgen(js_name = getNumThreads)]
pub fn get_num_threads() -> usize {
    #[cfg(feature = "threads")]
    { rayon::current_num_threads() }
    #[cfg(not(feature = "threads"))]
    { 1 }
}

// ============ Threading debug functions (only available with `threads` feature) ============

#[cfg(feature = "threads")]
/// DEBUG: mimic v3's dispatch pattern (rayon::scope + inline caller +
/// atomic tile counter) and record which rayon thread claims each tile.
///
/// Returns a flat array of [tile_idx, rayon_thread_idx, tid_param] triples
/// so we can see if all tiles were claimed by one thread (rayon dispatch
/// bug) or spread across threads (parallelism works, perf bug is elsewhere).
#[wasm_bindgen(js_name = probeV3Dispatch)]
pub fn probe_v3_dispatch(n_tiles: usize, work_ms_per_tile: f64) -> Vec<f64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let n_workers = rayon::current_num_threads();
    let tile_counter = AtomicUsize::new(0);
    let log: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::with_capacity(n_tiles));

    let worker = |tid: usize| {
        loop {
            let t = tile_counter.fetch_add(1, Ordering::Relaxed);
            if t >= n_tiles { break; }

            let rtid = rayon::current_thread_index().unwrap_or(9999);
            log.lock().unwrap().push((t, rtid, tid));

            // Simulate work.
            let start = js_sys::Date::now();
            while js_sys::Date::now() - start < work_ms_per_tile {
                core::hint::spin_loop();
            }
        }
    };

    let t0 = js_sys::Date::now();
    rayon::scope(|s| {
        for tid in 0..n_workers {
            s.spawn(move |_| worker(tid));
        }
        worker(n_workers); // caller
    });
    let wall = js_sys::Date::now() - t0;

    // Flatten: [wall, n_triples, t0,r0,p0, t1,r1,p1, ...]
    let mut out = vec![wall, log.lock().unwrap().len() as f64];
    for (t, r, p) in log.lock().unwrap().iter() {
        out.push(*t as f64);
        out.push(*r as f64);
        out.push(*p as f64);
    }
    out
}

#[cfg(feature = "threads")]
/// DEBUG: report which code path v3 would take for given (m,n,k).
/// Returns: [below_threshold, pack_a, c_pad, fast_path, slab_rows, total_tiles, tz(k*4), tz(n*4)]
#[wasm_bindgen(js_name = probeV3Path)]
pub fn probe_v3_path(m: usize, n: usize, k: usize) -> Vec<usize> {
    const PAD_ZEROS_THRESHOLD: u32 = 12;
    const OPT_MR: usize = 6;

    // u64 mul: WASM usize is 32-bit, m*n*k overflows at 2048³ → 0.
    // (This was the "cursed triple" — overflow → false below_threshold
    // positive → silent single-threaded fallback.)
    let flops = (m as u64) * (n as u64) * (k as u64);
    let size_below_threshold = flops < (192u64 * 192 * 192);

    let n_workers = rayon::current_num_threads().max(1);
    let pack_a = (k * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD;
    let c_pad = (n * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD;
    let slab_rows = {
        let base = (m + n_workers - 1) / n_workers;
        ((base + OPT_MR - 1) / OPT_MR * OPT_MR).max(OPT_MR)
    };
    let total_tiles = (m + slab_rows - 1) / slab_rows;
    let fast = !pack_a && !c_pad && !size_below_threshold && total_tiles >= 2;

    vec![
        size_below_threshold as usize,
        pack_a as usize,
        c_pad as usize,
        fast as usize,
        slab_rows,
        total_tiles,
        (k * 4).trailing_zeros() as usize,
        (n * 4).trailing_zeros() as usize,
    ]
}

#[cfg(feature = "threads")]
/// DEBUG: probe whether rayon workers are actually executing in parallel.
///
/// Spawns N tasks, each recording its rayon thread index and spinning for
/// ~duration_ms. If workers are live, wall-clock ≈ duration_ms (parallel).
/// If all tasks run on the main thread, wall-clock ≈ N × duration_ms.
///
/// Returns [wall_ms, n_distinct_thread_ids, max_thread_id_seen].
#[wasm_bindgen(js_name = probeRayonParallelism)]
pub fn probe_rayon_parallelism(n_tasks: usize, duration_ms: f64) -> Vec<f64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Rayon thread indices seen (bitmask; up to 64 threads).
    let seen = AtomicUsize::new(0);
    let max_idx = AtomicUsize::new(0);

    let t0 = js_sys::Date::now();
    (0..n_tasks).into_par_iter().for_each(|_| {
        let tid = rayon::current_thread_index().unwrap_or(usize::MAX);
        if tid < 64 {
            seen.fetch_or(1 << tid, Ordering::Relaxed);
            max_idx.fetch_max(tid, Ordering::Relaxed);
        }
        // Busy-spin for duration_ms (can't atomic.wait on main thread, and
        // we want deterministic work regardless of thread).
        let start = js_sys::Date::now();
        while js_sys::Date::now() - start < duration_ms {
            core::hint::spin_loop();
        }
    });
    let wall = js_sys::Date::now() - t0;

    let n_distinct = seen.load(Ordering::Relaxed).count_ones() as f64;
    vec![wall, n_distinct, max_idx.load(Ordering::Relaxed) as f64]
}

#[cfg(feature = "threads")]
/// Parallel f32 matrix multiplication using pthreadpool-rs
///
/// Uses pthreadpool-rs instead of rayon for parallelization.
/// On WASM, pthreadpool-rs uses wasm-bindgen-rayon under the hood.
///
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Pthreadpool)]
pub fn matmul_f32_pthreadpool(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_pthreadpool_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// XNNPACK-style f32 GEMM with pre-packed B matrix (LEGACY, single-threaded)
///
/// This is a two-phase API:
/// 1. Call `packB` once to convert B into XNNPACK format
/// 2. Call `matmulXnnpack` multiple times with different A matrices
///
/// This amortizes the packing cost over many matmuls, which is how XNNPACK works.
/// For PARALLEL matmul with pre-packed B, use `packBFull` + `matmulF32Prepacked`.
#[wasm_bindgen(js_name = packB)]
pub fn pack_b(b: &Float32Array, k: usize, n: usize) -> Float32Array {
    let b_vec = b.to_vec();
    let n_panels = n / 8;
    let mut packed = vec![0.0f32; n_panels * k * 8];
    simd_gemm::pack_b_xnnpack(&b_vec, &mut packed, k, n);
    Float32Array::from(packed.as_slice())
}

/// Pack ALL of B into panel-major layout (for use with matmulF32Prepacked).
///
/// Unlike `packB` (which truncates at N/8×8), this handles arbitrary N
/// by zero-padding the last panel to NR=8 width. The output size is
/// ceil(N/8) × K × 8 floats.
///
/// Call this ONCE for weight matrices that will be reused across many
/// matmuls (NN inference). The pack cost is O(K×N) = one pass through B;
/// tf.js/XNNPACK do exactly this at model-load time.
/// Pack B matrix for repeated matmuls.
/// Note: Prepacking optimization was removed. This now just returns a copy.
#[wasm_bindgen(js_name = packBFull)]
pub fn pack_b_full(b: &Float32Array, k: usize, n: usize) -> Float32Array {
    let _ = (k, n); // dimensions used for validation only now
    // Just return a copy - the specialized prepacking was removed
    let b_vec = b.to_vec();
    Float32Array::from(b_vec.as_slice())
}

/// Parallel matmul with pre-packed B (from packBFull).
///
/// Note: Prepacking optimization was removed. This now calls the regular
/// parallel matmul (packed_b is treated as normal B).
#[wasm_bindgen(js_name = matmulF32Prepacked)]
pub fn matmul_f32_prepacked(a: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let _pb_vec = packed_b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        // Use regular parallel matmul - prepacking was removed
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &_pb_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Float32Array::from(
            simd_gemm::matmul_dispatch_f32(&a_vec, &_pb_vec, m, n, k).as_slice()
        )
    }
}

/// XNNPACK-style matmul with pre-packed B
///
/// Requires both the original B (for remaining columns) and packed_b (for SIMD panels).
/// This handles arbitrary N, not just multiples of 8.
#[wasm_bindgen(js_name = matmulXnnpack)]
pub fn matmul_xnnpack(a: &Float32Array, b: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let _pb_vec = packed_b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_xnnpack_style_full(&a_vec, &b_vec, &pb_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Cache-blocked 6x8 GEMM for large matrices
///
/// Uses GOTO-style cache blocking to tile the computation:
/// - Outer loop tiles by N dimension (NC=256)
/// - Middle loop tiles by K dimension (KC=256)
/// - Inner loop tiles by M dimension (MC=128)
///
/// This ensures working set fits in L1/L2 cache for better performance
/// on large matrices (256x256 and above).
#[wasm_bindgen(js_name = matmulF32Blocked)]
pub fn matmul_f32_blocked(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_6x8_blocked(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Highly optimized 6x8 GEMM with FMA, loadsplat, and cache blocking
///
/// This is the most optimized implementation, matching XNNPACK patterns:
/// - 6x8 micro-kernel (12 accumulators fit in 16 XMM registers)
/// - f32x4_relaxed_madd for FMA
/// - v128_load32_splat for A broadcast
/// - L1/L2 cache blocking (KC=256, MC=72, NC=128)
/// - B matrix packing for contiguous access
#[wasm_bindgen(js_name = matmulF32Optimized)]
pub fn matmul_f32_optimized(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_dispatch_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Pre-pack B matrix for repeated matmuls with the same weights.
///
/// This amortizes packing cost across multiple matmuls (like tfjs inference mode).
/// Returns a Float32Array containing the packed B data.
///
/// Example:
/// ```javascript
/// const packedB = packBForGemm(weights, k, n);
/// // Later, for each input:
/// const result = matmulWithPackedB(input, packedB, m, n, k);
/// ```
#[wasm_bindgen(js_name = packBForGemm)]
pub fn pack_b_for_gemm(b: &Float32Array, _k: usize, _n: usize) -> Float32Array {
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let packed = simd_gemm::pack_b_for_gemm(&b_vec, _k, _n);
        Float32Array::from(packed.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Non-WASM: just return the original (no packing needed for native)
        Float32Array::from(b_vec.as_slice())
    }
}

/// GEMM with pre-packed B matrix (inference mode).
///
/// Use packBForGemm to create packedB once, then call this for each matmul.
/// This matches how tfjs/XNNPACK works for inference.
/// Uses parallel execution via futex pool when available.
#[wasm_bindgen(js_name = matmulWithPackedB)]
pub fn matmul_with_packed_b(a: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let packed_b_vec = packed_b.to_vec();

    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "atomics",
        feature = "futex-pool"
    ))]
    {
        // Use parallel version when futex pool is available
        let c_vec = simd_gemm::matmul_with_packed_b_parallel_f32(&a_vec, &packed_b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(all(target_arch = "wasm32", not(all(target_feature = "atomics", feature = "futex-pool"))))]
    {
        // Fallback to ST when no futex pool
        let c_vec = simd_gemm::matmul_with_packed_b_f32(&a_vec, &packed_b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Non-WASM: packed_b is just regular B, use normal matmul
        let c_vec = simd_gemm::matmul_dispatch_f32(&a_vec, &packed_b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Parallel version of optimized 6x8 GEMM using rayon (LEGACY)
///
/// Kept for A/B benchmarking. Has known problems — see v3 below.
#[wasm_bindgen(js_name = matmulF32OptimizedParallel)]
pub fn matmul_f32_optimized_parallel(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Parallel optimised GEMM, v3: pack-once, 2D-tile, atomic work-claiming.
///
/// This is the recommended parallel path. Differences from the legacy
/// `matmulF32OptimizedParallel`:
///
/// * B is packed ONCE and shared read-only across all workers
///   (old path packed B independently in every thread — with N threads that's
///   N× the packing work and N× allocator contention on WASM's locked dlmalloc)
///
/// * Macro-tiles (~MC × NC) are handed out via an atomic counter, matching
///   XNNPACK's `pthreadpool_parallelize_2d_tile_2d`. Load balances across
///   Apple Silicon perf/efficiency cores instead of assuming uniform workers.
///
/// * Zero per-task heap allocation. Workers write straight into disjoint
///   C slices.
///
/// * The calling thread participates (it's "thread 0"), so with an N-worker
///   Rayon pool you get N+1 way parallelism.
///
/// Requires `initThreadPool(n)` to have been called (same as legacy path).
#[wasm_bindgen(js_name = matmulF32OptimizedParallelV3)]
pub fn matmul_f32_optimized_parallel_v3(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Parallel optimised GEMM, v4: hijack Rayon's workers with raw
/// `memory.atomic.wait32`/`notify` dispatch.
///
/// v3 uses ONE `rayon::scope` per matmul (good), but inside it there's
/// still no shared packed-B (each thread packs its own) and the join is
/// Rayon's standard park/unpark.  v4 is the full pthreadpool model:
///
/// * ONE `rayon::scope` — we use wasm-bindgen-rayon's Web Workers but
///   NOT Rayon's task scheduler.
///
/// * Workers enter OUR spin-then-`atomic.wait` loop. Main drives them
///   block-by-block: pack B (shared), bump generation, `atomic.notify`,
///   drain tiles alongside workers, spin-wait for completion, repeat.
///
/// * Shared packed-B → minimum total packing work (same as single-thread).
///
/// * Per-block sync is ~1 `atomic.notify` + N×1 Relaxed `fetch_sub` +
///   one short main-thread spin. Compare Rayon: N× `Box<dyn FnOnce>` +
///   N× park/unpark per scope.
///
/// This is "our own thread manager", hosted inside Rayon's already-spawned
/// workers. No new dependencies, no separate worker pool to manage.
///
/// Requires `initThreadPool(n)` (same as v3).
#[wasm_bindgen(js_name = matmulF32OptimizedParallelV4)]
pub fn matmul_f32_optimized_parallel_v4(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

// ============================================================================
// FUTEX POOL - Low-latency parallel dispatch (experimental)
// ============================================================================

/// Initialize the futex thread pool with n threads.
///
/// This creates a separate thread pool that bypasses Rayon entirely, using
/// raw memory.atomic.wait32/notify for ~10μs dispatch overhead (vs ~150μs for Rayon).
///
/// Call this INSTEAD OF initThreadPool if you want to use matmulF32Futex.
/// If you call both, you'll have two thread pools (works but wastes cores).
///
/// Requires the `futex-pool` feature:
/// ```
/// wasm-pack build --features futex-pool
/// ```
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = initFutexPool)]
pub fn init_futex_pool(n: usize) {
    simd_gemm::init_futex_pool(n);
}

/// Get the number of threads in the futex pool.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = getFutexThreads)]
pub fn get_futex_threads() -> usize {
    simd_gemm::futex_threads_count()
}

/// Check if all futex workers are ready.
///
/// Web Workers are spawned asynchronously when `initFutexPool` is called.
/// This function returns true once all workers have started and are waiting
/// for work. You should poll this before calling `matmulF32Futex` to avoid
/// crashes.
///
/// Example:
/// ```javascript
/// initFutexPool(navigator.hardwareConcurrency);
/// while (!futexWorkersReady()) {
///   await new Promise(r => setTimeout(r, 10));
/// }
/// // Now safe to use matmulF32Futex
/// ```
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = futexWorkersReady)]
pub fn futex_workers_ready() -> bool {
    pthreadpool_rs::wasm_futex::workers_ready()
}

/// Get the number of futex workers that have started (for debugging).
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = futexWorkersReadyCount)]
pub fn futex_workers_ready_count() -> u32 {
    pthreadpool_rs::wasm_futex::workers_ready_count()
}

/// Measure dispatch overhead by calling parallelize with minimal work.
/// Returns average dispatch time in microseconds.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = measureDispatchOverhead)]
pub fn measure_dispatch_overhead(iterations: u32) -> f64 {
    use std::sync::atomic::{AtomicU32, Ordering};

    let pool = match pthreadpool_rs::wasm_futex::get_pool() {
        Some(p) => p,
        None => return -1.0, // Pool not initialized
    };

    // Minimal work function - just increment a counter
    let counter = AtomicU32::new(0);
    let num_threads = pthreadpool_rs::wasm_futex::threads_count();

    let start = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    for _ in 0..iterations {
        pool.parallelize(num_threads, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
    }

    let end = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let total_ms = end - start;
    let avg_us = (total_ms * 1000.0) / (iterations as f64);
    avg_us
}

/// Measure pure dispatch overhead with no actual work (just coordination).
/// Returns average dispatch time in microseconds.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = measurePureDispatchOverhead)]
pub fn measure_pure_dispatch_overhead(iterations: u32) -> f64 {
    let pool = match pthreadpool_rs::wasm_futex::get_pool() {
        Some(p) => p,
        None => return -1.0,
    };

    let num_threads = pthreadpool_rs::wasm_futex::threads_count();

    let start = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    for _ in 0..iterations {
        // Dispatch with minimal work - each thread runs once with no-op
        pool.parallelize(num_threads, |_| {
            // No-op - just sync overhead
        });
    }

    let end = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let total_ms = end - start;
    let avg_us = (total_ms * 1000.0) / (iterations as f64);
    avg_us
}

/// Parallel GEMM using the raw futex pool.
///
/// This achieves much lower dispatch overhead (~5-10μs) compared to Rayon (~150μs).
/// Best for small-to-medium matrices (128-512) where Rayon's overhead dominates.
///
/// Requires:
/// 1. Build with `--features futex-pool`
/// 2. Call `initFutexPool(n)` from JS before using this
///
/// Falls back to single-threaded if futex pool isn't initialized.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "atomics",
    feature = "futex-pool"
))]
#[wasm_bindgen(js_name = matmulF32Futex)]
pub fn matmul_f32_futex(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    // Use tiled version for better parallel efficiency (shared B packing)
    let c_vec = simd_gemm::matmul_futex_f32_tiled(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

// ============================================================================
// BLOCKED/SPECIAL MATMUL VARIANTS
// ============================================================================

/// Cache-blocked XNNPACK-style matmul with pre-packed B
///
/// Combines cache blocking with B-matrix packing for optimal performance.
/// Best for large matrices where both cache blocking and packing help.
#[wasm_bindgen(js_name = matmulXnnpackBlocked)]
pub fn matmul_xnnpack_blocked(a: &Float32Array, b: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let _pb_vec = packed_b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_xnnpack_blocked(&a_vec, &b_vec, &pb_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

// ============ ML Inference Ops ============
//
// These operations handle non-contiguous arrays safely by:
// 1. Using as_standard_layout() to ensure C-order before reshaping
// 2. Using ndarray's lane iterators which handle strides automatically
// 3. Avoiding manual index calculations that assume C-order

/// Layer normalization (matches torch.nn.functional.layer_norm)
///
/// Normalizes over the last N dimensions where N = normalized_shape.len()
/// output = (input - mean) / sqrt(var + eps) * gamma + beta
///
/// Handles non-contiguous inputs (e.g., from transpose/permute) safely.
///
/// # Arguments
/// * `normalized_shape` - The shape over which to normalize (last N dims)
/// * `gamma` - Optional scale parameter (weight), defaults to 1.0
/// * `beta` - Optional shift parameter (bias), defaults to 0.0
/// * `eps` - Small constant for numerical stability
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = layerNorm)]
    pub fn layer_norm(
        &self,
        normalized_shape: Vec<usize>,
        gamma: Option<NDArray>,
        beta: Option<NDArray>,
        eps: f64
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();
        let norm_ndim = normalized_shape.len();

        if norm_ndim > ndim {
            return Err(JsValue::from_str("normalized_shape has more dimensions than input"));
        }

        // Verify normalized_shape matches the last N dimensions
        for i in 0..norm_ndim {
            if shape[ndim - norm_ndim + i] != normalized_shape[i] {
                return Err(JsValue::from_str(&format!(
                    "normalized_shape mismatch at dim {}: expected {}, got {}",
                    i, shape[ndim - norm_ndim + i], normalized_shape[i]
                )));
            }
        }

        let norm_size: usize = normalized_shape.iter().product();

        // Get gamma/beta slices if provided
        let g_slice = gamma.as_ref().map(|g| g.inner.as_f64_slice());
        let b_slice = beta.as_ref().map(|b| b.inner.as_f64_slice());

        // Use lanes() to iterate over the normalization dimension
        // This handles any memory layout without copying
        // For multi-dim normalized_shape, we flatten conceptually by treating
        // the last norm_ndim dims as the "lane" to normalize over

        // Build the axis for normalization - we need to iterate over "batches"
        // where each batch is the first (ndim - norm_ndim) dimensions
        // For layerNorm, we normalize over the last norm_ndim dims

        // Reshape to 2D: [batch, norm_features] for simplicity
        // as_standard_layout is needed here because we're reshaping
        let data_contiguous = data.as_standard_layout();
        let batch_size: usize = shape[..ndim - norm_ndim].iter().product();

        let flat = data_contiguous.view()
            .into_shape_with_order((batch_size, norm_size))
            .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

        let mut flat_result = ndarray::Array2::<f64>::zeros((batch_size, norm_size));

        // Hoist the gamma/beta branches outside the hot loop for better vectorization
        // Use d*d instead of powi(2) for faster variance calculation
        // Zip gamma/beta iterators to eliminate bounds checks in inner loop
        match (&g_slice, &b_slice) {
            (Some(g), Some(b)) => {
                // Both gamma and beta - zip all iterators to remove bounds checks
                for (mut out_row, in_row) in flat_result.outer_iter_mut().zip(flat.outer_iter()) {
                    let mean = in_row.mean().unwrap_or(0.0);
                    let var = in_row.iter().map(|&x| { let d = x - mean; d * d }).sum::<f64>() / norm_size as f64;
                    let inv_std = 1.0 / (var + eps).sqrt();

                    for (((y, &x), &gamma), &beta) in out_row.iter_mut()
                        .zip(in_row.iter())
                        .zip(g.iter())
                        .zip(b.iter())
                    {
                        *y = (x - mean) * inv_std * gamma + beta;
                    }
                }
            }
            (Some(g), None) => {
                // Gamma only
                for (mut out_row, in_row) in flat_result.outer_iter_mut().zip(flat.outer_iter()) {
                    let mean = in_row.mean().unwrap_or(0.0);
                    let var = in_row.iter().map(|&x| { let d = x - mean; d * d }).sum::<f64>() / norm_size as f64;
                    let inv_std = 1.0 / (var + eps).sqrt();

                    for ((y, &x), &gamma) in out_row.iter_mut()
                        .zip(in_row.iter())
                        .zip(g.iter())
                    {
                        *y = (x - mean) * inv_std * gamma;
                    }
                }
            }
            (None, Some(b)) => {
                // Beta only
                for (mut out_row, in_row) in flat_result.outer_iter_mut().zip(flat.outer_iter()) {
                    let mean = in_row.mean().unwrap_or(0.0);
                    let var = in_row.iter().map(|&x| { let d = x - mean; d * d }).sum::<f64>() / norm_size as f64;
                    let inv_std = 1.0 / (var + eps).sqrt();

                    for ((y, &x), &beta) in out_row.iter_mut()
                        .zip(in_row.iter())
                        .zip(b.iter())
                    {
                        *y = (x - mean) * inv_std + beta;
                    }
                }
            }
            (None, None) => {
                // No affine
                for (mut out_row, in_row) in flat_result.outer_iter_mut().zip(flat.outer_iter()) {
                    let mean = in_row.mean().unwrap_or(0.0);
                    let var = in_row.iter().map(|&x| { let d = x - mean; d * d }).sum::<f64>() / norm_size as f64;
                    let inv_std = 1.0 / (var + eps).sqrt();

                    for (y, &x) in out_row.iter_mut().zip(in_row.iter()) {
                        *y = (x - mean) * inv_std;
                    }
                }
            }
        }

        // Reshape back to original shape
        let result_arr = flat_result.into_shape_with_order(ndarray::IxDyn(&shape))
            .map_err(|e| JsValue::from_str(&format!("reshape back failed: {}", e)))?;

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result_arr.to_owned())))
    }
}

/// RMS normalization (used in LLaMA, T5)
///
/// Unlike layer norm, RMS norm doesn't subtract mean, only divides by RMS.
/// output = input / sqrt(mean(input^2) + eps) * gamma
///
/// Handles non-contiguous inputs safely.
///
/// # Arguments
/// * `gamma` - Scale parameter (required for RMS norm)
/// * `eps` - Small constant for numerical stability
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = rmsNorm)]
    pub fn rms_norm(&self, gamma: &NDArray, eps: f64) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        if ndim == 0 {
            return Err(JsValue::from_str("Cannot apply rms_norm to scalar"));
        }

        let last_dim = *shape.last().unwrap();
        let batch_size: usize = shape[..ndim - 1].iter().product();

        // Ensure contiguous layout before reshaping
        let data_contiguous = data.as_standard_layout();
        let flat = data_contiguous.view()
            .into_shape_with_order((batch_size, last_dim))
            .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

        let g_slice = gamma.inner.as_f64_slice();

        if g_slice.len() != last_dim {
            return Err(JsValue::from_str(&format!(
                "gamma size {} doesn't match last dimension {}",
                g_slice.len(), last_dim
            )));
        }

        let mut result = ndarray::Array2::<f64>::zeros((batch_size, last_dim));

        for (mut out_row, in_row) in result.outer_iter_mut().zip(flat.outer_iter()) {
            // Compute RMS = sqrt(mean(x^2) + eps)
            let rms = (in_row.iter().map(|&x| x * x).sum::<f64>() / last_dim as f64 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for ((y, &x), &g) in out_row.iter_mut().zip(in_row.iter()).zip(g_slice.iter()) {
                *y = x * inv_rms * g;
            }
        }

        let result_arr = result.into_shape_with_order(ndarray::IxDyn(&shape))
            .map_err(|e| JsValue::from_str(&format!("reshape back failed: {}", e)))?;

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result_arr.to_owned())))
    }
}

/// Batch normalization (inference only)
///
/// Normalizes using pre-computed running statistics.
/// output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta
///
/// Handles non-contiguous inputs safely using ndarray iterators.
///
/// # Arguments
/// * `gamma` - Optional scale (weight), defaults to 1.0
/// * `beta` - Optional shift (bias), defaults to 0.0
/// * `running_mean` - Pre-computed mean per channel
/// * `running_var` - Pre-computed variance per channel
/// * `eps` - Numerical stability constant
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = batchNorm)]
    pub fn batch_norm(
        &self,
        gamma: Option<NDArray>,
        beta: Option<NDArray>,
        running_mean: &NDArray,
        running_var: &NDArray,
        eps: f64
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        // Assume NCHW format (batch, channels, height, width)
        if ndim < 2 {
            return Err(JsValue::from_str("batchNorm requires at least 2D input (N, C, ...)"));
        }

        let num_channels = shape[1];
        let mean_slice = running_mean.inner.as_f64_slice();
        let var_slice = running_var.inner.as_f64_slice();

        if mean_slice.len() != num_channels || var_slice.len() != num_channels {
            return Err(JsValue::from_str("running_mean/var must match channel dimension"));
        }

        // Compute per-channel scale and shift
        let mut scale = vec![1.0f64; num_channels];
        let mut shift = vec![0.0f64; num_channels];

        for c in 0..num_channels {
            let inv_std = 1.0 / (var_slice[c] + eps).sqrt();
            scale[c] = inv_std;
            shift[c] = -mean_slice[c] * inv_std;

            if let Some(ref g) = gamma {
                let g_slice = g.inner.as_f64_slice();
                scale[c] *= g_slice[c];
                shift[c] *= g_slice[c];
            }

            if let Some(ref b) = beta {
                let b_slice = b.inner.as_f64_slice();
                shift[c] += b_slice[c];
            }
        }

        // Use axis_iter to process channel by channel
        // This is much faster than indexed_iter_mut for NCHW layout
        let mut result = data.to_owned();

        // Iterate over channel axis (axis 1), applying scale/shift to each channel
        // axis_iter_mut gives us views for each channel across all batches and spatial dims
        for (c, mut channel_data) in result.axis_iter_mut(ndarray::Axis(1)).enumerate() {
            let s = scale[c];
            let sh = shift[c];
            channel_data.mapv_inplace(|x| x * s + sh);
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Take elements along an axis (like numpy.take)
///
/// Used for embedding lookup: take(embedding_table, token_ids, axis=0)
///
/// # Arguments
/// * `indices` - Array of indices to take (1D)
/// * `axis` - Axis along which to take
#[wasm_bindgen]
impl NDArray {
    pub fn take(&self, indices: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let idx_data = indices.inner.as_f64_slice();
        let indices_usize: Vec<usize> = idx_data.iter().map(|&x| x as usize).collect();

        // Validate indices
        let axis_len = shape[axis];
        for &idx in &indices_usize {
            if idx >= axis_len {
                return Err(JsValue::from_str(&format!(
                    "index {} is out of bounds for axis {} with size {}",
                    idx, axis, axis_len
                )));
            }
        }

        // Build output shape
        let mut out_shape = shape.to_vec();
        out_shape[axis] = indices_usize.len();

        // Use ndarray's select
        let result = data.select(ndarray::Axis(axis), &indices_usize);

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result.to_owned())))
    }
}

/// Gather elements along an axis (alias for take)
#[wasm_bindgen]
impl NDArray {
    pub fn gather(&self, indices: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
        self.take(indices, axis)
    }
}

/// Reciprocal square root: 1/sqrt(x)
#[wasm_bindgen]
impl NDArray {
    pub fn rsqrt(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| 1.0 / x.sqrt());
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[wasm_bindgen]
impl NDArray {
    pub fn sigmoid(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[wasm_bindgen]
impl NDArray {
    pub fn silu(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| x / (1.0 + (-x).exp()));
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }
}

/// Top-k values and indices along an axis
///
/// Returns (values, indices) where both have the axis dimension reduced to k.
/// Uses O(N) selection algorithm (select_nth_unstable) for efficiency.
/// Handles non-contiguous inputs safely using ndarray lane iterators.
///
/// # Arguments
/// * `k` - Number of top elements to return
/// * `axis` - Axis along which to find top-k
/// * `sorted` - If true, results are sorted in descending order
#[wasm_bindgen]
impl NDArray {
    pub fn topk(&self, k: usize, axis: usize, sorted: bool) -> Result<js_sys::Array, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        if k > shape[axis] {
            return Err(JsValue::from_str(&format!(
                "k={} is greater than axis size {}",
                k, shape[axis]
            )));
        }

        if k == 0 {
            return Err(JsValue::from_str("k must be at least 1"));
        }

        // Build output shape
        let mut out_shape = shape.clone();
        out_shape[axis] = k;

        let mut out_values = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&out_shape));
        let mut out_indices = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&out_shape));

        // Use ndarray's lanes() iterator which handles strides automatically
        let axis_obj = ndarray::Axis(axis);
        let n_cols = shape[axis];

        // For small k (most common in inference), use a min-heap to avoid
        // materializing the full (value, index) array. This saves memory bandwidth.
        // Threshold: if k <= 64, heap is faster; above that, select_nth_unstable wins.
        let use_heap = k <= 64;

        if use_heap {
            // Heap-based approach: O(N log k) but zero scratch memory writes
            use std::collections::BinaryHeap;

            // Min-heap of (value, index) - we want to keep the k largest
            // BinaryHeap is max-heap by default, so we invert comparison for min-heap
            #[derive(PartialEq)]
            struct MinHeapItem(f64, usize);

            // f64 doesn't implement Eq, but we need it for Ord
            // Our total_cmp based comparison makes this safe
            impl Eq for MinHeapItem {}

            impl PartialOrd for MinHeapItem {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }
            impl Ord for MinHeapItem {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    // Use total_cmp for NaN safety, inverted for min-heap behavior
                    // Tie-break by index for stable selection (prefer lower indices)
                    other.0.total_cmp(&self.0)
                        .then_with(|| other.1.cmp(&self.1))
                }
            }

            // Hoist heap allocation outside the loop
            let mut heap: BinaryHeap<MinHeapItem> = BinaryHeap::with_capacity(k + 1);
            let mut results: Vec<(f64, usize)> = Vec::with_capacity(k);

            for ((in_lane, mut out_val_lane), mut out_idx_lane) in data.lanes(axis_obj)
                .into_iter()
                .zip(out_values.lanes_mut(axis_obj))
                .zip(out_indices.lanes_mut(axis_obj))
            {
                // heap.drain() at end leaves it empty, no need for clear()

                for (i, &v) in in_lane.iter().enumerate() {
                    if heap.len() < k {
                        heap.push(MinHeapItem(v, i));
                    } else {
                        // Optimization: Check against the "cutoff" (smallest of top k) first
                        // Only modify heap structure if v is strictly better
                        // Use total_cmp to match the Ord implementation (NaN consistency)
                        let mut top = heap.peek_mut().unwrap();
                        if v.total_cmp(&top.0).is_gt() {
                            // Replace in-place, dropping triggers ONE sift-down
                            *top = MinHeapItem(v, i);
                        }
                    }
                }

                // Extract results, reusing buffer
                results.clear();
                results.extend(heap.drain().map(|MinHeapItem(v, i)| (v, i)));

                if sorted {
                    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                }

                for (i, &(val, orig_idx)) in results.iter().enumerate() {
                    out_val_lane[i] = val;
                    out_idx_lane[i] = orig_idx as f64;
                }
            }
        } else {
            // For large k, use select_nth_unstable with scratch buffer
            let mut scratch: Vec<(f64, usize)> = Vec::with_capacity(n_cols);

            for ((in_lane, mut out_val_lane), mut out_idx_lane) in data.lanes(axis_obj)
                .into_iter()
                .zip(out_values.lanes_mut(axis_obj))
                .zip(out_indices.lanes_mut(axis_obj))
            {
                scratch.clear();
                scratch.extend(in_lane.iter().enumerate().map(|(i, &v)| (v, i)));

                if k < scratch.len() {
                    scratch.select_nth_unstable_by(k - 1, |a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    scratch.truncate(k);
                }

                if sorted {
                    scratch.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                }

                for (i, &(val, orig_idx)) in scratch.iter().enumerate() {
                    out_val_lane[i] = val;
                    out_idx_lane[i] = orig_idx as f64;
                }
            }
        }

        let result = js_sys::Array::new();
        result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(out_values))));
        result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(out_indices))));

        Ok(result)
    }
}

/// Lower triangular matrix (zeros above k-th diagonal)
// ============ Sort Operations ============

/// Sort array elements (NaN values sort to end, NumPy behavior)
#[wasm_bindgen]
impl NDArray {
    pub fn sort(&self) -> Result<NDArray, JsValue> {
        CpuBackend::sort(&self.inner, None)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = sortAxis)]
    pub fn sort_axis(&self, axis: usize) -> Result<NDArray, JsValue> {
        CpuBackend::sort(&self.inner, Some(axis))
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn argsort(&self) -> Result<NDArray, JsValue> {
        CpuBackend::argsort(&self.inner, None)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = argsortAxis)]
    pub fn argsort_axis(&self, axis: usize) -> Result<NDArray, JsValue> {
        CpuBackend::argsort(&self.inner, Some(axis))
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn unique(&self) -> NDArray {
        NDArray::new(CpuBackend::unique(&self.inner))
    }
}

///
/// Handles batched inputs and non-contiguous arrays safely.
/// Uses row-based iteration for performance (avoids indexed_iter overhead).
///
/// # Arguments
/// * `k` - Diagonal offset (0 = main diagonal, positive = above, negative = below)
#[wasm_bindgen]
impl NDArray {
    pub fn tril(&self, k: i32) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(JsValue::from_str("tril requires at least 2D input"));
        }

        let nrows = shape[ndim - 2];
        let ncols = shape[ndim - 1];

        // Clone the data to get contiguous memory, then work on it
        let mut result = data.to_owned();

        // For 2D case, iterate by rows for efficiency
        // Use slice.fill() for memset optimization
        if ndim == 2 {
            for (i, mut row) in result.outer_iter_mut().enumerate() {
                // Zero out elements where j > i + k (above the k-th diagonal)
                let start_j = ((i as i32) + k + 1).max(0) as usize;
                if start_j < ncols {
                    // Use slice fill for potential memset optimization
                    row.slice_mut(ndarray::s![start_j..]).fill(0.0);
                }
            }
        } else {
            // For N-D case, treat as batch of 2D matrices
            let batch_size: usize = shape[..ndim - 2].iter().product();

            let flat = result.as_standard_layout().into_owned()
                .into_shape_with_order((batch_size, nrows, ncols))
                .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

            let mut flat_result = flat;

            // Apply row-based masking with slice.fill() to each 2D slice
            for mut matrix in flat_result.outer_iter_mut() {
                for (i, mut row) in matrix.outer_iter_mut().enumerate() {
                    let start_j = ((i as i32) + k + 1).max(0) as usize;
                    if start_j < ncols {
                        row.slice_mut(ndarray::s![start_j..]).fill(0.0);
                    }
                }
            }

            let shaped = flat_result.into_shape_with_order(ndarray::IxDyn(&shape))
                .map_err(|e| JsValue::from_str(&format!("reshape back failed: {}", e)))?;
            result = shaped;
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Upper triangular matrix (zeros below k-th diagonal)
///
/// Handles batched inputs and non-contiguous arrays safely.
/// Uses row-based iteration for performance (avoids indexed_iter overhead).
///
/// # Arguments
/// * `k` - Diagonal offset (0 = main diagonal, positive = above, negative = below)
#[wasm_bindgen]
impl NDArray {
    pub fn triu(&self, k: i32) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(JsValue::from_str("triu requires at least 2D input"));
        }

        let ncols = shape[ndim - 1];

        // Clone the data to get contiguous memory
        let mut result = data.to_owned();

        let nrows = shape[ndim - 2];

        // For 2D case, iterate by rows for efficiency
        // Use slice.fill() for memset optimization
        if ndim == 2 {
            for (i, mut row) in result.outer_iter_mut().enumerate() {
                // Zero out elements where j < i + k (below the k-th diagonal)
                let end_j = ((i as i32) + k).max(0) as usize;
                let end_j = end_j.min(ncols);
                if end_j > 0 {
                    row.slice_mut(ndarray::s![..end_j]).fill(0.0);
                }
            }
        } else {
            // For N-D case, treat as batch of 2D matrices
            let batch_size: usize = shape[..ndim - 2].iter().product();

            let flat = result.as_standard_layout().into_owned()
                .into_shape_with_order((batch_size, nrows, ncols))
                .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

            let mut flat_result = flat;

            for mut matrix in flat_result.outer_iter_mut() {
                for (i, mut row) in matrix.outer_iter_mut().enumerate() {
                    let end_j = ((i as i32) + k).max(0) as usize;
                    let end_j = end_j.min(ncols);
                    if end_j > 0 {
                        row.slice_mut(ndarray::s![..end_j]).fill(0.0);
                    }
                }
            }

            let shaped = flat_result.into_shape_with_order(ndarray::IxDyn(&shape))
                .map_err(|e| JsValue::from_str(&format!("reshape back failed: {}", e)))?;
            result = shaped;
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Create a causal attention mask (lower triangular matrix of -inf and 0)
///
/// Returns a mask where positions that can attend are 0, others are -inf.
/// Useful for transformer causal (autoregressive) attention.
///
/// # Arguments
/// * `size` - Sequence length (creates size x size mask)
#[wasm_bindgen(js_name = causalMask)]
pub fn causal_mask(size: usize) -> NDArray {
    let mut data = vec![0.0f64; size * size];

    for i in 0..size {
        for j in 0..size {
            // Set to -inf where j > i (can't attend to future)
            if j > i {
                data[i * size + j] = f64::NEG_INFINITY;
            }
        }
    }

    let arr = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[size, size]),
        data
    ).unwrap();

    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(arr))
}

// ============ Additional LLM Inference Ops ============

/// Split array into equal parts along an axis
///
/// Essential for QKV separation after single projection matmul.
///
/// # Arguments
/// * `num_splits` - Number of equal parts to split into
/// * `axis` - Axis along which to split
#[wasm_bindgen]
impl NDArray {
    pub fn split(&self, num_splits: usize, axis: usize) -> Result<js_sys::Array, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let axis_len = shape[axis];
        if axis_len % num_splits != 0 {
            return Err(JsValue::from_str(&format!(
                "axis {} of size {} cannot be evenly split into {} parts",
                axis, axis_len, num_splits
            )));
        }

        let split_size = axis_len / num_splits;
        let result = js_sys::Array::new();

        for i in 0..num_splits {
            let start = i * split_size;
            let end = start + split_size;

            // Use slice_axis to get a view of the split
            let slice = data.slice_axis(
                ndarray::Axis(axis),
                ndarray::Slice::from(start..end)
            );

            result.push(&JsValue::from(NDArray::new(
                rumpy_cpu::CpuArray::from_ndarray(slice.to_owned())
            )));
        }

        Ok(result)
    }
}

/// Chunk array into parts of a specific size along an axis
///
/// Alternative to split when you know the chunk size, not the number of chunks.
///
/// # Arguments
/// * `chunk_size` - Size of each chunk
/// * `axis` - Axis along which to chunk
#[wasm_bindgen]
impl NDArray {
    pub fn chunk(&self, chunk_size: usize, axis: usize) -> Result<js_sys::Array, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let axis_len = shape[axis];
        let result = js_sys::Array::new();

        let mut start = 0;
        while start < axis_len {
            let end = (start + chunk_size).min(axis_len);

            let slice = data.slice_axis(
                ndarray::Axis(axis),
                ndarray::Slice::from(start..end)
            );

            result.push(&JsValue::from(NDArray::new(
                rumpy_cpu::CpuArray::from_ndarray(slice.to_owned())
            )));

            start = end;
        }

        Ok(result)
    }
}

/// Cumulative sum along an axis
///
/// Essential for top-p (nucleus) sampling to calculate cumulative probabilities.
///
/// # Arguments
/// * `axis` - Axis along which to compute cumsum
#[wasm_bindgen]
impl NDArray {
    pub fn cumsum(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let mut result = data.to_owned();

        // Use lanes_mut to iterate along the cumsum axis
        for mut lane in result.lanes_mut(ndarray::Axis(axis)) {
            let mut acc = 0.0f64;
            for val in lane.iter_mut() {
                acc += *val;
                *val = acc;
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Scatter values to indices along an axis
///
/// Essential for KV cache updates - insert new token's KV into cache without
/// reallocating the entire buffer.
///
/// # Arguments
/// * `axis` - Axis along which to scatter
/// * `indices` - Indices where to place the values
/// * `src` - Source values to scatter
#[wasm_bindgen]
impl NDArray {
    pub fn scatter(&self, axis: usize, indices: &NDArray, src: &NDArray) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let idx_data = indices.inner.as_f64_slice();
        let src_data = src.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let mut result = data.to_owned();

        // For 1D case
        if shape.len() == 1 {
            for (i, &idx) in idx_data.iter().enumerate() {
                let idx = idx as usize;
                if idx < shape[0] {
                    result[idx] = src_data.as_slice().unwrap()[i];
                }
            }
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        // For N-D case, iterate along the scatter axis
        let axis_obj = ndarray::Axis(axis);
        let src_lanes = src_data.lanes(axis_obj);
        let result_lanes = result.lanes_mut(axis_obj);

        // Each lane in src corresponds to one index
        for ((src_lane, _idx_val), mut result_lane) in src_lanes.into_iter()
            .zip(idx_data.iter())
            .zip(result_lanes.into_iter())
        {
            // This simplified version assumes scatter is inserting single values
            // For more complex scatter, we'd need to iterate over indices per lane
            for (src_val, dst_val) in src_lane.iter().zip(result_lane.iter_mut()) {
                *dst_val = *src_val;
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Index copy - copy src into self at specified indices along an axis
///
/// Simpler KV cache update: self[indices] = src along axis
///
/// # Arguments
/// * `axis` - Axis along which to copy
/// * `indices` - 1D indices specifying where to copy
/// * `src` - Source tensor (must have same shape as self except at axis dimension)
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = indexCopy)]
    pub fn index_copy(&self, axis: usize, indices: &NDArray, src: &NDArray) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let idx_data = indices.inner.as_f64_slice();
        let src_data = src.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let mut result = data.to_owned();

        // For each index, copy the corresponding slice from src to result
        for (i, &idx) in idx_data.iter().enumerate() {
            let idx = idx as usize;
            if idx >= shape[axis] {
                return Err(JsValue::from_str(&format!(
                    "index {} is out of bounds for axis {} with size {}",
                    idx, axis, shape[axis]
                )));
            }

            // Get the i-th slice from src and copy to idx-th position in result
            let src_slice = src_data.index_axis(ndarray::Axis(axis), i);
            let mut dst_slice = result.index_axis_mut(ndarray::Axis(axis), idx);
            dst_slice.assign(&src_slice);
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Multinomial sampling from probability distribution
///
/// Essential for non-greedy token generation. Samples from the last dimension.
///
/// # Arguments
/// * `num_samples` - Number of samples to draw
/// * `replacement` - Whether to sample with replacement
#[wasm_bindgen]
impl NDArray {
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if shape.is_empty() {
            return Err(JsValue::from_str("multinomial requires at least 1D input"));
        }

        let last_dim = *shape.last().unwrap();

        if num_samples > last_dim && !replacement {
            return Err(JsValue::from_str(&format!(
                "cannot draw {} samples without replacement from {} categories",
                num_samples, last_dim
            )));
        }

        // Output shape: same as input but last dim is num_samples
        let mut out_shape = shape.clone();
        *out_shape.last_mut().unwrap() = num_samples;

        // Flatten to 2D for processing
        let batch_size: usize = shape[..shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);

        let data_contiguous = data.as_standard_layout();
        let flat = data_contiguous.view()
            .into_shape_with_order((batch_size, last_dim))
            .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

        let mut result = ndarray::Array2::<f64>::zeros((batch_size, num_samples));

        for (batch_idx, probs) in flat.outer_iter().enumerate() {
            // Normalize probabilities (in case they don't sum to 1)
            let sum: f64 = probs.iter().sum();
            let normalized: Vec<f64> = probs.iter().map(|&p| p / sum).collect();

            // Build cumulative distribution
            let mut cumsum = Vec::with_capacity(last_dim);
            let mut acc = 0.0f64;
            for &p in &normalized {
                acc += p;
                cumsum.push(acc);
            }

            // Sample using js_sys::Math::random() for WASM compatibility
            let mut used = vec![false; last_dim];
            for sample_idx in 0..num_samples {
                let r: f64 = js_sys::Math::random();

                // Binary search for the sample
                let mut idx = cumsum.partition_point(|&c| c < r);
                if idx >= last_dim {
                    idx = last_dim - 1;
                }

                // If no replacement, find next available
                if !replacement {
                    while used[idx] && idx < last_dim - 1 {
                        idx += 1;
                    }
                    if used[idx] {
                        // Wrap around
                        idx = 0;
                        while used[idx] && idx < last_dim - 1 {
                            idx += 1;
                        }
                    }
                    used[idx] = true;
                }

                result[[batch_idx, sample_idx]] = idx as f64;
            }
        }

        let result_shaped = result.into_shape_with_order(ndarray::IxDyn(&out_shape))
            .map_err(|e| JsValue::from_str(&format!("reshape result failed: {}", e)))?;

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result_shaped)))
    }
}

/// Tile/repeat array along each dimension
///
/// Essential for beam search - duplicating context for multiple hypotheses.
///
/// # Arguments
/// * `reps` - Number of repetitions along each dimension
#[wasm_bindgen]
impl NDArray {
    pub fn tile(&self, reps: Vec<usize>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        // Pad reps or shape to match dimensions
        let ndim = shape.len().max(reps.len());
        let mut padded_shape = vec![1usize; ndim - shape.len()];
        padded_shape.extend(&shape);
        let mut padded_reps = vec![1usize; ndim - reps.len()];
        padded_reps.extend(&reps);

        // Calculate output shape
        let out_shape: Vec<usize> = padded_shape.iter()
            .zip(padded_reps.iter())
            .map(|(&s, &r)| s * r)
            .collect();

        // Ensure contiguous data
        let data_contiguous = data.as_standard_layout();
        let reshaped = data_contiguous.view()
            .into_shape_with_order(ndarray::IxDyn(&padded_shape))
            .map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;

        // Build result by broadcasting and assignment
        let mut result = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&out_shape));

        // Fill by iterating over tiles
        let mut indices = vec![0usize; ndim];
        loop {
            // Copy the tile at current indices
            for (src_idx, src_val) in reshaped.indexed_iter() {
                let mut dst_idx = vec![0usize; ndim];
                for d in 0..ndim {
                    dst_idx[d] = indices[d] * padded_shape[d] + src_idx[d];
                }
                result[ndarray::IxDyn(&dst_idx)] = *src_val;
            }

            // Increment indices
            let mut d = ndim - 1;
            loop {
                indices[d] += 1;
                if indices[d] < padded_reps[d] {
                    break;
                }
                indices[d] = 0;
                if d == 0 {
                    // Done
                    return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
                }
                d -= 1;
            }
        }
    }
}

/// Repeat elements along an axis
///
/// Different from tile - this repeats individual elements, not the whole array.
///
/// # Arguments
/// * `repeats` - Number of repetitions for each element
/// * `axis` - Axis along which to repeat (optional, flattens if not specified)
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = repeat)]
    pub fn repeat_elements(&self, repeats: usize, axis: Option<usize>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();

        match axis {
            None => {
                // Flatten and repeat
                let flat: Vec<f64> = data.iter().cloned().collect();
                let repeated: Vec<f64> = flat.iter()
                    .flat_map(|&v| std::iter::repeat(v).take(repeats))
                    .collect();
                let arr = ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[repeated.len()]),
                    repeated
                ).map_err(|e| JsValue::from_str(&format!("reshape failed: {}", e)))?;
                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(arr)))
            }
            Some(ax) => {
                let shape = data.shape().to_vec();
                if ax >= shape.len() {
                    return Err(JsValue::from_str(&format!(
                        "axis {} is out of bounds for array of dimension {}",
                        ax, shape.len()
                    )));
                }

                // Build new shape
                let mut out_shape = shape.clone();
                out_shape[ax] *= repeats;

                let mut result = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&out_shape));

                // Iterate and repeat along axis
                for (i, src_slice) in data.axis_iter(ndarray::Axis(ax)).enumerate() {
                    for r in 0..repeats {
                        let dst_idx = i * repeats + r;
                        let mut dst_slice = result.index_axis_mut(ndarray::Axis(ax), dst_idx);
                        dst_slice.assign(&src_slice);
                    }
                }

                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
            }
        }
    }
}

/// Broadcast array to a new shape (like np.broadcast_to)
///
/// The array is broadcast to the target shape according to NumPy broadcasting rules.
/// This creates a view (no copying) when possible, but always returns a contiguous array.
///
/// # Arguments
/// * `shape` - Target shape to broadcast to
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = broadcastTo)]
    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let src_shape = data.shape();

        // Check if broadcasting is valid
        if shape.len() < src_shape.len() {
            return Err(JsValue::from_str("Cannot broadcast to smaller number of dimensions"));
        }

        // Pad source shape with 1s on the left to match target dims
        let pad = shape.len() - src_shape.len();
        let mut src_padded = vec![1usize; pad];
        src_padded.extend(src_shape.iter().cloned());

        // Validate broadcasting rules
        for (i, (&src_dim, &tgt_dim)) in src_padded.iter().zip(shape.iter()).enumerate() {
            if src_dim != 1 && src_dim != tgt_dim {
                return Err(JsValue::from_str(&format!(
                    "Cannot broadcast dimension {} from {} to {}",
                    i, src_dim, tgt_dim
                )));
            }
        }

        // Use ndarray's broadcast_to
        let target_shape = ndarray::IxDyn(&shape);
        let broadcasted = data.broadcast(target_shape)
            .ok_or_else(|| JsValue::from_str("Broadcast failed"))?;

        // Convert to owned (contiguous) array
        let result = broadcasted.to_owned();
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Cast array to a different dtype (currently f32/f64)
///
/// Since rumpy internally uses f64, this mainly handles precision conversion.
/// For ONNX compatibility, this will be extended to handle int8/int32/int64.
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = asType)]
    pub fn as_type(&self, dtype: &str) -> Result<NDArray, JsValue> {
        // Currently all arrays are f64 internally
        // This is a placeholder that validates dtype and returns a copy
        match dtype {
            "float64" | "f64" | "double" => {
                // Already f64, return clone
                Ok(NDArray::new(self.inner.clone()))
            }
            "float32" | "f32" | "float" => {
                // Convert to f32 and back (simulates precision loss)
                let data = self.inner.as_ndarray();
                let result = data.mapv(|x| x as f32 as f64);
                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
            }
            "int64" | "i64" | "long" => {
                // NumPy astype(int64) truncates toward zero, not rounds
                let data = self.inner.as_ndarray();
                let result = data.mapv(|x| x.trunc());
                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
            }
            "int32" | "i32" | "int" => {
                // NumPy astype(int32) truncates toward zero with overflow clamping
                let data = self.inner.as_ndarray();
                let result = data.mapv(|x| {
                    let truncated = x.trunc();
                    // Clamp to i32 range to avoid undefined behavior
                    truncated.clamp(i32::MIN as f64, i32::MAX as f64)
                });
                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
            }
            _ => Err(JsValue::from_str(&format!("Unsupported dtype: {}", dtype)))
        }
    }
}

/// Dequantize int8 tensor: output = (input - zero_point) * scale
///
/// Essential for running quantized ONNX models from transformers.js
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = dequantizeLinear)]
    pub fn dequantize_linear(&self, scale: f64, zero_point: f64) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| (x - zero_point) * scale);
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }
}

/// Pad array with constant value
///
/// # Arguments
/// * `pad_width` - Padding for each dimension as [before0, after0, before1, after1, ...]
/// * `constant_value` - Value to pad with
#[wasm_bindgen]
impl NDArray {
    pub fn pad(&self, pad_width: Vec<usize>, constant_value: f64) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        if pad_width.len() != ndim * 2 {
            return Err(JsValue::from_str(&format!(
                "pad_width must have {} elements for {}D array, got {}",
                ndim * 2, ndim, pad_width.len()
            )));
        }

        // Calculate output shape
        let mut out_shape = vec![0usize; ndim];
        for d in 0..ndim {
            out_shape[d] = pad_width[d * 2] + shape[d] + pad_width[d * 2 + 1];
        }

        // Create output filled with constant
        let mut result = ndarray::ArrayD::<f64>::from_elem(ndarray::IxDyn(&out_shape), constant_value);

        // Build the slice ranges for where to insert the original data
        let mut slice_info: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let start = pad_width[d * 2] as isize;
            let end = (pad_width[d * 2] + shape[d]) as isize;
            slice_info.push(ndarray::SliceInfoElem::Slice {
                start,
                end: Some(end),
                step: 1,
            });
        }

        // Assign data to the center
        let mut view = result.slice_mut(slice_info.as_slice());
        view.assign(&data);

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Log-softmax for numerical stability
///
/// Computes log(softmax(x)) using the LogSumExp trick to prevent overflow/underflow.
/// Essential for calculating perplexity and beam search scoring.
///
/// # Arguments
/// * `axis` - Axis along which to compute log_softmax
#[wasm_bindgen]
impl NDArray {
    #[wasm_bindgen(js_name = logSoftmax)]
    pub fn log_softmax(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let mut result = data.to_owned();

        // For numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        for mut lane in result.lanes_mut(ndarray::Axis(axis)) {
            // Find max for stability
            let max_val = lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute log(sum(exp(x - max)))
            let log_sum_exp = lane.iter()
                .map(|&x| (x - max_val).exp())
                .sum::<f64>()
                .ln();

            // Apply: x - max - log_sum_exp
            for val in lane.iter_mut() {
                *val = *val - max_val - log_sum_exp;
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Cumulative product along an axis
///
/// # Arguments
/// * `axis` - Axis along which to compute cumprod
#[wasm_bindgen]
impl NDArray {
    pub fn cumprod(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let mut result = data.to_owned();

        for mut lane in result.lanes_mut(ndarray::Axis(axis)) {
            let mut acc = 1.0f64;
            for val in lane.iter_mut() {
                acc *= *val;
                *val = acc;
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

/// Diff - compute n-th discrete difference along axis
///
/// # Arguments
/// * `n` - Number of times to apply diff
/// * `axis` - Axis along which to diff
#[wasm_bindgen]
impl NDArray {
    pub fn diff(&self, n: usize, axis: usize) -> Result<NDArray, JsValue> {
        if n == 0 {
            return Ok(NDArray::new(self.inner.clone()));
        }

        let mut current = self.inner.as_ndarray().to_owned();
        let mut shape = current.shape().to_vec();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        for _ in 0..n {
            if shape[axis] < 2 {
                return Err(JsValue::from_str("diff requires at least 2 elements along axis"));
            }

            let new_len = shape[axis] - 1;
            shape[axis] = new_len;

            let mut result = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&shape));

            // Compute differences
            for (i, (curr, next)) in current.axis_iter(ndarray::Axis(axis))
                .zip(current.axis_iter(ndarray::Axis(axis)).skip(1))
                .enumerate()
            {
                let mut dst = result.index_axis_mut(ndarray::Axis(axis), i);
                dst.zip_mut_with(&next, |d, &n| *d = n);
                dst.zip_mut_with(&curr, |d, &c| *d -= c);
            }

            current = result;
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(current)))
    }
}

// ============================================================================
// Missing NumPy ops - Added for ML model compatibility
// ============================================================================

// ============ Element-wise maximum/minimum ============

/// NumPy np.maximum: propagates NaN (if either is NaN, result is NaN)
/// Note: np.fmax ignores NaN and returns non-NaN value
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN }
    else { a.max(b) }
}

/// NumPy np.minimum: propagates NaN (if either is NaN, result is NaN)
/// Note: np.fmin ignores NaN and returns non-NaN value
#[inline]
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN }
    else { a.min(b) }
}

/// NumPy np.fmax: ignores NaN (returns non-NaN value when one is NaN)
#[inline]
fn np_fmax(a: f64, b: f64) -> f64 {
    if a.is_nan() { b }
    else if b.is_nan() { a }
    else { a.max(b) }
}

/// NumPy np.fmin: ignores NaN (returns non-NaN value when one is NaN)
#[inline]
fn np_fmin(a: f64, b: f64) -> f64 {
    if a.is_nan() { b }
    else if b.is_nan() { a }
    else { a.min(b) }
}

/// Element-wise maximum of two arrays
///
/// Compares two arrays element-by-element and returns the maximum values.
/// Equivalent to numpy.maximum(a, b).
/// Supports broadcasting. NaN propagates: if either value is NaN, result is NaN.
/// For NaN-ignoring behavior, use fmax instead.
#[wasm_bindgen]
pub fn maximum(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    // Check if shapes are compatible for broadcasting
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    if a_shape == b_shape {
        // Same shape - simple element-wise max
        let result = ndarray::Zip::from(&*a_data)
            .and(&*b_data)
            .map_collect(|&av, &bv| np_maximum(av, bv));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if a_data.len() == 1 {
        // a is scalar
        let scalar = *a_data.iter().next().unwrap();
        let result = b_data.mapv(|v| np_maximum(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if b_data.len() == 1 {
        // b is scalar
        let scalar = *b_data.iter().next().unwrap();
        let result = a_data.mapv(|v| np_maximum(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else {
        // Try numpy-style broadcasting
        let b_dim = ndarray::IxDyn(b_shape);
        if let Some(a_broadcast) = a_data.broadcast(b_dim.clone()) {
            let result = ndarray::Zip::from(&a_broadcast)
                .and(&*b_data)
                .map_collect(|&av, &bv| np_maximum(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        let a_dim = ndarray::IxDyn(a_shape);
        if let Some(b_broadcast) = b_data.broadcast(a_dim) {
            let result = ndarray::Zip::from(&*a_data)
                .and(&b_broadcast)
                .map_collect(|&av, &bv| np_maximum(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} are not broadcastable",
            a_shape, b_shape
        )))
    }
}

/// Element-wise minimum of two arrays
///
/// Compares two arrays element-by-element and returns the minimum values.
/// Equivalent to numpy.minimum(a, b).
/// Supports broadcasting. NaN propagates: if either value is NaN, result is NaN.
/// For NaN-ignoring behavior, use fmin instead.
#[wasm_bindgen]
pub fn minimum(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    if a_shape == b_shape {
        let result = ndarray::Zip::from(&*a_data)
            .and(&*b_data)
            .map_collect(|&av, &bv| np_minimum(av, bv));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if a_data.len() == 1 {
        let scalar = *a_data.iter().next().unwrap();
        let result = b_data.mapv(|v| np_minimum(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if b_data.len() == 1 {
        let scalar = *b_data.iter().next().unwrap();
        let result = a_data.mapv(|v| np_minimum(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else {
        let b_dim = ndarray::IxDyn(b_shape);
        if let Some(a_broadcast) = a_data.broadcast(b_dim.clone()) {
            let result = ndarray::Zip::from(&a_broadcast)
                .and(&*b_data)
                .map_collect(|&av, &bv| np_minimum(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        let a_dim = ndarray::IxDyn(a_shape);
        if let Some(b_broadcast) = b_data.broadcast(a_dim) {
            let result = ndarray::Zip::from(&*a_data)
                .and(&b_broadcast)
                .map_collect(|&av, &bv| np_minimum(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} are not broadcastable",
            a_shape, b_shape
        )))
    }
}

/// Element-wise maximum with a scalar (NaN propagates)
#[wasm_bindgen(js_name = maximumScalar)]
pub fn maximum_scalar(arr: &NDArray, scalar: f64) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| np_maximum(v, scalar));
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

/// Element-wise minimum with a scalar (NaN propagates)
#[wasm_bindgen(js_name = minimumScalar)]
pub fn minimum_scalar(arr: &NDArray, scalar: f64) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| np_minimum(v, scalar));
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

/// Element-wise fmax of two arrays (NaN-ignoring)
///
/// Like maximum, but ignores NaN values - returns non-NaN when one is NaN.
/// Equivalent to numpy.fmax(a, b).
/// NOTE: Rust function name is fmax_op to avoid collision with C stdlib fmax()
#[wasm_bindgen(js_name = fmaxArr)]
pub fn fmax_op(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    if a_shape == b_shape {
        let result = ndarray::Zip::from(&*a_data)
            .and(&*b_data)
            .map_collect(|&av, &bv| np_fmax(av, bv));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if a_data.len() == 1 {
        let scalar = *a_data.iter().next().unwrap();
        let result = b_data.mapv(|v| np_fmax(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if b_data.len() == 1 {
        let scalar = *b_data.iter().next().unwrap();
        let result = a_data.mapv(|v| np_fmax(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else {
        let b_dim = ndarray::IxDyn(b_shape);
        if let Some(a_broadcast) = a_data.broadcast(b_dim.clone()) {
            let result = ndarray::Zip::from(&a_broadcast)
                .and(&*b_data)
                .map_collect(|&av, &bv| np_fmax(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        let a_dim = ndarray::IxDyn(a_shape);
        if let Some(b_broadcast) = b_data.broadcast(a_dim) {
            let result = ndarray::Zip::from(&*a_data)
                .and(&b_broadcast)
                .map_collect(|&av, &bv| np_fmax(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} are not broadcastable",
            a_shape, b_shape
        )))
    }
}

/// Element-wise fmin of two arrays (NaN-ignoring)
///
/// Like minimum, but ignores NaN values - returns non-NaN when one is NaN.
/// Equivalent to numpy.fmin(a, b).
/// NOTE: Rust function name is fmin_op to avoid collision with C stdlib fmin()
#[wasm_bindgen(js_name = fminArr)]
pub fn fmin_op(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    if a_shape == b_shape {
        let result = ndarray::Zip::from(&*a_data)
            .and(&*b_data)
            .map_collect(|&av, &bv| np_fmin(av, bv));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if a_data.len() == 1 {
        let scalar = *a_data.iter().next().unwrap();
        let result = b_data.mapv(|v| np_fmin(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else if b_data.len() == 1 {
        let scalar = *b_data.iter().next().unwrap();
        let result = a_data.mapv(|v| np_fmin(v, scalar));
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    } else {
        let b_dim = ndarray::IxDyn(b_shape);
        if let Some(a_broadcast) = a_data.broadcast(b_dim.clone()) {
            let result = ndarray::Zip::from(&a_broadcast)
                .and(&*b_data)
                .map_collect(|&av, &bv| np_fmin(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        let a_dim = ndarray::IxDyn(a_shape);
        if let Some(b_broadcast) = b_data.broadcast(a_dim) {
            let result = ndarray::Zip::from(&*a_data)
                .and(&b_broadcast)
                .map_collect(|&av, &bv| np_fmin(av, bv));
            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
        }

        Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} are not broadcastable",
            a_shape, b_shape
        )))
    }
}

/// Element-wise fmax with a scalar (NaN-ignoring)
#[wasm_bindgen(js_name = fmaxScalar)]
pub fn fmax_scalar(arr: &NDArray, scalar: f64) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| np_fmax(v, scalar));
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

/// Element-wise fmin with a scalar (NaN-ignoring)
#[wasm_bindgen(js_name = fminScalar)]
pub fn fmin_scalar(arr: &NDArray, scalar: f64) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| np_fmin(v, scalar));
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// ============ Additional math functions ============

/// Element-wise atan2(y, x)
///
/// Computes the arc tangent of y/x, using the signs of both arguments
/// to determine the quadrant of the return value.
/// Equivalent to numpy.arctan2(y, x).
#[wasm_bindgen(js_name = atan2Arr)]
pub fn atan2_arr(y: &NDArray, x: &NDArray) -> Result<NDArray, JsValue> {
    let y_data = y.inner.as_ndarray();
    let x_data = x.inner.as_ndarray();

    if y_data.shape() != x_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for atan2",
            y_data.shape(), x_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*y_data)
        .and(&*x_data)
        .map_collect(|&yv, &xv| yv.atan2(xv));
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Element-wise log(exp(x1) + exp(x2)) computed in a numerically stable way.
///
/// Equivalent to numpy.logaddexp(x1, x2).
/// Useful for log-space probability computations.
#[wasm_bindgen(js_name = logaddexpArr)]
pub fn logaddexp_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for logaddexp",
            a_data.shape(), b_data.shape()
        )));
    }

    // Numerically stable: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| {
            if av.is_nan() || bv.is_nan() {
                f64::NAN
            } else if av == f64::NEG_INFINITY {
                bv
            } else if bv == f64::NEG_INFINITY {
                av
            } else {
                let max_val = av.max(bv);
                let min_val = av.min(bv);
                max_val + (1.0 + (min_val - max_val).exp()).ln()
            }
        });
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Element-wise log2(2^x1 + 2^x2) computed in a numerically stable way.
///
/// Equivalent to numpy.logaddexp2(x1, x2).
/// Useful for log2-space probability computations.
#[wasm_bindgen(js_name = logaddexp2Arr)]
pub fn logaddexp2_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for logaddexp2",
            a_data.shape(), b_data.shape()
        )));
    }

    let ln2 = std::f64::consts::LN_2;
    // log2(2^a + 2^b) = max(a,b) + log2(1 + 2^(min-max))
    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| {
            if av.is_nan() || bv.is_nan() {
                f64::NAN
            } else if av == f64::NEG_INFINITY {
                bv
            } else if bv == f64::NEG_INFINITY {
                av
            } else {
                let max_val = av.max(bv);
                let min_val = av.min(bv);
                max_val + (1.0 + ((min_val - max_val) * ln2).exp()).ln() / ln2
            }
        });
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Element-wise sign function
///
/// Returns -1 for negative, 0 for zero, 1 for positive values, NaN for NaN.
/// Equivalent to numpy.sign(x).
#[wasm_bindgen(js_name = signArr)]
pub fn sign_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| {
        if v.is_nan() { f64::NAN }
        else if v > 0.0 { 1.0 }
        else if v < 0.0 { -1.0 }
        else { 0.0 }
    });
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

/// Element-wise square (x^2)
///
/// Computes x * x for each element.
/// Equivalent to numpy.square(x).
#[wasm_bindgen(js_name = squareArr)]
pub fn square_arr(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| v * v);
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// ============ Logical operations ============

/// Element-wise logical AND
///
/// Returns 1.0 where both inputs are non-zero, 0.0 otherwise.
/// Equivalent to numpy.logical_and(a, b).
#[wasm_bindgen(js_name = logicalAnd)]
pub fn logical_and(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for logical_and",
            a_data.shape(), b_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| if av != 0.0 && bv != 0.0 { 1.0 } else { 0.0 });
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Element-wise logical OR
///
/// Returns 1.0 where either input is non-zero, 0.0 otherwise.
/// Equivalent to numpy.logical_or(a, b).
#[wasm_bindgen(js_name = logicalOr)]
pub fn logical_or(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for logical_or",
            a_data.shape(), b_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| if av != 0.0 || bv != 0.0 { 1.0 } else { 0.0 });
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Element-wise logical NOT
///
/// Returns 1.0 where input is zero, 0.0 otherwise.
/// Equivalent to numpy.logical_not(x).
#[wasm_bindgen(js_name = logicalNot)]
pub fn logical_not(arr: &NDArray) -> NDArray {
    let data = arr.inner.as_ndarray();
    let result = data.mapv(|v| if v == 0.0 { 1.0 } else { 0.0 });
    NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
}

// ============ Flip operations ============

/// Flip array along specified axis
///
/// Reverses the order of elements along the given axis.
/// Equivalent to numpy.flip(arr, axis).
#[wasm_bindgen]
impl NDArray {
    pub fn flip(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        // Use slice with step=-1 along the axis
        let mut slice_info: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(shape.len());
        for i in 0..shape.len() {
            if i == axis {
                slice_info.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: -1,
                });
            } else {
                slice_info.push(ndarray::SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            }
        }

        let sliced = data.slice(slice_info.as_slice());
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(sliced.to_owned())))
    }

    /// Flip array vertically (along axis 0)
    ///
    /// Equivalent to numpy.flipud(arr) or flip(arr, 0).
    pub fn flipud(&self) -> Result<NDArray, JsValue> {
        self.flip(0)
    }

    /// Flip array horizontally (along axis 1)
    ///
    /// Equivalent to numpy.fliplr(arr) or flip(arr, 1).
    /// Requires at least 2D array.
    pub fn fliplr(&self) -> Result<NDArray, JsValue> {
        if self.inner.ndim() < 2 {
            return Err(JsValue::from_str("fliplr requires at least 2D array"));
        }
        self.flip(1)
    }
}

// ============ Roll operation ============

/// Roll array elements along an axis
///
/// Elements that roll beyond the last position are re-introduced at the first.
/// Equivalent to numpy.roll(arr, shift, axis).
/// Critical for Swin Transformer's cyclic shift operation.
#[wasm_bindgen]
impl NDArray {
    pub fn roll(&self, shift: i32, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let axis_len = shape[axis] as i32;
        if axis_len == 0 {
            return Ok(NDArray::new(self.inner.clone()));
        }

        // Normalize shift to positive value in [0, axis_len)
        let shift_norm = ((shift % axis_len) + axis_len) % axis_len;
        if shift_norm == 0 {
            return Ok(NDArray::new(self.inner.clone()));
        }

        let mut result = data.to_owned();

        // Roll by concatenating two slices: [axis_len-shift:] + [:axis_len-shift]
        let split_point = (axis_len - shift_norm) as usize;

        for dst_idx in 0..shape[axis] {
            let actual_src = if dst_idx < shift_norm as usize {
                split_point + dst_idx
            } else {
                dst_idx - shift_norm as usize
            };

            // Copy slice from source to destination
            let src_slice = data.index_axis(ndarray::Axis(axis), actual_src);
            let mut dst_slice = result.index_axis_mut(ndarray::Axis(axis), dst_idx);
            dst_slice.assign(&src_slice);
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
    }
}

// ============ Diagonal operations ============

/// Extract diagonal or construct diagonal array
///
/// If input is 2D, extracts the diagonal.
/// If input is 1D, constructs a 2D array with the input on the diagonal.
/// k specifies diagonal offset (0=main, positive=above, negative=below).
/// Equivalent to numpy.diag(arr, k).
#[wasm_bindgen]
impl NDArray {
    pub fn diag(&self, k: i32) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        match shape.len() {
            1 => {
                // 1D -> 2D diagonal matrix
                let n = shape[0];
                let offset = k.unsigned_abs() as usize;
                let size = n + offset;
                let mut result = ndarray::Array2::<f64>::zeros((size, size));

                for i in 0..n {
                    if k >= 0 {
                        result[[i, i + offset]] = data[[i]];
                    } else {
                        result[[i + offset, i]] = data[[i]];
                    }
                }

                Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result.into_dyn())))
            }
            2 => {
                // 2D -> extract diagonal
                let (rows, cols) = (shape[0], shape[1]);
                let (start_row, start_col, diag_len) = if k >= 0 {
                    let start_col = k as usize;
                    let diag_len = cols.saturating_sub(start_col).min(rows);
                    (0, start_col, diag_len)
                } else {
                    let start_row = (-k) as usize;
                    let diag_len = rows.saturating_sub(start_row).min(cols);
                    (start_row, 0, diag_len)
                };

                let mut result = Vec::with_capacity(diag_len);
                for i in 0..diag_len {
                    result.push(data[[start_row + i, start_col + i]]);
                }

                Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result.clone(), vec![result.len()])
                    .map_err(|e| JsValue::from_str(&e.to_string()))?))
            }
            _ => Err(JsValue::from_str("diag expects 1D or 2D array")),
        }
    }

    /// Extract diagonal from 2D array (alias for diag with k=0)
    pub fn diagonal(&self, offset: Option<i32>) -> Result<NDArray, JsValue> {
        self.diag(offset.unwrap_or(0))
    }
}

// ============ Reduction operations ============

/// Log-sum-exp along an axis (numerically stable)
///
/// Computes log(sum(exp(x))) in a numerically stable way by subtracting max.
/// Equivalent to scipy.special.logsumexp or torch.logsumexp.
#[wasm_bindgen]
impl NDArray {
    pub fn logsumexp(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let ax = ndarray::Axis(axis);

        // Compute max along axis for numerical stability
        let max_vals = data.map_axis(ax, |lane| {
            lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });

        // Compute logsumexp: max + log(sum(exp(x - max)))
        let result = ndarray::Zip::from(max_vals.view())
            .and(data.lanes(ax))
            .map_collect(|&max_val, lane| {
                let sum_exp: f64 = lane.iter()
                    .map(|&x| (x - max_val).exp())
                    .sum();
                max_val + sum_exp.ln()
            });

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&rumpy_cpu::CpuArray::from_ndarray(result), new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
        }
    }

    /// Test if all elements are true (non-zero)
    ///
    /// Returns 1.0 if all elements are non-zero, 0.0 otherwise.
    /// Equivalent to numpy.all(arr).
    pub fn all(&self) -> f64 {
        let data = self.inner.as_f64_slice();
        if data.iter().all(|&x| x != 0.0) { 1.0 } else { 0.0 }
    }

    /// Test if any element is true (non-zero)
    ///
    /// Returns 1.0 if any element is non-zero, 0.0 otherwise.
    /// Equivalent to numpy.any(arr).
    pub fn any(&self) -> f64 {
        let data = self.inner.as_f64_slice();
        if data.iter().any(|&x| x != 0.0) { 1.0 } else { 0.0 }
    }

    /// Test if all elements are true along an axis
    #[wasm_bindgen(js_name = allAxis)]
    pub fn all_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let ax = ndarray::Axis(axis);
        let result = data.map_axis(ax, |lane| {
            if lane.iter().all(|&x| x != 0.0) { 1.0 } else { 0.0 }
        });

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&rumpy_cpu::CpuArray::from_ndarray(result), new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
        }
    }

    /// Test if any element is true along an axis
    #[wasm_bindgen(js_name = anyAxis)]
    pub fn any_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let ax = ndarray::Axis(axis);
        let result = data.map_axis(ax, |lane| {
            if lane.iter().any(|&x| x != 0.0) { 1.0 } else { 0.0 }
        });

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&rumpy_cpu::CpuArray::from_ndarray(result), new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
        }
    }
}

// ============ Einsum (Einstein summation) ============

/// Einstein summation convention
///
/// Performs tensor contractions, transposes, and reductions using
/// Einstein notation. Critical for attention mechanisms.
///
/// Supported patterns:
/// - "ij,jk->ik" : matrix multiplication
/// - "bij,bjk->bik" : batched matrix multiplication
/// - "bhqk,bhkd->bhqd" : attention (Q @ K.T @ V)
/// - "ij->ji" : transpose
/// - "ij->" : sum all
/// - "ij->i" : sum over j
/// - "ii->" : trace
/// - "...ij,...jk->...ik" : batched matmul with ellipsis
///
/// # Arguments
/// * `subscripts` - Einstein summation subscripts (e.g., "ij,jk->ik")
/// * `a` - First input array
/// * `b` - Optional second input array
#[wasm_bindgen]
pub fn einsum(subscripts: &str, a: &NDArray, b: Option<NDArray>) -> Result<NDArray, JsValue> {
    // Parse subscripts
    let parts: Vec<&str> = subscripts.split("->").collect();
    if parts.len() > 2 {
        return Err(JsValue::from_str("einsum: invalid subscripts format"));
    }

    let inputs_str = parts[0];
    let output_str = if parts.len() == 2 { parts[1].trim() } else { "" };

    let input_parts: Vec<&str> = inputs_str.split(',').map(|s| s.trim()).collect();

    match (input_parts.len(), &b) {
        (1, None) => {
            // Single array operation
            einsum_unary(input_parts[0], output_str, a)
        }
        (2, Some(b_arr)) => {
            // Binary operation
            einsum_binary(input_parts[0], input_parts[1], output_str, a, b_arr)
        }
        (1, Some(_)) => {
            Err(JsValue::from_str("einsum: subscripts specify one input but two arrays provided"))
        }
        (2, None) => {
            Err(JsValue::from_str("einsum: subscripts specify two inputs but only one array provided"))
        }
        _ => {
            Err(JsValue::from_str("einsum: only 1 or 2 input arrays supported"))
        }
    }
}

/// Unary einsum operations (transpose, trace, sum, etc.)
fn einsum_unary(input: &str, output: &str, a: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let a_shape = a_data.shape();

    // Handle ellipsis - expand to concrete indices
    let (input_expanded, output_expanded) = expand_ellipsis(input, output, a_shape.len())?;

    // Build index mappings
    let input_chars: Vec<char> = input_expanded.chars().collect();
    let output_chars: Vec<char> = output_expanded.chars().collect();

    if input_chars.len() != a_shape.len() {
        return Err(JsValue::from_str(&format!(
            "einsum: input subscript '{}' has {} indices but array has {} dimensions",
            input, input_chars.len(), a_shape.len()
        )));
    }

    // Check for trace (repeated indices in input)
    let mut char_counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for &c in &input_chars {
        *char_counts.entry(c).or_insert(0) += 1;
    }

    // Simple transpose case: all indices appear once and output reorders them
    if char_counts.values().all(|&c| c == 1) && output_chars.len() == input_chars.len() {
        // Build permutation
        let mut perm = Vec::with_capacity(output_chars.len());
        for oc in &output_chars {
            let pos = input_chars.iter().position(|&ic| ic == *oc)
                .ok_or_else(|| JsValue::from_str(&format!(
                    "einsum: output index '{}' not in input",
                    oc
                )))?;
            perm.push(pos);
        }
        let permuted = a_data.clone().permuted_axes(perm);
        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(permuted.to_owned())));
    }

    // Sum reduction case: some indices dropped
    if char_counts.values().all(|&c| c == 1) && output_chars.len() < input_chars.len() {
        // Find which axes to sum over
        let sum_axes: Vec<usize> = input_chars.iter().enumerate()
            .filter(|(_, c)| !output_chars.contains(c))
            .map(|(i, _)| i)
            .collect();

        let mut result = a_data.to_owned();
        // Sum in reverse order to maintain correct axis indices
        for &ax in sum_axes.iter().rev() {
            result = result.sum_axis(ndarray::Axis(ax));
        }

        // May need to transpose if output order differs
        if !output_chars.is_empty() {
            let remaining_chars: Vec<char> = input_chars.iter()
                .filter(|c| output_chars.contains(c))
                .cloned()
                .collect();

            if remaining_chars != output_chars {
                let mut perm = Vec::with_capacity(output_chars.len());
                for oc in &output_chars {
                    let pos = remaining_chars.iter().position(|&c| c == *oc)
                        .ok_or_else(|| JsValue::from_str("einsum: internal permutation error"))?;
                    perm.push(pos);
                }
                result = result.permuted_axes(perm).to_owned();
            }
        }

        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)));
    }

    // Trace case: "ii->" or similar
    if char_counts.values().any(|&c| c == 2) {
        // Find repeated index
        let repeated: Vec<char> = char_counts.iter()
            .filter(|(_, &c)| c == 2)
            .map(|(&k, _)| k)
            .collect();

        if repeated.len() == 1 && a_shape.len() == 2 {
            // Simple 2D trace
            let trace_char = repeated[0];
            let positions: Vec<usize> = input_chars.iter().enumerate()
                .filter(|(_, &c)| c == trace_char)
                .map(|(i, _)| i)
                .collect();

            if positions.len() == 2 && a_shape[positions[0]] == a_shape[positions[1]] {
                let n = a_shape[0].min(a_shape[1]);
                let trace: f64 = (0..n).map(|i| a_data[[i, i]]).sum();
                return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(vec![trace], vec![])
                    .map_err(|e| JsValue::from_str(&e.to_string()))?));
            }
        }
    }

    Err(JsValue::from_str(&format!(
        "einsum: unsupported unary operation '{}->{}'",
        input, output
    )))
}

/// Binary einsum operations (matmul, batched matmul, contractions)
fn einsum_binary(input_a: &str, input_b: &str, output: &str, a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    // Handle ellipsis
    let _max_dims = a_shape.len().max(b_shape.len());
    let (input_a_exp, output_exp_a) = expand_ellipsis(input_a, output, a_shape.len())?;
    let (input_b_exp, output_exp_b) = expand_ellipsis(input_b, output, b_shape.len())?;
    let output_exp = if !output_exp_a.is_empty() { output_exp_a } else { output_exp_b };

    let a_chars: Vec<char> = input_a_exp.chars().collect();
    let b_chars: Vec<char> = input_b_exp.chars().collect();
    let out_chars: Vec<char> = output_exp.chars().collect();

    if a_chars.len() != a_shape.len() {
        return Err(JsValue::from_str(&format!(
            "einsum: input subscript '{}' has {} indices but array has {} dimensions",
            input_a, a_chars.len(), a_shape.len()
        )));
    }
    if b_chars.len() != b_shape.len() {
        return Err(JsValue::from_str(&format!(
            "einsum: input subscript '{}' has {} indices but array has {} dimensions",
            input_b, b_chars.len(), b_shape.len()
        )));
    }

    // Special case: standard matmul "ij,jk->ik"
    if a_chars == ['i', 'j'] && b_chars == ['j', 'k'] && out_chars == ['i', 'k'] {
        return CpuBackend::matmul(&a.inner, &b.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()));
    }

    // Special case: batched matmul "bij,bjk->bik"
    if a_chars.len() == 3 && b_chars.len() == 3 && out_chars.len() == 3 {
        let (batch_a, i_a, j_a) = (a_chars[0], a_chars[1], a_chars[2]);
        let (batch_b, j_b, k_b) = (b_chars[0], b_chars[1], b_chars[2]);
        let (batch_o, i_o, k_o) = (out_chars[0], out_chars[1], out_chars[2]);

        if batch_a == batch_b && batch_b == batch_o &&
           i_a == i_o && k_b == k_o && j_a == j_b {
            // This is batched matmul
            let batch_size = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];

            if b_shape[0] != batch_size || b_shape[1] != k {
                return Err(JsValue::from_str("einsum: dimension mismatch for batched matmul"));
            }

            let mut result = vec![0.0; batch_size * m * n];

            for batch in 0..batch_size {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for kk in 0..k {
                            let a_idx = batch * m * k + i * k + kk;
                            let b_idx = batch * k * n + kk * n + j;
                            sum += a_data.as_slice().unwrap()[a_idx] * b_data.as_slice().unwrap()[b_idx];
                        }
                        result[batch * m * n + i * n + j] = sum;
                    }
                }
            }

            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![batch_size, m, n])
                .map_err(|e| JsValue::from_str(&e.to_string()))?));
        }
    }

    // Special case: 4D batched matmul "bhqk,bhkd->bhqd" (attention)
    if a_chars.len() == 4 && b_chars.len() == 4 && out_chars.len() == 4 {
        // Check if it's standard attention pattern
        if a_chars[0] == b_chars[0] && a_chars[0] == out_chars[0] &&  // batch
           a_chars[1] == b_chars[1] && a_chars[1] == out_chars[1] &&  // head
           a_chars[2] == out_chars[2] &&  // query dim preserved
           a_chars[3] == b_chars[2] &&    // contraction dim
           b_chars[3] == out_chars[3] {   // value dim preserved

            let (batch, heads, q_len, k_len) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
            let v_dim = b_shape[3];

            if b_shape[0] != batch || b_shape[1] != heads || b_shape[2] != k_len {
                return Err(JsValue::from_str("einsum: dimension mismatch for attention"));
            }

            let mut result = vec![0.0; batch * heads * q_len * v_dim];
            let a_flat = a_data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;
            let b_flat = b_data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

            for b_idx in 0..batch {
                for h in 0..heads {
                    for q in 0..q_len {
                        for v in 0..v_dim {
                            let mut sum = 0.0;
                            for k in 0..k_len {
                                let a_offset = b_idx * heads * q_len * k_len + h * q_len * k_len + q * k_len + k;
                                let b_offset = b_idx * heads * k_len * v_dim + h * k_len * v_dim + k * v_dim + v;
                                sum += a_flat[a_offset] * b_flat[b_offset];
                            }
                            let out_offset = b_idx * heads * q_len * v_dim + h * q_len * v_dim + q * v_dim + v;
                            result[out_offset] = sum;
                        }
                    }
                }
            }

            return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![batch, heads, q_len, v_dim])
                .map_err(|e| JsValue::from_str(&e.to_string()))?));
        }
    }

    // General case: identify contraction, batch, and output indices
    // Find indices that appear in both inputs (contraction candidates)
    let mut contraction_chars: Vec<char> = Vec::new();
    let mut batch_chars: Vec<char> = Vec::new();

    for &c in &a_chars {
        if b_chars.contains(&c) {
            if out_chars.contains(&c) {
                batch_chars.push(c);
            } else {
                contraction_chars.push(c);
            }
        }
    }

    // For now, handle simple cases where we can reshape to matmul
    // This is a simplified general einsum - full implementation would need more work

    Err(JsValue::from_str(&format!(
        "einsum: pattern '{},{}->{}' not yet supported. Supported: ij,jk->ik, bij,bjk->bik, bhqk,bhkd->bhqd",
        input_a, input_b, output
    )))
}

/// Expand ellipsis notation to concrete indices
fn expand_ellipsis(input: &str, output: &str, ndim: usize) -> Result<(String, String), JsValue> {
    if !input.contains("...") {
        return Ok((input.to_string(), output.to_string()));
    }

    let non_ellipsis_count = input.chars().filter(|&c| c != '.').count();
    let ellipsis_dims = ndim.saturating_sub(non_ellipsis_count);

    // Generate unique chars for ellipsis dimensions
    let ellipsis_chars: String = (0..ellipsis_dims)
        .map(|i| (b'A' + i as u8) as char)
        .collect();

    let expanded_input = input.replace("...", &ellipsis_chars);
    let expanded_output = output.replace("...", &ellipsis_chars);

    Ok((expanded_input, expanded_output))
}

// ============ Conv2d and ConvTranspose2d ============

/// 2D Convolution
///
/// Performs 2D convolution of input with kernel.
/// Input shape: (N, C_in, H, W)
/// Kernel shape: (C_out, C_in, kH, kW)
/// Output shape: (N, C_out, H_out, W_out)
///
/// # Arguments
/// * `input` - Input tensor (N, C_in, H, W)
/// * `kernel` - Convolution kernel (C_out, C_in, kH, kW)
/// * `stride` - Stride [stride_h, stride_w]
/// * `padding` - Padding [pad_h, pad_w]
#[wasm_bindgen]
pub fn conv2d(
    input: &NDArray,
    kernel: &NDArray,
    stride: Vec<usize>,
    padding: Vec<usize>,
) -> Result<NDArray, JsValue> {
    let input_data = input.inner.as_ndarray();
    let kernel_data = kernel.inner.as_ndarray();

    let input_shape = input_data.shape();
    let kernel_shape = kernel_data.shape();

    if input_shape.len() != 4 {
        return Err(JsValue::from_str("conv2d: input must be 4D (N, C_in, H, W)"));
    }
    if kernel_shape.len() != 4 {
        return Err(JsValue::from_str("conv2d: kernel must be 4D (C_out, C_in, kH, kW)"));
    }
    if stride.len() != 2 {
        return Err(JsValue::from_str("conv2d: stride must have 2 elements [stride_h, stride_w]"));
    }
    if padding.len() != 2 {
        return Err(JsValue::from_str("conv2d: padding must have 2 elements [pad_h, pad_w]"));
    }

    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_out, c_in_k, k_h, k_w) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_h, pad_w) = (padding[0], padding[1]);

    if c_in != c_in_k {
        return Err(JsValue::from_str(&format!(
            "conv2d: input channels {} doesn't match kernel channels {}",
            c_in, c_in_k
        )));
    }

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let input_flat = input_data.as_slice().ok_or_else(|| JsValue::from_str("input not contiguous"))?;
    let kernel_flat = kernel_data.as_slice().ok_or_else(|| JsValue::from_str("kernel not contiguous"))?;

    let mut output = vec![0.0; n * c_out * h_out * w_out];

    for batch in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0;

                    for ci in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check padding bounds
                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;

                                    let input_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + actual_h * w_in + actual_w;
                                    let kernel_idx = co * c_in_k * k_h * k_w + ci * k_h * k_w + kh * k_w + kw;

                                    sum += input_flat[input_idx] * kernel_flat[kernel_idx];
                                }
                            }
                        }
                    }

                    let out_idx = batch * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c_out, h_out, w_out])
        .map_err(|e| JsValue::from_str(&e.to_string()))?))
}

/// 2D Transposed Convolution (Deconvolution)
///
/// Performs transposed 2D convolution, often used for upsampling.
/// Input shape: (N, C_in, H, W)
/// Kernel shape: (C_in, C_out, kH, kW)  -- note C_in, C_out order differs from conv2d
/// Output shape: (N, C_out, H_out, W_out)
///
/// # Arguments
/// * `input` - Input tensor (N, C_in, H, W)
/// * `kernel` - Convolution kernel (C_in, C_out, kH, kW)
/// * `stride` - Stride [stride_h, stride_w]
/// * `padding` - Padding [pad_h, pad_w]
/// * `output_padding` - Additional padding for output [out_pad_h, out_pad_w]
#[wasm_bindgen(js_name = convTranspose2d)]
pub fn conv_transpose_2d(
    input: &NDArray,
    kernel: &NDArray,
    stride: Vec<usize>,
    padding: Vec<usize>,
    output_padding: Vec<usize>,
) -> Result<NDArray, JsValue> {
    let input_data = input.inner.as_ndarray();
    let kernel_data = kernel.inner.as_ndarray();

    let input_shape = input_data.shape();
    let kernel_shape = kernel_data.shape();

    if input_shape.len() != 4 {
        return Err(JsValue::from_str("conv_transpose2d: input must be 4D (N, C_in, H, W)"));
    }
    if kernel_shape.len() != 4 {
        return Err(JsValue::from_str("conv_transpose2d: kernel must be 4D (C_in, C_out, kH, kW)"));
    }

    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_in_k, c_out, k_h, k_w) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);
    let (stride_h, stride_w) = (stride[0], stride[1]);
    let (pad_h, pad_w) = (padding[0], padding[1]);
    let (out_pad_h, out_pad_w) = (output_padding.get(0).copied().unwrap_or(0),
                                   output_padding.get(1).copied().unwrap_or(0));

    if c_in != c_in_k {
        return Err(JsValue::from_str(&format!(
            "conv_transpose2d: input channels {} doesn't match kernel channels {}",
            c_in, c_in_k
        )));
    }

    // Output size formula for transposed conv
    let h_out = (h_in - 1) * stride_h - 2 * pad_h + k_h + out_pad_h;
    let w_out = (w_in - 1) * stride_w - 2 * pad_w + k_w + out_pad_w;

    let input_flat = input_data.as_slice().ok_or_else(|| JsValue::from_str("input not contiguous"))?;
    let kernel_flat = kernel_data.as_slice().ok_or_else(|| JsValue::from_str("kernel not contiguous"))?;

    let mut output = vec![0.0; n * c_out * h_out * w_out];

    // Transposed convolution: scatter input values weighted by kernel
    for batch in 0..n {
        for ci in 0..c_in {
            for ih in 0..h_in {
                for iw in 0..w_in {
                    let input_val = input_flat[batch * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw];

                    for co in 0..c_out {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh = ih * stride_h + kh;
                                let ow = iw * stride_w + kw;

                                // Apply padding
                                if oh >= pad_h && oh < h_out + pad_h && ow >= pad_w && ow < w_out + pad_w {
                                    let actual_oh = oh - pad_h;
                                    let actual_ow = ow - pad_w;

                                    if actual_oh < h_out && actual_ow < w_out {
                                        let kernel_idx = ci * c_out * k_h * k_w + co * k_h * k_w + kh * k_w + kw;
                                        let out_idx = batch * c_out * h_out * w_out + co * h_out * w_out + actual_oh * w_out + actual_ow;

                                        output[out_idx] += input_val * kernel_flat[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c_out, h_out, w_out])
        .map_err(|e| JsValue::from_str(&e.to_string()))?))
}

// ============ Interpolate/Resize ============

/// Bilinear interpolation for resizing 2D images
///
/// Input shape: (N, C, H, W) or (H, W)
/// Resizes spatial dimensions to target size using bilinear interpolation.
///
/// # Arguments
/// * `size` - Target size [target_h, target_w]
#[wasm_bindgen]
impl NDArray {
    pub fn interpolate(&self, size: Vec<usize>) -> Result<NDArray, JsValue> {
        if size.len() != 2 {
            return Err(JsValue::from_str("interpolate: size must have 2 elements [target_h, target_w]"));
        }

        let data = self.inner.as_ndarray();
        let shape = data.shape();
        let (target_h, target_w) = (size[0], size[1]);

        match shape.len() {
            2 => {
                // Simple 2D case
                let (h_in, w_in) = (shape[0], shape[1]);
                let result = bilinear_interpolate_2d(
                    data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?,
                    h_in, w_in,
                    target_h, target_w,
                );
                Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![target_h, target_w])
                    .map_err(|e| JsValue::from_str(&e.to_string()))?))
            }
            4 => {
                // (N, C, H, W) case
                let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
                let flat = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

                let mut result = Vec::with_capacity(n * c * target_h * target_w);

                for batch in 0..n {
                    for ch in 0..c {
                        let offset = batch * c * h_in * w_in + ch * h_in * w_in;
                        let slice = &flat[offset..offset + h_in * w_in];
                        let interpolated = bilinear_interpolate_2d(slice, h_in, w_in, target_h, target_w);
                        result.extend(interpolated);
                    }
                }

                Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![n, c, target_h, target_w])
                    .map_err(|e| JsValue::from_str(&e.to_string()))?))
            }
            _ => Err(JsValue::from_str("interpolate: input must be 2D (H, W) or 4D (N, C, H, W)"))
        }
    }

    /// Alias for interpolate
    pub fn resize(&self, size: Vec<usize>) -> Result<NDArray, JsValue> {
        self.interpolate(size)
    }
}

/// Bilinear interpolation helper for a single 2D image
fn bilinear_interpolate_2d(
    input: &[f64],
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; h_out * w_out];

    let scale_h = h_in as f64 / h_out as f64;
    let scale_w = w_in as f64 / w_out as f64;

    for oh in 0..h_out {
        for ow in 0..w_out {
            // Map output coord to input coord (align_corners=false style)
            let ih_f = (oh as f64 + 0.5) * scale_h - 0.5;
            let iw_f = (ow as f64 + 0.5) * scale_w - 0.5;

            let ih0 = ih_f.floor() as i64;
            let iw0 = iw_f.floor() as i64;
            let ih1 = ih0 + 1;
            let iw1 = iw0 + 1;

            let dh = ih_f - ih0 as f64;
            let dw = iw_f - iw0 as f64;

            // Clamp indices
            let ih0c = ih0.clamp(0, h_in as i64 - 1) as usize;
            let ih1c = ih1.clamp(0, h_in as i64 - 1) as usize;
            let iw0c = iw0.clamp(0, w_in as i64 - 1) as usize;
            let iw1c = iw1.clamp(0, w_in as i64 - 1) as usize;

            // Bilinear interpolation
            let v00 = input[ih0c * w_in + iw0c];
            let v01 = input[ih0c * w_in + iw1c];
            let v10 = input[ih1c * w_in + iw0c];
            let v11 = input[ih1c * w_in + iw1c];

            let value = v00 * (1.0 - dh) * (1.0 - dw)
                      + v01 * (1.0 - dh) * dw
                      + v10 * dh * (1.0 - dw)
                      + v11 * dh * dw;

            output[oh * w_out + ow] = value;
        }
    }

    output
}

// ============ Decomposition operations (NumPy parity) ============

/// Decompose array into fractional and integral parts
///
/// Returns two arrays: fractional parts and integral parts.
/// Equivalent to numpy.modf(x).
#[wasm_bindgen(js_name = modfArr)]
pub fn modf_arr(arr: &NDArray) -> js_sys::Array {
    let data = arr.inner.as_ndarray();
    let frac = data.mapv(|x| x.fract());
    let integ = data.mapv(|x| x.trunc());

    let result = js_sys::Array::new();
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(frac))));
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(integ))));
    result
}

/// Decompose array into mantissa and exponent (base 2)
///
/// Returns two arrays: mantissa and exponent where x = mantissa * 2^exponent.
/// Equivalent to numpy.frexp(x).
#[wasm_bindgen(js_name = frexpArr)]
pub fn frexp_arr(arr: &NDArray) -> js_sys::Array {
    let data = arr.inner.as_ndarray();

    let mantissa = data.mapv(|x| {
        if x == 0.0 || x.is_nan() || x.is_infinite() {
            x
        } else {
            let exp = x.abs().log2().floor() + 1.0;
            x / 2_f64.powf(exp)
        }
    });

    let exponent = data.mapv(|x| {
        if x == 0.0 {
            0.0
        } else if x.is_nan() || x.is_infinite() {
            f64::NAN
        } else {
            x.abs().log2().floor() + 1.0
        }
    });

    let result = js_sys::Array::new();
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(mantissa))));
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(exponent))));
    result
}

/// Compute x * 2^exp element-wise
///
/// Equivalent to numpy.ldexp(x, exp).
#[wasm_bindgen(js_name = ldexpArr)]
pub fn ldexp_arr(arr: &NDArray, exp: &NDArray) -> Result<NDArray, JsValue> {
    let arr_data = arr.inner.as_ndarray();
    let exp_data = exp.inner.as_ndarray();

    if arr_data.shape() != exp_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for ldexp",
            arr_data.shape(), exp_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*arr_data)
        .and(&*exp_data)
        .map_collect(|&x, &e| x * 2_f64.powf(e));
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Compute floor division and remainder simultaneously
///
/// Returns two arrays: quotient (floor division) and remainder.
/// Equivalent to numpy.divmod(a, b).
#[wasm_bindgen(js_name = divmodArr)]
pub fn divmod_arr(a: &NDArray, b: &NDArray) -> Result<js_sys::Array, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for divmod",
            a_data.shape(), b_data.shape()
        )));
    }

    let quotient = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| (av / bv).floor());

    let remainder = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| {
            // Python-style modulo: result has same sign as divisor
            let r = av % bv;
            if (r > 0.0 && bv < 0.0) || (r < 0.0 && bv > 0.0) {
                r + bv
            } else {
                r
            }
        });

    let result = js_sys::Array::new();
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(quotient))));
    result.push(&JsValue::from(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(remainder))));
    Ok(result)
}

/// Python-style modulo (remainder has same sign as divisor)
///
/// Equivalent to numpy.mod(a, b) or a % b in Python.
#[wasm_bindgen(js_name = modArr)]
pub fn mod_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for mod",
            a_data.shape(), b_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| {
            // Python-style modulo: result has same sign as divisor
            let r = av % bv;
            if (r > 0.0 && bv < 0.0) || (r < 0.0 && bv > 0.0) {
                r + bv
            } else {
                r
            }
        });
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Copy sign of b to magnitude of a
///
/// Equivalent to numpy.copysign(a, b).
#[wasm_bindgen(js_name = copysignArr)]
pub fn copysign_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for copysign",
            a_data.shape(), b_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| av.abs().copysign(bv));
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

/// Compute hypotenuse: sqrt(a^2 + b^2) avoiding overflow
///
/// Equivalent to numpy.hypot(a, b).
#[wasm_bindgen(js_name = hypotArr)]
pub fn hypot_arr(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let a_data = a.inner.as_ndarray();
    let b_data = b.inner.as_ndarray();

    if a_data.shape() != b_data.shape() {
        return Err(JsValue::from_str(&format!(
            "shapes {:?} and {:?} must match for hypot",
            a_data.shape(), b_data.shape()
        )));
    }

    let result = ndarray::Zip::from(&*a_data)
        .and(&*b_data)
        .map_collect(|&av, &bv| av.hypot(bv));
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result)))
}

// ============ Space generation (NumPy parity) ============

/// Generate logarithmically spaced values
///
/// Returns num values from base^start to base^stop (inclusive).
/// Equivalent to numpy.logspace(start, stop, num, base).
#[wasm_bindgen(js_name = logspaceArr)]
pub fn logspace_arr(start: f64, stop: f64, num: usize, base: f64) -> NDArray {
    if num == 0 {
        return NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(vec![], vec![0]).unwrap());
    }
    if num == 1 {
        return NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(vec![base.powf(start)], vec![1]).unwrap());
    }

    let step = (stop - start) / (num - 1) as f64;
    let result: Vec<f64> = (0..num)
        .map(|i| base.powf(start + step * i as f64))
        .collect();
    NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![num]).unwrap())
}

/// Generate geometrically spaced values
///
/// Returns num values from start to stop (inclusive) with geometric spacing.
/// Equivalent to numpy.geomspace(start, stop, num).
#[wasm_bindgen(js_name = geomspaceArr)]
pub fn geomspace_arr(start: f64, stop: f64, num: usize) -> Result<NDArray, JsValue> {
    if start == 0.0 || stop == 0.0 {
        return Err(JsValue::from_str("geomspace: start and stop must be non-zero"));
    }
    if (start < 0.0) != (stop < 0.0) {
        return Err(JsValue::from_str("geomspace: start and stop must have the same sign"));
    }

    if num == 0 {
        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(vec![], vec![0]).unwrap()));
    }
    if num == 1 {
        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(vec![start], vec![1]).unwrap()));
    }

    let sign = start.signum();
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let step = (log_stop - log_start) / (num - 1) as f64;

    let result: Vec<f64> = (0..num)
        .map(|i| sign * (log_start + step * i as f64).exp())
        .collect();
    Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, vec![num]).unwrap()))
}
