//! torchjs-wasm - Neural network operations for torchjs
//!
//! This crate provides WASM bindings for PyTorch-style neural network operations,
//! including convolutions, pooling, batch normalization, and activation functions.
//!
//! Mirrors torch.nn.functional API for JavaScript/WASM usage.

use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Float64Array};

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ============ Activation Functions ============

/// ReLU activation: max(0, x)
/// NaN values are propagated
#[wasm_bindgen]
pub fn relu_f64(input: &Float64Array) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x > 0.0 { x }
        else { 0.0 }
    }).collect();
    Float64Array::from(&result[..])
}

/// ReLU activation for f32
#[wasm_bindgen]
pub fn relu_f32(input: &Float32Array) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let result: Vec<f32> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x > 0.0 { x }
        else { 0.0 }
    }).collect();
    Float32Array::from(&result[..])
}

/// ReLU6 activation: min(max(0, x), 6)
/// Clamps output to [0, 6] range. Used in MobileNetV2 architectures.
#[wasm_bindgen]
pub fn relu6_f64(input: &Float64Array) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x < 0.0 { 0.0 }
        else if x > 6.0 { 6.0 }
        else { x }
    }).collect();
    Float64Array::from(&result[..])
}

/// ReLU6 for f32
#[wasm_bindgen]
pub fn relu6_f32(input: &Float32Array) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let result: Vec<f32> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x < 0.0 { 0.0 }
        else if x > 6.0 { 6.0 }
        else { x }
    }).collect();
    Float32Array::from(&result[..])
}

/// Leaky ReLU activation: x if x > 0, else alpha * x
#[wasm_bindgen]
pub fn leaky_relu_f64(input: &Float64Array, alpha: f64) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x > 0.0 { x }
        else { alpha * x }
    }).collect();
    Float64Array::from(&result[..])
}

/// Leaky ReLU for f32
#[wasm_bindgen]
pub fn leaky_relu_f32(input: &Float32Array, alpha: f32) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let result: Vec<f32> = data.iter().map(|&x| {
        if x.is_nan() { x }
        else if x > 0.0 { x }
        else { alpha * x }
    }).collect();
    Float32Array::from(&result[..])
}

/// GELU activation (Gaussian Error Linear Unit)
/// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[wasm_bindgen]
pub fn gelu_f64(input: &Float64Array) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    let result: Vec<f64> = data.iter().map(|&x| {
        let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
        x * 0.5 * (1.0 + inner.tanh())
    }).collect();
    Float64Array::from(&result[..])
}

/// GELU for f32
#[wasm_bindgen]
pub fn gelu_f32(input: &Float32Array) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let result: Vec<f32> = data.iter().map(|&x| {
        let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
        x * 0.5 * (1.0 + inner.tanh())
    }).collect();
    Float32Array::from(&result[..])
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[wasm_bindgen]
pub fn sigmoid_f64(input: &Float64Array) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Float64Array::from(&result[..])
}

/// Sigmoid for f32
#[wasm_bindgen]
pub fn sigmoid_f32(input: &Float32Array) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Float32Array::from(&result[..])
}

/// Softmax along the last axis
/// For flattened data with specified axis_size
#[wasm_bindgen]
pub fn softmax_f64(input: &Float64Array, axis_size: usize) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let num_batches = data.len() / axis_size;
    let mut result = vec![0.0f64; data.len()];

    for b in 0..num_batches {
        let start = b * axis_size;
        let end = start + axis_size;

        // Find max for numerical stability
        let max_val = data[start..end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for i in start..end {
            let exp_val = (data[i] - max_val).exp();
            result[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for i in start..end {
            result[i] /= sum;
        }
    }

    Float64Array::from(&result[..])
}

/// Softmax for f32
#[wasm_bindgen]
pub fn softmax_f32(input: &Float32Array, axis_size: usize) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let num_batches = data.len() / axis_size;
    let mut result = vec![0.0f32; data.len()];

    for b in 0..num_batches {
        let start = b * axis_size;
        let end = start + axis_size;

        let max_val = data[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for i in start..end {
            let exp_val = (data[i] - max_val).exp();
            result[i] = exp_val;
            sum += exp_val;
        }

        for i in start..end {
            result[i] /= sum;
        }
    }

    Float32Array::from(&result[..])
}

// ============ Convolution Operations ============

/// 2D Convolution
///
/// Input shape: (N, C_in, H, W) - flattened
/// Kernel shape: (C_out, C_in, kH, kW) - flattened
/// Output shape: (N, C_out, H_out, W_out) - flattened
#[wasm_bindgen]
pub fn conv2d_f64(
    input: &Float64Array,
    kernel: &Float64Array,
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float64Array {
    let input_data: Vec<f64> = input.to_vec();
    let kernel_data: Vec<f64> = kernel.to_vec();

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut output = vec![0.0f64; n * c_out * h_out * w_out];

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

                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;

                                    let input_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + actual_h * w_in + actual_w;
                                    let kernel_idx = co * c_in * k_h * k_w + ci * k_h * k_w + kh * k_w + kw;

                                    sum += input_data[input_idx] * kernel_data[kernel_idx];
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

    Float64Array::from(&output[..])
}

/// 2D Convolution for f32
#[wasm_bindgen]
pub fn conv2d_f32(
    input: &Float32Array,
    kernel: &Float32Array,
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float32Array {
    let input_data: Vec<f32> = input.to_vec();
    let kernel_data: Vec<f32> = kernel.to_vec();

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut output = vec![0.0f32; n * c_out * h_out * w_out];

    for batch in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;

                    for ci in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;

                                    let input_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + actual_h * w_in + actual_w;
                                    let kernel_idx = co * c_in * k_h * k_w + ci * k_h * k_w + kh * k_w + kw;

                                    sum += input_data[input_idx] * kernel_data[kernel_idx];
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

    Float32Array::from(&output[..])
}

/// 2D Depthwise Convolution
///
/// Input shape: (N, C, H, W) - flattened
/// Kernel shape: (C, depth_multiplier, kH, kW) - flattened
/// Output shape: (N, C * depth_multiplier, H_out, W_out) - flattened
#[wasm_bindgen]
pub fn depthwise_conv2d_f64(
    input: &Float64Array,
    kernel: &Float64Array,
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    depth_mult: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float64Array {
    let input_data: Vec<f64> = input.to_vec();
    let kernel_data: Vec<f64> = kernel.to_vec();

    let c_out = c_in * depth_mult;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut output = vec![0.0f64; n * c_out * h_out * w_out];

    for batch in 0..n {
        for ci in 0..c_in {
            for dm in 0..depth_mult {
                let co = ci * depth_mult + dm;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;

                                    let input_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + actual_h * w_in + actual_w;
                                    let kernel_idx = ci * depth_mult * k_h * k_w + dm * k_h * k_w + kh * k_w + kw;

                                    sum += input_data[input_idx] * kernel_data[kernel_idx];
                                }
                            }
                        }

                        let out_idx = batch * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    Float64Array::from(&output[..])
}

/// 2D Depthwise Convolution for f32
#[wasm_bindgen]
pub fn depthwise_conv2d_f32(
    input: &Float32Array,
    kernel: &Float32Array,
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    depth_mult: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float32Array {
    let input_data: Vec<f32> = input.to_vec();
    let kernel_data: Vec<f32> = kernel.to_vec();

    let c_out = c_in * depth_mult;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    let mut output = vec![0.0f32; n * c_out * h_out * w_out];

    for batch in 0..n {
        for ci in 0..c_in {
            for dm in 0..depth_mult {
                let co = ci * depth_mult + dm;

                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;

                                    let input_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + actual_h * w_in + actual_w;
                                    let kernel_idx = ci * depth_mult * k_h * k_w + dm * k_h * k_w + kh * k_w + kw;

                                    sum += input_data[input_idx] * kernel_data[kernel_idx];
                                }
                            }
                        }

                        let out_idx = batch * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    Float32Array::from(&output[..])
}

// ============ Pooling Operations ============

/// Max Pooling 2D
///
/// Input shape: (N, C, H, W) - flattened
/// Output shape: (N, C, H_out, W_out) - flattened
#[wasm_bindgen]
pub fn max_pool2d_f64(
    input: &Float64Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float64Array {
    let input_data: Vec<f64> = input.to_vec();

    let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = vec![f64::NEG_INFINITY; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f64::NEG_INFINITY;

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                let actual_h = ih - pad_h;
                                let actual_w = iw - pad_w;
                                let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                max_val = max_val.max(input_data[idx]);
                            }
                        }
                    }

                    if max_val == f64::NEG_INFINITY {
                        max_val = 0.0;
                    }

                    let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    Float64Array::from(&output[..])
}

/// Max Pooling 2D for f32
#[wasm_bindgen]
pub fn max_pool2d_f32(
    input: &Float32Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float32Array {
    let input_data: Vec<f32> = input.to_vec();

    let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = vec![f32::NEG_INFINITY; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                let actual_h = ih - pad_h;
                                let actual_w = iw - pad_w;
                                let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                max_val = max_val.max(input_data[idx]);
                            }
                        }
                    }

                    if max_val == f32::NEG_INFINITY {
                        max_val = 0.0;
                    }

                    let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    Float32Array::from(&output[..])
}

/// Average Pooling 2D
#[wasm_bindgen]
pub fn avg_pool2d_f64(
    input: &Float64Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float64Array {
    let input_data: Vec<f64> = input.to_vec();

    let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = vec![0.0f64; n * c * h_out * w_out];

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

                            if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                let actual_h = ih - pad_h;
                                let actual_w = iw - pad_w;
                                let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                sum += input_data[idx];
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

    Float64Array::from(&output[..])
}

/// Average Pooling 2D for f32
#[wasm_bindgen]
pub fn avg_pool2d_f32(
    input: &Float32Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Float32Array {
    let input_data: Vec<f32> = input.to_vec();

    let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

    let mut output = vec![0.0f32; n * c * h_out * w_out];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                let actual_h = ih - pad_h;
                                let actual_w = iw - pad_w;
                                let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                sum += input_data[idx];
                                count += 1;
                            }
                        }
                    }

                    let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                    output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }

    Float32Array::from(&output[..])
}

// ============ Batch Normalization ============

/// Batch Normalization (inference only)
///
/// output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta
///
/// Input: flattened NCHW format
/// gamma, beta, running_mean, running_var: 1D arrays of size C
#[wasm_bindgen]
pub fn batch_norm_f64(
    input: &Float64Array,
    gamma: &Float64Array,
    beta: &Float64Array,
    running_mean: &Float64Array,
    running_var: &Float64Array,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f64,
) -> Float64Array {
    let input_data: Vec<f64> = input.to_vec();
    let gamma_data: Vec<f64> = gamma.to_vec();
    let beta_data: Vec<f64> = beta.to_vec();
    let mean_data: Vec<f64> = running_mean.to_vec();
    let var_data: Vec<f64> = running_var.to_vec();

    // Pre-compute scale and shift per channel
    let mut scale = vec![0.0f64; c];
    let mut shift = vec![0.0f64; c];

    for ch in 0..c {
        let inv_std = 1.0 / (var_data[ch] + eps).sqrt();
        scale[ch] = gamma_data[ch] * inv_std;
        shift[ch] = beta_data[ch] - mean_data[ch] * scale[ch];
    }

    let mut output = vec![0.0f64; input_data.len()];
    let hw = h * w;

    for batch in 0..n {
        for ch in 0..c {
            let s = scale[ch];
            let sh = shift[ch];
            for i in 0..hw {
                let idx = batch * c * hw + ch * hw + i;
                output[idx] = input_data[idx] * s + sh;
            }
        }
    }

    Float64Array::from(&output[..])
}

/// Batch Normalization for f32
#[wasm_bindgen]
pub fn batch_norm_f32(
    input: &Float32Array,
    gamma: &Float32Array,
    beta: &Float32Array,
    running_mean: &Float32Array,
    running_var: &Float32Array,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
) -> Float32Array {
    let input_data: Vec<f32> = input.to_vec();
    let gamma_data: Vec<f32> = gamma.to_vec();
    let beta_data: Vec<f32> = beta.to_vec();
    let mean_data: Vec<f32> = running_mean.to_vec();
    let var_data: Vec<f32> = running_var.to_vec();

    let mut scale = vec![0.0f32; c];
    let mut shift = vec![0.0f32; c];

    for ch in 0..c {
        let inv_std = 1.0 / (var_data[ch] + eps).sqrt();
        scale[ch] = gamma_data[ch] * inv_std;
        shift[ch] = beta_data[ch] - mean_data[ch] * scale[ch];
    }

    let mut output = vec![0.0f32; input_data.len()];
    let hw = h * w;

    for batch in 0..n {
        for ch in 0..c {
            let s = scale[ch];
            let sh = shift[ch];
            for i in 0..hw {
                let idx = batch * c * hw + ch * hw + i;
                output[idx] = input_data[idx] * s + sh;
            }
        }
    }

    Float32Array::from(&output[..])
}

// ============ PReLU Activation ============

/// PReLU activation: x if x > 0, else weight * x
/// Weight can be per-channel or per-element
#[wasm_bindgen]
pub fn prelu_f32(input: &Float32Array, weight: &Float32Array) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let w: Vec<f32> = weight.to_vec();
    let w_len = w.len();

    let result: Vec<f32> = data.iter().enumerate().map(|(i, &x)| {
        if x > 0.0 { x } else { w[i % w_len] * x }
    }).collect();

    Float32Array::from(&result[..])
}

/// PReLU for f64
#[wasm_bindgen]
pub fn prelu_f64(input: &Float64Array, weight: &Float64Array) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let w: Vec<f64> = weight.to_vec();
    let w_len = w.len();

    let result: Vec<f64> = data.iter().enumerate().map(|(i, &x)| {
        if x > 0.0 { x } else { w[i % w_len] * x }
    }).collect();

    Float64Array::from(&result[..])
}

// ============ Image Resize Operations ============

/// Bilinear resize for NCHW format
/// Input shape: (N, C, H_in, W_in)
/// Output shape: (N, C, H_out, W_out)
#[wasm_bindgen]
pub fn resize_bilinear_f32(
    input: &Float32Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let mut output = vec![0.0f32; n * c * h_out * w_out];

    let scale_y = h_in as f32 / h_out as f32;
    let scale_x = w_in as f32 / w_out as f32;

    for batch in 0..n {
        for ch in 0..c {
            let in_base = batch * c * h_in * w_in + ch * h_in * w_in;
            let out_base = batch * c * h_out * w_out + ch * h_out * w_out;

            for oy in 0..h_out {
                for ox in 0..w_out {
                    // Compute source coordinates (center-aligned)
                    let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
                    let src_x = (ox as f32 + 0.5) * scale_x - 0.5;

                    let y0 = src_y.floor().max(0.0) as usize;
                    let x0 = src_x.floor().max(0.0) as usize;
                    let y1 = (y0 + 1).min(h_in - 1);
                    let x1 = (x0 + 1).min(w_in - 1);

                    let ly = src_y - y0 as f32;
                    let lx = src_x - x0 as f32;
                    let hy = 1.0 - ly;
                    let hx = 1.0 - lx;

                    let v00 = data[in_base + y0 * w_in + x0];
                    let v01 = data[in_base + y0 * w_in + x1];
                    let v10 = data[in_base + y1 * w_in + x0];
                    let v11 = data[in_base + y1 * w_in + x1];

                    let val = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
                    output[out_base + oy * w_out + ox] = val;
                }
            }
        }
    }

    Float32Array::from(&output[..])
}

/// Bilinear resize for f64
#[wasm_bindgen]
pub fn resize_bilinear_f64(
    input: &Float64Array,
    n: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let mut output = vec![0.0f64; n * c * h_out * w_out];

    let scale_y = h_in as f64 / h_out as f64;
    let scale_x = w_in as f64 / w_out as f64;

    for batch in 0..n {
        for ch in 0..c {
            let in_base = batch * c * h_in * w_in + ch * h_in * w_in;
            let out_base = batch * c * h_out * w_out + ch * h_out * w_out;

            for oy in 0..h_out {
                for ox in 0..w_out {
                    let src_y = (oy as f64 + 0.5) * scale_y - 0.5;
                    let src_x = (ox as f64 + 0.5) * scale_x - 0.5;

                    let y0 = src_y.floor().max(0.0) as usize;
                    let x0 = src_x.floor().max(0.0) as usize;
                    let y1 = (y0 + 1).min(h_in - 1);
                    let x1 = (x0 + 1).min(w_in - 1);

                    let ly = src_y - y0 as f64;
                    let lx = src_x - x0 as f64;
                    let hy = 1.0 - ly;
                    let hx = 1.0 - lx;

                    let v00 = data[in_base + y0 * w_in + x0];
                    let v01 = data[in_base + y0 * w_in + x1];
                    let v10 = data[in_base + y1 * w_in + x0];
                    let v11 = data[in_base + y1 * w_in + x1];

                    let val = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
                    output[out_base + oy * w_out + ox] = val;
                }
            }
        }
    }

    Float64Array::from(&output[..])
}

// ============ Global Average Pooling ============

/// Global Average Pooling 2D - averages entire spatial dimension
/// Input shape: (N, C, H, W)
/// Output shape: (N, C, 1, 1)
#[wasm_bindgen]
pub fn global_avg_pool2d_f32(
    input: &Float32Array,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Float32Array {
    let data: Vec<f32> = input.to_vec();
    let hw = h * w;
    let mut output = vec![0.0f32; n * c];

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * hw + ch * hw;
            let sum: f32 = (0..hw).map(|i| data[base + i]).sum();
            output[batch * c + ch] = sum / hw as f32;
        }
    }

    Float32Array::from(&output[..])
}

/// Global Average Pooling for f64
#[wasm_bindgen]
pub fn global_avg_pool2d_f64(
    input: &Float64Array,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Float64Array {
    let data: Vec<f64> = input.to_vec();
    let hw = h * w;
    let mut output = vec![0.0f64; n * c];

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * hw + ch * hw;
            let sum: f64 = (0..hw).map(|i| data[base + i]).sum();
            output[batch * c + ch] = sum / hw as f64;
        }
    }

    Float64Array::from(&output[..])
}

// ============ Helper Functions ============

/// Get output dimensions for conv2d
#[wasm_bindgen]
pub fn conv2d_output_size(h_in: usize, w_in: usize, k_h: usize, k_w: usize, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize) -> Vec<usize> {
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    vec![h_out, w_out]
}
