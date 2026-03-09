#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Generate test cases from NumPy to verify rumpy-ts correctness.
Outputs JSON that can be loaded by the JS test runner.
"""

import numpy as np
import json

np.random.seed(42)

test_cases = []

def add_test(name, op_name, inputs, expected, shape=None):
    """Add a test case."""
    test_cases.append({
        "name": name,
        "op": op_name,
        "inputs": inputs,
        "expected": expected.flatten().tolist() if hasattr(expected, 'flatten') else expected,
        "expected_shape": list(expected.shape) if hasattr(expected, 'shape') else shape
    })

# ==================== SPRINT 1: Structural Operations ====================

# Test permute/transpose
arr_2x3 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
add_test("permute 2x3 -> 3x2", "permute",
         {"data": arr_2x3.flatten().tolist(), "shape": [2, 3], "axes": [1, 0]},
         arr_2x3.T)

arr_2x3x4 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
add_test("permute 2x3x4 -> 4x2x3", "permute",
         {"data": arr_2x3x4.flatten().tolist(), "shape": [2, 3, 4], "axes": [2, 0, 1]},
         arr_2x3x4.transpose(2, 0, 1))

# Test sumAxis
arr_3x4 = np.arange(12, dtype=np.float64).reshape(3, 4)
add_test("sumAxis(0) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": False},
         arr_3x4.sum(axis=0))

add_test("sumAxis(1) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 1, "keepdims": False},
         arr_3x4.sum(axis=1))

add_test("sumAxis(0, keepdims) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": True},
         arr_3x4.sum(axis=0, keepdims=True))

# Test meanAxis
add_test("meanAxis(0) 3x4", "meanAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": False},
         arr_3x4.mean(axis=0))

add_test("meanAxis(1) 3x4", "meanAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 1, "keepdims": False},
         arr_3x4.mean(axis=1))

# Test maxAxis / minAxis
arr_rand = np.random.rand(4, 5).astype(np.float64)
add_test("maxAxis(0) 4x5", "maxAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 0, "keepdims": False},
         arr_rand.max(axis=0))

add_test("maxAxis(1) 4x5", "maxAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 1, "keepdims": False},
         arr_rand.max(axis=1))

add_test("minAxis(0) 4x5", "minAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 0, "keepdims": False},
         arr_rand.min(axis=0))

add_test("minAxis(1) 4x5", "minAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 1, "keepdims": False},
         arr_rand.min(axis=1))

# Test softmax
def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

arr_softmax = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
add_test("softmax axis=1", "softmax",
         {"data": arr_softmax.flatten().tolist(), "shape": [2, 3], "axis": 1},
         softmax(arr_softmax, axis=1))

add_test("softmax axis=0", "softmax",
         {"data": arr_softmax.flatten().tolist(), "shape": [2, 3], "axis": 0},
         softmax(arr_softmax, axis=0))

# Test relu
arr_relu = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("relu", "relu",
         {"data": arr_relu.tolist(), "shape": [5]},
         np.maximum(arr_relu, 0))

# Test gelu (approximate)
def gelu_approx(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

arr_gelu = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("gelu", "gelu",
         {"data": arr_gelu.tolist(), "shape": [5]},
         gelu_approx(arr_gelu))

# Test argmax/argmin
arr_arg = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
add_test("argmax", "argmax",
         {"data": arr_arg.tolist(), "shape": [8]},
         int(np.argmax(arr_arg)), shape=[])

add_test("argmin", "argmin",
         {"data": arr_arg.tolist(), "shape": [8]},
         int(np.argmin(arr_arg)), shape=[])

# ==================== SPRINT 2: CNN Operations ====================

# Test slice
arr_10 = np.arange(10, dtype=np.float64)
add_test("slice [2:5]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [2], "stops": [5], "steps": [1]},
         arr_10[2:5])

add_test("slice [-3:]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [-3], "stops": [2147483647], "steps": [1]},
         arr_10[-3:])

add_test("slice [::2]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [0], "stops": [2147483647], "steps": [2]},
         arr_10[::2])

# 2D slice
arr_5x5 = np.arange(25, dtype=np.float64).reshape(5, 5)
add_test("slice 2D [1:4, 2:5]", "slice",
         {"data": arr_5x5.flatten().tolist(), "shape": [5, 5],
          "starts": [1, 2], "stops": [4, 5], "steps": [1, 1]},
         arr_5x5[1:4, 2:5])

# Test im2col
# Simple 1x1x4x4 image
img_4x4 = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)

def im2col_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of im2col."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    # Pad if needed
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0,0), (0,0), (ph,ph), (pw,pw)), mode='constant')

    cols = np.zeros((n * h_out * w_out, c * kh * kw))

    for batch in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                row_idx = batch * h_out * w_out + oh * w_out + ow
                patch = img[batch, :, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                cols[row_idx] = patch.flatten()

    return cols

add_test("im2col 4x4 k=2 s=1 p=0", "im2col",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0},
         im2col_numpy(img_4x4, 2, 2, 1, 1, 0, 0))

add_test("im2col 4x4 k=2 s=2 p=0", "im2col",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         im2col_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# Test maxPool2d
def maxpool2d_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of max pooling."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    # Pad if needed
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0,0), (0,0), (ph,ph), (pw,pw)), mode='constant', constant_values=-np.inf)

    out = np.zeros((n, c, h_out, w_out))

    for batch in range(n):
        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = img[batch, ch, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                    out[batch, ch, oh, ow] = np.max(patch)

    return out

add_test("maxPool2d 4x4 k=2 s=2", "maxPool2d",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         maxpool2d_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# Test avgPool2d
def avgpool2d_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of average pooling."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    out = np.zeros((n, c, h_out, w_out))

    for batch in range(n):
        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    count = 0
                    total = 0.0
                    for ki in range(kh):
                        for kj in range(kw):
                            ih = oh * sh + ki - ph
                            iw = ow * sw + kj - pw
                            if 0 <= ih < h and 0 <= iw < w:
                                total += img[batch, ch, ih, iw]
                                count += 1
                    out[batch, ch, oh, ow] = total / count if count > 0 else 0

    return out

add_test("avgPool2d 4x4 k=2 s=2", "avgPool2d",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         avgpool2d_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# ==================== SPRINT 3: Boolean Masking ====================

# Test comparisons
arr_cmp = np.array([1, 2, 3, 4, 5], dtype=np.float64)
add_test("gtScalar(3)", "gtScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp > 3).astype(np.float64))

add_test("ltScalar(3)", "ltScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp < 3).astype(np.float64))

add_test("eqScalar(3)", "eqScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp == 3).astype(np.float64))

# Test getByMask
arr_mask_test = np.array([1, 2, 3, 4, 5], dtype=np.float64)
mask = arr_mask_test > 2
add_test("getByMask (x > 2)", "getByMask",
         {"data": arr_mask_test.tolist(), "shape": [5],
          "mask": mask.astype(np.float64).tolist()},
         arr_mask_test[mask])

# Test setByMask
result_set = arr_mask_test.copy()
result_set[mask] = 0
add_test("setByMask (x > 2, 0)", "setByMask",
         {"data": arr_mask_test.tolist(), "shape": [5],
          "mask": mask.astype(np.float64).tolist(), "value": 0},
         result_set)

# Test where
cond = np.array([1, 0, 1, 0, 1], dtype=np.float64)
x = np.array([10, 20, 30, 40, 50], dtype=np.float64)
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
add_test("where", "where_",
         {"cond": cond.tolist(), "x": x.tolist(), "y": y.tolist(), "shape": [5]},
         np.where(cond.astype(bool), x, y))

# Test clip
arr_clip = np.array([-5, 0, 5, 10, 15], dtype=np.float64)
add_test("clip(0, 10)", "clip",
         {"data": arr_clip.tolist(), "shape": [5], "min": 0, "max": 10},
         np.clip(arr_clip, 0, 10))

# Test isnan/isinf
arr_special = np.array([1, np.nan, np.inf, -np.inf, 0], dtype=np.float64)
add_test("isNan", "isNan",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isnan(arr_special).astype(np.float64))

add_test("isInf", "isInf",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isinf(arr_special).astype(np.float64))

add_test("isFinite", "isFinite",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isfinite(arr_special).astype(np.float64))

# Test countNonzero
arr_nz = np.array([0, 1, 0, 2, 0, 3], dtype=np.float64)
add_test("countNonzero", "countNonzero",
         {"data": arr_nz.tolist(), "shape": [6]},
         int(np.count_nonzero(arr_nz)), shape=[])

# ==================== SPRINT 4: ML Inference Ops ====================

# Test layerNorm
def layer_norm_numpy(x, normalized_shape, gamma=None, beta=None, eps=1e-5):
    """NumPy reference implementation of layer normalization."""
    shape = x.shape
    ndim = len(shape)
    norm_ndim = len(normalized_shape)

    norm_size = np.prod(normalized_shape)
    batch_size = np.prod(shape[:ndim - norm_ndim])

    flat = x.reshape(batch_size, norm_size)
    mean = flat.mean(axis=1, keepdims=True)
    var = flat.var(axis=1, keepdims=True)

    normalized = (flat - mean) / np.sqrt(var + eps)

    if gamma is not None:
        normalized = normalized * gamma.flatten()
    if beta is not None:
        normalized = normalized + beta.flatten()

    return normalized.reshape(shape)

arr_ln = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)
add_test("layerNorm 2x4 no affine", "layerNorm",
         {"data": arr_ln.flatten().tolist(), "shape": [2, 4],
          "normalized_shape": [4], "gamma": None, "beta": None, "eps": 1e-5},
         layer_norm_numpy(arr_ln, [4], eps=1e-5))

gamma_ln = np.array([1, 2, 1, 2], dtype=np.float64)
beta_ln = np.array([0, 1, 0, 1], dtype=np.float64)
add_test("layerNorm 2x4 with gamma/beta", "layerNorm",
         {"data": arr_ln.flatten().tolist(), "shape": [2, 4],
          "normalized_shape": [4],
          "gamma": gamma_ln.tolist(), "beta": beta_ln.tolist(), "eps": 1e-5},
         layer_norm_numpy(arr_ln, [4], gamma_ln, beta_ln, eps=1e-5))

# Test rmsNorm
def rms_norm_numpy(x, gamma, eps=1e-5):
    """NumPy reference implementation of RMS normalization."""
    shape = x.shape
    last_dim = shape[-1]
    batch_size = np.prod(shape[:-1])

    flat = x.reshape(batch_size, last_dim)
    rms = np.sqrt(np.mean(flat ** 2, axis=1, keepdims=True) + eps)
    normalized = flat / rms

    return (normalized * gamma.flatten()).reshape(shape)

arr_rms = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
gamma_rms = np.array([1, 2, 3], dtype=np.float64)
add_test("rmsNorm 2x3", "rmsNorm",
         {"data": arr_rms.flatten().tolist(), "shape": [2, 3],
          "gamma": gamma_rms.tolist(), "eps": 1e-5},
         rms_norm_numpy(arr_rms, gamma_rms, eps=1e-5))

# Test batchNorm (inference mode)
def batch_norm_numpy(x, gamma, beta, running_mean, running_var, eps=1e-5):
    """NumPy reference implementation of batch normalization (inference)."""
    # x shape: NCHW
    n, c = x.shape[0], x.shape[1]
    spatial = np.prod(x.shape[2:])

    x_flat = x.reshape(n, c, spatial)
    result = np.zeros_like(x_flat)

    for ch in range(c):
        scale = 1.0 / np.sqrt(running_var[ch] + eps)
        if gamma is not None:
            scale *= gamma[ch]

        shift = -running_mean[ch] * scale
        if gamma is not None and beta is not None:
            # Corrected formula: shift = gamma * (-mean/std) + beta
            shift = gamma[ch] * (-running_mean[ch] / np.sqrt(running_var[ch] + eps)) + beta[ch]
        elif beta is not None:
            shift += beta[ch]

        result[:, ch, :] = x_flat[:, ch, :] * scale + shift

    return result.reshape(x.shape)

arr_bn = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2)  # N=2, C=3, H=2, W=2
running_mean_bn = np.array([1, 2, 3], dtype=np.float64)
running_var_bn = np.array([0.5, 1.0, 1.5], dtype=np.float64)
gamma_bn = np.array([1, 1, 1], dtype=np.float64)
beta_bn = np.array([0, 0, 0], dtype=np.float64)

add_test("batchNorm NCHW", "batchNorm",
         {"data": arr_bn.flatten().tolist(), "shape": [2, 3, 2, 2],
          "gamma": gamma_bn.tolist(), "beta": beta_bn.tolist(),
          "running_mean": running_mean_bn.tolist(), "running_var": running_var_bn.tolist(),
          "eps": 1e-5},
         batch_norm_numpy(arr_bn, gamma_bn, beta_bn, running_mean_bn, running_var_bn, eps=1e-5))

# Test take (embedding lookup)
embedding_table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float64)
indices_take = np.array([0, 2, 1], dtype=np.float64)

add_test("take axis=0 (embedding lookup)", "take",
         {"data": embedding_table.flatten().tolist(), "shape": [4, 3],
          "indices": indices_take.tolist(), "axis": 0},
         np.take(embedding_table, [0, 2, 1], axis=0))

arr_2d = np.arange(12, dtype=np.float64).reshape(3, 4)
add_test("take axis=1", "take",
         {"data": arr_2d.flatten().tolist(), "shape": [3, 4],
          "indices": [0, 2].tolist() if isinstance([0, 2], np.ndarray) else [0.0, 2.0], "axis": 1},
         np.take(arr_2d, [0, 2], axis=1))

# Test rsqrt
arr_rsqrt = np.array([1, 4, 9, 16, 25], dtype=np.float64)
add_test("rsqrt", "rsqrt",
         {"data": arr_rsqrt.tolist(), "shape": [5]},
         1.0 / np.sqrt(arr_rsqrt))

# Test sigmoid
def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

arr_sigmoid = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("sigmoid", "sigmoid",
         {"data": arr_sigmoid.tolist(), "shape": [5]},
         sigmoid_numpy(arr_sigmoid))

# Test silu (swish)
def silu_numpy(x):
    return x / (1.0 + np.exp(-x))

arr_silu = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("silu", "silu",
         {"data": arr_silu.tolist(), "shape": [5]},
         silu_numpy(arr_silu))

# Test topk
arr_topk = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
sorted_idx = np.argsort(arr_topk)[::-1][:3]  # Top 3 indices
sorted_vals = arr_topk[sorted_idx]

add_test("topk k=3", "topk",
         {"data": arr_topk.tolist(), "shape": [8], "k": 3, "axis": 0, "sorted": True},
         {"values": sorted_vals.tolist(), "indices": sorted_idx.tolist()}, shape=[3])

# 2D topk
arr_topk_2d = np.array([[3, 1, 4], [1, 5, 9]], dtype=np.float64)
# topk along axis=1
vals_0 = np.sort(arr_topk_2d[0])[::-1][:2]
vals_1 = np.sort(arr_topk_2d[1])[::-1][:2]
idx_0 = np.argsort(arr_topk_2d[0])[::-1][:2]
idx_1 = np.argsort(arr_topk_2d[1])[::-1][:2]

add_test("topk 2D k=2 axis=1", "topk",
         {"data": arr_topk_2d.flatten().tolist(), "shape": [2, 3], "k": 2, "axis": 1, "sorted": True},
         {"values": np.array([vals_0, vals_1]).tolist(),
          "indices": np.array([idx_0, idx_1]).tolist()}, shape=[2, 2])

# Test tril
arr_tril = np.arange(9, dtype=np.float64).reshape(3, 3)
add_test("tril k=0", "tril",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": 0},
         np.tril(arr_tril, k=0))

add_test("tril k=1", "tril",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": 1},
         np.tril(arr_tril, k=1))

add_test("tril k=-1", "tril",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": -1},
         np.tril(arr_tril, k=-1))

# Test triu
add_test("triu k=0", "triu",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": 0},
         np.triu(arr_tril, k=0))

add_test("triu k=1", "triu",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": 1},
         np.triu(arr_tril, k=1))

add_test("triu k=-1", "triu",
         {"data": arr_tril.flatten().tolist(), "shape": [3, 3], "k": -1},
         np.triu(arr_tril, k=-1))

# Test causalMask
def causal_mask_numpy(size):
    mask = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if j > i:
                mask[i, j] = float('-inf')
    return mask

add_test("causalMask size=4", "causalMask",
         {"size": 4},
         causal_mask_numpy(4))

# ==================== Non-contiguous Input Tests ====================
# These tests verify that operations work correctly on transposed arrays

# Test layerNorm on transposed input
arr_ln_t = np.array([[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.float64).T  # 2x4 transposed from 4x2
add_test("layerNorm transposed input", "layerNorm",
         {"data": arr_ln_t.flatten().tolist(), "shape": [2, 4],
          "normalized_shape": [4], "gamma": None, "beta": None, "eps": 1e-5},
         layer_norm_numpy(arr_ln_t, [4], eps=1e-5))

# Test tril on transposed input
arr_tril_t = np.arange(9, dtype=np.float64).reshape(3, 3).T
add_test("tril transposed k=0", "tril",
         {"data": arr_tril_t.flatten().tolist(), "shape": [3, 3], "k": 0},
         np.tril(arr_tril_t, k=0))

# Test topk on transposed 2D array
arr_topk_t = np.array([[3, 1], [1, 5], [4, 9]], dtype=np.float64).T  # 2x3 from 3x2
vals_t0 = np.sort(arr_topk_t[0])[::-1][:2]
vals_t1 = np.sort(arr_topk_t[1])[::-1][:2]
idx_t0 = np.argsort(arr_topk_t[0])[::-1][:2]
idx_t1 = np.argsort(arr_topk_t[1])[::-1][:2]
add_test("topk transposed 2D k=2", "topk",
         {"data": arr_topk_t.flatten().tolist(), "shape": [2, 3], "k": 2, "axis": 1, "sorted": True},
         {"values": np.array([vals_t0, vals_t1]).tolist(),
          "indices": np.array([idx_t0, idx_t1]).tolist()}, shape=[2, 2])

# Custom JSON encoder to handle NaN/Inf
# ==================== SPRINT 5: Additional LLM Ops ====================

# Test split
arr_split = np.arange(12, dtype=np.float64).reshape(2, 6)
splits = np.split(arr_split, 3, axis=1)
add_test("split 2x6 into 3 along axis 1", "split",
         {"data": arr_split.flatten().tolist(), "shape": [2, 6], "num_splits": 3, "axis": 1},
         splits[0])  # Just test first split for simplicity

# Test chunk (using array_split for uneven)
arr_chunk = np.arange(10, dtype=np.float64)
chunks = [arr_chunk[:4], arr_chunk[4:8], arr_chunk[8:]]
add_test("chunk size 4", "chunk",
         {"data": arr_chunk.flatten().tolist(), "shape": [10], "chunk_size": 4, "axis": 0},
         chunks[0])  # Test first chunk

# Test cumsum
arr_cumsum = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
add_test("cumsum axis=1", "cumsum",
         {"data": arr_cumsum.flatten().tolist(), "shape": [2, 3], "axis": 1},
         np.cumsum(arr_cumsum, axis=1))

add_test("cumsum axis=0", "cumsum",
         {"data": arr_cumsum.flatten().tolist(), "shape": [2, 3], "axis": 0},
         np.cumsum(arr_cumsum, axis=0))

# Test cumprod
add_test("cumprod axis=1", "cumprod",
         {"data": arr_cumsum.flatten().tolist(), "shape": [2, 3], "axis": 1},
         np.cumprod(arr_cumsum, axis=1))

# Test tile
arr_tile = np.array([[1, 2], [3, 4]], dtype=np.float64)
add_test("tile (2,3)", "tile",
         {"data": arr_tile.flatten().tolist(), "shape": [2, 2], "reps": [2, 3]},
         np.tile(arr_tile, (2, 3)))

# Test repeat
arr_repeat = np.array([1, 2, 3], dtype=np.float64)
add_test("repeat 3 times", "repeat",
         {"data": arr_repeat.flatten().tolist(), "shape": [3], "repeats": 3, "axis": None},
         np.repeat(arr_repeat, 3))

add_test("repeat axis=0", "repeat",
         {"data": arr_tile.flatten().tolist(), "shape": [2, 2], "repeats": 2, "axis": 0},
         np.repeat(arr_tile, 2, axis=0))

# Test pad
arr_pad = np.array([[1, 2], [3, 4]], dtype=np.float64)
add_test("pad constant", "pad",
         {"data": arr_pad.flatten().tolist(), "shape": [2, 2],
          "pad_width": [1, 1, 2, 2], "constant_value": 0},
         np.pad(arr_pad, ((1, 1), (2, 2)), mode='constant', constant_values=0))

# Test log_softmax (using scipy.special.log_softmax equivalent)
arr_logsoftmax = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64)
# Manual log_softmax for numpy
def log_softmax_np(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))

add_test("log_softmax axis=1", "logSoftmax",
         {"data": arr_logsoftmax.flatten().tolist(), "shape": [2, 3], "axis": 1},
         log_softmax_np(arr_logsoftmax, axis=1))

# Test diff
arr_diff = np.array([1, 3, 6, 10, 15], dtype=np.float64)
add_test("diff n=1", "diff",
         {"data": arr_diff.flatten().tolist(), "shape": [5], "n": 1, "axis": 0},
         np.diff(arr_diff, n=1))

add_test("diff n=2", "diff",
         {"data": arr_diff.flatten().tolist(), "shape": [5], "n": 2, "axis": 0},
         np.diff(arr_diff, n=2))

# Test indexCopy (scatter-like)
arr_base = np.zeros((5, 3), dtype=np.float64)
arr_src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
indices = np.array([1, 3], dtype=np.float64)
result_indexcopy = arr_base.copy()
result_indexcopy[1] = arr_src[0]
result_indexcopy[3] = arr_src[1]
add_test("indexCopy axis=0", "indexCopy",
         {"data": arr_base.flatten().tolist(), "shape": [5, 3],
          "indices": indices.tolist(), "indices_shape": [2],
          "src": arr_src.flatten().tolist(), "src_shape": [2, 3],
          "axis": 0},
         result_indexcopy)


# ==================== SPRINT 6: Transformer Ops ====================

# Test tanh
arr_tanh = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("tanh", "tanhArr",
         {"data": arr_tanh.tolist(), "shape": [5]},
         np.tanh(arr_tanh))

# Test neg
arr_neg = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("neg", "negArr",
         {"data": arr_neg.tolist(), "shape": [5]},
         -arr_neg)

# Test sinh
arr_sinh = np.array([-1, 0, 1], dtype=np.float64)
add_test("sinh", "sinhArr",
         {"data": arr_sinh.tolist(), "shape": [3]},
         np.sinh(arr_sinh))

# Test cosh
arr_cosh = np.array([-1, 0, 1], dtype=np.float64)
add_test("cosh", "coshArr",
         {"data": arr_cosh.tolist(), "shape": [3]},
         np.cosh(arr_cosh))

# Test broadcast_to
arr_bc = np.array([[1, 2, 3]], dtype=np.float64)  # shape (1, 3)
add_test("broadcastTo (1,3) -> (4,3)", "broadcastTo",
         {"data": arr_bc.flatten().tolist(), "shape": [1, 3], "target_shape": [4, 3]},
         np.broadcast_to(arr_bc, (4, 3)))

arr_bc_scalar = np.array([5], dtype=np.float64)  # shape (1,)
add_test("broadcastTo (1,) -> (5,)", "broadcastTo",
         {"data": arr_bc_scalar.tolist(), "shape": [1], "target_shape": [5]},
         np.broadcast_to(arr_bc_scalar, (5,)))

# Test broadcast_to with implicit dimension expansion
arr_bc_1d = np.array([1, 2, 3], dtype=np.float64)  # shape (3,)
add_test("broadcastTo (3,) -> (2,3)", "broadcastTo",
         {"data": arr_bc_1d.tolist(), "shape": [3], "target_shape": [2, 3]},
         np.broadcast_to(arr_bc_1d, (2, 3)))

# Test asType (cast)
arr_cast = np.array([1.5, 2.7, 3.2], dtype=np.float64)
add_test("asType float32", "asType",
         {"data": arr_cast.tolist(), "shape": [3], "dtype": "float32"},
         arr_cast.astype(np.float32).astype(np.float64))  # f64 -> f32 -> f64 to match behavior

add_test("asType int64", "asType",
         {"data": arr_cast.tolist(), "shape": [3], "dtype": "int64"},
         np.trunc(arr_cast))  # NumPy astype(int64) truncates toward zero

# Test dequantizeLinear
# ONNX DequantizeLinear: output = (input - zero_point) * scale
arr_quant = np.array([0, 64, 128, 192, 255], dtype=np.float64)
scale = 0.01
zero_point = 128.0
add_test("dequantizeLinear", "dequantizeLinear",
         {"data": arr_quant.tolist(), "shape": [5], "scale": scale, "zero_point": zero_point},
         (arr_quant - zero_point) * scale)

# Test gather (alias for take) - critical for embeddings
embedding = np.random.rand(1000, 256).astype(np.float64)  # vocab_size=1000, embed_dim=256
token_ids = np.array([0, 42, 100, 999], dtype=np.float64)
add_test("gather (embedding lookup)", "gather",
         {"data": embedding.flatten().tolist(), "shape": [1000, 256],
          "indices": token_ids.tolist(), "axis": 0},
         np.take(embedding, token_ids.astype(int), axis=0))


# ==================== EDGE CASE TESTS ====================

# Edge cases for unary ops: NaN, inf, -inf, zeros
edge_cases = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e308, 1e-308, -1e308], dtype=np.float64)

add_test("tanh edge cases", "tanhArr",
         {"data": edge_cases.tolist(), "shape": [8]},
         np.tanh(edge_cases))

add_test("neg edge cases", "negArr",
         {"data": edge_cases.tolist(), "shape": [8]},
         -edge_cases)

add_test("sinh edge cases", "sinhArr",
         {"data": edge_cases.tolist(), "shape": [8]},
         np.sinh(edge_cases))

add_test("cosh edge cases", "coshArr",
         {"data": edge_cases.tolist(), "shape": [8]},
         np.cosh(edge_cases))

# Sinh/cosh overflow test (values > 710 overflow to inf)
overflow_test = np.array([700, 710, 720, -700, -710, -720], dtype=np.float64)
add_test("sinh overflow", "sinhArr",
         {"data": overflow_test.tolist(), "shape": [6]},
         np.sinh(overflow_test))

add_test("cosh overflow", "coshArr",
         {"data": overflow_test.tolist(), "shape": [6]},
         np.cosh(overflow_test))

# asType int64 with negative numbers and edge cases
arr_cast_negative = np.array([1.5, -1.5, 2.7, -2.7, 0.9, -0.9], dtype=np.float64)
add_test("asType int64 negative", "asType",
         {"data": arr_cast_negative.tolist(), "shape": [6], "dtype": "int64"},
         np.trunc(arr_cast_negative))

# asType int32 with overflow clamping
arr_cast_overflow = np.array([1e15, -1e15, 2147483647, -2147483648, 2147483650, -2147483650], dtype=np.float64)
# Note: NumPy actually wraps on overflow, but we clamp for safety
# Our implementation clamps to i32 range
i32_min, i32_max = -2147483648, 2147483647
expected_clamped = np.clip(np.trunc(arr_cast_overflow), i32_min, i32_max)
add_test("asType int32 overflow", "asType",
         {"data": arr_cast_overflow.tolist(), "shape": [6], "dtype": "int32"},
         expected_clamped)

# broadcastTo edge cases: 0-dim, multi-1-dim
arr_bc_multi = np.array([[1], [2], [3]], dtype=np.float64)  # shape (3, 1)
add_test("broadcastTo (3,1) -> (3,4)", "broadcastTo",
         {"data": arr_bc_multi.flatten().tolist(), "shape": [3, 1], "target_shape": [3, 4]},
         np.broadcast_to(arr_bc_multi, (3, 4)))

# dequantizeLinear with edge scale/zero_point
arr_dq_edge = np.array([0, 127, 255], dtype=np.float64)
add_test("dequantizeLinear scale=0.5", "dequantizeLinear",
         {"data": arr_dq_edge.tolist(), "shape": [3], "scale": 0.5, "zero_point": 127.0},
         (arr_dq_edge - 127.0) * 0.5)

add_test("dequantizeLinear scale=0.001", "dequantizeLinear",
         {"data": arr_dq_edge.tolist(), "shape": [3], "scale": 0.001, "zero_point": 0.0},
         arr_dq_edge * 0.001)


# ==================== MISSING OP TESTS (from audit) ====================

# Test var (population variance, ddof=0)
arr_var = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=np.float64)
add_test("var population", "var",
         {"data": arr_var.tolist(), "shape": [8]},
         np.var(arr_var), shape=[])

# Test var with ddof=1 (sample variance)
add_test("varDdof sample", "varDdof",
         {"data": arr_var.tolist(), "shape": [8], "ddof": 1},
         np.var(arr_var, ddof=1), shape=[])

# Test std (population std, ddof=0)
add_test("std population", "std",
         {"data": arr_var.tolist(), "shape": [8]},
         np.std(arr_var), shape=[])

# Test std with ddof=1 (sample std)
add_test("stdDdof sample", "stdDdof",
         {"data": arr_var.tolist(), "shape": [8], "ddof": 1},
         np.std(arr_var, ddof=1), shape=[])

# Test prod
arr_prod = np.array([1, 2, 3, 4, 5], dtype=np.float64)
add_test("prod", "prod",
         {"data": arr_prod.tolist(), "shape": [5]},
         np.prod(arr_prod), shape=[])

# Test argmin/argmax with NaN (should not panic, return index of first non-NaN min/max)
arr_nan_arg = np.array([3.0, np.nan, 1.0, 4.0, np.nan], dtype=np.float64)
# NumPy's argmin returns the first NaN index, but our implementation returns first non-NaN min
# We're testing that it doesn't panic
add_test("argmin with NaN", "argmin",
         {"data": arr_nan_arg.tolist(), "shape": [5]},
         int(np.nanargmin(arr_nan_arg)), shape=[])  # Use nanargmin for expected

add_test("argmax with NaN", "argmax",
         {"data": arr_nan_arg.tolist(), "shape": [5]},
         int(np.nanargmax(arr_nan_arg)), shape=[])  # Use nanargmax for expected

# Test relu with NaN (should propagate NaN)
arr_relu_nan = np.array([-2, -1, 0, np.nan, 1, 2], dtype=np.float64)
add_test("relu with NaN", "relu",
         {"data": arr_relu_nan.tolist(), "shape": [6]},
         np.where(np.isnan(arr_relu_nan), np.nan, np.maximum(arr_relu_nan, 0)))

# Test reshapeInfer (-1 dimension)
arr_reshape = np.arange(12, dtype=np.float64)
add_test("reshapeInfer (-1, 4)", "reshapeInfer",
         {"data": arr_reshape.tolist(), "shape": [12], "new_shape": [-1, 4]},
         arr_reshape.reshape(-1, 4))

add_test("reshapeInfer (3, -1)", "reshapeInfer",
         {"data": arr_reshape.tolist(), "shape": [12], "new_shape": [3, -1]},
         arr_reshape.reshape(3, -1))

# Test min/max with NaN
arr_minmax_nan = np.array([3.0, np.nan, 1.0, 4.0], dtype=np.float64)
add_test("min with NaN", "min",
         {"data": arr_minmax_nan.tolist(), "shape": [4]},
         np.nanmin(arr_minmax_nan), shape=[])  # Our min ignores NaN like numpy's f64::min

add_test("max with NaN", "max",
         {"data": arr_minmax_nan.tolist(), "shape": [4]},
         np.nanmax(arr_minmax_nan), shape=[])  # Our max ignores NaN like numpy's f64::max

# Test matmul (basic)
a_mat = np.array([[1, 2], [3, 4]], dtype=np.float64)
b_mat = np.array([[5, 6], [7, 8]], dtype=np.float64)
add_test("matmul 2x2", "matmul",
         {"a": a_mat.flatten().tolist(), "a_shape": [2, 2],
          "b": b_mat.flatten().tolist(), "b_shape": [2, 2]},
         np.matmul(a_mat, b_mat))

# Test matmul non-square
a_rect = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # 2x3
b_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)  # 3x2
add_test("matmul 2x3 @ 3x2", "matmul",
         {"a": a_rect.flatten().tolist(), "a_shape": [2, 3],
          "b": b_rect.flatten().tolist(), "b_shape": [3, 2]},
         np.matmul(a_rect, b_rect))


# ==================== AUDIT ROUND 2 TESTS ====================

# Test sort with NaN (NaN should sort to end)
arr_sort_nan = np.array([3.0, np.nan, 1.0, np.nan, 2.0], dtype=np.float64)
add_test("sort with NaN", "sort",
         {"data": arr_sort_nan.tolist(), "shape": [5]},
         np.sort(arr_sort_nan))  # NumPy puts NaN at end

# Test argsort with NaN
add_test("argsort with NaN", "argsort",
         {"data": arr_sort_nan.tolist(), "shape": [5]},
         np.argsort(arr_sort_nan).astype(np.float64))

# Test sign with NaN (should propagate NaN)
arr_sign_nan = np.array([-2, -1, 0, np.nan, 1, 2], dtype=np.float64)
add_test("sign with NaN", "signArr",
         {"data": arr_sign_nan.tolist(), "shape": [6]},
         np.sign(arr_sign_nan))

# Test maximum with NaN (non-NaN wins)
arr_max_a = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64)
arr_max_b = np.array([2.0, 2.0, np.nan, np.nan], dtype=np.float64)
add_test("maximum with NaN", "maximum",
         {"a": arr_max_a.tolist(), "a_shape": [4],
          "b": arr_max_b.tolist(), "b_shape": [4]},
         np.maximum(arr_max_a, arr_max_b))

# Test minimum with NaN (non-NaN wins)
add_test("minimum with NaN", "minimum",
         {"a": arr_max_a.tolist(), "a_shape": [4],
          "b": arr_max_b.tolist(), "b_shape": [4]},
         np.minimum(arr_max_a, arr_max_b))

# Test unique with NaN (should have one NaN at end)
arr_unique_nan = np.array([1.0, np.nan, 2.0, 1.0, np.nan, 3.0], dtype=np.float64)
add_test("unique with NaN", "unique",
         {"data": arr_unique_nan.tolist(), "shape": [6]},
         np.unique(arr_unique_nan))  # NumPy puts one NaN at end


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        return super().default(obj)

def sanitize_for_json(obj):
    """Replace NaN/Inf with string markers in nested structures."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj

# Write output
output = {
    "generated_by": "numpy",
    "numpy_version": np.__version__,
    "test_count": len(test_cases),
    "tests": sanitize_for_json(test_cases)
}

with open("numpy-test-cases.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(test_cases)} test cases")
print(f"Saved to numpy-test-cases.json")
