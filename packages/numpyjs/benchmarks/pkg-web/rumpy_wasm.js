/* @ts-self-types="./rumpy_wasm.d.ts" */
import { startWorkers } from './snippets/wasm-bindgen-rayon-38edf6e439f6d70d/src/workerHelpers.js';

/**
 * WASM-resident f32 buffer. Wraps a `Vec<f32>` that lives in WASM linear
 * memory. JS can get a zero-copy `Float32Array` view via `ptr()` + the
 * shared memory buffer.
 *
 * The buffer stays valid until `free()` is called or the object is GC'd.
 * Memory growth does NOT invalidate the buffer (the Vec's address is
 * stable), only JS-side views of `wasmMemory().buffer` need re-deriving.
 */
export class F32Buffer {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(F32Buffer.prototype);
        obj.__wbg_ptr = ptr;
        F32BufferFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        F32BufferFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_f32buffer_free(ptr, 0);
    }
    /**
     * Copy data FROM a JS Float32Array INTO this buffer.
     * Useful for the first fill if you can't construct data directly into
     * a zero-copy view (e.g. data comes from a WebGL readback).
     * @param {Float32Array} src
     */
    copyFrom(src) {
        wasm.f32buffer_copyFrom(this.__wbg_ptr, src);
    }
    /**
     * Copy data FROM this buffer TO a JS Float32Array.
     * For the zero-copy path you don't need this — construct a view
     * instead. This exists for cases where the result needs to go to a
     * non-shared ArrayBuffer (e.g. postMessage to a context without SAB).
     * @param {Float32Array} dst
     */
    copyTo(dst) {
        wasm.f32buffer_copyTo(this.__wbg_ptr, dst);
    }
    /**
     * Explicitly free this buffer's memory. The handle is consumed.
     */
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.f32buffer_free(ptr);
    }
    /**
     * @returns {number}
     */
    len() {
        const ret = wasm.f32buffer_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Byte offset into WASM linear memory where this buffer's data starts.
     *
     * Use with `wasmMemory().buffer` to construct a zero-copy view:
     *   new Float32Array(wasmMemory().buffer, buf.ptr(), buf.len())
     * @returns {number}
     */
    ptr() {
        const ret = wasm.f32buffer_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) F32Buffer.prototype[Symbol.dispose] = F32Buffer.prototype.free;

/**
 * N-dimensional array type exposed to JavaScript
 */
export class NDArray {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(NDArray.prototype);
        obj.__wbg_ptr = ptr;
        NDArrayFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NDArrayFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ndarray_free(ptr, 0);
    }
    /**
     * @param {NDArray} other
     * @returns {NDArray}
     */
    add(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_add(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} scalar
     * @returns {NDArray}
     */
    addScalar(scalar) {
        const ret = wasm.ndarray_addScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * Test if all elements are true (non-zero)
     *
     * Returns 1.0 if all elements are non-zero, 0.0 otherwise.
     * Equivalent to numpy.all(arr).
     * @returns {number}
     */
    all() {
        const ret = wasm.ndarray_all(this.__wbg_ptr);
        return ret;
    }
    /**
     * Test if all elements are true along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    allAxis(axis, keepdims) {
        const ret = wasm.ndarray_allAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Test if any element is true (non-zero)
     *
     * Returns 1.0 if any element is non-zero, 0.0 otherwise.
     * Equivalent to numpy.any(arr).
     * @returns {number}
     */
    any() {
        const ret = wasm.ndarray_any(this.__wbg_ptr);
        return ret;
    }
    /**
     * Test if any element is true along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    anyAxis(axis, keepdims) {
        const ret = wasm.ndarray_anyAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Argmax - index of maximum value (flattened)
     * @returns {number}
     */
    argmax() {
        const ret = wasm.ndarray_argmax(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Argmin - index of minimum value (flattened)
     * @returns {number}
     */
    argmin() {
        const ret = wasm.ndarray_argmin(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {string} dtype
     * @returns {NDArray}
     */
    asType(dtype) {
        const ptr0 = passStringToWasm0(dtype, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_asType(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Average pooling 2D
     *
     * Input shape: (N, C, H, W)
     * Output shape: (N, C, H_out, W_out)
     * @param {number} kernel_h
     * @param {number} kernel_w
     * @param {number} stride_h
     * @param {number} stride_w
     * @param {number} pad_h
     * @param {number} pad_w
     * @returns {NDArray}
     */
    avgPool2d(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) {
        const ret = wasm.ndarray_avgPool2d(this.__wbg_ptr, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {NDArray | null | undefined} gamma
     * @param {NDArray | null | undefined} beta
     * @param {NDArray} running_mean
     * @param {NDArray} running_var
     * @param {number} eps
     * @returns {NDArray}
     */
    batchNorm(gamma, beta, running_mean, running_var, eps) {
        let ptr0 = 0;
        if (!isLikeNone(gamma)) {
            _assertClass(gamma, NDArray);
            ptr0 = gamma.__destroy_into_raw();
        }
        let ptr1 = 0;
        if (!isLikeNone(beta)) {
            _assertClass(beta, NDArray);
            ptr1 = beta.__destroy_into_raw();
        }
        _assertClass(running_mean, NDArray);
        _assertClass(running_var, NDArray);
        const ret = wasm.ndarray_batchNorm(this.__wbg_ptr, ptr0, ptr1, running_mean.__wbg_ptr, running_var.__wbg_ptr, eps);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {Uint32Array} shape
     * @returns {NDArray}
     */
    broadcastTo(shape) {
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_broadcastTo(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} chunk_size
     * @param {number} axis
     * @returns {Array<any>}
     */
    chunk(chunk_size, axis) {
        const ret = wasm.ndarray_chunk(this.__wbg_ptr, chunk_size, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Clip values to a range
     * @param {number} min
     * @param {number} max
     * @returns {NDArray}
     */
    clip(min, max) {
        const ret = wasm.ndarray_clip(this.__wbg_ptr, min, max);
        return NDArray.__wrap(ret);
    }
    /**
     * Clone the array
     * @returns {NDArray}
     */
    clone() {
        const ret = wasm.ndarray_clone(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * Count non-zero elements
     * @returns {number}
     */
    countNonzero() {
        const ret = wasm.ndarray_countNonzero(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} axis
     * @returns {NDArray}
     */
    cumprod(axis) {
        const ret = wasm.ndarray_cumprod(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} axis
     * @returns {NDArray}
     */
    cumsum(axis) {
        const ret = wasm.ndarray_cumsum(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Get pointer to the underlying data buffer
     *
     * Returns the byte offset into WASM linear memory where this array's data begins.
     * Use with `memory()` to create a zero-copy TypedArray view.
     *
     * WARNING: The pointer is only valid while this NDArray exists and WASM memory
     * hasn't been resized. Cache invalidation is the caller's responsibility.
     * @returns {number}
     */
    dataPtr() {
        const ret = wasm.ndarray_dataPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} scale
     * @param {number} zero_point
     * @returns {NDArray}
     */
    dequantizeLinear(scale, zero_point) {
        const ret = wasm.ndarray_dequantizeLinear(this.__wbg_ptr, scale, zero_point);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} k
     * @returns {NDArray}
     */
    diag(k) {
        const ret = wasm.ndarray_diag(this.__wbg_ptr, k);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Extract diagonal from 2D array (alias for diag with k=0)
     * @param {number | null} [offset]
     * @returns {NDArray}
     */
    diagonal(offset) {
        const ret = wasm.ndarray_diagonal(this.__wbg_ptr, isLikeNone(offset) ? 0x100000001 : (offset) >> 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} n
     * @param {number} axis
     * @returns {NDArray}
     */
    diff(n, axis) {
        const ret = wasm.ndarray_diff(this.__wbg_ptr, n, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {NDArray} other
     * @returns {NDArray}
     */
    div(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_div(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} scalar
     * @returns {NDArray}
     */
    divScalar(scalar) {
        const ret = wasm.ndarray_divScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * Data type
     * @returns {string}
     */
    get dtype() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.ndarray_dtype(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Comparison: equal (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    eq(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_eq(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: equal
     * @param {number} scalar
     * @returns {NDArray}
     */
    eqScalar(scalar) {
        const ret = wasm.ndarray_eqScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * Expand dims - add axis of length 1
     * @param {number} axis
     * @returns {NDArray}
     */
    expandDims(axis) {
        const ret = wasm.ndarray_expandDims(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @returns {NDArray}
     */
    flatten() {
        const ret = wasm.ndarray_flatten(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} axis
     * @returns {NDArray}
     */
    flip(axis) {
        const ret = wasm.ndarray_flip(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Flip array horizontally (along axis 1)
     *
     * Equivalent to numpy.fliplr(arr) or flip(arr, 1).
     * Requires at least 2D array.
     * @returns {NDArray}
     */
    fliplr() {
        const ret = wasm.ndarray_fliplr(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Flip array vertically (along axis 0)
     *
     * Equivalent to numpy.flipud(arr) or flip(arr, 0).
     * @returns {NDArray}
     */
    flipud() {
        const ret = wasm.ndarray_flipud(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Explicitly free the array memory
     *
     * After calling this, the NDArray is consumed and cannot be used.
     * This is useful for deterministic memory cleanup without waiting for GC.
     */
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.ndarray_free(ptr);
    }
    /**
     * @param {NDArray} indices
     * @param {number} axis
     * @returns {NDArray}
     */
    gather(indices, axis) {
        _assertClass(indices, NDArray);
        const ret = wasm.ndarray_gather(this.__wbg_ptr, indices.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Comparison: greater than or equal (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    ge(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_ge(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: greater than or equal
     * @param {number} scalar
     * @returns {NDArray}
     */
    geScalar(scalar) {
        const ret = wasm.ndarray_geScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * GELU activation (Gaussian Error Linear Unit)
     * Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     * @returns {NDArray}
     */
    gelu() {
        const ret = wasm.ndarray_gelu(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * Get elements where mask is non-zero (truthy)
     *
     * Returns a 1D array of selected elements.
     * Mask must be same shape as self, or broadcastable.
     *
     * Example: arr.getByMask(arr.gt_scalar(0.5)) returns all elements > 0.5
     * @param {NDArray} mask
     * @returns {NDArray}
     */
    getByMask(mask) {
        _assertClass(mask, NDArray);
        const ret = wasm.ndarray_getByMask(this.__wbg_ptr, mask.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Get element at flat index
     * @param {number} index
     * @returns {number}
     */
    getFlat(index) {
        const ret = wasm.ndarray_getFlat(this.__wbg_ptr, index);
        return ret;
    }
    /**
     * Comparison: greater than (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    gt(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_gt(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: greater than
     * @param {number} scalar
     * @returns {NDArray}
     */
    gtScalar(scalar) {
        const ret = wasm.ndarray_gtScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * im2col: Convert image patches to columns for convolution via GEMM
     *
     * Input shape: (N, C, H, W) - batch, channels, height, width
     * Output shape: (N * H_out * W_out, C * kernel_h * kernel_w)
     *
     * This transforms the convolution operation into a matrix multiplication:
     *   output = im2col(input) @ weights.reshape(out_channels, -1).T
     * @param {number} kernel_h
     * @param {number} kernel_w
     * @param {number} stride_h
     * @param {number} stride_w
     * @param {number} pad_h
     * @param {number} pad_w
     * @returns {NDArray}
     */
    im2col(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) {
        const ret = wasm.ndarray_im2col(this.__wbg_ptr, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} axis
     * @param {NDArray} indices
     * @param {NDArray} src
     * @returns {NDArray}
     */
    indexCopy(axis, indices, src) {
        _assertClass(indices, NDArray);
        _assertClass(src, NDArray);
        const ret = wasm.ndarray_indexCopy(this.__wbg_ptr, axis, indices.__wbg_ptr, src.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {Uint32Array} size
     * @returns {NDArray}
     */
    interpolate(size) {
        const ptr0 = passArray32ToWasm0(size, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_interpolate(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Check if array is empty
     * @returns {boolean}
     */
    isEmpty() {
        const ret = wasm.ndarray_isEmpty(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check for finite values (not NaN, not Inf)
     * @returns {NDArray}
     */
    isFinite() {
        const ret = wasm.ndarray_isFinite(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * Check for infinite values
     * @returns {NDArray}
     */
    isInf() {
        const ret = wasm.ndarray_isInf(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * Check for NaN values
     * @returns {NDArray}
     */
    isNan() {
        const ret = wasm.ndarray_isNan(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {Uint32Array} normalized_shape
     * @param {NDArray | null | undefined} gamma
     * @param {NDArray | null | undefined} beta
     * @param {number} eps
     * @returns {NDArray}
     */
    layerNorm(normalized_shape, gamma, beta, eps) {
        const ptr0 = passArray32ToWasm0(normalized_shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        let ptr1 = 0;
        if (!isLikeNone(gamma)) {
            _assertClass(gamma, NDArray);
            ptr1 = gamma.__destroy_into_raw();
        }
        let ptr2 = 0;
        if (!isLikeNone(beta)) {
            _assertClass(beta, NDArray);
            ptr2 = beta.__destroy_into_raw();
        }
        const ret = wasm.ndarray_layerNorm(this.__wbg_ptr, ptr0, len0, ptr1, ptr2, eps);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Comparison: less than or equal (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    le(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_le(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: less than or equal
     * @param {number} scalar
     * @returns {NDArray}
     */
    leScalar(scalar) {
        const ret = wasm.ndarray_leScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * Get the number of elements in the array
     * @returns {number}
     */
    len() {
        const ret = wasm.ndarray_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} axis
     * @returns {NDArray}
     */
    logSoftmax(axis) {
        const ret = wasm.ndarray_logSoftmax(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    logsumexp(axis, keepdims) {
        const ret = wasm.ndarray_logsumexp(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Comparison: less than (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    lt(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_lt(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: less than
     * @param {number} scalar
     * @returns {NDArray}
     */
    ltScalar(scalar) {
        const ret = wasm.ndarray_ltScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    max() {
        const ret = wasm.ndarray_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * Max along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    maxAxis(axis, keepdims) {
        const ret = wasm.ndarray_maxAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Max pooling 2D
     *
     * Input shape: (N, C, H, W)
     * Output shape: (N, C, H_out, W_out)
     * @param {number} kernel_h
     * @param {number} kernel_w
     * @param {number} stride_h
     * @param {number} stride_w
     * @param {number} pad_h
     * @param {number} pad_w
     * @returns {NDArray}
     */
    maxPool2d(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) {
        const ret = wasm.ndarray_maxPool2d(this.__wbg_ptr, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.ndarray_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Mean along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    meanAxis(axis, keepdims) {
        const ret = wasm.ndarray_meanAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @returns {number}
     */
    min() {
        const ret = wasm.ndarray_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * Min along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    minAxis(axis, keepdims) {
        const ret = wasm.ndarray_minAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {NDArray} other
     * @returns {NDArray}
     */
    mul(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_mul(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} scalar
     * @returns {NDArray}
     */
    mulScalar(scalar) {
        const ret = wasm.ndarray_mulScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} num_samples
     * @param {boolean} replacement
     * @returns {NDArray}
     */
    multinomial(num_samples, replacement) {
        const ret = wasm.ndarray_multinomial(this.__wbg_ptr, num_samples, replacement);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Get total size in bytes
     * @returns {number}
     */
    nbytes() {
        const ret = wasm.ndarray_nbytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of dimensions
     * @returns {number}
     */
    get ndim() {
        const ret = wasm.ndarray_ndim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Comparison: not equal (element-wise)
     * @param {NDArray} other
     * @returns {NDArray}
     */
    ne(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_ne(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Scalar comparison: not equal
     * @param {number} scalar
     * @returns {NDArray}
     */
    neScalar(scalar) {
        const ret = wasm.ndarray_neScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * Get indices of non-zero elements (flat indices)
     * @returns {Uint32Array}
     */
    nonzeroFlat() {
        const ret = wasm.ndarray_nonzeroFlat(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {Uint32Array} pad_width
     * @param {number} constant_value
     * @returns {NDArray}
     */
    pad(pad_width, constant_value) {
        const ptr0 = passArray32ToWasm0(pad_width, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_pad(this.__wbg_ptr, ptr0, len0, constant_value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Permute array dimensions
     * axes specifies the new order of dimensions
     * e.g., permute([1, 0, 2]) swaps first two dimensions
     * @param {Uint32Array} axes
     * @returns {NDArray}
     */
    permute(axes) {
        const ptr0 = passArray32ToWasm0(axes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_permute(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} scalar
     * @returns {NDArray}
     */
    powScalar(scalar) {
        const ret = wasm.ndarray_powScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    prod() {
        const ret = wasm.ndarray_prod(this.__wbg_ptr);
        return ret;
    }
    /**
     * ReLU activation: max(0, x)
     * NaN values are propagated (NumPy behavior)
     * @returns {NDArray}
     */
    relu() {
        const ret = wasm.ndarray_relu(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} repeats
     * @param {number | null} [axis]
     * @returns {NDArray}
     */
    repeat(repeats, axis) {
        const ret = wasm.ndarray_repeat(this.__wbg_ptr, repeats, isLikeNone(axis) ? 0x100000001 : (axis) >>> 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {Uint32Array} shape
     * @returns {NDArray}
     */
    reshape(shape) {
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_reshape(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Reshape with support for -1 dimension inference (NumPy-style)
     * One dimension can be -1, which will be inferred from the total size
     * @param {BigInt64Array} shape
     * @returns {NDArray}
     */
    reshapeInfer(shape) {
        const ptr0 = passArray64ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_reshapeInfer(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Alias for interpolate
     * @param {Uint32Array} size
     * @returns {NDArray}
     */
    resize(size) {
        const ptr0 = passArray32ToWasm0(size, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_resize(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {NDArray} gamma
     * @param {number} eps
     * @returns {NDArray}
     */
    rmsNorm(gamma, eps) {
        _assertClass(gamma, NDArray);
        const ret = wasm.ndarray_rmsNorm(this.__wbg_ptr, gamma.__wbg_ptr, eps);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} shift
     * @param {number} axis
     * @returns {NDArray}
     */
    roll(shift, axis) {
        const ret = wasm.ndarray_roll(this.__wbg_ptr, shift, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @returns {NDArray}
     */
    rsqrt() {
        const ret = wasm.ndarray_rsqrt(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} axis
     * @param {NDArray} indices
     * @param {NDArray} src
     * @returns {NDArray}
     */
    scatter(axis, indices, src) {
        _assertClass(indices, NDArray);
        _assertClass(src, NDArray);
        const ret = wasm.ndarray_scatter(this.__wbg_ptr, axis, indices.__wbg_ptr, src.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Set elements where mask is non-zero to a scalar value
     *
     * Returns a new array with selected elements replaced.
     * @param {NDArray} mask
     * @param {number} value
     * @returns {NDArray}
     */
    setByMask(mask, value) {
        _assertClass(mask, NDArray);
        const ret = wasm.ndarray_setByMask(this.__wbg_ptr, mask.__wbg_ptr, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Get array shape
     * @returns {Uint32Array}
     */
    get shape() {
        const ret = wasm.ndarray_shape(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {NDArray}
     */
    sigmoid() {
        const ret = wasm.ndarray_sigmoid(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @returns {NDArray}
     */
    silu() {
        const ret = wasm.ndarray_silu(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * Total number of elements
     * @returns {number}
     */
    get size() {
        const ret = wasm.ndarray_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Slice the array with start:stop:step for each dimension
     *
     * Uses parallel i32 arrays for starts, stops, steps.
     * - Negative indices work like Python (count from end)
     * - i32::MAX (2147483647) for stop means "to the end" (like : in Python)
     * - Missing dimensions in arrays assume full range
     *
     * Example: arr[1:3, :, 2:5] with shape [10, 10, 10]
     *   starts = [1, 0, 2]
     *   stops = [3, 2147483647, 5]  // MAX_INT for ":"
     *   steps = [1, 1, 1]
     * @param {Int32Array} starts
     * @param {Int32Array} stops
     * @param {Int32Array} steps
     * @returns {NDArray}
     */
    slice(starts, stops, steps) {
        const ptr0 = passArray32ToWasm0(starts, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(stops, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(steps, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_slice(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Slice along a single axis (simpler API for common case)
     *
     * Equivalent to arr[:, :, start:stop] when axis=2
     * @param {number} axis
     * @param {number} start
     * @param {number} stop
     * @returns {NDArray}
     */
    sliceAxis(axis, start, stop) {
        const ret = wasm.ndarray_sliceAxis(this.__wbg_ptr, axis, start, stop);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Softmax along an axis
     * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
     * @param {number} axis
     * @returns {NDArray}
     */
    softmax(axis) {
        const ret = wasm.ndarray_softmax(this.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} num_splits
     * @param {number} axis
     * @returns {Array<any>}
     */
    split(num_splits, axis) {
        const ret = wasm.ndarray_split(this.__wbg_ptr, num_splits, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Squeeze - remove axes of length 1
     * @returns {NDArray}
     */
    squeeze() {
        const ret = wasm.ndarray_squeeze(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    std() {
        const ret = wasm.ndarray_std(this.__wbg_ptr);
        return ret;
    }
    /**
     * Standard deviation with degrees of freedom adjustment
     * ddof=0 for population std (default), ddof=1 for sample std
     * @param {number} ddof
     * @returns {number}
     */
    stdDdof(ddof) {
        const ret = wasm.ndarray_stdDdof(this.__wbg_ptr, ddof);
        return ret;
    }
    /**
     * @param {NDArray} other
     * @returns {NDArray}
     */
    sub(other) {
        _assertClass(other, NDArray);
        const ret = wasm.ndarray_sub(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} scalar
     * @returns {NDArray}
     */
    subScalar(scalar) {
        const ret = wasm.ndarray_subScalar(this.__wbg_ptr, scalar);
        return NDArray.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    sum() {
        const ret = wasm.ndarray_sum(this.__wbg_ptr);
        return ret;
    }
    /**
     * Sum along an axis
     * @param {number} axis
     * @param {boolean | null} [keepdims]
     * @returns {NDArray}
     */
    sumAxis(axis, keepdims) {
        const ret = wasm.ndarray_sumAxis(this.__wbg_ptr, axis, isLikeNone(keepdims) ? 0xFFFFFF : keepdims ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {NDArray} indices
     * @param {number} axis
     * @returns {NDArray}
     */
    take(indices, axis) {
        _assertClass(indices, NDArray);
        const ret = wasm.ndarray_take(this.__wbg_ptr, indices.__wbg_ptr, axis);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {Uint32Array} reps
     * @returns {NDArray}
     */
    tile(reps) {
        const ptr0 = passArray32ToWasm0(reps, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.ndarray_tile(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * Convert to Float64Array (creates a copy)
     *
     * This method always works but involves copying data from WASM memory to JS.
     * For zero-copy access when SharedArrayBuffer is available, use `asTypedArrayView()`.
     * @returns {Float64Array}
     */
    toTypedArray() {
        const ret = wasm.ndarray_toTypedArray(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} k
     * @param {number} axis
     * @param {boolean} sorted
     * @returns {Array<any>}
     */
    topk(k, axis, sorted) {
        const ret = wasm.ndarray_topk(this.__wbg_ptr, k, axis, sorted);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {NDArray}
     */
    transpose() {
        const ret = wasm.ndarray_transpose(this.__wbg_ptr);
        return NDArray.__wrap(ret);
    }
    /**
     * @param {number} k
     * @returns {NDArray}
     */
    tril(k) {
        const ret = wasm.ndarray_tril(this.__wbg_ptr, k);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @param {number} k
     * @returns {NDArray}
     */
    triu(k) {
        const ret = wasm.ndarray_triu(this.__wbg_ptr, k);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return NDArray.__wrap(ret[0]);
    }
    /**
     * @returns {number}
     */
    var() {
        const ret = wasm.ndarray_var(this.__wbg_ptr);
        return ret;
    }
    /**
     * Variance with degrees of freedom adjustment
     * ddof=0 for population variance (default), ddof=1 for sample variance
     * @param {number} ddof
     * @returns {number}
     */
    varDdof(ddof) {
        const ret = wasm.ndarray_varDdof(this.__wbg_ptr, ddof);
        return ret;
    }
}
if (Symbol.dispose) NDArray.prototype[Symbol.dispose] = NDArray.prototype.free;

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function absArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.absArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Allocate an f32 buffer of the given length inside WASM memory.
 * Contents are uninitialised — write before reading.
 * @param {number} len
 * @returns {F32Buffer}
 */
export function allocF32(len) {
    const ret = wasm.allocF32(len);
    return F32Buffer.__wrap(ret);
}

/**
 * @param {number} start
 * @param {number} stop
 * @param {number} step
 * @returns {NDArray}
 */
export function arange(start, stop, step) {
    const ret = wasm.arange(start, stop, step);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {Float64Array} data
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function arrayFromTyped(data, shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.arrayFromTyped(data, ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Element-wise atan2(y, x)
 *
 * Computes the arc tangent of y/x, using the signs of both arguments
 * to determine the quadrant of the return value.
 * Equivalent to numpy.arctan2(y, x).
 * @param {NDArray} y
 * @param {NDArray} x
 * @returns {NDArray}
 */
export function atan2Arr(y, x) {
    _assertClass(y, NDArray);
    _assertClass(x, NDArray);
    const ret = wasm.atan2Arr(y.__wbg_ptr, x.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Create a causal attention mask (lower triangular matrix of -inf and 0)
 *
 * Returns a mask where positions that can attend are 0, others are -inf.
 * Useful for transformer causal (autoregressive) attention.
 *
 * # Arguments
 * * `size` - Sequence length (creates size x size mask)
 * @param {number} size
 * @returns {NDArray}
 */
export function causalMask(size) {
    const ret = wasm.causalMask(size);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function ceilArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.ceilArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Compute checksum (sum of all elements) for verification
 * @param {Float32Array} a
 * @returns {number}
 */
export function checksum(a) {
    const ret = wasm.checksum(a);
    return ret;
}

/**
 * Concatenate two arrays along an axis
 * @param {NDArray} a
 * @param {NDArray} b
 * @param {number} axis
 * @returns {NDArray}
 */
export function concatenate2(a, b, axis) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.concatenate2(a.__wbg_ptr, b.__wbg_ptr, axis);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Concatenate three arrays along an axis
 * @param {NDArray} a
 * @param {NDArray} b
 * @param {NDArray} c
 * @param {number} axis
 * @returns {NDArray}
 */
export function concatenate3(a, b, c, axis) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    _assertClass(c, NDArray);
    const ret = wasm.concatenate3(a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr, axis);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * 2D Convolution
 *
 * Performs 2D convolution of input with kernel.
 * Input shape: (N, C_in, H, W)
 * Kernel shape: (C_out, C_in, kH, kW)
 * Output shape: (N, C_out, H_out, W_out)
 *
 * # Arguments
 * * `input` - Input tensor (N, C_in, H, W)
 * * `kernel` - Convolution kernel (C_out, C_in, kH, kW)
 * * `stride` - Stride [stride_h, stride_w]
 * * `padding` - Padding [pad_h, pad_w]
 * @param {NDArray} input
 * @param {NDArray} kernel
 * @param {Uint32Array} stride
 * @param {Uint32Array} padding
 * @returns {NDArray}
 */
export function conv2d(input, kernel, stride, padding) {
    _assertClass(input, NDArray);
    _assertClass(kernel, NDArray);
    const ptr0 = passArray32ToWasm0(stride, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(padding, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.conv2d(input.__wbg_ptr, kernel.__wbg_ptr, ptr0, len0, ptr1, len1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * 2D Transposed Convolution (Deconvolution)
 *
 * Performs transposed 2D convolution, often used for upsampling.
 * Input shape: (N, C_in, H, W)
 * Kernel shape: (C_in, C_out, kH, kW)  -- note C_in, C_out order differs from conv2d
 * Output shape: (N, C_out, H_out, W_out)
 *
 * # Arguments
 * * `input` - Input tensor (N, C_in, H, W)
 * * `kernel` - Convolution kernel (C_in, C_out, kH, kW)
 * * `stride` - Stride [stride_h, stride_w]
 * * `padding` - Padding [pad_h, pad_w]
 * * `output_padding` - Additional padding for output [out_pad_h, out_pad_w]
 * @param {NDArray} input
 * @param {NDArray} kernel
 * @param {Uint32Array} stride
 * @param {Uint32Array} padding
 * @param {Uint32Array} output_padding
 * @returns {NDArray}
 */
export function convTranspose2d(input, kernel, stride, padding, output_padding) {
    _assertClass(input, NDArray);
    _assertClass(kernel, NDArray);
    const ptr0 = passArray32ToWasm0(stride, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(padding, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(output_padding, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.convTranspose2d(input.__wbg_ptr, kernel.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function cosArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.cosArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function coshArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.coshArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {number}
 */
export function det(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.det(arr.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function dot(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.dot(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Einstein summation convention
 *
 * Performs tensor contractions, transposes, and reductions using
 * Einstein notation. Critical for attention mechanisms.
 *
 * Supported patterns:
 * - "ij,jk->ik" : matrix multiplication
 * - "bij,bjk->bik" : batched matrix multiplication
 * - "bhqk,bhkd->bhqd" : attention (Q @ K.T @ V)
 * - "ij->ji" : transpose
 * - "ij->" : sum all
 * - "ij->i" : sum over j
 * - "ii->" : trace
 * - "...ij,...jk->...ik" : batched matmul with ellipsis
 *
 * # Arguments
 * * `subscripts` - Einstein summation subscripts (e.g., "ij,jk->ik")
 * * `a` - First input array
 * * `b` - Optional second input array
 * @param {string} subscripts
 * @param {NDArray} a
 * @param {NDArray | null} [b]
 * @returns {NDArray}
 */
export function einsum(subscripts, a, b) {
    const ptr0 = passStringToWasm0(subscripts, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    _assertClass(a, NDArray);
    let ptr1 = 0;
    if (!isLikeNone(b)) {
        _assertClass(b, NDArray);
        ptr1 = b.__destroy_into_raw();
    }
    const ret = wasm.einsum(ptr0, len0, a.__wbg_ptr, ptr1);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function expArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.expArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise exp(x) - 1
 *
 * Computes exp(x) - 1 with better precision for small x.
 * Equivalent to numpy.expm1(x).
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function expm1Arr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.expm1Arr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {number} n
 * @returns {NDArray}
 */
export function eye(n) {
    const ret = wasm.eye(n);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function floorArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.floorArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {Uint32Array} shape
 * @param {number} value
 * @returns {NDArray}
 */
export function full(shape, value) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.full(ptr0, len0, value);
    return NDArray.__wrap(ret);
}

/**
 * Get the current number of rayon threads
 * @returns {number}
 */
export function getNumThreads() {
    const ret = wasm.getNumThreads();
    return ret >>> 0;
}

/**
 * Check if SharedArrayBuffer is available
 *
 * Returns true if the environment supports SharedArrayBuffer (COOP/COEP headers set).
 * When false, `asTypedArrayView()` will not work and you should use `toTypedArray()`.
 * @returns {boolean}
 */
export function hasSharedArrayBuffer() {
    const ret = wasm.hasSharedArrayBuffer();
    return ret !== 0;
}

/**
 * Horizontal stack (concatenate along axis 1 for 2D+, axis 0 for 1D)
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function hstack2(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.hstack2(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Initialize panic hook for better error messages
 */
export function init() {
    wasm.init();
}

/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return ret;
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function inv(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.inv(arr.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {number} start
 * @param {number} stop
 * @param {number} num
 * @returns {NDArray}
 */
export function linspace(start, stop, num) {
    const ret = wasm.linspace(start, stop, num);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise log(1 + x)
 *
 * Computes log(1 + x) with better precision for small x.
 * Equivalent to numpy.log1p(x).
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function log1pArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.log1pArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function logArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.logArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise logical AND
 *
 * Returns 1.0 where both inputs are non-zero, 0.0 otherwise.
 * Equivalent to numpy.logical_and(a, b).
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function logicalAnd(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.logicalAnd(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Element-wise logical NOT
 *
 * Returns 1.0 where input is zero, 0.0 otherwise.
 * Equivalent to numpy.logical_not(x).
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function logicalNot(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.logicalNot(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise logical OR
 *
 * Returns 1.0 where either input is non-zero, 0.0 otherwise.
 * Equivalent to numpy.logical_or(a, b).
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function logicalOr(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.logicalOr(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function matmul(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.matmul(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Fast f32 matrix multiplication using WASM SIMD
 *
 * This is a direct binding to the SIMD-optimized GEMM kernel, matching XNNPACK's approach.
 * Uses f32 (4 elements per v128) instead of f64 (2 elements per v128) for 2x throughput.
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32(a, b, m, n, k) {
    const ret = wasm.matmulF32(a, b, m, n, k);
    return ret;
}

/**
 * 5x8 kernel specifically for matrices where M is divisible by 5
 *
 * Optimized for 100x100 case (and similar).
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF325x8(a, b, m, n, k) {
    const ret = wasm.matmulF325x8(a, b, m, n, k);
    return ret;
}

/**
 * Auto-tuned f32 matrix multiplication
 *
 * Automatically selects the best kernel based on matrix dimensions:
 * - 5x8 kernel for matrices where M % 5 == 0 (like 100x100)
 * - FMA for medium matrices (packing overhead not amortized)
 * - FMA + packed for large matrices (packing overhead amortized)
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Auto(a, b, m, n, k) {
    const ret = wasm.matmulF32Auto(a, b, m, n, k);
    return ret;
}

/**
 * Cache-blocked 6x8 GEMM for large matrices
 *
 * Uses GOTO-style cache blocking to tile the computation:
 * - Outer loop tiles by N dimension (NC=256)
 * - Middle loop tiles by K dimension (KC=256)
 * - Inner loop tiles by M dimension (MC=128)
 *
 * This ensures working set fits in L1/L2 cache for better performance
 * on large matrices (256x256 and above).
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Blocked(a, b, m, n, k) {
    const ret = wasm.matmulF32Blocked(a, b, m, n, k);
    return ret;
}

/**
 * Fast f32 matrix multiplication with FMA (fused multiply-add)
 *
 * Uses relaxed-simd f32x4_relaxed_madd for better throughput.
 * FMA computes a*b+c in one instruction instead of two (mul + add).
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32FMA(a, b, m, n, k) {
    const ret = wasm.matmulF32FMA(a, b, m, n, k);
    return ret;
}

/**
 * Fast f32 matrix multiplication with FMA + packed B
 *
 * Combines both optimizations: FMA instructions and B matrix packing.
 * This is the fastest kernel for large matrices.
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32FMAPacked(a, b, m, n, k) {
    const ret = wasm.matmulF32FMAPacked(a, b, m, n, k);
    return ret;
}

/**
 * Highly optimized 6x8 GEMM with FMA, loadsplat, and cache blocking
 *
 * This is the most optimized implementation, matching XNNPACK patterns:
 * - 6x8 micro-kernel (12 accumulators fit in 16 XMM registers)
 * - f32x4_relaxed_madd for FMA
 * - v128_load32_splat for A broadcast
 * - L1/L2 cache blocking (KC=256, MC=72, NC=128)
 * - B matrix packing for contiguous access
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Optimized(a, b, m, n, k) {
    const ret = wasm.matmulF32Optimized(a, b, m, n, k);
    return ret;
}

/**
 * Parallel version of optimized 6x8 GEMM using rayon (LEGACY)
 *
 * Kept for A/B benchmarking. Has known problems — see v3 below.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32OptimizedParallel(a, b, m, n, k) {
    const ret = wasm.matmulF32OptimizedParallel(a, b, m, n, k);
    return ret;
}

/**
 * Parallel optimised GEMM, v3: pack-once, 2D-tile, atomic work-claiming.
 *
 * This is the recommended parallel path. Differences from the legacy
 * `matmulF32OptimizedParallel`:
 *
 * * B is packed ONCE and shared read-only across all workers
 *   (old path packed B independently in every thread — with N threads that's
 *   N× the packing work and N× allocator contention on WASM's locked dlmalloc)
 *
 * * Macro-tiles (~MC × NC) are handed out via an atomic counter, matching
 *   XNNPACK's `pthreadpool_parallelize_2d_tile_2d`. Load balances across
 *   Apple Silicon perf/efficiency cores instead of assuming uniform workers.
 *
 * * Zero per-task heap allocation. Workers write straight into disjoint
 *   C slices.
 *
 * * The calling thread participates (it's "thread 0"), so with an N-worker
 *   Rayon pool you get N+1 way parallelism.
 *
 * Requires `initThreadPool(n)` to have been called (same as legacy path).
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32OptimizedParallelV3(a, b, m, n, k) {
    const ret = wasm.matmulF32OptimizedParallelV3(a, b, m, n, k);
    return ret;
}

/**
 * Parallel optimised GEMM, v4: hijack Rayon's workers with raw
 * `memory.atomic.wait32`/`notify` dispatch.
 *
 * v3 uses ONE `rayon::scope` per matmul (good), but inside it there's
 * still no shared packed-B (each thread packs its own) and the join is
 * Rayon's standard park/unpark.  v4 is the full pthreadpool model:
 *
 * * ONE `rayon::scope` — we use wasm-bindgen-rayon's Web Workers but
 *   NOT Rayon's task scheduler.
 *
 * * Workers enter OUR spin-then-`atomic.wait` loop. Main drives them
 *   block-by-block: pack B (shared), bump generation, `atomic.notify`,
 *   drain tiles alongside workers, spin-wait for completion, repeat.
 *
 * * Shared packed-B → minimum total packing work (same as single-thread).
 *
 * * Per-block sync is ~1 `atomic.notify` + N×1 Relaxed `fetch_sub` +
 *   one short main-thread spin. Compare Rayon: N× `Box<dyn FnOnce>` +
 *   N× park/unpark per scope.
 *
 * This is "our own thread manager", hosted inside Rayon's already-spawned
 * workers. No new dependencies, no separate worker pool to manage.
 *
 * Requires `initThreadPool(n)` (same as v3).
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32OptimizedParallelV4(a, b, m, n, k) {
    const ret = wasm.matmulF32OptimizedParallelV4(a, b, m, n, k);
    return ret;
}

/**
 * Fast f32 matrix multiplication with explicit matrix packing
 *
 * This version always uses matrix packing regardless of size, for benchmarking.
 * Packing reorders B matrix into cache-friendly column panels.
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Packed(a, b, m, n, k) {
    const ret = wasm.matmulF32Packed(a, b, m, n, k);
    return ret;
}

/**
 * Parallel f32 matrix multiplication using rayon + Web Workers
 *
 * Uses rayon to parallelize across the M dimension with native WASM threads.
 * MUST call `initThreadPool(num_threads)` from JS before using this function!
 *
 * For large matrices (256+), this scales with available cores.
 * Falls back to single-threaded for small matrices.
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Parallel(a, b, m, n, k) {
    const ret = wasm.matmulF32Parallel(a, b, m, n, k);
    return ret;
}

/**
 * Parallel f32 matrix multiplication V2 using rayon + Web Workers (zero-allocation)
 *
 * This is an improved version that writes directly to pre-allocated memory,
 * avoiding per-thread allocations. This is significantly faster than V1
 * for large matrices.
 *
 * MUST call `initThreadPool(num_threads)` from JS before using this function!
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32ParallelV2(a, b, m, n, k) {
    const ret = wasm.matmulF32ParallelV2(a, b, m, n, k);
    return ret;
}

/**
 * Parallel matmul with pre-packed B (from packBFull).
 *
 * Note: Prepacking optimization was removed. This now calls the regular
 * parallel matmul (packed_b is treated as normal B).
 * @param {Float32Array} a
 * @param {Float32Array} packed_b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Prepacked(a, packed_b, m, n, k) {
    const ret = wasm.matmulF32Prepacked(a, packed_b, m, n, k);
    return ret;
}

/**
 * Parallel matmul with pre-packed B, ZERO JS↔WASM copies.
 *
 * The leanest call path: A and packed-B already in WASM memory, C
 * written directly, no per-call packing. This is the tf.js-equivalent
 * path for NN inference.
 *
 * Note: The specialized prepacked kernel was removed during a refactor.
 * This now just calls the regular parallel matmul (packed_b is treated as B).
 * @param {F32Buffer} a
 * @param {F32Buffer} packed_b
 * @param {F32Buffer} c
 * @param {number} m
 * @param {number} n
 * @param {number} k
 */
export function matmulF32PrepackedZeroCopy(a, packed_b, c, m, n, k) {
    _assertClass(a, F32Buffer);
    _assertClass(packed_b, F32Buffer);
    _assertClass(c, F32Buffer);
    wasm.matmulF32PrepackedZeroCopy(a.__wbg_ptr, packed_b.__wbg_ptr, c.__wbg_ptr, m, n, k);
}

/**
 * Parallel f32 matrix multiplication using pthreadpool-rs
 *
 * Uses pthreadpool-rs instead of rayon for parallelization.
 * On WASM, pthreadpool-rs uses wasm-bindgen-rayon under the hood.
 *
 * MUST call `initThreadPool(num_threads)` from JS before using this function!
 *
 * Parameters:
 * - a: Float32Array, row-major, shape [m, k]
 * - b: Float32Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float32Array of shape [m, n]
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulF32Pthreadpool(a, b, m, n, k) {
    const ret = wasm.matmulF32Pthreadpool(a, b, m, n, k);
    return ret;
}

/**
 * Parallel matmul, ZERO JS↔WASM copies.
 *
 * A, B, C all live in WASM memory (F32Buffers). B is packed on-the-fly
 * (same behaviour as matmulF32OptimizedParallelV3 but without the
 * Float32Array round-trips). C is overwritten.
 *
 * This is the general API — B can vary call-to-call. For constant B
 * (NN inference), use matmulF32PrepackedZeroCopy which skips the pack.
 * @param {F32Buffer} a
 * @param {F32Buffer} b
 * @param {F32Buffer} c
 * @param {number} m
 * @param {number} n
 * @param {number} k
 */
export function matmulF32ZeroCopy(a, b, c, m, n, k) {
    _assertClass(a, F32Buffer);
    _assertClass(b, F32Buffer);
    _assertClass(c, F32Buffer);
    wasm.matmulF32ZeroCopy(a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr, m, n, k);
}

/**
 * Fast f64 matrix multiplication using WASM SIMD
 *
 * Direct binding to the SIMD-optimized GEMM kernel for f64.
 * Uses f64x2 (2 elements per v128).
 *
 * Parameters:
 * - a: Float64Array, row-major, shape [m, k]
 * - b: Float64Array, row-major, shape [k, n]
 * - m, n, k: matrix dimensions
 *
 * Returns: Float64Array of shape [m, n]
 * @param {Float64Array} a
 * @param {Float64Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float64Array}
 */
export function matmulF64(a, b, m, n, k) {
    const ret = wasm.matmulF64(a, b, m, n, k);
    return ret;
}

/**
 * Use the gemm crate for highly optimized GEMM
 *
 * The gemm crate uses BLIS-style optimizations:
 * - Cache-blocking at L1/L2/L3 levels
 * - Optimized micro-kernels
 * - Smart packing strategies
 *
 * This should be as fast as or faster than our hand-written SIMD kernels.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulGemm(a, b, m, n, k) {
    const ret = wasm.matmulGemm(a, b, m, n, k);
    return ret;
}

/**
 * GEMM with pre-packed B matrix (inference mode).
 *
 * Use packBForGemm to create packedB once, then call this for each matmul.
 * This matches how tfjs/XNNPACK works for inference.
 * Uses parallel execution via futex pool when available.
 * @param {Float32Array} a
 * @param {Float32Array} packed_b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulWithPackedB(a, packed_b, m, n, k) {
    const ret = wasm.matmulWithPackedB(a, packed_b, m, n, k);
    return ret;
}

/**
 * XNNPACK-style matmul with pre-packed B
 *
 * Requires both the original B (for remaining columns) and packed_b (for SIMD panels).
 * This handles arbitrary N, not just multiples of 8.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} packed_b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulXnnpack(a, b, packed_b, m, n, k) {
    const ret = wasm.matmulXnnpack(a, b, packed_b, m, n, k);
    return ret;
}

/**
 * Cache-blocked XNNPACK-style matmul with pre-packed B
 *
 * Combines cache blocking with B-matrix packing for optimal performance.
 * Best for large matrices where both cache blocking and packing help.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} packed_b
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Float32Array}
 */
export function matmulXnnpackBlocked(a, b, packed_b, m, n, k) {
    const ret = wasm.matmulXnnpackBlocked(a, b, packed_b, m, n, k);
    return ret;
}

/**
 * Verify correctness: compute max absolute difference between two f32 arrays
 *
 * Returns the maximum |a[i] - b[i]| across all elements.
 * Use this to verify that different kernels produce the same results.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function maxAbsDiff(a, b) {
    const ret = wasm.maxAbsDiff(a, b);
    return ret;
}

/**
 * Element-wise maximum of two arrays
 *
 * Compares two arrays element-by-element and returns the maximum values.
 * Equivalent to numpy.maximum(a, b).
 * Supports broadcasting.
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function maximum(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.maximum(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Element-wise maximum with a scalar
 * @param {NDArray} arr
 * @param {number} scalar
 * @returns {NDArray}
 */
export function maximumScalar(arr, scalar) {
    _assertClass(arr, NDArray);
    const ret = wasm.maximumScalar(arr.__wbg_ptr, scalar);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise minimum of two arrays
 *
 * Compares two arrays element-by-element and returns the minimum values.
 * Equivalent to numpy.minimum(a, b).
 * Supports broadcasting.
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function minimum(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.minimum(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Element-wise minimum with a scalar
 * @param {NDArray} arr
 * @param {number} scalar
 * @returns {NDArray}
 */
export function minimumScalar(arr, scalar) {
    _assertClass(arr, NDArray);
    const ret = wasm.minimumScalar(arr.__wbg_ptr, scalar);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function negArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.negArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function ones(shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.ones(ptr0, len0);
    return NDArray.__wrap(ret);
}

/**
 * XNNPACK-style f32 GEMM with pre-packed B matrix (LEGACY, single-threaded)
 *
 * This is a two-phase API:
 * 1. Call `packB` once to convert B into XNNPACK format
 * 2. Call `matmulXnnpack` multiple times with different A matrices
 *
 * This amortizes the packing cost over many matmuls, which is how XNNPACK works.
 * For PARALLEL matmul with pre-packed B, use `packBFull` + `matmulF32Prepacked`.
 * @param {Float32Array} b
 * @param {number} k
 * @param {number} n
 * @returns {Float32Array}
 */
export function packB(b, k, n) {
    const ret = wasm.packB(b, k, n);
    return ret;
}

/**
 * Pre-pack B matrix for repeated matmuls with the same weights.
 *
 * This amortizes packing cost across multiple matmuls (like tfjs inference mode).
 * Returns a Float32Array containing the packed B data.
 *
 * Example:
 * ```javascript
 * const packedB = packBForGemm(weights, k, n);
 * // Later, for each input:
 * const result = matmulWithPackedB(input, packedB, m, n, k);
 * ```
 * @param {Float32Array} b
 * @param {number} k
 * @param {number} n
 * @returns {Float32Array}
 */
export function packBForGemm(b, k, n) {
    const ret = wasm.packBForGemm(b, k, n);
    return ret;
}

/**
 * Pack ALL of B into panel-major layout (for use with matmulF32Prepacked).
 *
 * Unlike `packB` (which truncates at N/8×8), this handles arbitrary N
 * by zero-padding the last panel to NR=8 width. The output size is
 * ceil(N/8) × K × 8 floats.
 *
 * Call this ONCE for weight matrices that will be reused across many
 * matmuls (NN inference). The pack cost is O(K×N) = one pass through B;
 * tf.js/XNNPACK do exactly this at model-load time.
 * Pack B matrix for repeated matmuls.
 * Note: Prepacking optimization was removed. This now just returns a copy.
 * @param {Float32Array} b
 * @param {number} k
 * @param {number} n
 * @returns {Float32Array}
 */
export function packBFull(b, k, n) {
    const ret = wasm.packBFull(b, k, n);
    return ret;
}

/**
 * Pack B (in an F32Buffer) into panel-major layout (in another F32Buffer).
 *
 * Call once per weight matrix; reuse packedB across many matmuls.
 * Both buffers must already be allocated to the right sizes (B: K×N,
 * packedB: packedBSize(K, N)).
 *
 * Note: Currently just copies B to packed_b. The specialized packing was
 * removed during a refactor. The matmul still works (just re-packs internally).
 * @param {F32Buffer} b
 * @param {F32Buffer} packed_b
 * @param {number} k
 * @param {number} n
 */
export function packBInPlace(b, packed_b, k, n) {
    _assertClass(b, F32Buffer);
    _assertClass(packed_b, F32Buffer);
    wasm.packBInPlace(b.__wbg_ptr, packed_b.__wbg_ptr, k, n);
}

/**
 * Size (in f32 elements) of a fully-packed B buffer for matmulF32PrepackedZeroCopy.
 *
 * = ceil(N/8) × K × 8.  For N divisible by 8 (most cases), equals K × N.
 * @param {number} k
 * @param {number} n
 * @returns {number}
 */
export function packedBSize(k, n) {
    const ret = wasm.packedBSize(k, n);
    return ret >>> 0;
}

/**
 * DEBUG: probe whether rayon workers are actually executing in parallel.
 *
 * Spawns N tasks, each recording its rayon thread index and spinning for
 * ~duration_ms. If workers are live, wall-clock ≈ duration_ms (parallel).
 * If all tasks run on the main thread, wall-clock ≈ N × duration_ms.
 *
 * Returns [wall_ms, n_distinct_thread_ids, max_thread_id_seen].
 * @param {number} n_tasks
 * @param {number} duration_ms
 * @returns {Float64Array}
 */
export function probeRayonParallelism(n_tasks, duration_ms) {
    const ret = wasm.probeRayonParallelism(n_tasks, duration_ms);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * DEBUG: mimic v3's dispatch pattern (rayon::scope + inline caller +
 * atomic tile counter) and record which rayon thread claims each tile.
 *
 * Returns a flat array of [tile_idx, rayon_thread_idx, tid_param] triples
 * so we can see if all tiles were claimed by one thread (rayon dispatch
 * bug) or spread across threads (parallelism works, perf bug is elsewhere).
 * @param {number} n_tiles
 * @param {number} work_ms_per_tile
 * @returns {Float64Array}
 */
export function probeV3Dispatch(n_tiles, work_ms_per_tile) {
    const ret = wasm.probeV3Dispatch(n_tiles, work_ms_per_tile);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * DEBUG: report which code path v3 would take for given (m,n,k).
 * Returns: [below_threshold, pack_a, c_pad, fast_path, slab_rows, total_tiles, tz(k*4), tz(n*4)]
 * @param {number} m
 * @param {number} n
 * @param {number} k
 * @returns {Uint32Array}
 */
export function probeV3Path(m, n, k) {
    const ret = wasm.probeV3Path(m, n, k);
    var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * @param {number} loc
 * @param {number} scale
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function randomNormal(loc, scale, shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.randomNormal(loc, scale, ptr0, len0);
    return NDArray.__wrap(ret);
}

/**
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function randomRand(shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.randomRand(ptr0, len0);
    return NDArray.__wrap(ret);
}

/**
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function randomRandn(shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.randomRandn(ptr0, len0);
    return NDArray.__wrap(ret);
}

/**
 * @param {bigint} seed
 */
export function randomSeed(seed) {
    wasm.randomSeed(seed);
}

/**
 * @param {number} low
 * @param {number} high
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function randomUniform(low, high, shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.randomUniform(low, high, ptr0, len0);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function roundArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.roundArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise sign function
 *
 * Returns -1 for negative, 0 for zero, 1 for positive values.
 * Equivalent to numpy.sign(x).
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function signArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.signArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function sinArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.sinArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function sinhArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.sinhArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function solve(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.solve(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function sqrtArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.sqrtArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Element-wise square (x^2)
 *
 * Computes x * x for each element.
 * Equivalent to numpy.square(x).
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function squareArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.squareArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Stack two arrays along a new axis
 * @param {NDArray} a
 * @param {NDArray} b
 * @param {number} axis
 * @returns {NDArray}
 */
export function stack2(a, b, axis) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.stack2(a.__wbg_ptr, b.__wbg_ptr, axis);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Stack three arrays along a new axis
 * @param {NDArray} a
 * @param {NDArray} b
 * @param {NDArray} c
 * @param {number} axis
 * @returns {NDArray}
 */
export function stack3(a, b, c, axis) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    _assertClass(c, NDArray);
    const ret = wasm.stack3(a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr, axis);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function tanArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.tanArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * @param {NDArray} arr
 * @returns {NDArray}
 */
export function tanhArr(arr) {
    _assertClass(arr, NDArray);
    const ret = wasm.tanhArr(arr.__wbg_ptr);
    return NDArray.__wrap(ret);
}

/**
 * Vertical stack (concatenate along axis 0)
 * @param {NDArray} a
 * @param {NDArray} b
 * @returns {NDArray}
 */
export function vstack2(a, b) {
    _assertClass(a, NDArray);
    _assertClass(b, NDArray);
    const ret = wasm.vstack2(a.__wbg_ptr, b.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * Get WASM linear memory for zero-copy access
 *
 * Returns the WebAssembly.Memory object that backs all arrays.
 * Use with `dataPtr()` and `len()` to create zero-copy TypedArray views:
 *
 * ```javascript
 * const wasmMemory = rumpy.wasmMemory();
 * const ptr = array.dataPtr();
 * const len = array.len();
 * const view = new Float64Array(wasmMemory.buffer, ptr, len);
 * // view is now a zero-copy view into the array's data
 * ```
 *
 * Note: Views are invalidated if WASM memory grows. Monitor memory size
 * or recreate views after operations that might allocate.
 * @returns {any}
 */
export function wasmMemory() {
    const ret = wasm.wasmMemory();
    return ret;
}

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}

/**
 * Numpy-style where: select x where condition is true, else y
 *
 * condition, x, y must have compatible shapes (broadcasting supported).
 * Returns x where condition != 0, else y.
 * @param {NDArray} condition
 * @param {NDArray} x
 * @param {NDArray} y
 * @returns {NDArray}
 */
export function where_(condition, x, y) {
    _assertClass(condition, NDArray);
    _assertClass(x, NDArray);
    _assertClass(y, NDArray);
    const ret = wasm.where_(condition.__wbg_ptr, x.__wbg_ptr, y.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return NDArray.__wrap(ret[0]);
}

/**
 * @param {Uint32Array} shape
 * @returns {NDArray}
 */
export function zeros(shape) {
    const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.zeros(ptr0, len0);
    return NDArray.__wrap(ret);
}

function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_undefined_c18285b9fc34cb7d: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_f1258f0b3cab52b2: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_module_39ff3d28752148a9: function() {
            const ret = wasmModule;
            return ret;
        },
        __wbg___wbindgen_throw_39bc967c0e5a9b58: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_b2176991427f6db8: function() { return handleError(function (arg0) {
            globalThis.crypto.getRandomValues(arg0);
        }, arguments); },
        __wbg_has_14f08fae2dc367dc: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.has(arg0, arg1);
            return ret;
        }, arguments); },
        __wbg_instanceof_Window_4aba49e4d1a12365: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_326999dcd07f2163: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_5855c1f289dfffc1: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_c2e7f800270db256: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_ndarray_new: function(arg0) {
            const ret = NDArray.__wrap(arg0);
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_cbee8c0d5c479eac: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_b1617cc9f69683c5: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_e21686f285806d67: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_c8449d782396d344: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_now_edd718b3004d8631: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_prototypesetcall_0d860ddc26c33f4b: function(arg0, arg1, arg2) {
            Float64Array.prototype.set.call(getArrayF64FromWasm0(arg0, arg1), arg2);
        },
        __wbg_prototypesetcall_75794f1851d5d9c5: function(arg0, arg1, arg2) {
            Float32Array.prototype.set.call(getArrayF32FromWasm0(arg0, arg1), arg2);
        },
        __wbg_prototypesetcall_f034d444741426c3: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_push_a6f9488ffd3fae3b: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_random_2b7bed8995d680fb: function() {
            const ret = Math.random();
            return ret;
        },
        __wbg_set_15686f5d9ca06dee: function(arg0, arg1, arg2) {
            arg0.set(getArrayF32FromWasm0(arg1, arg2));
        },
        __wbg_slice_be0f51f7e6b41197: function(arg0, arg1, arg2) {
            const ret = arg0.slice(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_startWorkers_8b582d57e92bd2d4: function(arg0, arg1, arg2) {
            const ret = startWorkers(arg0, arg1, wbg_rayon_PoolBuilder.__wrap(arg2));
            return ret;
        },
        __wbg_static_accessor_GLOBAL_THIS_14325d8cca34bb77: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_f3a1e69f9c5a7e8e: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_50cdb5b517789aca: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_d6c4126e4c244380: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_082d802304b82ac3: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_subarray_7ad5f01d4a9c1c4d: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
        memory: memory || new WebAssembly.Memory({initial:18,maximum:16384,shared:true}),
    };
    return {
        __proto__: null,
        "./rumpy_wasm_bg.js": import0,
    };
}

const F32BufferFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_f32buffer_free(ptr >>> 0, 1));
const NDArrayFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ndarray_free(ptr >>> 0, 1));
const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
    if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
    }
    return cachedBigUint64ArrayMemory0;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getBigUint64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasm = instance.exports;
    wasmModule = module;
    cachedBigUint64ArrayMemory0 = null;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('rumpy_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };
