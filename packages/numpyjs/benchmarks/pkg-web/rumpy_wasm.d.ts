/* tslint:disable */
/* eslint-disable */

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
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Copy data FROM a JS Float32Array INTO this buffer.
     * Useful for the first fill if you can't construct data directly into
     * a zero-copy view (e.g. data comes from a WebGL readback).
     */
    copyFrom(src: Float32Array): void;
    /**
     * Copy data FROM this buffer TO a JS Float32Array.
     * For the zero-copy path you don't need this — construct a view
     * instead. This exists for cases where the result needs to go to a
     * non-shared ArrayBuffer (e.g. postMessage to a context without SAB).
     */
    copyTo(dst: Float32Array): void;
    /**
     * Explicitly free this buffer's memory. The handle is consumed.
     */
    free(): void;
    len(): number;
    /**
     * Byte offset into WASM linear memory where this buffer's data starts.
     *
     * Use with `wasmMemory().buffer` to construct a zero-copy view:
     *   new Float32Array(wasmMemory().buffer, buf.ptr(), buf.len())
     */
    ptr(): number;
}

/**
 * N-dimensional array type exposed to JavaScript
 */
export class NDArray {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    add(other: NDArray): NDArray;
    addScalar(scalar: number): NDArray;
    /**
     * Test if all elements are true (non-zero)
     *
     * Returns 1.0 if all elements are non-zero, 0.0 otherwise.
     * Equivalent to numpy.all(arr).
     */
    all(): number;
    /**
     * Test if all elements are true along an axis
     */
    allAxis(axis: number, keepdims?: boolean | null): NDArray;
    /**
     * Test if any element is true (non-zero)
     *
     * Returns 1.0 if any element is non-zero, 0.0 otherwise.
     * Equivalent to numpy.any(arr).
     */
    any(): number;
    /**
     * Test if any element is true along an axis
     */
    anyAxis(axis: number, keepdims?: boolean | null): NDArray;
    /**
     * Argmax - index of maximum value (flattened)
     */
    argmax(): number;
    /**
     * Argmin - index of minimum value (flattened)
     */
    argmin(): number;
    asType(dtype: string): NDArray;
    /**
     * Average pooling 2D
     *
     * Input shape: (N, C, H, W)
     * Output shape: (N, C, H_out, W_out)
     */
    avgPool2d(kernel_h: number, kernel_w: number, stride_h: number, stride_w: number, pad_h: number, pad_w: number): NDArray;
    batchNorm(gamma: NDArray | null | undefined, beta: NDArray | null | undefined, running_mean: NDArray, running_var: NDArray, eps: number): NDArray;
    broadcastTo(shape: Uint32Array): NDArray;
    chunk(chunk_size: number, axis: number): Array<any>;
    /**
     * Clip values to a range
     */
    clip(min: number, max: number): NDArray;
    /**
     * Clone the array
     */
    clone(): NDArray;
    /**
     * Count non-zero elements
     */
    countNonzero(): number;
    cumprod(axis: number): NDArray;
    cumsum(axis: number): NDArray;
    /**
     * Get pointer to the underlying data buffer
     *
     * Returns the byte offset into WASM linear memory where this array's data begins.
     * Use with `memory()` to create a zero-copy TypedArray view.
     *
     * WARNING: The pointer is only valid while this NDArray exists and WASM memory
     * hasn't been resized. Cache invalidation is the caller's responsibility.
     */
    dataPtr(): number;
    dequantizeLinear(scale: number, zero_point: number): NDArray;
    diag(k: number): NDArray;
    /**
     * Extract diagonal from 2D array (alias for diag with k=0)
     */
    diagonal(offset?: number | null): NDArray;
    diff(n: number, axis: number): NDArray;
    div(other: NDArray): NDArray;
    divScalar(scalar: number): NDArray;
    /**
     * Comparison: equal (element-wise)
     */
    eq(other: NDArray): NDArray;
    /**
     * Scalar comparison: equal
     */
    eqScalar(scalar: number): NDArray;
    /**
     * Expand dims - add axis of length 1
     */
    expandDims(axis: number): NDArray;
    flatten(): NDArray;
    flip(axis: number): NDArray;
    /**
     * Flip array horizontally (along axis 1)
     *
     * Equivalent to numpy.fliplr(arr) or flip(arr, 1).
     * Requires at least 2D array.
     */
    fliplr(): NDArray;
    /**
     * Flip array vertically (along axis 0)
     *
     * Equivalent to numpy.flipud(arr) or flip(arr, 0).
     */
    flipud(): NDArray;
    /**
     * Explicitly free the array memory
     *
     * After calling this, the NDArray is consumed and cannot be used.
     * This is useful for deterministic memory cleanup without waiting for GC.
     */
    free(): void;
    gather(indices: NDArray, axis: number): NDArray;
    /**
     * Comparison: greater than or equal (element-wise)
     */
    ge(other: NDArray): NDArray;
    /**
     * Scalar comparison: greater than or equal
     */
    geScalar(scalar: number): NDArray;
    /**
     * GELU activation (Gaussian Error Linear Unit)
     * Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    gelu(): NDArray;
    /**
     * Get elements where mask is non-zero (truthy)
     *
     * Returns a 1D array of selected elements.
     * Mask must be same shape as self, or broadcastable.
     *
     * Example: arr.getByMask(arr.gt_scalar(0.5)) returns all elements > 0.5
     */
    getByMask(mask: NDArray): NDArray;
    /**
     * Get element at flat index
     */
    getFlat(index: number): number;
    /**
     * Comparison: greater than (element-wise)
     */
    gt(other: NDArray): NDArray;
    /**
     * Scalar comparison: greater than
     */
    gtScalar(scalar: number): NDArray;
    /**
     * im2col: Convert image patches to columns for convolution via GEMM
     *
     * Input shape: (N, C, H, W) - batch, channels, height, width
     * Output shape: (N * H_out * W_out, C * kernel_h * kernel_w)
     *
     * This transforms the convolution operation into a matrix multiplication:
     *   output = im2col(input) @ weights.reshape(out_channels, -1).T
     */
    im2col(kernel_h: number, kernel_w: number, stride_h: number, stride_w: number, pad_h: number, pad_w: number): NDArray;
    indexCopy(axis: number, indices: NDArray, src: NDArray): NDArray;
    interpolate(size: Uint32Array): NDArray;
    /**
     * Check if array is empty
     */
    isEmpty(): boolean;
    /**
     * Check for finite values (not NaN, not Inf)
     */
    isFinite(): NDArray;
    /**
     * Check for infinite values
     */
    isInf(): NDArray;
    /**
     * Check for NaN values
     */
    isNan(): NDArray;
    layerNorm(normalized_shape: Uint32Array, gamma: NDArray | null | undefined, beta: NDArray | null | undefined, eps: number): NDArray;
    /**
     * Comparison: less than or equal (element-wise)
     */
    le(other: NDArray): NDArray;
    /**
     * Scalar comparison: less than or equal
     */
    leScalar(scalar: number): NDArray;
    /**
     * Get the number of elements in the array
     */
    len(): number;
    logSoftmax(axis: number): NDArray;
    logsumexp(axis: number, keepdims?: boolean | null): NDArray;
    /**
     * Comparison: less than (element-wise)
     */
    lt(other: NDArray): NDArray;
    /**
     * Scalar comparison: less than
     */
    ltScalar(scalar: number): NDArray;
    max(): number;
    /**
     * Max along an axis
     */
    maxAxis(axis: number, keepdims?: boolean | null): NDArray;
    /**
     * Max pooling 2D
     *
     * Input shape: (N, C, H, W)
     * Output shape: (N, C, H_out, W_out)
     */
    maxPool2d(kernel_h: number, kernel_w: number, stride_h: number, stride_w: number, pad_h: number, pad_w: number): NDArray;
    mean(): number;
    /**
     * Mean along an axis
     */
    meanAxis(axis: number, keepdims?: boolean | null): NDArray;
    min(): number;
    /**
     * Min along an axis
     */
    minAxis(axis: number, keepdims?: boolean | null): NDArray;
    mul(other: NDArray): NDArray;
    mulScalar(scalar: number): NDArray;
    multinomial(num_samples: number, replacement: boolean): NDArray;
    /**
     * Get total size in bytes
     */
    nbytes(): number;
    /**
     * Comparison: not equal (element-wise)
     */
    ne(other: NDArray): NDArray;
    /**
     * Scalar comparison: not equal
     */
    neScalar(scalar: number): NDArray;
    /**
     * Get indices of non-zero elements (flat indices)
     */
    nonzeroFlat(): Uint32Array;
    pad(pad_width: Uint32Array, constant_value: number): NDArray;
    /**
     * Permute array dimensions
     * axes specifies the new order of dimensions
     * e.g., permute([1, 0, 2]) swaps first two dimensions
     */
    permute(axes: Uint32Array): NDArray;
    powScalar(scalar: number): NDArray;
    prod(): number;
    /**
     * ReLU activation: max(0, x)
     * NaN values are propagated (NumPy behavior)
     */
    relu(): NDArray;
    repeat(repeats: number, axis?: number | null): NDArray;
    reshape(shape: Uint32Array): NDArray;
    /**
     * Reshape with support for -1 dimension inference (NumPy-style)
     * One dimension can be -1, which will be inferred from the total size
     */
    reshapeInfer(shape: BigInt64Array): NDArray;
    /**
     * Alias for interpolate
     */
    resize(size: Uint32Array): NDArray;
    rmsNorm(gamma: NDArray, eps: number): NDArray;
    roll(shift: number, axis: number): NDArray;
    rsqrt(): NDArray;
    scatter(axis: number, indices: NDArray, src: NDArray): NDArray;
    /**
     * Set elements where mask is non-zero to a scalar value
     *
     * Returns a new array with selected elements replaced.
     */
    setByMask(mask: NDArray, value: number): NDArray;
    sigmoid(): NDArray;
    silu(): NDArray;
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
     */
    slice(starts: Int32Array, stops: Int32Array, steps: Int32Array): NDArray;
    /**
     * Slice along a single axis (simpler API for common case)
     *
     * Equivalent to arr[:, :, start:stop] when axis=2
     */
    sliceAxis(axis: number, start: number, stop: number): NDArray;
    /**
     * Softmax along an axis
     * softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
     */
    softmax(axis: number): NDArray;
    split(num_splits: number, axis: number): Array<any>;
    /**
     * Squeeze - remove axes of length 1
     */
    squeeze(): NDArray;
    std(): number;
    /**
     * Standard deviation with degrees of freedom adjustment
     * ddof=0 for population std (default), ddof=1 for sample std
     */
    stdDdof(ddof: number): number;
    sub(other: NDArray): NDArray;
    subScalar(scalar: number): NDArray;
    sum(): number;
    /**
     * Sum along an axis
     */
    sumAxis(axis: number, keepdims?: boolean | null): NDArray;
    take(indices: NDArray, axis: number): NDArray;
    tile(reps: Uint32Array): NDArray;
    /**
     * Convert to Float64Array (creates a copy)
     *
     * This method always works but involves copying data from WASM memory to JS.
     * For zero-copy access when SharedArrayBuffer is available, use `asTypedArrayView()`.
     */
    toTypedArray(): Float64Array;
    topk(k: number, axis: number, sorted: boolean): Array<any>;
    transpose(): NDArray;
    tril(k: number): NDArray;
    triu(k: number): NDArray;
    var(): number;
    /**
     * Variance with degrees of freedom adjustment
     * ddof=0 for population variance (default), ddof=1 for sample variance
     */
    varDdof(ddof: number): number;
    /**
     * Data type
     */
    readonly dtype: string;
    /**
     * Number of dimensions
     */
    readonly ndim: number;
    /**
     * Get array shape
     */
    readonly shape: Uint32Array;
    /**
     * Total number of elements
     */
    readonly size: number;
}

export function absArr(arr: NDArray): NDArray;

/**
 * Allocate an f32 buffer of the given length inside WASM memory.
 * Contents are uninitialised — write before reading.
 */
export function allocF32(len: number): F32Buffer;

export function arange(start: number, stop: number, step: number): NDArray;

export function arrayFromTyped(data: Float64Array, shape: Uint32Array): NDArray;

/**
 * Element-wise atan2(y, x)
 *
 * Computes the arc tangent of y/x, using the signs of both arguments
 * to determine the quadrant of the return value.
 * Equivalent to numpy.arctan2(y, x).
 */
export function atan2Arr(y: NDArray, x: NDArray): NDArray;

/**
 * Create a causal attention mask (lower triangular matrix of -inf and 0)
 *
 * Returns a mask where positions that can attend are 0, others are -inf.
 * Useful for transformer causal (autoregressive) attention.
 *
 * # Arguments
 * * `size` - Sequence length (creates size x size mask)
 */
export function causalMask(size: number): NDArray;

export function ceilArr(arr: NDArray): NDArray;

/**
 * Compute checksum (sum of all elements) for verification
 */
export function checksum(a: Float32Array): number;

/**
 * Concatenate two arrays along an axis
 */
export function concatenate2(a: NDArray, b: NDArray, axis: number): NDArray;

/**
 * Concatenate three arrays along an axis
 */
export function concatenate3(a: NDArray, b: NDArray, c: NDArray, axis: number): NDArray;

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
 */
export function conv2d(input: NDArray, kernel: NDArray, stride: Uint32Array, padding: Uint32Array): NDArray;

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
 */
export function convTranspose2d(input: NDArray, kernel: NDArray, stride: Uint32Array, padding: Uint32Array, output_padding: Uint32Array): NDArray;

export function cosArr(arr: NDArray): NDArray;

export function coshArr(arr: NDArray): NDArray;

export function det(arr: NDArray): number;

export function dot(a: NDArray, b: NDArray): NDArray;

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
 */
export function einsum(subscripts: string, a: NDArray, b?: NDArray | null): NDArray;

export function expArr(arr: NDArray): NDArray;

/**
 * Element-wise exp(x) - 1
 *
 * Computes exp(x) - 1 with better precision for small x.
 * Equivalent to numpy.expm1(x).
 */
export function expm1Arr(arr: NDArray): NDArray;

export function eye(n: number): NDArray;

export function floorArr(arr: NDArray): NDArray;

export function full(shape: Uint32Array, value: number): NDArray;

/**
 * Get the current number of rayon threads
 */
export function getNumThreads(): number;

/**
 * Check if SharedArrayBuffer is available
 *
 * Returns true if the environment supports SharedArrayBuffer (COOP/COEP headers set).
 * When false, `asTypedArrayView()` will not work and you should use `toTypedArray()`.
 */
export function hasSharedArrayBuffer(): boolean;

/**
 * Horizontal stack (concatenate along axis 1 for 2D+, axis 0 for 1D)
 */
export function hstack2(a: NDArray, b: NDArray): NDArray;

/**
 * Initialize panic hook for better error messages
 */
export function init(): void;

export function initThreadPool(num_threads: number): Promise<any>;

export function inv(arr: NDArray): NDArray;

export function linspace(start: number, stop: number, num: number): NDArray;

/**
 * Element-wise log(1 + x)
 *
 * Computes log(1 + x) with better precision for small x.
 * Equivalent to numpy.log1p(x).
 */
export function log1pArr(arr: NDArray): NDArray;

export function logArr(arr: NDArray): NDArray;

/**
 * Element-wise logical AND
 *
 * Returns 1.0 where both inputs are non-zero, 0.0 otherwise.
 * Equivalent to numpy.logical_and(a, b).
 */
export function logicalAnd(a: NDArray, b: NDArray): NDArray;

/**
 * Element-wise logical NOT
 *
 * Returns 1.0 where input is zero, 0.0 otherwise.
 * Equivalent to numpy.logical_not(x).
 */
export function logicalNot(arr: NDArray): NDArray;

/**
 * Element-wise logical OR
 *
 * Returns 1.0 where either input is non-zero, 0.0 otherwise.
 * Equivalent to numpy.logical_or(a, b).
 */
export function logicalOr(a: NDArray, b: NDArray): NDArray;

export function matmul(a: NDArray, b: NDArray): NDArray;

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
 */
export function matmulF32(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * 5x8 kernel specifically for matrices where M is divisible by 5
 *
 * Optimized for 100x100 case (and similar).
 */
export function matmulF325x8(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32Auto(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32Blocked(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32FMA(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32FMAPacked(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Highly optimized 6x8 GEMM with FMA, loadsplat, and cache blocking
 *
 * This is the most optimized implementation, matching XNNPACK patterns:
 * - 6x8 micro-kernel (12 accumulators fit in 16 XMM registers)
 * - f32x4_relaxed_madd for FMA
 * - v128_load32_splat for A broadcast
 * - L1/L2 cache blocking (KC=256, MC=72, NC=128)
 * - B matrix packing for contiguous access
 */
export function matmulF32Optimized(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Parallel version of optimized 6x8 GEMM using rayon (LEGACY)
 *
 * Kept for A/B benchmarking. Has known problems — see v3 below.
 */
export function matmulF32OptimizedParallel(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32OptimizedParallelV3(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32OptimizedParallelV4(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32Packed(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32Parallel(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

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
 */
export function matmulF32ParallelV2(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Parallel matmul with pre-packed B (from packBFull).
 *
 * Note: Prepacking optimization was removed. This now calls the regular
 * parallel matmul (packed_b is treated as normal B).
 */
export function matmulF32Prepacked(a: Float32Array, packed_b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Parallel matmul with pre-packed B, ZERO JS↔WASM copies.
 *
 * The leanest call path: A and packed-B already in WASM memory, C
 * written directly, no per-call packing. This is the tf.js-equivalent
 * path for NN inference.
 *
 * Note: The specialized prepacked kernel was removed during a refactor.
 * This now just calls the regular parallel matmul (packed_b is treated as B).
 */
export function matmulF32PrepackedZeroCopy(a: F32Buffer, packed_b: F32Buffer, c: F32Buffer, m: number, n: number, k: number): void;

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
 */
export function matmulF32Pthreadpool(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Parallel matmul, ZERO JS↔WASM copies.
 *
 * A, B, C all live in WASM memory (F32Buffers). B is packed on-the-fly
 * (same behaviour as matmulF32OptimizedParallelV3 but without the
 * Float32Array round-trips). C is overwritten.
 *
 * This is the general API — B can vary call-to-call. For constant B
 * (NN inference), use matmulF32PrepackedZeroCopy which skips the pack.
 */
export function matmulF32ZeroCopy(a: F32Buffer, b: F32Buffer, c: F32Buffer, m: number, n: number, k: number): void;

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
 */
export function matmulF64(a: Float64Array, b: Float64Array, m: number, n: number, k: number): Float64Array;

/**
 * Use the gemm crate for highly optimized GEMM
 *
 * The gemm crate uses BLIS-style optimizations:
 * - Cache-blocking at L1/L2/L3 levels
 * - Optimized micro-kernels
 * - Smart packing strategies
 *
 * This should be as fast as or faster than our hand-written SIMD kernels.
 */
export function matmulGemm(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * GEMM with pre-packed B matrix (inference mode).
 *
 * Use packBForGemm to create packedB once, then call this for each matmul.
 * This matches how tfjs/XNNPACK works for inference.
 * Uses parallel execution via futex pool when available.
 */
export function matmulWithPackedB(a: Float32Array, packed_b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * XNNPACK-style matmul with pre-packed B
 *
 * Requires both the original B (for remaining columns) and packed_b (for SIMD panels).
 * This handles arbitrary N, not just multiples of 8.
 */
export function matmulXnnpack(a: Float32Array, b: Float32Array, packed_b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Cache-blocked XNNPACK-style matmul with pre-packed B
 *
 * Combines cache blocking with B-matrix packing for optimal performance.
 * Best for large matrices where both cache blocking and packing help.
 */
export function matmulXnnpackBlocked(a: Float32Array, b: Float32Array, packed_b: Float32Array, m: number, n: number, k: number): Float32Array;

/**
 * Verify correctness: compute max absolute difference between two f32 arrays
 *
 * Returns the maximum |a[i] - b[i]| across all elements.
 * Use this to verify that different kernels produce the same results.
 */
export function maxAbsDiff(a: Float32Array, b: Float32Array): number;

/**
 * Element-wise maximum of two arrays
 *
 * Compares two arrays element-by-element and returns the maximum values.
 * Equivalent to numpy.maximum(a, b).
 * Supports broadcasting.
 */
export function maximum(a: NDArray, b: NDArray): NDArray;

/**
 * Element-wise maximum with a scalar
 */
export function maximumScalar(arr: NDArray, scalar: number): NDArray;

/**
 * Element-wise minimum of two arrays
 *
 * Compares two arrays element-by-element and returns the minimum values.
 * Equivalent to numpy.minimum(a, b).
 * Supports broadcasting.
 */
export function minimum(a: NDArray, b: NDArray): NDArray;

/**
 * Element-wise minimum with a scalar
 */
export function minimumScalar(arr: NDArray, scalar: number): NDArray;

export function negArr(arr: NDArray): NDArray;

export function ones(shape: Uint32Array): NDArray;

/**
 * XNNPACK-style f32 GEMM with pre-packed B matrix (LEGACY, single-threaded)
 *
 * This is a two-phase API:
 * 1. Call `packB` once to convert B into XNNPACK format
 * 2. Call `matmulXnnpack` multiple times with different A matrices
 *
 * This amortizes the packing cost over many matmuls, which is how XNNPACK works.
 * For PARALLEL matmul with pre-packed B, use `packBFull` + `matmulF32Prepacked`.
 */
export function packB(b: Float32Array, k: number, n: number): Float32Array;

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
 */
export function packBForGemm(b: Float32Array, k: number, n: number): Float32Array;

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
 */
export function packBFull(b: Float32Array, k: number, n: number): Float32Array;

/**
 * Pack B (in an F32Buffer) into panel-major layout (in another F32Buffer).
 *
 * Call once per weight matrix; reuse packedB across many matmuls.
 * Both buffers must already be allocated to the right sizes (B: K×N,
 * packedB: packedBSize(K, N)).
 *
 * Note: Currently just copies B to packed_b. The specialized packing was
 * removed during a refactor. The matmul still works (just re-packs internally).
 */
export function packBInPlace(b: F32Buffer, packed_b: F32Buffer, k: number, n: number): void;

/**
 * Size (in f32 elements) of a fully-packed B buffer for matmulF32PrepackedZeroCopy.
 *
 * = ceil(N/8) × K × 8.  For N divisible by 8 (most cases), equals K × N.
 */
export function packedBSize(k: number, n: number): number;

/**
 * DEBUG: probe whether rayon workers are actually executing in parallel.
 *
 * Spawns N tasks, each recording its rayon thread index and spinning for
 * ~duration_ms. If workers are live, wall-clock ≈ duration_ms (parallel).
 * If all tasks run on the main thread, wall-clock ≈ N × duration_ms.
 *
 * Returns [wall_ms, n_distinct_thread_ids, max_thread_id_seen].
 */
export function probeRayonParallelism(n_tasks: number, duration_ms: number): Float64Array;

/**
 * DEBUG: mimic v3's dispatch pattern (rayon::scope + inline caller +
 * atomic tile counter) and record which rayon thread claims each tile.
 *
 * Returns a flat array of [tile_idx, rayon_thread_idx, tid_param] triples
 * so we can see if all tiles were claimed by one thread (rayon dispatch
 * bug) or spread across threads (parallelism works, perf bug is elsewhere).
 */
export function probeV3Dispatch(n_tiles: number, work_ms_per_tile: number): Float64Array;

/**
 * DEBUG: report which code path v3 would take for given (m,n,k).
 * Returns: [below_threshold, pack_a, c_pad, fast_path, slab_rows, total_tiles, tz(k*4), tz(n*4)]
 */
export function probeV3Path(m: number, n: number, k: number): Uint32Array;

export function randomNormal(loc: number, scale: number, shape: Uint32Array): NDArray;

export function randomRand(shape: Uint32Array): NDArray;

export function randomRandn(shape: Uint32Array): NDArray;

export function randomSeed(seed: bigint): void;

export function randomUniform(low: number, high: number, shape: Uint32Array): NDArray;

export function roundArr(arr: NDArray): NDArray;

/**
 * Element-wise sign function
 *
 * Returns -1 for negative, 0 for zero, 1 for positive values.
 * Equivalent to numpy.sign(x).
 */
export function signArr(arr: NDArray): NDArray;

export function sinArr(arr: NDArray): NDArray;

export function sinhArr(arr: NDArray): NDArray;

export function solve(a: NDArray, b: NDArray): NDArray;

export function sqrtArr(arr: NDArray): NDArray;

/**
 * Element-wise square (x^2)
 *
 * Computes x * x for each element.
 * Equivalent to numpy.square(x).
 */
export function squareArr(arr: NDArray): NDArray;

/**
 * Stack two arrays along a new axis
 */
export function stack2(a: NDArray, b: NDArray, axis: number): NDArray;

/**
 * Stack three arrays along a new axis
 */
export function stack3(a: NDArray, b: NDArray, c: NDArray, axis: number): NDArray;

export function tanArr(arr: NDArray): NDArray;

export function tanhArr(arr: NDArray): NDArray;

/**
 * Vertical stack (concatenate along axis 0)
 */
export function vstack2(a: NDArray, b: NDArray): NDArray;

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
 */
export function wasmMemory(): any;

export class wbg_rayon_PoolBuilder {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    build(): void;
    numThreads(): number;
    receiver(): number;
}

export function wbg_rayon_start_worker(receiver: number): void;

/**
 * Numpy-style where: select x where condition is true, else y
 *
 * condition, x, y must have compatible shapes (broadcasting supported).
 * Returns x where condition != 0, else y.
 */
export function where_(condition: NDArray, x: NDArray, y: NDArray): NDArray;

export function zeros(shape: Uint32Array): NDArray;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly __wbg_f32buffer_free: (a: number, b: number) => void;
    readonly __wbg_ndarray_free: (a: number, b: number) => void;
    readonly absArr: (a: number) => number;
    readonly allocF32: (a: number) => number;
    readonly arange: (a: number, b: number, c: number) => [number, number, number];
    readonly arrayFromTyped: (a: any, b: number, c: number) => [number, number, number];
    readonly atan2Arr: (a: number, b: number) => [number, number, number];
    readonly causalMask: (a: number) => number;
    readonly ceilArr: (a: number) => number;
    readonly checksum: (a: any) => number;
    readonly concatenate2: (a: number, b: number, c: number) => [number, number, number];
    readonly concatenate3: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly conv2d: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly convTranspose2d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly cosArr: (a: number) => number;
    readonly coshArr: (a: number) => number;
    readonly det: (a: number) => [number, number, number];
    readonly dot: (a: number, b: number) => [number, number, number];
    readonly einsum: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly expArr: (a: number) => number;
    readonly expm1Arr: (a: number) => number;
    readonly eye: (a: number) => number;
    readonly f32buffer_copyFrom: (a: number, b: any) => void;
    readonly f32buffer_copyTo: (a: number, b: any) => void;
    readonly f32buffer_free: (a: number) => void;
    readonly f32buffer_len: (a: number) => number;
    readonly f32buffer_ptr: (a: number) => number;
    readonly floorArr: (a: number) => number;
    readonly full: (a: number, b: number, c: number) => number;
    readonly hasSharedArrayBuffer: () => number;
    readonly hstack2: (a: number, b: number) => [number, number, number];
    readonly inv: (a: number) => [number, number, number];
    readonly linspace: (a: number, b: number, c: number) => number;
    readonly log1pArr: (a: number) => number;
    readonly logArr: (a: number) => number;
    readonly logicalAnd: (a: number, b: number) => [number, number, number];
    readonly logicalNot: (a: number) => number;
    readonly logicalOr: (a: number, b: number) => [number, number, number];
    readonly matmul: (a: number, b: number) => [number, number, number];
    readonly matmulF32: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF325x8: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32Auto: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32Blocked: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32FMA: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32FMAPacked: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32Optimized: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32OptimizedParallel: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32Packed: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32Parallel: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32ParallelV2: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32PrepackedZeroCopy: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly matmulF32Pthreadpool: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32ZeroCopy: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly matmulF64: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulGemm: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulWithPackedB: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulXnnpack: (a: any, b: any, c: any, d: number, e: number, f: number) => any;
    readonly matmulXnnpackBlocked: (a: any, b: any, c: any, d: number, e: number, f: number) => any;
    readonly maxAbsDiff: (a: any, b: any) => number;
    readonly maximum: (a: number, b: number) => [number, number, number];
    readonly maximumScalar: (a: number, b: number) => number;
    readonly minimum: (a: number, b: number) => [number, number, number];
    readonly minimumScalar: (a: number, b: number) => number;
    readonly ndarray_add: (a: number, b: number) => [number, number, number];
    readonly ndarray_addScalar: (a: number, b: number) => number;
    readonly ndarray_all: (a: number) => number;
    readonly ndarray_allAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_any: (a: number) => number;
    readonly ndarray_anyAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_argmax: (a: number) => number;
    readonly ndarray_argmin: (a: number) => number;
    readonly ndarray_asType: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_avgPool2d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly ndarray_batchNorm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly ndarray_broadcastTo: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_chunk: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_clip: (a: number, b: number, c: number) => number;
    readonly ndarray_clone: (a: number) => number;
    readonly ndarray_countNonzero: (a: number) => number;
    readonly ndarray_cumprod: (a: number, b: number) => [number, number, number];
    readonly ndarray_cumsum: (a: number, b: number) => [number, number, number];
    readonly ndarray_dataPtr: (a: number) => number;
    readonly ndarray_dequantizeLinear: (a: number, b: number, c: number) => number;
    readonly ndarray_diag: (a: number, b: number) => [number, number, number];
    readonly ndarray_diagonal: (a: number, b: number) => [number, number, number];
    readonly ndarray_diff: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_div: (a: number, b: number) => [number, number, number];
    readonly ndarray_divScalar: (a: number, b: number) => number;
    readonly ndarray_dtype: (a: number) => [number, number];
    readonly ndarray_eq: (a: number, b: number) => [number, number, number];
    readonly ndarray_eqScalar: (a: number, b: number) => number;
    readonly ndarray_expandDims: (a: number, b: number) => [number, number, number];
    readonly ndarray_flatten: (a: number) => number;
    readonly ndarray_flip: (a: number, b: number) => [number, number, number];
    readonly ndarray_fliplr: (a: number) => [number, number, number];
    readonly ndarray_flipud: (a: number) => [number, number, number];
    readonly ndarray_free: (a: number) => void;
    readonly ndarray_gather: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_ge: (a: number, b: number) => [number, number, number];
    readonly ndarray_geScalar: (a: number, b: number) => number;
    readonly ndarray_gelu: (a: number) => number;
    readonly ndarray_getByMask: (a: number, b: number) => [number, number, number];
    readonly ndarray_getFlat: (a: number, b: number) => number;
    readonly ndarray_gt: (a: number, b: number) => [number, number, number];
    readonly ndarray_gtScalar: (a: number, b: number) => number;
    readonly ndarray_im2col: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly ndarray_indexCopy: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly ndarray_interpolate: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_isEmpty: (a: number) => number;
    readonly ndarray_isFinite: (a: number) => number;
    readonly ndarray_isInf: (a: number) => number;
    readonly ndarray_isNan: (a: number) => number;
    readonly ndarray_layerNorm: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly ndarray_le: (a: number, b: number) => [number, number, number];
    readonly ndarray_leScalar: (a: number, b: number) => number;
    readonly ndarray_len: (a: number) => number;
    readonly ndarray_logSoftmax: (a: number, b: number) => [number, number, number];
    readonly ndarray_logsumexp: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_lt: (a: number, b: number) => [number, number, number];
    readonly ndarray_ltScalar: (a: number, b: number) => number;
    readonly ndarray_max: (a: number) => number;
    readonly ndarray_maxAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_maxPool2d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly ndarray_mean: (a: number) => number;
    readonly ndarray_meanAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_min: (a: number) => number;
    readonly ndarray_minAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_mul: (a: number, b: number) => [number, number, number];
    readonly ndarray_mulScalar: (a: number, b: number) => number;
    readonly ndarray_multinomial: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_nbytes: (a: number) => number;
    readonly ndarray_ndim: (a: number) => number;
    readonly ndarray_ne: (a: number, b: number) => [number, number, number];
    readonly ndarray_neScalar: (a: number, b: number) => number;
    readonly ndarray_nonzeroFlat: (a: number) => [number, number];
    readonly ndarray_pad: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly ndarray_permute: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_powScalar: (a: number, b: number) => number;
    readonly ndarray_prod: (a: number) => number;
    readonly ndarray_relu: (a: number) => number;
    readonly ndarray_repeat: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_reshape: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_reshapeInfer: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_rmsNorm: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_roll: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_rsqrt: (a: number) => number;
    readonly ndarray_scatter: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly ndarray_setByMask: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_shape: (a: number) => [number, number];
    readonly ndarray_sigmoid: (a: number) => number;
    readonly ndarray_silu: (a: number) => number;
    readonly ndarray_slice: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly ndarray_sliceAxis: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly ndarray_softmax: (a: number, b: number) => [number, number, number];
    readonly ndarray_split: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_squeeze: (a: number) => number;
    readonly ndarray_std: (a: number) => number;
    readonly ndarray_stdDdof: (a: number, b: number) => number;
    readonly ndarray_sub: (a: number, b: number) => [number, number, number];
    readonly ndarray_subScalar: (a: number, b: number) => number;
    readonly ndarray_sum: (a: number) => number;
    readonly ndarray_sumAxis: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_tile: (a: number, b: number, c: number) => [number, number, number];
    readonly ndarray_toTypedArray: (a: number) => any;
    readonly ndarray_topk: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly ndarray_transpose: (a: number) => number;
    readonly ndarray_tril: (a: number, b: number) => [number, number, number];
    readonly ndarray_triu: (a: number, b: number) => [number, number, number];
    readonly ndarray_var: (a: number) => number;
    readonly ndarray_varDdof: (a: number, b: number) => number;
    readonly negArr: (a: number) => number;
    readonly ones: (a: number, b: number) => number;
    readonly packB: (a: any, b: number, c: number) => any;
    readonly packBForGemm: (a: any, b: number, c: number) => any;
    readonly packBFull: (a: any, b: number, c: number) => any;
    readonly packBInPlace: (a: number, b: number, c: number, d: number) => void;
    readonly packedBSize: (a: number, b: number) => number;
    readonly probeRayonParallelism: (a: number, b: number) => [number, number];
    readonly probeV3Dispatch: (a: number, b: number) => [number, number];
    readonly probeV3Path: (a: number, b: number, c: number) => [number, number];
    readonly randomNormal: (a: number, b: number, c: number, d: number) => number;
    readonly randomRand: (a: number, b: number) => number;
    readonly randomRandn: (a: number, b: number) => number;
    readonly randomUniform: (a: number, b: number, c: number, d: number) => number;
    readonly roundArr: (a: number) => number;
    readonly signArr: (a: number) => number;
    readonly sinArr: (a: number) => number;
    readonly sinhArr: (a: number) => number;
    readonly solve: (a: number, b: number) => [number, number, number];
    readonly sqrtArr: (a: number) => number;
    readonly squareArr: (a: number) => number;
    readonly stack2: (a: number, b: number, c: number) => [number, number, number];
    readonly stack3: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly tanArr: (a: number) => number;
    readonly tanhArr: (a: number) => number;
    readonly vstack2: (a: number, b: number) => [number, number, number];
    readonly where_: (a: number, b: number, c: number) => [number, number, number];
    readonly zeros: (a: number, b: number) => number;
    readonly init: () => void;
    readonly ndarray_size: (a: number) => number;
    readonly ndarray_take: (a: number, b: number, c: number) => [number, number, number];
    readonly randomSeed: (a: bigint) => void;
    readonly ndarray_resize: (a: number, b: number, c: number) => [number, number, number];
    readonly getNumThreads: () => number;
    readonly wasmMemory: () => any;
    readonly matmulF32Prepacked: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32OptimizedParallelV3: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly matmulF32OptimizedParallelV4: (a: any, b: any, c: number, d: number, e: number) => any;
    readonly __wbg_wbg_rayon_poolbuilder_free: (a: number, b: number) => void;
    readonly initThreadPool: (a: number) => any;
    readonly wbg_rayon_poolbuilder_build: (a: number) => void;
    readonly wbg_rayon_poolbuilder_numThreads: (a: number) => number;
    readonly wbg_rayon_poolbuilder_receiver: (a: number) => number;
    readonly wbg_rayon_start_worker: (a: number) => void;
    readonly memory: WebAssembly.Memory;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_thread_destroy: (a?: number, b?: number, c?: number) => void;
    readonly __wbindgen_start: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number }} module - Passing `SyncInitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput, memory?: WebAssembly.Memory, thread_stack_size?: number } | SyncInitInput, memory?: WebAssembly.Memory): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number }} module_or_path - Passing `InitInput` directly is deprecated.
 * @param {WebAssembly.Memory} memory - Deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput>, memory?: WebAssembly.Memory, thread_stack_size?: number } | InitInput | Promise<InitInput>, memory?: WebAssembly.Memory): Promise<InitOutput>;
