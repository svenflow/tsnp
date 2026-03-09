# tsnp (rumpy-ts) NumPy Parity Audit Report

## Executive Summary

This audit examines the `tsnp` library (formerly rumpy-ts), a NumPy-like TypeScript library with WASM (CPU) and WebGPU backends. The library has a well-structured architecture with Rust traits defining backend interfaces and multiple backend implementations.

**Overall Status:**
- **CPU/WASM Backend:** Comprehensive implementation with ~120+ operations across all categories
- **WebGPU Backend:** Partial implementation focused on elementwise ops and matmul, with CPU fallbacks for complex operations
- **NumPy Parity:** Good coverage of core operations; significant gaps in advanced functionality

---

## 1. Operations Coverage Summary

### 1.1 Implemented Operations by Category

| Category | Ops Defined | CPU Implemented | WebGPU Shaders | WebGPU Async |
|----------|-------------|-----------------|----------------|--------------|
| Creation | 10 | 10 | 0 (CPU fallback) | 0 |
| Math (Unary) | 26 | 26 | 22 | 0 |
| Math (Binary) | 15 | 15 | 7 | 0 |
| Math (Scalar) | 6 | 6 | 5 | 0 |
| Stats | 32 | 32 | 10 | 0 |
| Linalg | 22 | 22 | 6 | 1 (matmul) |
| Manipulation | 16 | 16 | 0 (CPU fallback) | 0 |
| Random | 8 | 8 | 0 (CPU fallback) | 0 |
| Sort/Search | 9 | 9 | 0 (CPU fallback) | 0 |
| Compare | 12 | 12 | 0 (CPU fallback) | 0 |

### 1.2 WebGPU Shader Coverage Detail

**Elementwise Ops with GPU Shaders:**
- Unary: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `round`, `neg`, `reciprocal`, `square`, `cbrt`, `log10`
- Binary: `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`
- Scalar: `addScalar`, `subScalar`, `mulScalar`, `divScalar`, `powScalar`

**Reduction Ops with GPU Shaders:**
- `sum`, `prod`, `min`, `max` (parallel reduction with shared memory)
- `argmin`, `argmax` (value + index tracking)
- `all`, `any` (boolean reductions)
- `cumsum`, `cumprod` (naive serial implementation per thread)
- `sumAxis` (for 2D arrays, axis 0 and 1)

**Linalg Ops with GPU Shaders:**
- `matmul` (multiple optimized implementations: tiled, vec4, tfjs-32)
- `transpose` (2D)
- `outer` product
- `dot` product (1D)
- `trace`
- `norm` (L2)

---

## 2. Missing NumPy Operations (Prioritized)

### 2.1 HIGH Priority - Commonly Used

| NumPy Function | Status | Notes |
|----------------|--------|-------|
| `np.empty()` | MISSING | Array creation without initialization |
| `np.empty_like()` | MISSING | Create uninitialized array with same shape |
| `np.zeros_like()` | MISSING | Create zeros array with same shape |
| `np.ones_like()` | MISSING | Create ones array with same shape |
| `np.asarray()` | PARTIAL | `array()` exists but no dtype conversion |
| `np.astype()` | MISSING | Dtype conversion (currently f64 only) |
| `np.copy()` | MISSING | Explicit copy (clone exists on array) |
| `np.swapaxes()` | MISSING | Swap two axes |
| `np.moveaxis()` | MISSING | Move axis to new position |
| `np.atleast_1d/2d/3d()` | MISSING | Ensure minimum dimensions |
| `np.broadcast_to()` | MISSING | Explicit broadcasting |
| `np.broadcast_arrays()` | MISSING | Broadcast multiple arrays |
| `np.einsum()` | MISSING | Einstein summation (very important for ML) |
| `np.dstack()` | MISSING | Stack along third axis |
| `np.column_stack()` | MISSING | Stack 1D as columns |
| `np.row_stack()` | MISSING | Alias for vstack |
| `np.array_split()` | MISSING | Split into n parts |
| `np.dsplit()`, `np.vsplit()`, `np.hsplit()` | MISSING | Split shortcuts |
| `np.insert()` | MISSING | Insert values |
| `np.delete()` | MISSING | Delete values |
| `np.append()` | MISSING | Append to array |
| `np.resize()` | MISSING | Resize array |
| `np.squeeze()` with negative axis | PARTIAL | Negative axis indexing not supported |
| `np.gradient()` | MISSING | Compute gradient |
| `np.ediff1d()` | MISSING | 1D differences |
| `np.cross()` | MISSING | Cross product |
| `np.cov()` | MISSING | Covariance matrix |
| `np.corrcoef()` | MISSING | Correlation coefficients |
| `np.count_nonzero()` | PARTIAL | In WASM but not in Backend trait |

### 2.2 MEDIUM Priority - Important for ML/Scientific

| NumPy Function | Status | Notes |
|----------------|--------|-------|
| `np.convolve()` | MISSING | 1D convolution |
| `np.correlate()` | MISSING | 1D correlation |
| `np.fft.*` | MISSING | Fast Fourier Transform family |
| `np.linalg.matrix_rank()` | EXISTS | As `rank()` |
| `np.linalg.cond()` | MISSING | Condition number |
| `np.linalg.matrix_power()` | MISSING | Matrix power |
| `np.linalg.slogdet()` | MISSING | Sign and log of determinant |
| `np.linalg.multi_dot()` | MISSING | Efficient multi-matrix multiply |
| `np.kron()` | MISSING | Kronecker product |
| `np.polyfit()` | MISSING | Polynomial fitting |
| `np.polyval()` | MISSING | Polynomial evaluation |
| `np.roots()` | MISSING | Polynomial roots |
| `np.interp()` | MISSING | Linear interpolation |
| `np.bincount()` | MISSING | Count occurrences |
| `np.apply_along_axis()` | MISSING | Apply function along axis |
| `np.vectorize()` | MISSING | Vectorize a function |
| `np.piecewise()` | MISSING | Piecewise functions |
| `np.select()` | MISSING | Conditional selection |
| `np.extract()` | MISSING | Extract based on condition |
| `np.place()` | MISSING | Change elements based on condition |
| `np.compress()` | MISSING | Return selected slices |
| `np.partition()` | MISSING | Partial sort |
| `np.argpartition()` | MISSING | Indices for partial sort |
| `np.lexsort()` | MISSING | Indirect sort with keys |
| `np.tensordot()` with tuple axes | PARTIAL | Only integer axes supported |

### 2.3 LOW Priority - Less Common

| NumPy Function | Status | Notes |
|----------------|--------|-------|
| `np.packbits()` / `np.unpackbits()` | MISSING | Bit packing |
| `np.frexp()` / `np.ldexp()` | MISSING | Float decomposition |
| `np.modf()` | MISSING | Fractional and integer parts |
| `np.divmod()` | MISSING | Division and remainder |
| `np.trunc()` | MISSING | Truncate to integer |
| `np.fix()` | MISSING | Round towards zero |
| `np.spacing()` | MISSING | ULP of x |
| `np.nextafter()` | MISSING | Next floating point value |
| `np.copysign()` | MISSING | Copy sign of y to x |
| `np.signbit()` | MISSING | Sign bit test |
| `np.heaviside()` | MISSING | Heaviside step function |
| `np.sinc()` | MISSING | Sinc function |
| `np.i0()` | MISSING | Modified Bessel function |
| `np.arcsinh()`, `np.arccosh()`, `np.arctanh()` | MISSING | Inverse hyperbolic |
| `np.unwrap()` | MISSING | Unwrap phase angles |
| `np.conj()` | MISSING | Complex conjugate |
| `np.real()`, `np.imag()` | MISSING | Complex parts |
| `np.angle()` | MISSING | Phase angle |
| `np.nan_to_num()` | MISSING | Replace NaN/inf |
| `np.flatnonzero()` | MISSING | Non-zero indices (flat) |
| `np.argwhere()` | MISSING | Indices where condition is true |
| `np.setdiff1d()` | MISSING | Set difference |
| `np.union1d()` | MISSING | Set union |
| `np.intersect1d()` | MISSING | Set intersection |
| `np.in1d()` | MISSING | Test membership |
| `np.isin()` | MISSING | Test membership (modern) |

---

## 3. NumPy Parity Issues Found

### 3.1 Data Type Limitations

**Issue:** Library only supports `f64` (Float64) dtype.

NumPy supports:
- `float16`, `float32`, `float64`, `float128`
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `complex64`, `complex128`
- `bool`

**Impact:**
- Cannot do integer arithmetic with proper overflow behavior
- Cannot represent boolean arrays efficiently
- No complex number support
- Memory inefficient for large arrays that could use smaller types

**Current Workaround:** Boolean operations use `0.0`/`1.0` convention

### 3.2 Negative Index Handling

**Status:** Partial support

- `reshape_infer` supports `-1` for dimension inference (NumPy behavior)
- `slice` supports negative indices
- Most axis parameters do NOT support negative indexing

**NumPy Behavior:**
```python
arr.sum(axis=-1)  # Works - last axis
arr.squeeze(axis=-1)  # Works - last axis
```

**Current tsnp Behavior:**
```typescript
arr.sumAxis(-1)  // Likely fails or returns incorrect result
```

### 3.3 NaN Handling Issues

**Good NaN Handling (Matches NumPy):**
- `np.nansum`, `np.nanmean`, `np.nanvar`, `np.nanstd`, `np.nanmin`, `np.nanmax` - all implemented
- `np.maximum`/`np.minimum` - correctly propagate NaN
- `np.fmax`/`np.fmin` - correctly ignore NaN
- `np.isnan`, `np.isinf`, `np.isfinite` - implemented
- `argmin`/`argmax` - NaN treated as greater/less than all values (NumPy behavior)
- Sorting - NaN values sort to end

**Potential NaN Issues:**
- `clip` with NaN bounds returns NaN (correct)
- WebGPU shaders use `x != x` trick for NaN detection in f32 (works)

### 3.4 Rounding Behavior

**Round (np.round):**
- NumPy uses round-half-away-from-zero for `np.round()` with 0 decimals
- tsnp correctly implements this

**Rint (np.rint):**
- NumPy uses banker's rounding (round half to even)
- tsnp correctly implements banker's rounding

**WebGPU Note:**
- WGSL `round()` uses banker's rounding
- Comment in code claims this matches NumPy, but NumPy's `round()` rounds away from zero
- This could cause subtle differences when using WebGPU backend

### 3.5 Empty Array Handling

**Good Handling (Matches NumPy):**
- `min()`/`max()` on empty array returns error (NumPy raises ValueError)
- `mean()` on empty array returns NaN
- `sum()` on empty array returns 0.0

**Missing:**
- `np.empty()` function

### 3.6 Division Behavior

- Integer division semantics not applicable (f64 only)
- Division by zero returns `Inf` or `NaN` (IEEE 754 behavior, matches NumPy)

### 3.7 Broadcasting Edge Cases

**Implemented:** Full NumPy-style broadcasting for binary operations

**Potential Issues:**
- Broadcasting with scalar arrays (shape `[]`) may not be tested
- Higher dimensional broadcasting (>3D) should be verified

### 3.8 SVD/Eigendecomposition Limitations

**SVD:**
- Uses eigendecomposition of A^T A (simplified approach)
- Comment acknowledges numerical issues for ill-conditioned matrices
- Not using proper Golub-Kahan bidiagonalization

**Eigendecomposition:**
- Uses power iteration with deflation
- Fixed 1000 max iterations
- Designed for symmetric matrices; may fail for non-symmetric

**NumPy uses LAPACK for production-quality implementations**

---

## 4. WebGPU vs CPU Coverage Analysis

### 4.1 Operations with GPU Acceleration

| Operation | CPU (WASM) | WebGPU Shader | Notes |
|-----------|------------|---------------|-------|
| **Elementwise Math** |
| sin, cos, tan | SIMD | GPU | Full parity |
| asin, acos, atan | SIMD | GPU | Full parity |
| sinh, cosh, tanh | SIMD | GPU | Full parity |
| exp, exp2, log, log2, log10 | SIMD | GPU | log10 uses approximation |
| sqrt, cbrt | SIMD | GPU | cbrt uses pow approximation |
| abs, sign, neg | SIMD | GPU | Full parity |
| floor, ceil, round | SIMD | GPU | round() semantics differ |
| reciprocal, square | SIMD | GPU | Full parity |
| expm1, log1p | SIMD | CPU only | Numerically stable versions |
| rint | SIMD | CPU only | Banker's rounding |
| deg2rad, rad2deg | SIMD | CPU only | Simple conversion |
| **Binary Ops** |
| add, sub, mul, div | SIMD | GPU | With broadcasting |
| pow | SIMD | GPU | GPU handles negative bases |
| maximum, minimum | SIMD | GPU | NaN propagation |
| fmax, fmin | SIMD | CPU only | NaN ignoring |
| arctan2, hypot | SIMD | CPU only | Two-arg functions |
| mod, fmod, remainder | SIMD | CPU only | Different semantics |
| **Reductions** |
| sum, prod | SIMD | GPU | Parallel reduction |
| min, max | SIMD | GPU | With NaN propagation |
| mean | SIMD | GPU (sum/size) | Computed from sum |
| var, std | SIMD | CPU only | Two-pass algorithm |
| argmin, argmax | SIMD | GPU | Track index |
| cumsum, cumprod | SIMD | GPU (naive) | O(n^2) on GPU |
| all, any | SIMD | GPU | Boolean reduction |
| **Linalg** |
| matmul | SIMD + faer | GPU (optimized) | Multiple strategies |
| dot | SIMD | GPU | 1D multiply-sum |
| transpose | ndarray | GPU | 2D only on GPU |
| trace | Loop | GPU | Diagonal sum |
| norm | SIMD | GPU (L2 only) | Other norms CPU |
| outer | Loop | GPU | Full matrix |
| inv, det, solve | faer | CPU only | Gaussian elimination |
| qr, lu, cholesky | faer | CPU only | Decompositions |
| svd, eig, eigvals | Custom | CPU only | Eigenvalue methods |
| **Creation** | All CPU | CPU fallback | No GPU creation |
| **Manipulation** | All CPU | CPU fallback | reshape, concat, etc |
| **Random** | All CPU | CPU fallback | PRNG is inherently serial |
| **Sort/Search** | All CPU | CPU fallback | sort, argsort, etc |
| **Compare** | All CPU | CPU fallback | eq, lt, etc |

### 4.2 GPU Performance Notes

**Matmul Performance (from CLAUDE.md):**
| Size | rumpy GFLOPS | tfjs GFLOPS | Speedup |
|------|-------------|-------------|---------|
| 512x512 | 224 | 206 | 1.08x |
| 1024x1024 | 1023 | 467 | 2.19x |
| 2048x2048 | 2070 | 1273 | 1.63x |
| 4096x4096 | 5799 | 2264 | 2.56x |

**Key Optimizations:**
1. Store A as vec4 along K dimension (not M)
2. B-value register caching
3. Autotune selects optimal shader per size
4. tfjs-32 shader (32x32 tiles) for 1024/2048
5. small-16 shader for 512/4096

**GPU Overhead Note:**
- Playwright Chromium adds ~15% GPU overhead vs native Chrome
- Always benchmark in native browser

---

## 5. Recommendations

### 5.1 High Priority (Core Functionality)

1. **Add dtype support** - At minimum `float32` for WebGPU (currently using f64->f32 conversion)
2. **Implement `einsum`** - Critical for ML frameworks
3. **Add negative axis indexing** - Consistent with NumPy behavior
4. **Implement `broadcast_to` and `broadcast_arrays`** - Explicit broadcasting utilities
5. **Add `empty()`, `empty_like()`, `zeros_like()`, `ones_like()`** - Common creation patterns

### 5.2 Medium Priority (Feature Completeness)

1. **FFT module** - Essential for signal processing
2. **Convolution operations** - `convolve`, `correlate`
3. **More linalg functions** - `cond`, `matrix_power`, `slogdet`
4. **Statistical functions** - `cov`, `corrcoef`
5. **More manipulation** - `swapaxes`, `moveaxis`, array splitting variants
6. **Improve SVD/eigendecomposition** - Use proper algorithms or LAPACK bindings

### 5.3 WebGPU Backend Expansion

1. **Port remaining unary ops** - `expm1`, `log1p`, `rint`, `deg2rad`, `rad2deg`
2. **Port binary ops** - `arctan2`, `hypot`, `mod`, `fmod`, `fmax`, `fmin`
3. **Port variance/std** - Two-pass algorithms on GPU
4. **Broadcasting** - GPU implementation for binary ops with different shapes
5. **Fix round() semantics** - WGSL banker's rounding vs NumPy round-away-from-zero

### 5.4 Testing & Quality

1. **Edge case tests** - Empty arrays, single-element arrays, very large arrays
2. **NaN/Inf tests** - Comprehensive coverage across all operations
3. **Broadcasting tests** - High-dimensional cases
4. **Numerical accuracy tests** - Compare with NumPy reference values
5. **WebGPU vs WASM parity tests** - Ensure backends produce identical results

---

## 6. Architecture Observations

### 6.1 Strengths

- Clean trait-based backend architecture (`Backend`, `CreationOps`, `MathOps`, etc.)
- Comprehensive CPU implementation using ndarray + faer
- Well-optimized SIMD GEMM kernels for WASM
- Multiple WebGPU matmul strategies with autotune
- Good NaN handling throughout
- Proper NumPy behavior for edge cases (empty arrays, ddof, etc.)

### 6.2 Areas for Improvement

- WebGPU backend is in TypeScript, not Rust (architectural inconsistency)
- Rust `rumpy-webgpu` crate is a placeholder
- No shared test suite between Rust and TypeScript (mentioned but not verified)
- Limited dtype support
- Some linalg implementations use simplified algorithms

---

## File Locations Referenced

- Backend traits: `/Users/sven/code/tsnp/crates/rumpy-core/src/ops.rs`
- CPU implementation: `/Users/sven/code/tsnp/crates/rumpy-cpu/src/` (math.rs, stats.rs, linalg.rs, etc.)
- WASM bindings: `/Users/sven/code/tsnp/crates/rumpy-wasm/src/lib.rs`
- WebGPU backend: `/Users/sven/code/tsnp/tests/webgpu-backend.ts`
- Test utilities: `/Users/sven/code/tsnp/tests/test-utils.ts`

---

*Report generated: 2026-03-04*
*Library version: tsnp (renamed from rumpy-ts)*
