# GEMM Optimization Notes

## 🏆 CURRENT STATUS: WE BEAT XNNPACK! (2026-02-26)

### Single-Threaded Performance - WE WIN ACROSS THE BOARD

Our new **optimized 6x8 micro-kernel** beats TensorFlow.js XNNPACK at ALL matrix sizes:

| Size | tfjs XNNPACK | rumpy optimized | Speedup | Notes |
|------|--------------|-----------------|---------|-------|
| 128x128 | 0.16ms | 0.10ms | **42% FASTER** | Small matrix optimization |
| 256x256 | 0.86ms | 0.45ms | **48% FASTER** | Best improvement |
| 512x512 | 4.09ms | 3.37ms | **18% FASTER** | |
| 768x768 | 13.9ms | 11.3ms | **19% FASTER** | |
| 1024x1024 | 32.8ms | 26.4ms | **20% FASTER** | |
| 2048x2048 | 262ms | 214ms | **19% FASTER** | Large matrix |
| 4096x4096 | 2167ms | 1832ms | **15% FASTER** | Very large matrix |

### Key Optimizations That Made the Difference

1. **MR=6, NR=8 micro-kernel** - 6 rows × 8 cols per tile, uses 12 v128 accumulator registers (fits perfectly in WASM's 16 XMM registers)

2. **`v128_load32_splat` instead of `f32x4_splat`** - Dedicated instruction for broadcasting scalar to vector, faster than load+shuffle

3. **`f32x4_relaxed_madd` (FMA)** - Fused multiply-add in tight inner loop (note: FMA is faster when used WITH the other optimizations, but was slower in isolation before)

4. **L1/L2 cache blocking** - KC=256 (fits K-panel in L1), MC=72 (6 rows × 12 tiles), NC=128 (fits N-panel in L2)

5. **B matrix packing** - Pack B once, reuse for all M-panels. Sequential memory access in inner loop.

### Functions

- `matmulF32Optimized` - Single-threaded optimized kernel (FASTEST single-threaded)
- `matmulF32OptimizedParallel` - Multi-threaded version
- Previous kernels still available for comparison

---

## Multi-threaded Optimization Journey (2026-02-26)

### Problem: Multi-threaded was 5x SLOWER than tfjs

With the old rayon-based parallelization, we were consistently 1.5-5x slower than tfjs multi-threaded (which uses XNNPACK + pthreadpool).

### Root Causes Identified

1. **wasm-bindgen-rayon initialization**: `initThreadPool()` was hanging because our benchmark server wasn't properly serving the workerHelpers.js module. Fixed by routing `/pkg/` requests to `/pkg/rumpy_wasm.js`.

2. **Allocator contention**: Original code allocated `packed_b` buffer inside `par_iter().for_each()`. With hundreds of tasks, threads serialized waiting for the global heap lock. Fixed by using `for_each_init()` to allocate buffers once per thread.

3. **1D vs 2D tiling**: 1D row partitioning means all threads read the entire B matrix → cache thrashing. 2D tiling reduces shared memory access.

### 🔴 THE BIG DISCOVERY: 32-bit `usize` Overflow (NOT cache aliasing)

**Symptom**: Parallelization worked perfectly up to 1792×1792 (5–7× speedup), but dropped to **exactly 1.0×** at 2048×2048 and 4096×4096.

**Initial misdiagnosis** (preserved for posterity — 6 hours were spent on this):
We first believed this was cache-set aliasing from power-of-2 row strides. Implemented stride padding, A-packing, 2D tiling, per-thread scratch jitter, buffer spacers. None of it helped. `probeV3Dispatch()` confirmed workers were running on 8 distinct rayon threads — so dispatch was fine, yet every worker took single-threaded time.

**Actual root cause** (found via `probeV3Path()`):

```rust
if m * n * k < 128 * 128 * 128 {  // ← THE BUG
    return single_threaded();
}
```

WASM `usize` is **32-bit**. At `(2048, 2048, 2048)`, `m * n * k = 8,589,934,592` which **overflows to exactly 0**. `0 < threshold` → true → silent single-threaded fallback. At 2047, `m*n*k = 8,576,582,623 mod 2³² = 4,281,615,327` — non-zero, goes parallel. At 2056, similar. **The "any non-pow2 dim fixes it" pattern was coincidence**, not causation.

**Evidence** (now correctly interpreted):
| (M, N, K) | `m*n*k mod 2³²` | Path taken | Speedup |
|---|---:|---|---:|
| (2047, 2047, 2047) | 4,281,615,327 | parallel | 3.75× |
| (2048, 2048, 2048) | **0** | **single-threaded** | **1.00×** |
| (2049, 2049, 2049) | 12,614,337 | parallel | 4.21× |
| (4096, 4096, 4096) | **0** | **single-threaded** | **1.00×** |

**The fix** — cast to `u64` before multiplying:
```rust
if (m as u64) * (n as u64) * (k as u64) < (128u64 * 128 * 128) {
```

After the fix, 2048³ goes from 1.00× → **6.6× scaling** with no other changes.

**Cache aliasing IS a real phenomenon** — a 6-row micro-kernel at stride 8192 B does create L1 set conflicts, costing ~10–20% per thread. But it doesn't cause a total collapse to 1.0×. That specific pattern (exactly 1.0×, cliff not slope, pow2-triggered) is the overflow fingerprint.

**⚠️ This bug is currently BACK IN MAIN** — the PR #2 squash-merge dropped `simd_gemm.rs` changes in a conflict. See `docs/GEMM-NEXT-STEPS.md` for the re-fix.

### Current Multi-threaded Results (with fixes, 10 threads)

| Size | tfjs multi | rumpy parallel | vs tfjs | Parallel Speedup |
|------|-----------|----------------|---------|------------------|
| 256x256 | 0.14ms | 0.17ms | 1.23x slower | 2.6x |
| 512x512 | 0.74ms | 0.80ms | 1.07x slower | 4.3x |
| 1024x1024 | 5.33ms | 4.67ms | **0.88x FASTER** | 5.7x |
| 2049x2049 | ~42ms | ~50ms | 1.2x slower | 4.2x |
| 2056x2056 | ~42ms | ~45ms | ~1.07x slower | 4.8x |

**We now beat tfjs at 1024x1024!** Larger sizes need padding to avoid cache aliasing.

### Lessons Learned

1. **Power-of-2 matrix dimensions are DANGEROUS** in multi-threaded code. Always test 2^n-1, 2^n, and 2^n+1.

2. **wasm-bindgen-rayon is picky** about module resolution. The workerHelpers.js import `../../..` must resolve correctly.

3. **Allocate buffers per-thread, not per-task** using `for_each_init()` to avoid allocator serialization.

4. **Pre-allocate WASM memory** to avoid `memory.grow` during parallel execution (causes global pause).

---

## Historical Data (2026-02-25)

### Previous Performance (BEFORE optimization)

Old results with `matmulXnnpack` (our previous best):

| Size | rumpy XNNPACK | tfjs 1-thread | Ratio | Notes |
|------|---------------|---------------|-------|-------|
| 32x32 | 0.013ms | 0.016ms | **0.81x FASTER** | Small overhead wins |
| 64x64 | 0.022ms | 0.017ms | 1.29x slower | Overhead dominates |
| 100x100 | 0.067ms | 0.044ms | 1.53x slower | N%8 handled correctly |
| 128x128 | 0.081ms | 0.070ms | 1.16x slower | |
| 256x256 | 0.672ms | 0.531ms | 1.27x slower | Main gap area |
| 384x384 | 1.70ms | 1.77ms | **0.96x FASTER** | |
| 512x512 | 4.30ms | 4.21ms | 1.02x slower | Nearly matched |
| 768x768 | 13.4ms | 16.0ms | **0.84x FASTER** | |
| 1024x1024 | 37.0ms | 33.7ms | 1.10x slower | Close |

### Multi-threaded Performance (old rayon-based, both 14 threads)

| Size | rumpy V2 | tfjs 14-thread | Ratio | Notes |
|------|----------|----------------|-------|-------|
| 32x32 | 0.007ms | 0.020ms | **2.9x FASTER** | Small matrix overhead |
| 64x64 | 0.024ms | 0.019ms | 1.3x slower | Thread overhead |
| 100x100 | 0.166ms | 0.031ms | 5.4x slower | |
| 128x128 | 0.223ms | 0.041ms | 5.4x slower | |
| 256x256 | 0.318ms | 0.116ms | 2.7x slower | |
| 384x384 | 0.529ms | 0.326ms | 1.6x slower | |
| 512x512 | 1.17ms | 0.74ms | 1.6x slower | |
| 768x768 | 3.41ms | 2.31ms | 1.5x slower | |
| 1024x1024 | 8.95ms | 5.35ms | 1.7x slower | |

**NOTE**: Multi-threaded still needs testing with the new optimized kernel!

**We are 1.5-5x SLOWER than tfjs at most sizes with fair threading!**

---

## 🔧 Parallel V3 (2026-02-26) — ROOT CAUSE ANALYSIS

After deep-diving `matmul_optimized_f32_parallel` and comparing against how
XNNPACK drives `pthreadpool_parallelize_2d_tile_2d`, we found **four**
independent sources of waste in the legacy parallel path. Each one is a
~1.5× hit; multiplied together they explain the 1.5–5× gap.

### Bug #1: Every thread re-packs B (N× redundant work)

```rust
c.par_chunks_mut(rows_per_thread * n).for_each(|(_, c_chunk)| {
    let local_c = matmul_optimized_f32(a_slice, b, local_m, n, k);  // ← packs B!
    c_chunk.copy_from_slice(&local_c);
});
```

`matmul_optimized_f32` is a complete, self-contained GEMM — it allocates a
packing buffer and packs every KC×NC block of B internally. With 14 threads
you do 14× the packing. For 1024² @ KC=256, NC=128, that's packing 8 blocks
of 32 K floats each × 14 threads = **3.5 M extra float stores** per matmul,
vs 250 K for a pack-once path.

### Bug #2: Per-thread heap allocation under a global lock

Both `Vec::with_capacity(m*n)` for C and `Vec::with_capacity(KC*NC)` for
packed_b go through WASM's `dlmalloc`, which is globally mutexed. 14 threads
simultaneously trying to allocate serialise on that lock. This is *worse*
than native: on Linux you'd get per-thread arenas; on WASM you get nothing.

### Bug #3: `par_chunks_mut` caller doesn't compute

When called from outside the Rayon pool (i.e. from the JS main thread),
`par_iter` family functions dispatch all work to pool workers and block the
caller. With N Rayon workers you get N-way parallelism, not N+1. Contrast
with pthreadpool where the caller *is* thread 0 and does its share.

### Bug #4: Extra `copy_from_slice` at the end

Each thread computes into a fresh `local_c: Vec<f32>` then copies back
into the real C. For 1024² @ 14 threads, that's 14 × (1024/14) × 1024 = 1 M
extra f32 stores.

### Bug #5 (architectural): 1D slab partitioning = zero load balancing

Splitting M into N_threads equal slabs means each thread gets exactly one
task. If one core is slow (Apple Silicon efficiency cores!), it holds
everyone back — there's nothing to steal. XNNPACK tiles into ~MC×NC chunks
and lets workers drain a queue, so slow cores just do fewer tiles.

### The fix: `matmul_optimized_f32_parallel_v3`

| | legacy | v3 |
|---|---|---|
| B packing | per-thread (N×) | once, shared read-only |
| per-task alloc | 2× `Vec` per thread | 0 (pointers into pre-alloc'd buffers) |
| caller computes | no (waits) | yes (thread 0 via `rayon::scope` inline) |
| work distribution | N fixed slabs | atomic counter, ~15 tiles/block, stealable |
| final copy | `copy_from_slice` | workers write directly to C |

**Expected speedup**: 2–4× over legacy. May match/beat tf.js at 256–1024.

### The harder fix: raw-futex pool (`wasm_futex.rs`)

v3 still goes through Rayon's `scope`/`spawn`, which `Box`es each spawned
closure and uses `park()`/`unpark()` for the join barrier. For sub-ms GEMM
blocks this is measurable.

The `wasm_futex` module reimplements the pthreadpool algorithm natively:
workers spin ~100 μs then `memory.atomic.wait32`; the caller bumps a
generation counter + one `atomic.notify` to dispatch; one atomic `fetch_sub`
per claimed tile.  For back-to-back GEMM blocks (the KC loop dispatches
every ~200 μs) workers never actually park — they're still spinning when
the next generation lands.

**Build**: `wasm-pack build --features futex-pool` — pulls in `wasm_thread`
for Web Worker spawning (`std::thread::spawn` panics on wasm32).

**Caveat**: the futex pool spawns its *own* workers separate from Rayon's.
If you're using both, don't size both to full `navigator.hardwareConcurrency`
or you oversubscribe. For a pure-rumpy app, use futex-pool alone and skip
`initThreadPool`.

---

### V2 vs V1 Parallel (both 14 threads)

| Size | V1 | V2 | V2 Speedup | Notes |
|------|----|----|------------|-------|
| 64x64 | 0.122ms | 0.024ms | **4.8x** | Huge win from no-alloc |
| 100x100 | 0.231ms | 0.166ms | 1.4x | |
| 128x128 | 0.230ms | 0.223ms | 1.0x | |
| 256x256 | 0.353ms | 0.318ms | 1.1x | |
| 512x512 | 1.19ms | 1.17ms | 1.0x | |
| 1024x1024 | 9.03ms | 8.95ms | 1.0x | |

V2's zero-allocation approach helps most at small sizes where allocation overhead dominates

## WASM Binary Analysis (2026-02-25)

Disassembled both rumpy_wasm_bg.wasm (470KB) and tfjs-backend-wasm-simd.wasm (415KB) using wasm2wat.

### Instruction Counts Comparison

| Instruction | Rumpy | TFJS | Analysis |
|-------------|-------|------|----------|
| **f32x4.relaxed_madd (FMA)** | 166 | 0 | We use FMA, they don't |
| f32x4.mul | 677 | 1,585 | They use separate mul |
| f32x4.add | 567 | 1,738 | They use separate add |
| v128.load | 2,119 | 3,631 | They load more data |
| v128.store | 1,955 | 1,646 | We store more often |
| **Load/Store ratio** | 1.08 | 2.21 | They reuse data 2x more per store |
| i8x16.shuffle | 324 | 888 | They do more data reorganization |
| Branch instructions | 8,062 | 6,000 | We have more conditionals |
| Loop instructions | 1,676 | 1,815 | Similar loop count |

### Key Findings

1. **FMA vs mul+add**: We use `f32x4.relaxed_madd` (166 instances), TFJS uses ZERO FMA. They use separate `f32x4.mul` + `f32x4.add`. V8's WASM runtime may optimize mul+add better than relaxed_madd.

2. **Load/Store Ratio**: Our ratio is 1.08 (almost 1:1 load:store), theirs is 2.21 (load 2.2x per store). This indicates TFJS reuses loaded data much better - likely through register blocking and better accumulator management.

3. **Branch Count**: We have 8,062 branches vs their 6,000 (35% more). This suggests more conditional paths, possibly from bounds checking, edge case handling, or less aggressive loop unrolling.

4. **Shuffle Operations**: TFJS uses 888 shuffles vs our 324. More shuffles may indicate more sophisticated data reorganization for cache-friendly access patterns.

### Implications

- Our FMA usage may actually be hurting performance (matches micro-benchmark findings)
- We need better register blocking to improve load/store ratio
- Reducing branches could help - consider more aggressive inlining and loop unrolling
- The fundamental difference may be XNNPACK's mature C→WASM vs our Rust→WASM codegen

## Key Learnings

### 1. Pre-packing B is Critical
The single biggest win was pre-packing B matrix once and reusing it. Without pre-packing, we were 6-8x slower because we packed B on every matmul call.

With pre-packing API (`packB` + `matmulXnnpack`), we match XNNPACK at most sizes.

### 2. FMA is Not Always Faster
Counterintuitively, `f32x4_relaxed_madd` (FMA) is slightly SLOWER than `f32x4_mul` + `f32x4_add` in WASM:
- 1024x1024: FMA 53ms vs mul+add 47ms

This might be because:
- WASM JIT may not have optimal FMA codegen
- Instruction scheduling may be worse with FMA
- XNNPACK itself uses mul+add, not FMA

### 3. Cache Blocking Adds Overhead in WASM
Implemented cache blocking (KC=256, MC=128, NC=256) but it's SLOWER than non-blocked:
- Overhead of zeroing C and load-add-store per tile
- WASM memory model may be different from native
- XNNPACK likely has more sophisticated streaming/prefetch that WASM can't express

The blocked kernels (`matmulF32Blocked`, `matmulXnnpackBlocked`) are available but not recommended.

### 4. Rayon Threading Overhead in WASM is Brutal

We investigated why our parallel implementation is 1.5-5x slower than TFJS parallel:

**Our V1 problems (now partially fixed in V2):**
- Each thread allocated its own `Vec<f32>` result
- Results concatenated with `c.extend(chunk)` at the end
- 1D row partitioning (all threads read entire B matrix)

**V2 fixes:**
- Uses `par_chunks_mut` to write directly to pre-allocated output
- No per-thread allocation, no final copy
- Still uses 1D row partitioning (TODO: 2D tiling)

**Why XNNPACK/pthreadpool is still faster:**
1. **pthreadpool vs rayon**: pthreadpool uses futex-based synchronization, wait-free work items, and persistent thread pools. Rayon uses work-stealing with atomic latches that have higher overhead in WASM.
2. **2D tiling**: XNNPACK uses `parallelize_2d_tile_2d()` for M×N tiles. Each thread processes tiles and can steal from others. B columns stay hot in L2 cache.
3. **Apple Silicon issue**: wasm-bindgen-rayon has known issues on M1/M4 - efficiency cores drag performance (see github.com/GoogleChromeLabs/wasm-bindgen-rayon/issues/16)
4. **Atomics overhead**: WASM atomics are slower on Apple Silicon vs x86

**Potential fixes not yet tried:**
- 2D tile partitioning for better cache reuse
- Reduce thread count (use 4 perf cores instead of 14 total)
- Consider pthreadpool instead of rayon for WASM builds
- JS Web Workers approach (each worker loads own WASM instance)

### 5. XNNPACK's Multi-threading is Built-in
TensorFlow.js WASM backend uses XNNPACK which has built-in multi-threading via pthreadpool. When comparing, we need to ensure fair thread counts.

### 5. Tile Size Matters
We use 6x8 tiles (6 rows, 8 cols = 2 v128s per row). XNNPACK also uses 6x8 or similar. The tile size affects:
- Register pressure (12 accumulators = 12 v128 registers)
- Cache efficiency
- Alignment (n must be divisible by 8 for SIMD panels)

### 6. Memory Layout
XNNPACK packs B in a specific format: for each panel of 8 columns, K values are interleaved:
```
panel0[k0:col0-7, k1:col0-7, k2:col0-7, ...]
panel1[k0:col8-15, ...]
```

This gives sequential memory access in the inner loop.

## Resolved Issues

### N % 8 != 0 Bug (FIXED)
The XNNPACK-style kernel now handles arbitrary matrix dimensions:
- `matmul_simd_f32_xnnpack_style_full` takes both original B and packed_b
- SIMD panels use packed_b, remaining columns use original B
- 100x100 matrices now work correctly

## Outstanding Issues

### 1. 256x256 Performance Gap
At 256x256, we're 1.27x slower than tfjs. This is the main area for improvement.

Possible causes:
- Not enough work to amortize packing overhead
- Sub-optimal K loop unrolling at this size
- XNNPACK may have special-cased this size

### 2. Small Matrix Overhead (64-128)
At 64x64 and 128x128, we're 1.16-1.29x slower. Function call and packing overhead dominates.

Potential fixes:
- Inline small matrices
- Skip packing for small sizes
- Use simpler 4x8 kernel for sizes < 128

### 3. Parallel Overhead at Small Sizes
At small sizes (64x64), parallel is much slower than single-threaded due to thread spawn/join overhead. Current heuristic requires m*n*k >= 64^3 to use parallel.

## Architecture

```
matmulF32 (default)
  -> matmul_dispatch_f32
    -> if m >= 4, n >= 8: matmul_simd_f32 (mul+add kernel)
    -> else: matmul_scalar_f32

matmulF32Parallel
  -> rayon parallel_iter over row chunks
    -> each thread calls matmul_dispatch_f32

matmulXnnpack (fastest single-threaded)
  -> packB (pre-pack B once)
  -> matmul_simd_f32_xnnpack_style_full (6x8 kernel with mul+add)
    -> SIMD for panels 0..(n/8)
    -> scalar fallback for remaining columns

matmulF32Blocked (experimental, not recommended)
  -> Zero C
  -> Triple-nested blocking (NC, KC, MC)
  -> Load-add-store per tile
```

## Recommendations

**For production use:**
1. **Single matmul**: Use `matmulF32` for simplicity, or `matmulXnnpack` with pre-packed B if weights are reused
2. **Neural network inference**: Pre-pack weights with `packB`, use `matmulXnnpack` for forward passes
3. **Large matrices**: Use `matmulF32Parallel` for matrices >= 256 (3-4x faster than tfjs single-threaded)
4. **Small matrices**: Use `matmulF32` (parallel overhead not worth it)

## Build Commands

```bash
# Build with threads+SIMD
rustup default nightly
wasm-pack build crates/rumpy-wasm --target web --out-dir ../../benchmarks/pkg-web --release

# Run benchmarks
cd benchmarks
npm run dev  # or: npx vite
node run-f32-benchmark.js
```

## Future Work

1. **Tune small matrix path** - Special-case 64-256 to reduce overhead
2. **Investigate 256 gap** - Profile why this specific size is slower
3. **Parallel XNNPACK** - Combine pre-packing with multi-threading for best of both
4. **WebGPU GEMM** - For matrices > 1024, GPU should be faster
