# GEMM: What to Do Next

_Deep research synthesis, 2026-02-27. Covers micro-kernel codegen analysis, XNNPACK internals, algorithmic techniques, and what the literature says._

---

## 🚨 FIRST: Regression in main

The PR #2 squash-merge (`b5c0355`) **dropped `crates/rumpy-cpu/src/simd_gemm.rs` changes** in a merge conflict. Main currently has:

```rust
// Line 2222, 2271, 2321, 2383, 2879:
if m * n * k < 64 * 64 * 64 {   // ← STILL OVERFLOWS AT 2048³
```

At `(2048, 2048, 2048)`, `m * n * k = 8.6e9` wraps 32-bit `usize` to **0**, silently routing to single-threaded. This is the same bug we fixed in commit `0320aad`. The "cache associativity aliasing" explanation in `GEMM-OPTIMIZATION-NOTES.md` is a red herring — the 2040/2048/2056 pattern it describes is **exactly** what overflow-masquerading-as-something-else looks like (any non-pow2 dim coincidentally produces non-zero `m*n*k`).

**Fix (5 lines each):**
```rust
// Replace all `if m * n * k < THRESHOLD` with:
if (m as u64) * (n as u64) * (k as u64) < (THRESHOLD as u64) {
```

Also missing from main: `matmul_optimized_f32_parallel_v3`, `pack_b_full_xnnpack`, `matmul_slab_prepackedb`, `micro_kernel_6x8_fma_pa`, `pack_a_6xkc`. The `F32Buffer` API in `rumpy-wasm` was rewired to call the old `matmul_optimized_f32_parallel` instead. **Re-cherry-pick commits `360f04c` → `5e0a02d` from `perf/parallel-gemm-v3` (they're still on origin).**

---

## Why we beat XNNPACK (from disassembling their WASM)

Disassembled `tfjs-backend-wasm-threaded-simd.wasm` (func 415, the 5×8 minmax kernel):

| Dimension | XNNPACK (tf.js 4.22) | rumpy `micro_kernel_6x8_fma` | Impact |
|---|---|---|---|
| **FMA** | `f32x4.mul` + `f32x4.add` (separate, **NO relaxed-simd build ships**) | `f32x4_relaxed_madd` → native `vfmadd231ps` (1 μop) | **2× fewer arith instructions** |
| **MR × NR** | 5×8 (10 accumulators) | 6×8 (12 accumulators) | **+20% compute per B-load** |
| **K-unroll** | 4× (amortizes their `load`+`shuffle` A-broadcast) | 1× (our `load32_splat` is already 1-μop) | wash |
| **A-broadcast** | `v128.load` + 4× `i8x16.shuffle` per row | `v128.load32_splat` | same memory ops, ours is 0 port-5 pressure |

The **relaxed-simd absence is tf.js's core disadvantage**. They'll fix this eventually — when they do, our lead shrinks. The MR=6 vs MR=5 advantage is permanent unless V8 spills us (we're at the 15-XMM ceiling exactly).

From `XNNPACK/src/configs/gemm-config.c:942`:
```c
if (hardware_config->is_x86) {
  // x86: 4×8 LOADSPLAT — they NEVER TRIED 6×8 loadsplat on x86
  ...ukernel_4x8__wasmrelaxedsimd_fma_loadsplat;
} else {  // ARM
  ...ukernel_6x8__wasmrelaxedsimd_fma_splat;  // K=4, load+shuffle
}
```

**We found a hole in their search space.** They tested {4×8 loadsplat, 6×8 splat} on x86 but not 6×8 loadsplat.

---

## Our codegen inefficiencies (from disassembling `rumpy_wasm_bg.wasm`)

The 6×8 micro-kernel was inlined 12 times. All instances show the same issues:

### 1. A-pointer chain recomputed every K (highest cost)

Source has 6 precomputed row pointers (`a0..a5`). LLVM **spilled them** and emits a chained `i32.add lda` sequence every iteration:

```wat
local.get 3              ; a_base
...load32_splat          ; row 0
local.get 3; local.get 39; i32.add; local.tee 17  ; temp = a_base + lda
...load32_splat          ; row 1
local.get 17; local.get 39; i32.add; local.tee 17  ; temp += lda
...load32_splat          ; row 2
...(×3 more)
```

= **14 extra WASM instructions per K** (5 adds + 5 gets + 4 tees). Native impact: ~5 extra `lea`/`add` per K-iter.

**Fix:** keep the 6 pointers live across iterations. Options in rough order of nastiness:
- Bump them by 4 each iteration instead of re-deriving: `a0 = a0.add(1); a1 = a1.add(1); ...` — 6 adds but no chain dependency
- Wrap pointers in `core::hint::black_box` after setup to prevent rematerialization
- Inline asm barrier: `asm!("", inout(reg) ptr, options(nostack, nomem))` for each

### 2. LLVM didn't K-unroll

```wat
;; 11 instructions of pure loop overhead per K:
local.get 12; i32.const 32; i32.add; local.set 12   ; b_run += 32
local.get 3;  i32.const 4;  i32.add; local.set 3    ; a_ptr += 4
local.get 14; i32.const 1;  i32.sub; local.tee 14; br_if 0
```

That's 11 instructions to do `add; add; dec; jnz` — WASM stack-machine tax. Unrolling by 2 halves the per-K cost.

**Fix:** manual K-unroll by 2 in source. LLVM's WASM backend doesn't auto-unroll (confirmed from `--print-code`). The tricky part: K%2 tail. Pad K to even at pack time (we control packed-B) or add a scalar tail.

**Caveat:** unrolling by 2 adds ~12 accumulator-FMA instructions to the loop body (24 FMAs instead of 12), increasing i-cache footprint. At K=256 (typical KC), we go from ~100 loop iterations to ~50 — probably worth it. At K<64 (head-dim attention), maybe not.

### 3. `v128.load offset=N` not folded

```wat
local.get 12; i32.const 16; i32.add; v128.load align=1   ; ← should be v128.load offset=16
```

3 extra instructions for `b_run+16`. Same issue in packed-A: 5 offsets × 3 instructions = 15 wasted. This is an **LLVM WASM backend limitation** — SIMD load offset-folding is incomplete.

**Fix:** known LLVM issue. Workaround is to use raw pointer arithmetic in a pattern LLVM recognizes:
```rust
// Instead of:
let vb1 = v128_load(b_run.add(4) as *const v128);
// Try:
let vb1 = v128_load((b_run as *const u8).add(16) as *const v128);
```
Or file an LLVM bug — the fold is straightforward, probably just a missing pattern.

### 4. `align=1` on everything

Packed B and scratch buffers ARE 16-aligned (we allocate them), but the `as *const v128` cast from `*const f32` drops alignment info. On modern x86 `movups` vs `movaps` is ~free, but on some ARM cores it matters.

**Fix:** `#[repr(align(16))]` wrapper for packed buffers, or `v128_load(...)` where the intrinsic's documented alignment matches.

### Summary: instructions per K

| | Ideal | Current | With fixes 1+2+3 |
|---|---:|---:|---:|
| FMA | 12 | 12 | 24 (K-unroll 2) |
| Loads | 8 | 8 | 16 |
| Pointer arith | 2 | 8 | 4 |
| Loop overhead | 2 | 11 | 11 (amortized over 2K) |
| **"Real" ops / K** | **23** | **31** | **~27.5** |
| Stack-machine noise (local.*) | — | 75 | ~130 total / 2K = 65/K |

Expected gain from fixes 1–3: **5–10%**. Not huge, but essentially free.

---

## Tier 1: High-ROI next optimizations

### A. GEMMFIP — fused first-pass packing (5–15%)

_Xu, Van Zee, van de Geijn, ICS 2023, [arXiv:2302.08417](https://arxiv.org/abs/2302.08417)_

First time the kernel touches a micro-panel of A, read from the source matrix **and** write to the pack buffer in the same pass. Subsequent K-iterations read from packed. Single-threaded software pipelining — no pack thread, no barrier.

**Why this beats our current approach:** we have a heuristic "pack A when K≥4096" that's always wrong for some workloads. GEMMFIP makes the **first-pass cost the same as unpacked**, and all subsequent passes get packed performance. Best-of-both unconditionally.

**Implementation sketch:**
```rust
// Two micro-kernel variants:
//   _first_pass: reads A[row, k], writes pack[k*MR + row], does FMAs
//   _subsequent: reads pack[k*MR + row], does FMAs
// Bitmap tracks which (ii, p) panels are packed. Cleared per matmul.
unsafe fn micro_kernel_6x8_first_pass(
    k_size: usize,
    a_strided: *const f32, lda: usize,
    a_pack_dest: *mut f32,  // ← writes here as side-effect
    b_packed: *const f32,
    c: *mut f32, ldc: usize, beta: f32,
) { /* load A strided, store to pack, FMA */ }
```

**Complexity:** 3.5/5. Needs the two kernel variants + a packed-panel bitmap + scratch pre-allocation. BLIS sandbox has source to crib from.

### B. K-splitting for skinny shapes (up to 3–4× where we currently get 1-core)

_oneDNN `gemm_driver.cpp:1300`_

For `[small M, small N, huge K]` (transformer QKV reduction, attention score·V), M×N partitioning gives fewer tiles than threads. Split K instead:

```
trigger: (m/64 + n/64) < n_threads  AND  k >= 2048
split K up to 4 ways → each thread does full M×N on its K-slice
reduce via parallel SAXPY (partial-C buffers, never atomics)
```

For `[64, 4096] × [4096, 64]` with 8 threads: old path gets 1 M-tile × 1 N-tile = 1-core; K-split gives 4×64×1024 sub-GEMMs → 4-core utilization, then a 64×64×4 SAXPY reduce.

**WASM notes:** pre-allocate per-worker partial-C (scratch arena indexed by worker ID). The SAXPY reduce is embarrassingly parallel. **Remember the u64 cast** on `m*n*k` in the trigger check.

**Complexity:** 3/5.

### C. Per-browser kernel selection (5% on Firefox)

From [jott.live matmul analysis](https://jott.live/markdown/mm_wasm):

| Browser | Best (MR, NR, K-unroll) @ N=128 | Throughput |
|---|---|---|
| Chrome | (2, 8, 1) | 46 GFLOP/s |
| Firefox | (2, 4, **16**) | 44 GFLOP/s |

Chrome prefers shallow unroll + wide N; Firefox prefers deep K-unroll + narrow N. SpiderMonkey's WASM backend has different register pressure heuristics than V8.

**Fix:** ship 2 kernel variants, detect via `navigator.userAgent` (yes, really — there's no feature-detection for JIT characteristics). One-time 5 ms micro-benchmark at init is the nicer way but adds startup latency.

**Complexity:** 2/5.

### D. Int8 with zero-point folding (2–3× vs naive int8)

If you add int8 GEMM, do it the XNNPACK QS8 way from the start (`src/reference/packing.cc`):

```
At pack-B time, fold into bias:  bias'[n] = bias[n] − input_zp × Σₖ W[n,k]
Hot loop becomes raw  acc += a[k] × w[k]  — zero per-iteration zp arithmetic
Pad packed-B tail with kzp (not 0) so overflow reads contribute zero
```

Don't do the Ruy-style separate-sums-array — it's slower and only needed if zp changes per-call (it doesn't for inference).

**Complexity:** 2/5 for the math, 4/5 for the `i32x4.dot_i16x8_s` kernel (WASM relaxed-simd has it, V8 lowers to `vpmaddwd`).

---

## Tier 2: Worth an experiment

### E. Hilbert-curve block scheduling (new, Jan 2026)

_Georganas, Heinecke, Dubey (Intel), [arXiv:2601.16294](https://arxiv.org/abs/2601.16294)_

Replace 5-nested-loop block iteration with Hilbert-curve traversal of the output-block grid. Geomean **1.4× over oneDNN** on Emerald Rapids, **2.0× on 128-core Granite Rapids**. Core logic: ~30 lines.

Why it works: adjacent Hilbert indices → spatially adjacent blocks → cache locality without per-shape tile tuning. **Shape-obliviousness** is the WASM-relevant property — you can't JIT-tune tile sizes, so a parameter-free schedule matters.

Ruy already does a lighter version (`block_map.cc` — Hilbert when working set > LLC, linear otherwise).

**Complexity:** 2.5/5. Hilbert decode is 4 shuffles/block, textbook code. **Single-threaded benefit unclear** — paper's wins scale with core count.

### F. tinyBLAS-style k-vectorized kernel for small matrices

llamafile's `tinyBLAS` computes `C = Aᵀ·B` — k-vectorized, both operands loaded as v128 along K, horizontal sum at tile end. Zero packing, zero allocation. For 128-bit: 4×3 tile, 12 vector accumulators, one `hsum` per output tile.

**Nobody has ported this to WASM.** The design avoids all our "when to pack" headaches for matrices where packing doesn't amortize (`m,n,k < ~200`).

**Complexity:** 3/5. Write a 4×3 k-vectorized kernel, wire it as the <200 path replacing the scalar fallback.

### G. LLVM RegStackify mitigation (unknown gain, but it's a landmine)

[llvm#98631](https://github.com/llvm/llvm-project/issues/98631) — **open bug, affects all WASM SIMD code**. LLVM's WASM backend reorders instructions to build deep expression stacks, **undoing manual FMA interleaving**. Measured 1.6–1.8× slowdown on XNNPACK dwconv kernels.

The fix ([llvm#97283](https://github.com/llvm/llvm-project/pull/97283), `-webassembly-max-reg-stackify-depth`) is still unmerged.

**Our WAT analysis shows LLVM DID reschedule** our kernel (phase 1: loads interleaved with 6 FMAs; phase 2: 6 more FMAs batched). Whether this hurts depends on V8's downstream scheduling. V8's `turbo_instruction_scheduling` is **OFF by default** — it trusts your instruction order.

**Fix:** `asm!("" : "+r"(acc))` barriers between FMA groups, or the XNNPACK macro:
```rust
macro_rules! force_realize {
    ($x:expr) => { unsafe { core::arch::asm!("", in("m") &$x, options(nostack, readonly)) } };
}
```

**Complexity:** 1/5 for the barrier. Testing whether it helps: 1 hour.

---

## Tier 3: Don't bother (with receipts)

| Technique | Why not | Source |
|---|---|---|
| **Strassen** | Crossover n≈500–1300, wins 3–8% at n=2048. **Zero production ML libraries ship it** — oneDNN issue #1971 explicitly rejected ("DL workloads don't benefit" — shapes are rank-k / fat-skinny, not square). | SC16 "Strassen Reloaded" [arXiv:1605.01078](https://arxiv.org/abs/1605.01078); AlphaTensor's 47-mul is GF(2)-only |
| **Prefetch** | WASM proposal **benchmarked and rejected by its own author**: −2 to −8% on x86. Opcodes removed from V8. `__builtin_prefetch` → no-op. | [WebAssembly/simd #352](https://github.com/WebAssembly/simd/pull/352) |
| **Non-temporal stores** | **Neither BLIS nor oneDNN uses NT stores in GEMM.** McCalpin: NT stores hurt single-thread by 17–28%. GEMM is compute-bound; C micro-tile is 3 cache lines with reuse. | oneDNN uses NT *loads* (`tileloaddt1`), backward from folklore |
| **Runtime low-rank detection** | Searched torch.compile, XLA, TVM, oneDNN, Ruy, XNNPACK: nobody does it. Pretrained dense weights are full-rank. Structured (Monarch, butterfly, 2:4) is user-declared at model design time. | — |
| **Z-order / Morton storage** | Nobody uses it for GEMM *storage* (only Ruy for block *iteration order*). Goto panel packing achieves same locality simpler. | — |
| **Runtime auto-tuning** | No production library does it. BLIS has closed-form tile sizes from cache geometry. | Low et al. TOMS 2016 "Analytical Modeling Is Enough" |
| **BLR / H-matrix** | Structural mismatch — NN weights have no spatial smoothness. | — |
| **K=4 unroll with load+shuffle** | Our `load32_splat` is already 1 μop / 0 port-5. The 6 extra persistent A registers would spill at MR=6. Only makes sense if dropping to MR=4. | V8 `macro-assembler-shared-ia32-x64.cc:1286`: `vbroadcastss xmm, [m32]` |

---

## Decision tree for "what next"

```
Did the usize fix make it into main?
├─ NO → apply it first (5 minutes, highest ROI by far)
└─ YES ↓

Do you do int8 inference?
├─ YES → XNNPACK-style zp-folded packing [D]
└─ NO ↓

Do you have skinny K-reduction shapes (attention score·V, RoPE, [M,N]<64 with K>1024)?
├─ YES → K-splitting [B]
└─ NO ↓

Trying to squeeze the last 5-10%?
├─ Codegen fixes (pointer hoisting, K-unroll-2, offset folding) — free-ish
├─ GEMMFIP [A] — eliminates the "when to pack" heuristic
├─ RegStackify barriers [G] — 1-hour experiment
└─ Hilbert scheduling [E] — if multi-threaded scaling plateaus
```

---

## Watch for: tf.js adding relaxed-simd

Our 15–24% lead is ~half structural (`relaxed_madd` vs `mul+add`). When tf.js ships a `-wasmrelaxedsimd-fma` build (XNNPACK already has the kernels, they just don't deploy them), our lead drops to MR=6 vs MR=5 = ~20% theoretical, probably ~10% real after their K=4 unroll advantage.

Monitor: [tfjs-backend-wasm package](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm) for a fourth `.wasm` file.

---

## Key files for reference

| What | Where |
|---|---|
| XNNPACK WASM kernel selection | [`gemm-config.c:942`](https://github.com/google/XNNPACK/blob/master/src/configs/gemm-config.c) |
| XNNPACK 6×8 splat (K=4) template | [`wasmsimd-splat.c.in`](https://github.com/google/XNNPACK/blob/master/src/f32-gemm/wasmsimd-splat.c.in) |
| XNNPACK s4 rotate trick | [`f32-gemm-6x8s4-minmax-wasmrelaxedsimd-fma.c`](https://github.com/google/XNNPACK/blob/master/src/f32-gemm/gen/f32-gemm-6x8s4-minmax-wasmrelaxedsimd-fma.c) |
| XNNPACK zp-folding reference | [`src/reference/packing.cc`](https://github.com/google/XNNPACK/blob/master/src/reference/packing.cc) |
| V8 FMA lowering | [`macro-assembler-shared-ia32-x64.h:27-53`](https://github.com/v8/v8/blob/main/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.h) |
| V8 load32_splat → vbroadcastss | [`macro-assembler-shared-ia32-x64.cc:1286`](https://github.com/v8/v8/blob/main/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.cc) |
| V8 15-XMM limit | [`register-x64.h:152`](https://github.com/v8/v8/blob/main/src/codegen/x64/register-x64.h) |
| LLVM RegStackify bug | [#98631](https://github.com/llvm/llvm-project/issues/98631), fix: [#97283](https://github.com/llvm/llvm-project/pull/97283) |
| GEMMFIP paper | [arXiv:2302.08417](https://arxiv.org/abs/2302.08417) |
| Hilbert GEMM paper | [arXiv:2601.16294](https://arxiv.org/abs/2601.16294) |
| BLIS analytical tile sizing | Low et al. TOMS 2016 "Analytical Modeling Is Enough" |
| WASM prefetch rejection | [WebAssembly/simd #352](https://github.com/WebAssembly/simd/pull/352) |
| Ruy Hilbert block iteration | [`block_map.cc`](https://github.com/google/ruy/blob/master/ruy/block_map.cc) |
| Our disassembled kernel | `/tmp/rumpy.wat:5586` (strided), `:4227` (packed-A) — 105 vs 101 WASM instructions/K |
| tf.js disassembled kernel | `/tmp/tfjs.wat:func 415` — MR=5×NR=8, K-unroll=4, mul+add, **no relaxed-simd** |
