# WebGPU Matrix Multiplication Optimization Plan v3

## Goal
Match or beat TensorFlow.js WebGPU matmul performance across all matrix sizes.

## Current State

### Our Performance
- 512x512: 4.70ms (57 GFLOPS)
- 1024x1024: 12.00ms (179 GFLOPS)
- 2048x2048: 49.23ms (349 GFLOPS)
- 4096x4096: 308.92ms (445 GFLOPS)

### TensorFlow.js Performance
- 512x512: 1.33ms (202 GFLOPS) - 3.5x faster
- 1024x1024: 3.35ms (641 GFLOPS) - 3.6x faster
- 2048x2048: 9.60ms (1,789 GFLOPS) - 5.1x faster
- 4096x4096: 64.05ms (2,146 GFLOPS) - 4.8x faster

---

## Key Concepts & Goals

### Memory Hierarchy Optimization Goals
1. **Coalesced Global Memory Reads**: Consecutive threads should access consecutive memory addresses to maximize memory bandwidth. Achieved by transposing matrix A during shared memory load.

2. **Shared Memory Bank Conflict Avoidance**: Shared memory is divided into banks (typically 32). Strided access patterns can serialize reads, killing performance. Mitigated through padding (e.g., `tile[16][65]` instead of `tile[16][64]`).

3. **Register Blocking**: Keep partial results in fastest memory (registers) to minimize shared memory traffic.

4. **Memory Latency Hiding**: Advanced technique using double-buffering to overlap computation with memory loads.

### Autotuning Rationale
No single shader configuration is optimal for all matrix sizes and GPUs. The key to matching TensorFlow.js is a **zoo of specialized kernels** with a runtime dispatcher selecting the best one.

---

## Implementation Plan

### Phase 1: Foundational Tiling (Target: ~1,000 GFLOPS)

**Configuration:**
- Workgroup: 16×16 (256 threads)
- Output tile per workgroup: 32×32 (each thread computes 2×2)
- Inner (K) dimension tile: 32
- Shared memory: 32×33 + 32×32 = 4.2KB (padding for bank conflicts)

**Key Implementation Details:**
1. Load 32×32 tiles of A and B into shared memory
2. **Transpose A during load** to achieve coalesced global memory reads
3. **Add padding** to A tile (32×33 instead of 32×32) to avoid bank conflicts
4. workgroupBarrier() synchronization
5. Compute 2×2 output per thread from shared memory tiles
6. Bounds checking for edge cases

**WGSL Structure:**
```wgsl
const TILE_SIZE: u32 = 32u;
const TILE_SIZE_PADDED: u32 = 33u;  // Padding to avoid bank conflicts

var<workgroup> As: array<f32, 1056>;  // 32×33 (transposed, padded)
var<workgroup> Bs: array<f32, 1024>;  // 32×32

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {

  let tx = lid.x;
  let ty = lid.y;

  // Output coordinates
  let outRow = wid.y * TILE_SIZE + ty * 2u;
  let outCol = wid.x * TILE_SIZE + tx * 2u;

  // 2×2 accumulators
  var acc00: f32 = 0.0;
  var acc01: f32 = 0.0;
  var acc10: f32 = 0.0;
  var acc11: f32 = 0.0;

  let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    // Load A (transposed) into padded shared memory
    // Global A[row, k] → Shared As[k, row] with padding
    // ... loading code with bounds checking ...

    // Load B (row-major) into shared memory
    // ... loading code with bounds checking ...

    workgroupBarrier();

    // Inner loop over K dimension
    for (var k: u32 = 0u; k < TILE_SIZE; k++) {
      // Read from transposed+padded As
      let a0 = As[k * TILE_SIZE_PADDED + ty * 2u];
      let a1 = As[k * TILE_SIZE_PADDED + ty * 2u + 1u];
      let b0 = Bs[k * TILE_SIZE + tx * 2u];
      let b1 = Bs[k * TILE_SIZE + tx * 2u + 1u];

      acc00 = fma(a0, b0, acc00);
      acc01 = fma(a0, b1, acc01);
      acc10 = fma(a1, b0, acc10);
      acc11 = fma(a1, b1, acc11);
    }

    workgroupBarrier();
  }

  // Write output with bounds checking
  // ...
}
```

---

### Phase 2: Register Blocking (Target: ~2,000 GFLOPS)

**Configuration:**
- Each thread computes 4×4 output tile (16 f32 register accumulators)
- Output tile per workgroup: 64×64 (16×16 threads × 4×4 each)
- Inner (K) dimension tile: **16** (conservative for shared memory)
- Shared memory: (64×17 + 16×64) × 4 bytes = ~5.4KB

**Trade-off Analysis:**
- Smaller K tile (16 vs 32): Fewer elements loaded per barrier, but more barriers total
- This is a **key autotuning parameter** - some GPUs prefer K=32, others K=16
- The autotuner will benchmark both configurations

**Key Changes:**
1. 16 accumulators per thread in registers
2. Manually unroll inner loop for register reuse
3. Fewer global memory loads per output element (higher arithmetic intensity)
4. Add padding to shared memory tiles

---

### Phase 3: Vectorization (Target: ~2,600 GFLOPS)

**Configuration:**
- Use vec4<f32> for memory operations
- 4 floats per transaction
- Requires matrix dimensions divisible by 4 (or JS-side padding)

**Key Changes:**
1. Storage buffers: `array<vec4<f32>>`
2. Each load fetches 4 floats per transaction
3. Vectorized tile loading: `As[idx] = A_vec4[row * (K/4) + k/4];`
4. May require JS-side padding for non-divisible dimensions

---

### Phase 4: Autotuning Framework & Dispatch (Target: ~3,200+ GFLOPS)

**This phase builds the infrastructure to systematically find optimal configurations.**

#### 4.1 Offline Benchmarking Harness

Build a systematic exploration framework:

```typescript
interface TuningConfig {
  TILE_M: number;      // 32, 64, 128
  TILE_N: number;      // 32, 64, 128
  TILE_K: number;      // 8, 16, 32
  THREAD_TILE_M: number;  // 2, 4, 8
  THREAD_TILE_N: number;  // 2, 4, 8
  WORKGROUP_X: number;    // 8, 16, 32
  WORKGROUP_Y: number;    // 8, 16, 32
  USE_VEC4: boolean;
  A_TILE_PADDING: number; // 0, 1
}

async function benchmarkConfig(
  config: TuningConfig,
  testSizes: [number, number, number][]  // [M, N, K] tuples
): Promise<Map<string, number>> {
  const shader = generateShader(config);
  const pipeline = await compileShader(shader);

  const results = new Map<string, number>();
  for (const [M, N, K] of testSizes) {
    const gflops = await runBenchmark(pipeline, M, N, K);
    results.set(`${M}x${N}x${K}`, gflops);
  }
  return results;
}

// Explore parameter space
const configs: TuningConfig[] = [
  { TILE_M: 32, TILE_N: 32, TILE_K: 32, THREAD_TILE_M: 2, THREAD_TILE_N: 2, ... },
  { TILE_M: 64, TILE_N: 64, TILE_K: 16, THREAD_TILE_M: 4, THREAD_TILE_N: 4, ... },
  { TILE_M: 128, TILE_N: 32, TILE_K: 16, THREAD_TILE_M: 8, THREAD_TILE_N: 2, ... },
  // ... dozens more combinations
];

const testSizes = [
  [128, 128, 128], [256, 256, 256], [512, 512, 512],
  [1024, 1024, 1024], [2048, 2048, 2048], [4096, 4096, 4096],
  [128, 4096, 128], [4096, 128, 4096],  // Tall-skinny and short-fat
];
```

#### 4.2 Heuristic Generation

Analyze benchmark results to generate dispatch rules:

```typescript
// After running all benchmarks, find optimal config per size range
const optimalConfigs: Map<string, TuningConfig> = findOptimalConfigs(allResults);

// Output: dispatch heuristics
function generateDispatcher(): string {
  return `
    function selectShader(M: number, N: number, K: number): string {
      const maxDim = Math.max(M, N, K);
      const ratio = M / N;

      if (maxDim < 256) {
        return 'small_32x32x32_2x2';  // Low overhead for small matrices
      } else if (ratio > 4) {
        return 'tall_128x32x16_8x2';  // Optimized for tall-skinny
      } else if (ratio < 0.25) {
        return 'fat_32x128x16_2x8';   // Optimized for short-fat
      } else if (maxDim >= 2048) {
        return 'large_64x64x16_4x4_vec4';  // Maximum throughput
      } else {
        return 'medium_64x64x16_4x4';
      }
    }
  `;
}
```

#### 4.3 Pre-Built Shader Variants

Based on expected optimal configurations:

| Variant Name | TILE (M×N×K) | Thread Tile | Workgroup | Vec4 | Best For |
|--------------|--------------|-------------|-----------|------|----------|
| small_32x32x32_2x2 | 32×32×32 | 2×2 | 16×16 | No | M,N,K < 256 |
| medium_64x64x16_4x4 | 64×64×16 | 4×4 | 16×16 | No | 256-1024 |
| large_64x64x16_4x4_vec4 | 64×64×16 | 4×4 | 16×16 | Yes | ≥2048 |
| tall_128x32x16_8x2 | 128×32×16 | 8×2 | 16×16 | No | M >> N |
| fat_32x128x16_2x8 | 32×128×16 | 2×8 | 16×16 | No | M << N |

---

### Phase 5: Polish & Advanced Features

1. **f16 (Half-Precision) Support**
   - Enable `"shader-f16"` WebGPU feature
   - Theoretically **double GFLOPS** on supported hardware
   - Use behind a flag for precision-tolerant workloads

2. **Edge Case Cleanup Shader**
   - Separate shader with bounds checking for non-divisible dimensions
   - Run fast unchecked kernel on main body, checked kernel on edges

3. **Full GEMM Support**
   - Implement `C = alpha * A * B + beta * C`
   - Add alpha and beta uniforms

4. **Transposed Input Support**
   - Handle transposed A and/or B without explicit transpose
   - Adjust indexing based on transpose flags

### Phase 6 (Future): Double-Buffering for Latency Hiding

**Advanced optimization to overlap computation with memory loads:**

While computing on tile N in registers, prefetch tile N+1 from global memory into a separate shared memory buffer. This hides global memory latency behind computation.

**Implementation sketch:**
```wgsl
var<workgroup> As0: array<f32, TILE_SIZE>;  // Buffer 0
var<workgroup> As1: array<f32, TILE_SIZE>;  // Buffer 1

// Load first tile into buffer 0
loadTileA(As0, 0);
workgroupBarrier();

for (var t: u32 = 0u; t < numTiles - 1u; t++) {
  let currentBuffer = select(As0, As1, t % 2 == 0);
  let nextBuffer = select(As1, As0, t % 2 == 0);

  // Start async load of next tile
  loadTileA(nextBuffer, t + 1);

  // Compute on current tile
  computeTile(currentBuffer);

  workgroupBarrier();
}
// Handle last tile
```

---

## Success Criteria

| Metric | Target |
|--------|--------|
| 512×512 | ≤1.5ms (match tfjs) |
| 1024×1024 | ≤4ms (match tfjs) |
| 2048×2048 | ≤10ms (match tfjs) |
| 4096×4096 | <60ms (beat tfjs) |
| Peak GFLOPS | >2,500 |
| Numerical accuracy | Within 1e-4 of CPU |
| All existing tests | Pass |

---

## Implementation Checklist

- [ ] **Phase 1: Foundational Tiling**
  - [ ] Implement 32×32 tiled shader
  - [ ] Add A transpose during load (coalesced global reads)
  - [ ] Add shared memory padding (bank conflict avoidance)
  - [ ] Bounds checking for edge cases
  - [ ] Benchmark: target ~1,000 GFLOPS

- [ ] **Phase 2: Register Blocking**
  - [ ] 4×4 per-thread output (16 accumulators)
  - [ ] K tile = 16 (shared memory safe)
  - [ ] Manual fma() unrolling
  - [ ] Benchmark: target ~2,000 GFLOPS

- [ ] **Phase 3: Vectorization**
  - [ ] vec4<f32> storage buffers
  - [ ] Vectorized tile loading
  - [ ] JS-side dimension padding
  - [ ] Benchmark: target ~2,600 GFLOPS

- [ ] **Phase 4: Autotuning Framework**
  - [ ] Build offline benchmarking harness
  - [ ] Create shader template system
  - [ ] Explore parameter space (configs × sizes)
  - [ ] Generate dispatch heuristics
  - [ ] Implement runtime dispatcher
  - [ ] Target: ~3,200+ GFLOPS

- [ ] **Phase 5: Polish**
  - [ ] f16 support (optional)
  - [ ] Edge case cleanup shader
  - [ ] Full GEMM (alpha, beta)
  - [ ] Transposed input support

- [ ] **Phase 6 (Future): Double-Buffering**
  - [ ] Implement prefetching
  - [ ] Overlap load/compute

---

## References

1. Simon Boehm - CUDA Matrix Multiplication Optimization (siboehm.com)
2. WebGPU Fundamentals - Compute Shaders (webgpufundamentals.org)
3. Surma - WebGPU All of the cores (surma.dev)
4. TensorFlow.js matmul_packed_webgpu.ts (GitHub)
5. Nuss and Bolts - Optimizing WebGPU Matmul for 1TFLOP+ (nuss-and-bolts.com)
6. Salykova - Advanced Matrix Multiplication Optimization on NVIDIA GPUs
7. Gemini 2.5 Pro Review (v1, v2 feedback integrated)
