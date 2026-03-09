# tsnp (TypeScript NumPy)

High-performance NumPy-like library for TypeScript, powered by Rust/WASM and WebGPU.

## ⚠️ CRITICAL: Backend Purity Rules

### WASM Backend: 100% Rust Implementation

**ZERO JavaScript math implementations allowed.**

- ALL math operations MUST be implemented in Rust, compiled to WASM
- `wasm-backend.ts` is ONLY a thin wrapper - no math logic whatsoever
- If a method calls `Math.*` or does computation in TypeScript, it's WRONG
- If WASM module doesn't export an op, ADD IT TO RUST - never implement in TS

```typescript
// ❌ WRONG - math in TypeScript
trunc(arr: IFaceNDArray): IFaceNDArray {
  const result = new Float64Array(arr.data.length);
  for (let i = 0; i < arr.data.length; i++) {
    result[i] = Math.trunc(arr.data[i]);
  }
  return this.array(Array.from(result), arr.shape);
}

// ✅ CORRECT - delegate to WASM
trunc(arr: IFaceNDArray): IFaceNDArray {
  return this.wrap(this.wasm.truncArr(this.unwrap(arr)));
}
```

### WebGPU Backend: 100% GPU Shaders

**ZERO CPU fallbacks for math operations.**

- ALL math operations MUST have WGSL compute shaders
- `webgpu-backend.ts` contains ONLY shader orchestration code
- Sync methods can block on GPU readback, but computation is ALWAYS on GPU
- If a shader doesn't exist, WRITE ONE - never fall back to CPU

```typescript
// ❌ WRONG - CPU fallback
sin(arr: IFaceNDArray): IFaceNDArray {
  const result = new Float64Array(arr.data.length);
  for (let i = 0; i < arr.data.length; i++) {
    result[i] = Math.sin(arr.data[i]);
  }
  return this.array(Array.from(result), arr.shape);
}

// ✅ CORRECT - GPU shader
sin(arr: IFaceNDArray): IFaceNDArray {
  return this.runElementwiseShader('sin', arr);
}
```

### Adding New Operations

1. Add method signature to `Backend` interface in `test-utils.ts`
2. TypeScript compiler errors show what's missing in each backend
3. **WASM**: Add to `crates/rumpy-wasm/src/lib.rs`, rebuild with `wasm-pack build`
4. **WebGPU**: Add WGSL shader to `webgpu-backend.ts`
5. Add test to appropriate test file (runs against ALL backends)
6. **NEVER** implement math in the TypeScript wrapper layer

## Test Parameterization

Tests MUST run against ALL backends with identical expectations:

```typescript
// ✅ CORRECT - parameterized test
describe.each(backends)('$name backend', (B) => {
  it('computes sin', async () => {
    const arr = B.array([0, Math.PI/2, Math.PI], [3]);
    const result = B.sin(arr);
    expect(await getData(result, B)).toEqual([0, 1, 0]);
  });
});

// ❌ WRONG - backend-specific test
describe('wasm backend', () => {
  it('computes sin', () => { /* wasm-specific */ });
});
describe('webgpu backend', () => {
  it('computes sin', () => { /* different implementation */ });
});
```

**If a backend can't pass a test, FIX THE BACKEND - don't skip the test.**

## Development

**Always use `bun`, never raw `npm`:**

```bash
bun install
bun run build
bun test
```

## Build Commands

```bash
# Build WASM (from project root)
wasm-pack build crates/rumpy-wasm --target web

# Copy WASM to tests directory
cp -r crates/rumpy-wasm/pkg tests/wasm-pkg

# Run tests
cd tests && bun test
```

## Architecture

```
tsnp/
├── crates/
│   ├── rumpy-core/      # Backend traits and common types
│   ├── rumpy-cpu/       # CPU backend (ndarray/faer)
│   ├── rumpy-wasm/      # WASM bindings - ALL math here
│   └── pthreadpool-rs/  # Thread pool for parallel computation
├── tests/
│   ├── test-utils.ts    # Backend interface (185 methods)
│   ├── wasm-backend.ts  # THIN wrapper only - no math
│   ├── webgpu-backend.ts # Shader orchestration only
│   ├── *.test.ts        # Parameterized tests
│   └── wasm-pkg/        # Compiled WASM module
└── benchmarks/
```

## WebGPU Performance

Beats tfjs-webgpu at all matrix sizes (up to 2.56x faster at 4096x4096).

Key insights:
- Store A as vec4 along K dimension (not M)
- B-value register caching for ILP
- Autotune selects optimal shader per size

**CRITICAL: Test WebGPU in native Chrome, NOT Playwright (adds ~15-20% overhead)**

## Checklist Before Committing

- [ ] `bun test` passes with 0 failures
- [ ] No `Math.*` calls in wasm-backend.ts (except test utilities)
- [ ] No CPU loops over array data in webgpu-backend.ts math methods
- [ ] All new ops added to Backend interface
- [ ] Tests are parameterized across all backends
