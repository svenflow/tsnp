# torchjs

PyTorch-style neural network operations for JavaScript - torch.nn.functional equivalent.

## Philosophy

**This package mirrors PyTorch's torch.nn.functional API.** It provides low-level neural network primitives for inference, not the full PyTorch training framework.

## Package Structure

```
torchjs/
├── src/
│   ├── index.ts           # Unified API + pure JS fallback
│   ├── types.ts           # Tensor and backend interfaces
│   ├── webgpu-backend.ts  # WebGPU compute shaders (WGSL)
│   └── wasm-backend.ts    # Rust/WASM wrapper
├── crates/
│   └── torchjs-wasm/      # Rust WASM implementation
│       └── src/lib.rs
├── Cargo.toml             # Rust workspace
└── package.json           # npm package
```

## Dual Backend Architecture

**WASM + WebGPU** - every op should have both implementations:

1. **WASM (torchjs-wasm)**: Rust compiled to WebAssembly. Works everywhere.
2. **WebGPU (webgpu-backend.ts)**: GPU compute shaders in WGSL. Fast on browsers with WebGPU support.

### Backend Selection

```typescript
import { nn, createWebGPUBackend } from 'torchjs';

// Pure JS fallback (works anywhere)
const output = nn.relu(input);
const conv = nn.conv2d(input, weight, bias, [1, 1], [1, 1]);

// WebGPU backend (GPU acceleration)
const gpuBackend = await createWebGPUBackend();
const gpuOutput = gpuBackend.relu(input);
```

## Available Operations

### Activation Functions
- `relu`, `relu6`, `leakyRelu`, `gelu`, `sigmoid`, `softmax`, `prelu`

### Convolutions
- `conv2d` - 2D convolution with stride, padding, bias
- `depthwiseConv2d` - Depthwise separable convolution (MobileNet-style)

### Pooling
- `maxPool2d`, `avgPool2d` - Strided 2D pooling
- `globalAvgPool2d` - Global average pooling

### Normalization
- `batchNorm` - Batch normalization (inference mode)

### Image Processing
- `resizeBilinear` - Bilinear interpolation resize

### Tensor Creation
- `tensor(data, shape)` - Create tensor from array
- `zeros(shape)`, `ones(shape)` - Initialize tensors
- `rand(shape)`, `randn(shape)` - Random tensors

## Quick Start

```typescript
import { nn, tensor, createWebGPUBackend } from 'torchjs';

// Create input tensor [N, C, H, W]
const input = tensor(imageData, [1, 3, 224, 224]);
const weight = tensor(convWeights, [64, 3, 3, 3]);
const bias = tensor(convBias, [64]);

// Convolution + activation
const conv = nn.conv2d(input, weight, bias, [1, 1], [1, 1]);
const activated = nn.relu(conv);

// Batch normalization
const normalized = nn.batchNorm(
  activated, gamma, beta, runningMean, runningVar, 1e-5
);

// Max pooling
const pooled = nn.maxPool2d(normalized, [2, 2], [2, 2], [0, 0]);
```

## WebGPU Backend Details

The WebGPU backend uses WGSL compute shaders for GPU-accelerated operations:

- **Data Format**: All tensors stored as f32 in GPUBuffer (NCHW layout)
- **Async Operations**: Shaders run asynchronously; use `await tensor.getData()` to read results
- **Shader Caching**: Pipelines cached for repeated ops
- **Workgroup Sizes**: Optimized per-op (8x8 for conv/pool, 256 for elementwise)

### GPU Tensor API

```typescript
import { WebGPUTensor, createWebGPUBackend } from 'torchjs';

const backend = await createWebGPUBackend();
const gpuTensor = WebGPUTensor.fromArray(cpuData, [1, 64, 56, 56], device);
const result = backend.conv2d(gpuTensor, weight, bias, [1, 1], [1, 1]);
const cpuData = await result.getData();  // Async readback
```

## Why torchjs exists

numpyjs and scikitlearnjs mirror Python's numpy and sklearn 1:1. But neither has:
- conv2d (not in numpy, not in sklearn)
- batch_norm (not in numpy, not in sklearn)
- pooling layers (not in numpy, sklearn MLP has none)
- prelu activation (not in numpy, sklearn MLPs use relu/tanh/logistic only)

For MediaPipe HandPose inference and similar CNN models, you need PyTorch-style ops. That's torchjs.

## What Does NOT Belong Here

- Full PyTorch training API (autograd, optimizer, etc.)
- Layer classes (nn.Conv2d, nn.Linear) - use functional ops directly
- Model definitions - build your own inference pipeline
- Matmul, transpose, reshape - those are in numpyjs

## Building

```bash
# TypeScript
cd packages/torchjs
pnpm build          # Build TS → dist/

# Rust/WASM
pnpm build:wasm     # Build Rust → pkg/
# Or manually:
wasm-pack build crates/torchjs-wasm --target web --out-dir ../../pkg
```

## Testing Methodology

Tests verify JS outputs match PyTorch outputs exactly:

1. Run operation in PyTorch with specific inputs
2. Copy exact numerical output
3. Hardcode in JS test as expected values
4. Test JS implementation against those values

```bash
# Get expected values from PyTorch
uv run python3 -c "
import torch
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print('relu output:', F.relu(x).tolist())
"

# Run tests
pnpm test
```

Use `toBeCloseTo(expected, 5)` for f32 comparisons (5 decimal places).

## Checklist Before Committing

- [ ] `pnpm typecheck` passes
- [ ] `pnpm build` succeeds
- [ ] `pnpm test` passes
- [ ] `cargo check` passes in torchjs/
- [ ] API matches torch.nn.functional style
- [ ] Both WASM and WebGPU backends implement the op
- [ ] Tests added with PyTorch-verified expected values
- [ ] WebGPU shader tested in browser
