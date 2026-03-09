# pydatajs

**PyData ecosystem for JavaScript** - High-performance scientific computing and machine learning library.

## Repository Structure

This is a monorepo containing multiple packages:

```
pydatajs/
├── packages/
│   ├── numpyjs/        # NumPy-like array operations (Rust/WASM + WebGPU)
│   └── scikitlearnjs/  # scikit-learn ML library
├── package.json        # Root workspace configuration
└── README.md
```

## Packages

| Package | Description | npm |
|---------|-------------|-----|
| numpyjs | NumPy-compatible array operations with Rust/WASM backend | `npm install numpyjs` |
| scikitlearnjs | Machine learning library (depends on numpyjs) | `npm install scikitlearnjs` |

## Development

### Prerequisites
- Node.js 18+
- Rust toolchain (for WASM builds)
- wasm-pack

### Build

```bash
# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test
```

## Publishing

Packages are published to npm under:
- `numpyjs`
- `scikitlearnjs`

**Publisher:** svenflow (npm account)

## License

MIT
