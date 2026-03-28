# pydatajs

**PyData ecosystem for JavaScript** - High-performance scientific computing and machine learning.

## Packages

| Package                                   | Description                 | Status         |
| ----------------------------------------- | --------------------------- | -------------- |
| [numpyjs](./packages/numpyjs)             | NumPy-like array operations | In Development |
| [scikitlearnjs](./packages/scikitlearnjs) | scikit-learn ML library     | Planned        |
| scipyjs                                   | SciPy scientific computing  | Planned        |
| pandasjs                                  | Pandas-like DataFrames      | Planned        |

## Installation

```bash
npm install numpyjs
npm install scikitlearnjs
```

## Features

- **NumPy-compatible API** - Familiar syntax for Python developers
- **High Performance** - WebGPU (3,272 GFLOPS), WASM SIMD (625 GFLOPS), and pure JS backends
- **Faster than tfjs** - 1.45× faster WebGPU matmul on M4, beats tfjs on iPhone too
- **TypeScript First** - Full type safety and IDE support
- **Browser & Node.js** - Works everywhere JavaScript runs

> **[Run the benchmark yourself →](https://svenflow.github.io/pydatajs/bench.html)**

## License

MIT
