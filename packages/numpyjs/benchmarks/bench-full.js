#!/usr/bin/env node
/**
 * Comprehensive Node.js benchmark runner
 * Tests rumpy-ts WASM vs ml-matrix, mathjs, ndarray, native JS
 * Outputs JSON file with all results
 */

import { writeFileSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { Matrix, inverse } from 'ml-matrix';
import * as math from 'mathjs';
import ndarray from 'ndarray';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_FILE = join(__dirname, 'benchmark-results.json');

// Load rumpy-ts WASM (auto-loads in nodejs target)
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const rumpy = require('./pkg/rumpy_wasm.js');

function benchmark(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 5; i++) fn();

  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);
  const trimmed = times.slice(Math.floor(times.length * 0.05), Math.floor(times.length * 0.95));
  const mean = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  const std = Math.sqrt(trimmed.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) / trimmed.length);

  return { mean, std, min: trimmed[0], max: trimmed[trimmed.length - 1] };
}

function randomArray(size) {
  return new Float64Array(size).map(() => Math.random());
}

async function runBenchmarks() {
  const size = 10000;
  const iterations = 100;
  const matrixSize = 100;

  console.log(`\n=== rumpy-ts Comprehensive Benchmark ===`);
  console.log(`Array size: ${size}, Matrix: ${matrixSize}x${matrixSize}, Iterations: ${iterations}\n`);

  const results = {
    timestamp: new Date().toISOString(),
    config: { size, matrixSize, iterations },
    libraries: ['rumpy-ts', 'ml-matrix', 'mathjs', 'ndarray', 'native'],
    operations: {}
  };

  const data1 = randomArray(size);
  const data2 = randomArray(size);

  // ============ Element-wise Operations ============
  console.log('--- Element-wise Operations ---');

  // Create rumpy-ts arrays
  const rArr1 = rumpy.arrayFromTyped(data1, new Uint32Array([size]));
  const rArr2 = rumpy.arrayFromTyped(data2, new Uint32Array([size]));

  // ml-matrix
  const m1 = Matrix.from1DArray(size, 1, Array.from(data1));
  const m2 = Matrix.from1DArray(size, 1, Array.from(data2));

  // mathjs
  const mj1 = Array.from(data1);
  const mj2 = Array.from(data2);

  // ndarray
  const nd1 = ndarray(data1, [size]);
  const nd2 = ndarray(data2, [size]);

  // ADD
  results.operations['add'] = {};

  const addRumpy = benchmark('rumpy-ts add', () => {
    const result = rArr1.add(rArr2);
    result.free();
  }, iterations);
  console.log(`rumpy-ts add: ${addRumpy.mean.toFixed(3)}ms ± ${addRumpy.std.toFixed(3)}ms`);
  results.operations['add']['rumpy-ts'] = addRumpy.mean;

  const addMl = benchmark('ml-matrix add', () => Matrix.add(m1, m2), iterations);
  console.log(`ml-matrix add: ${addMl.mean.toFixed(3)}ms ± ${addMl.std.toFixed(3)}ms`);
  results.operations['add']['ml-matrix'] = addMl.mean;

  const addMath = benchmark('mathjs add', () => math.add(mj1, mj2), iterations);
  console.log(`mathjs add: ${addMath.mean.toFixed(3)}ms ± ${addMath.std.toFixed(3)}ms`);
  results.operations['add']['mathjs'] = addMath.mean;

  const addNd = benchmark('ndarray add', () => {
    const result = ndarray(new Float64Array(size), [size]);
    for (let i = 0; i < size; i++) result.set(i, nd1.get(i) + nd2.get(i));
  }, iterations);
  console.log(`ndarray add: ${addNd.mean.toFixed(3)}ms ± ${addNd.std.toFixed(3)}ms`);
  results.operations['add']['ndarray'] = addNd.mean;

  const addNative = benchmark('native add', () => {
    const result = new Float64Array(size);
    for (let i = 0; i < size; i++) result[i] = data1[i] + data2[i];
  }, iterations);
  console.log(`native add: ${addNative.mean.toFixed(3)}ms ± ${addNative.std.toFixed(3)}ms`);
  results.operations['add']['native'] = addNative.mean;

  // MUL
  results.operations['mul'] = {};

  const mulRumpy = benchmark('rumpy-ts mul', () => {
    const result = rArr1.mul(rArr2);
    result.free();
  }, iterations);
  console.log(`rumpy-ts mul: ${mulRumpy.mean.toFixed(3)}ms ± ${mulRumpy.std.toFixed(3)}ms`);
  results.operations['mul']['rumpy-ts'] = mulRumpy.mean;

  const mulMl = benchmark('ml-matrix mul', () => Matrix.mul(m1, m2), iterations);
  console.log(`ml-matrix mul: ${mulMl.mean.toFixed(3)}ms ± ${mulMl.std.toFixed(3)}ms`);
  results.operations['mul']['ml-matrix'] = mulMl.mean;

  const mulMath = benchmark('mathjs mul', () => math.dotMultiply(mj1, mj2), iterations);
  console.log(`mathjs mul: ${mulMath.mean.toFixed(3)}ms ± ${mulMath.std.toFixed(3)}ms`);
  results.operations['mul']['mathjs'] = mulMath.mean;

  const mulNative = benchmark('native mul', () => {
    const result = new Float64Array(size);
    for (let i = 0; i < size; i++) result[i] = data1[i] * data2[i];
  }, iterations);
  console.log(`native mul: ${mulNative.mean.toFixed(3)}ms ± ${mulNative.std.toFixed(3)}ms`);
  results.operations['mul']['native'] = mulNative.mean;

  // Note: rumpy-ts trig/exp/log ops have a WASM libm issue - using intrinsics not available in WASM
  // See: https://docs.rs/libm/latest/libm/ - need to add libm feature to fix

  // ============ Matrix Operations ============
  console.log('\n--- Matrix Operations ---');
  const n = matrixSize;
  const matData1 = randomArray(n * n);
  const matData2 = randomArray(n * n);

  // rumpy-ts matrices
  const rMat1 = rumpy.arrayFromTyped(matData1, new Uint32Array([n, n]));
  const rMat2 = rumpy.arrayFromTyped(matData2, new Uint32Array([n, n]));

  // ml-matrix matrices
  const mat1 = new Matrix(n, n);
  const mat2 = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      mat1.set(i, j, matData1[i * n + j]);
      mat2.set(i, j, matData2[i * n + j]);
    }
  }

  // mathjs matrices
  const arr1 = [];
  const arr2 = [];
  for (let i = 0; i < n; i++) {
    arr1.push(Array.from(matData1.slice(i * n, (i + 1) * n)));
    arr2.push(Array.from(matData2.slice(i * n, (i + 1) * n)));
  }

  // MATMUL
  results.operations['matmul'] = {};

  const matmulRumpy = benchmark('rumpy-ts matmul', () => {
    const result = rumpy.matmul(rMat1, rMat2);
    result.free();
  }, Math.floor(iterations / 2));
  console.log(`rumpy-ts matmul: ${matmulRumpy.mean.toFixed(3)}ms ± ${matmulRumpy.std.toFixed(3)}ms`);
  results.operations['matmul']['rumpy-ts'] = matmulRumpy.mean;

  const matmulMl = benchmark('ml-matrix matmul', () => mat1.mmul(mat2), Math.floor(iterations / 2));
  console.log(`ml-matrix matmul: ${matmulMl.mean.toFixed(3)}ms ± ${matmulMl.std.toFixed(3)}ms`);
  results.operations['matmul']['ml-matrix'] = matmulMl.mean;

  const matmulMath = benchmark('mathjs matmul', () => math.multiply(arr1, arr2), Math.floor(iterations / 2));
  console.log(`mathjs matmul: ${matmulMath.mean.toFixed(3)}ms ± ${matmulMath.std.toFixed(3)}ms`);
  results.operations['matmul']['mathjs'] = matmulMath.mean;

  // TRANSPOSE
  results.operations['transpose'] = {};

  const transposeRumpy = benchmark('rumpy-ts transpose', () => {
    const result = rMat1.transpose();
    result.free();
  }, iterations);
  console.log(`rumpy-ts transpose: ${transposeRumpy.mean.toFixed(3)}ms ± ${transposeRumpy.std.toFixed(3)}ms`);
  results.operations['transpose']['rumpy-ts'] = transposeRumpy.mean;

  const transposeMl = benchmark('ml-matrix transpose', () => mat1.transpose(), iterations);
  console.log(`ml-matrix transpose: ${transposeMl.mean.toFixed(3)}ms ± ${transposeMl.std.toFixed(3)}ms`);
  results.operations['transpose']['ml-matrix'] = transposeMl.mean;

  const transposeMath = benchmark('mathjs transpose', () => math.transpose(arr1), iterations);
  console.log(`mathjs transpose: ${transposeMath.mean.toFixed(3)}ms ± ${transposeMath.std.toFixed(3)}ms`);
  results.operations['transpose']['mathjs'] = transposeMath.mean;

  // ============ Reductions ============
  console.log('\n--- Reductions ---');

  // SUM
  results.operations['sum'] = {};

  const sumRumpy = benchmark('rumpy-ts sum', () => rArr1.sum(), iterations);
  console.log(`rumpy-ts sum: ${sumRumpy.mean.toFixed(3)}ms ± ${sumRumpy.std.toFixed(3)}ms`);
  results.operations['sum']['rumpy-ts'] = sumRumpy.mean;

  const sumMath = benchmark('mathjs sum', () => math.sum(mj1), iterations);
  console.log(`mathjs sum: ${sumMath.mean.toFixed(3)}ms ± ${sumMath.std.toFixed(3)}ms`);
  results.operations['sum']['mathjs'] = sumMath.mean;

  const sumNative = benchmark('native sum', () => data1.reduce((a, b) => a + b, 0), iterations);
  console.log(`native sum: ${sumNative.mean.toFixed(3)}ms ± ${sumNative.std.toFixed(3)}ms`);
  results.operations['sum']['native'] = sumNative.mean;

  // MEAN
  results.operations['mean'] = {};

  const meanRumpy = benchmark('rumpy-ts mean', () => rArr1.mean(), iterations);
  console.log(`rumpy-ts mean: ${meanRumpy.mean.toFixed(3)}ms ± ${meanRumpy.std.toFixed(3)}ms`);
  results.operations['mean']['rumpy-ts'] = meanRumpy.mean;

  const meanMath = benchmark('mathjs mean', () => math.mean(mj1), iterations);
  console.log(`mathjs mean: ${meanMath.mean.toFixed(3)}ms ± ${meanMath.std.toFixed(3)}ms`);
  results.operations['mean']['mathjs'] = meanMath.mean;

  // MAX
  results.operations['max'] = {};

  const maxRumpy = benchmark('rumpy-ts max', () => rArr1.max(), iterations);
  console.log(`rumpy-ts max: ${maxRumpy.mean.toFixed(3)}ms ± ${maxRumpy.std.toFixed(3)}ms`);
  results.operations['max']['rumpy-ts'] = maxRumpy.mean;

  const maxMath = benchmark('mathjs max', () => math.max(mj1), iterations);
  console.log(`mathjs max: ${maxMath.mean.toFixed(3)}ms ± ${maxMath.std.toFixed(3)}ms`);
  results.operations['max']['mathjs'] = maxMath.mean;

  const maxNative = benchmark('native max', () => Math.max(...data1), iterations);
  console.log(`native max: ${maxNative.mean.toFixed(3)}ms ± ${maxNative.std.toFixed(3)}ms`);
  results.operations['max']['native'] = maxNative.mean;

  // ============ Linear Regression Pipeline ============
  console.log('\n--- Linear Regression Pipeline ---');

  // Generate synthetic data: y = 3x + 2 + noise
  const nSamples = 1000;
  const X = new Float64Array(nSamples);
  const y = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    X[i] = Math.random() * 10;
    y[i] = 3 * X[i] + 2 + (Math.random() - 0.5) * 2;
  }

  results.operations['linear-regression'] = {};

  // rumpy-ts linear regression (closed-form solution: w = (X^T X)^-1 X^T y)
  const lrRumpy = benchmark('rumpy-ts linreg', () => {
    // Add bias column to X
    const XwBias = new Float64Array(nSamples * 2);
    for (let i = 0; i < nSamples; i++) {
      XwBias[i * 2] = X[i];
      XwBias[i * 2 + 1] = 1;
    }
    const Xmat = rumpy.arrayFromTyped(XwBias, new Uint32Array([nSamples, 2]));
    const Ymat = rumpy.arrayFromTyped(y, new Uint32Array([nSamples, 1]));

    // w = (X^T X)^-1 X^T y
    const Xt = Xmat.transpose();
    const XtX = rumpy.matmul(Xt, Xmat);
    const XtXinv = rumpy.inv(XtX);
    const XtY = rumpy.matmul(Xt, Ymat);
    const w = rumpy.matmul(XtXinv, XtY);

    // Cleanup
    Xmat.free();
    Ymat.free();
    Xt.free();
    XtX.free();
    XtXinv.free();
    XtY.free();
    w.free();
  }, Math.floor(iterations / 2));
  console.log(`rumpy-ts linreg: ${lrRumpy.mean.toFixed(3)}ms ± ${lrRumpy.std.toFixed(3)}ms`);
  results.operations['linear-regression']['rumpy-ts'] = lrRumpy.mean;

  // ml-matrix linear regression
  const lrMl = benchmark('ml-matrix linreg', () => {
    const Xmat = new Matrix(nSamples, 2);
    for (let i = 0; i < nSamples; i++) {
      Xmat.set(i, 0, X[i]);
      Xmat.set(i, 1, 1);
    }
    const Ymat = Matrix.columnVector(Array.from(y));

    // w = (X^T X)^-1 X^T y
    const Xt = Xmat.transpose();
    const XtX = Xt.mmul(Xmat);
    const XtXinv = inverse(XtX);
    const XtY = Xt.mmul(Ymat);
    const w = XtXinv.mmul(XtY);
  }, Math.floor(iterations / 2));
  console.log(`ml-matrix linreg: ${lrMl.mean.toFixed(3)}ms ± ${lrMl.std.toFixed(3)}ms`);
  results.operations['linear-regression']['ml-matrix'] = lrMl.mean;

  // mathjs linear regression
  const lrMath = benchmark('mathjs linreg', () => {
    const Xmat = [];
    for (let i = 0; i < nSamples; i++) {
      Xmat.push([X[i], 1]);
    }
    const Ymat = Array.from(y).map(v => [v]);

    // w = (X^T X)^-1 X^T y
    const Xt = math.transpose(Xmat);
    const XtX = math.multiply(Xt, Xmat);
    const XtXinv = math.inv(XtX);
    const XtY = math.multiply(Xt, Ymat);
    const w = math.multiply(XtXinv, XtY);
  }, Math.floor(iterations / 2));
  console.log(`mathjs linreg: ${lrMath.mean.toFixed(3)}ms ± ${lrMath.std.toFixed(3)}ms`);
  results.operations['linear-regression']['mathjs'] = lrMath.mean;

  // ============ Write Results ============
  writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\n✅ Results written to ${OUTPUT_FILE}`);

  // ============ Summary ============
  console.log('\n=== SUMMARY (times in ms, lower is better) ===\n');

  const libs = ['rumpy-ts', 'ml-matrix', 'mathjs', 'ndarray', 'native'];
  const colWidth = 12;

  // Header
  console.log('Operation'.padEnd(18) + libs.map(l => l.padEnd(colWidth)).join(''));
  console.log('-'.repeat(18 + libs.length * colWidth));

  // Find fastest for each operation
  for (const [op, times] of Object.entries(results.operations)) {
    const values = libs.map(lib => times[lib]);
    const validValues = values.filter(v => v !== undefined);
    const fastest = Math.min(...validValues);

    const row = libs.map(lib => {
      const val = times[lib];
      if (val === undefined) return '—'.padEnd(colWidth);
      const formatted = val.toFixed(3);
      if (val === fastest && validValues.length > 1) {
        return `${formatted}*`.padEnd(colWidth);
      }
      return formatted.padEnd(colWidth);
    });

    console.log(op.padEnd(18) + row.join(''));
  }

  console.log('\n* = fastest\n');

  // Cleanup
  rArr1.free();
  rArr2.free();
  rMat1.free();
  rMat2.free();

  return results;
}

runBenchmarks().catch(console.error);
