#!/usr/bin/env node
/**
 * Node.js benchmark runner (no browser needed)
 * Tests ml-matrix, mathjs, native JS against rumpy-ts WASM
 */

import { writeFileSync } from 'fs';
import { Matrix } from 'ml-matrix';
import * as math from 'mathjs';

const OUTPUT_FILE = 'benchmark-results.json';

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

  console.log(`\n=== rumpy-ts Benchmark (Node.js) ===`);
  console.log(`Array size: ${size}, Matrix: ${matrixSize}x${matrixSize}, Iterations: ${iterations}\n`);

  const results = {
    timestamp: new Date().toISOString(),
    config: { size, matrixSize, iterations },
    operations: {}
  };

  const data1 = randomArray(size);
  const data2 = randomArray(size);

  // ============ Element-wise Operations ============
  console.log('--- Element-wise Operations ---');

  // ml-matrix
  const m1 = Matrix.from1DArray(size, 1, Array.from(data1));
  const m2 = Matrix.from1DArray(size, 1, Array.from(data2));

  const addMl = benchmark('ml-matrix add', () => Matrix.add(m1, m2), iterations);
  console.log(`ml-matrix add: ${addMl.mean.toFixed(3)}ms ± ${addMl.std.toFixed(3)}ms`);
  results.operations['add'] = { 'ml-matrix': addMl.mean };

  const mulMl = benchmark('ml-matrix mul', () => Matrix.mul(m1, m2), iterations);
  console.log(`ml-matrix mul: ${mulMl.mean.toFixed(3)}ms ± ${mulMl.std.toFixed(3)}ms`);
  results.operations['mul'] = { 'ml-matrix': mulMl.mean };

  // mathjs
  const mj1 = Array.from(data1);
  const mj2 = Array.from(data2);

  const addMath = benchmark('mathjs add', () => math.add(mj1, mj2), iterations);
  console.log(`mathjs add: ${addMath.mean.toFixed(3)}ms ± ${addMath.std.toFixed(3)}ms`);
  results.operations['add']['mathjs'] = addMath.mean;

  const mulMath = benchmark('mathjs mul', () => math.dotMultiply(mj1, mj2), iterations);
  console.log(`mathjs mul: ${mulMath.mean.toFixed(3)}ms ± ${mulMath.std.toFixed(3)}ms`);
  results.operations['mul']['mathjs'] = mulMath.mean;

  // Native JS
  const addNative = benchmark('native add', () => {
    const result = new Float64Array(size);
    for (let i = 0; i < size; i++) result[i] = data1[i] + data2[i];
  }, iterations);
  console.log(`native add: ${addNative.mean.toFixed(3)}ms ± ${addNative.std.toFixed(3)}ms`);
  results.operations['add']['native'] = addNative.mean;

  // ============ Matrix Operations ============
  console.log('\n--- Matrix Operations ---');
  const n = matrixSize;
  const matData1 = randomArray(n * n);
  const matData2 = randomArray(n * n);

  const mat1 = new Matrix(n, n);
  const mat2 = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      mat1.set(i, j, matData1[i * n + j]);
      mat2.set(i, j, matData2[i * n + j]);
    }
  }

  const matmulMl = benchmark('ml-matrix matmul', () => mat1.mmul(mat2), Math.floor(iterations / 2));
  console.log(`ml-matrix matmul: ${matmulMl.mean.toFixed(3)}ms ± ${matmulMl.std.toFixed(3)}ms`);
  results.operations['matmul'] = { 'ml-matrix': matmulMl.mean };

  const transposeMl = benchmark('ml-matrix transpose', () => mat1.transpose(), iterations);
  console.log(`ml-matrix transpose: ${transposeMl.mean.toFixed(3)}ms ± ${transposeMl.std.toFixed(3)}ms`);
  results.operations['transpose'] = { 'ml-matrix': transposeMl.mean };

  // mathjs matmul
  const arr1 = [];
  const arr2 = [];
  for (let i = 0; i < n; i++) {
    arr1.push(Array.from(matData1.slice(i * n, (i + 1) * n)));
    arr2.push(Array.from(matData2.slice(i * n, (i + 1) * n)));
  }

  const matmulMath = benchmark('mathjs matmul', () => math.multiply(arr1, arr2), Math.floor(iterations / 2));
  console.log(`mathjs matmul: ${matmulMath.mean.toFixed(3)}ms ± ${matmulMath.std.toFixed(3)}ms`);
  results.operations['matmul']['mathjs'] = matmulMath.mean;

  const transposeMath = benchmark('mathjs transpose', () => math.transpose(arr1), iterations);
  console.log(`mathjs transpose: ${transposeMath.mean.toFixed(3)}ms ± ${transposeMath.std.toFixed(3)}ms`);
  results.operations['transpose']['mathjs'] = transposeMath.mean;

  // ============ Reductions ============
  console.log('\n--- Reductions ---');

  const sumMath = benchmark('mathjs sum', () => math.sum(mj1), iterations);
  console.log(`mathjs sum: ${sumMath.mean.toFixed(3)}ms ± ${sumMath.std.toFixed(3)}ms`);
  results.operations['sum'] = { 'mathjs': sumMath.mean };

  const meanMath = benchmark('mathjs mean', () => math.mean(mj1), iterations);
  console.log(`mathjs mean: ${meanMath.mean.toFixed(3)}ms ± ${meanMath.std.toFixed(3)}ms`);
  results.operations['mean'] = { 'mathjs': meanMath.mean };

  const sumNative = benchmark('native sum', () => data1.reduce((a, b) => a + b, 0), iterations);
  console.log(`native sum: ${sumNative.mean.toFixed(3)}ms ± ${sumNative.std.toFixed(3)}ms`);
  results.operations['sum']['native'] = sumNative.mean;

  // Write results
  writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\nResults written to ${OUTPUT_FILE}`);

  // Summary
  console.log('\n=== SUMMARY ===');
  console.log('Operation | ml-matrix | mathjs | native');
  console.log('-'.repeat(50));
  for (const [op, times] of Object.entries(results.operations)) {
    const vals = [
      times['ml-matrix'] ? `${times['ml-matrix'].toFixed(3)}` : '—',
      times['mathjs'] ? `${times['mathjs'].toFixed(3)}` : '—',
      times['native'] ? `${times['native'].toFixed(3)}` : '—'
    ];
    console.log(`${op.padEnd(12)} | ${vals[0].padEnd(10)} | ${vals[1].padEnd(10)} | ${vals[2]}`);
  }

  return results;
}

runBenchmarks().catch(console.error);
