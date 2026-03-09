/**
 * rumpy-ts Comprehensive Benchmark Suite
 *
 * Compares:
 * - rumpy-ts (Rust/WASM)
 * - TensorFlow.js (WASM + WebGL)
 * - ml-matrix (pure JS)
 * - mathjs (pure JS)
 * - ndarray (pure JS)
 */

import * as tf from '@tensorflow/tfjs';
import { Matrix } from 'ml-matrix';
import * as math from 'mathjs';
import ndarray from 'ndarray';

// Import rumpy-ts (will be built separately)
let rumpy = null;

const COLORS = {
  rumpy: '#6366f1',
  tfjs: '#22c55e',
  mlMatrix: '#f59e0b',
  mathjs: '#ef4444',
  ndarray: '#8b5cf6'
};

const results = {
  elementwise: {},
  matrix: {},
  reductions: {},
  pipeline: {}
};

let charts = {};

function log(msg) {
  const status = document.getElementById('status');
  status.textContent += '\n' + msg;
  status.scrollTop = status.scrollHeight;
  console.log(msg);
}

function clearLog() {
  document.getElementById('status').textContent = '';
}

// Benchmark utilities
function benchmark(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 5; i++) fn();

  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }

  // Remove outliers (top/bottom 5%)
  times.sort((a, b) => a - b);
  const trimmed = times.slice(Math.floor(times.length * 0.05), Math.floor(times.length * 0.95));

  const mean = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  const std = Math.sqrt(trimmed.map(t => (t - mean) ** 2).reduce((a, b) => a + b, 0) / trimmed.length);

  return { mean, std, min: trimmed[0], max: trimmed[trimmed.length - 1] };
}

// Generate random data
function randomArray(size) {
  return new Float64Array(size).map(() => Math.random());
}

function randomMatrix(rows, cols) {
  return new Float64Array(rows * cols).map(() => Math.random());
}

// =====================
// Benchmark Functions
// =====================

async function benchmarkElementwise(size, iterations) {
  const data1 = randomArray(size);
  const data2 = randomArray(size);
  const results = {};

  log(`\n--- Element-wise Operations (${size} elements, ${iterations} iterations) ---`);

  // TensorFlow.js
  try {
    const tf1 = tf.tensor1d(data1);
    const tf2 = tf.tensor1d(data2);

    results.add_tfjs = benchmark('tf.add', () => {
      const r = tf.add(tf1, tf2);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js add: ${results.add_tfjs.mean.toFixed(3)}ms ± ${results.add_tfjs.std.toFixed(3)}ms`);

    results.mul_tfjs = benchmark('tf.mul', () => {
      const r = tf.mul(tf1, tf2);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js mul: ${results.mul_tfjs.mean.toFixed(3)}ms ± ${results.mul_tfjs.std.toFixed(3)}ms`);

    results.exp_tfjs = benchmark('tf.exp', () => {
      const r = tf.exp(tf1);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js exp: ${results.exp_tfjs.mean.toFixed(3)}ms ± ${results.exp_tfjs.std.toFixed(3)}ms`);

    tf1.dispose();
    tf2.dispose();
  } catch (e) {
    log(`TensorFlow.js error: ${e.message}`);
  }

  // ml-matrix
  try {
    const m1 = Matrix.from1DArray(size, 1, Array.from(data1));
    const m2 = Matrix.from1DArray(size, 1, Array.from(data2));

    results.add_mlMatrix = benchmark('ml-matrix.add', () => {
      Matrix.add(m1, m2);
    }, iterations);
    log(`ml-matrix add: ${results.add_mlMatrix.mean.toFixed(3)}ms ± ${results.add_mlMatrix.std.toFixed(3)}ms`);

    results.mul_mlMatrix = benchmark('ml-matrix.mul', () => {
      Matrix.mul(m1, m2);
    }, iterations);
    log(`ml-matrix mul: ${results.mul_mlMatrix.mean.toFixed(3)}ms ± ${results.mul_mlMatrix.std.toFixed(3)}ms`);
  } catch (e) {
    log(`ml-matrix error: ${e.message}`);
  }

  // mathjs
  try {
    const mj1 = Array.from(data1);
    const mj2 = Array.from(data2);

    results.add_mathjs = benchmark('mathjs.add', () => {
      math.add(mj1, mj2);
    }, iterations);
    log(`mathjs add: ${results.add_mathjs.mean.toFixed(3)}ms ± ${results.add_mathjs.std.toFixed(3)}ms`);

    results.mul_mathjs = benchmark('mathjs.multiply', () => {
      math.dotMultiply(mj1, mj2);
    }, iterations);
    log(`mathjs mul: ${results.mul_mathjs.mean.toFixed(3)}ms ± ${results.mul_mathjs.std.toFixed(3)}ms`);
  } catch (e) {
    log(`mathjs error: ${e.message}`);
  }

  // Native JS (baseline)
  results.add_native = benchmark('native add', () => {
    const result = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      result[i] = data1[i] + data2[i];
    }
  }, iterations);
  log(`Native JS add: ${results.add_native.mean.toFixed(3)}ms ± ${results.add_native.std.toFixed(3)}ms`);

  return results;
}

async function benchmarkMatrix(matrixSize, iterations) {
  const n = matrixSize;
  const data1 = randomMatrix(n, n);
  const data2 = randomMatrix(n, n);
  const results = {};

  log(`\n--- Matrix Operations (${n}x${n}, ${iterations} iterations) ---`);

  // TensorFlow.js
  try {
    const tf1 = tf.tensor2d(data1, [n, n]);
    const tf2 = tf.tensor2d(data2, [n, n]);

    results.matmul_tfjs = benchmark('tf.matMul', () => {
      const r = tf.matMul(tf1, tf2);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js matmul: ${results.matmul_tfjs.mean.toFixed(3)}ms ± ${results.matmul_tfjs.std.toFixed(3)}ms`);

    results.transpose_tfjs = benchmark('tf.transpose', () => {
      const r = tf.transpose(tf1);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js transpose: ${results.transpose_tfjs.mean.toFixed(3)}ms ± ${results.transpose_tfjs.std.toFixed(3)}ms`);

    tf1.dispose();
    tf2.dispose();
  } catch (e) {
    log(`TensorFlow.js error: ${e.message}`);
  }

  // ml-matrix
  try {
    const m1 = new Matrix(n, n);
    const m2 = new Matrix(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        m1.set(i, j, data1[i * n + j]);
        m2.set(i, j, data2[i * n + j]);
      }
    }

    results.matmul_mlMatrix = benchmark('ml-matrix.mmul', () => {
      m1.mmul(m2);
    }, iterations);
    log(`ml-matrix matmul: ${results.matmul_mlMatrix.mean.toFixed(3)}ms ± ${results.matmul_mlMatrix.std.toFixed(3)}ms`);

    results.transpose_mlMatrix = benchmark('ml-matrix.transpose', () => {
      m1.transpose();
    }, iterations);
    log(`ml-matrix transpose: ${results.transpose_mlMatrix.mean.toFixed(3)}ms ± ${results.transpose_mlMatrix.std.toFixed(3)}ms`);
  } catch (e) {
    log(`ml-matrix error: ${e.message}`);
  }

  // mathjs
  try {
    // Convert to 2D array for mathjs
    const arr1 = [];
    const arr2 = [];
    for (let i = 0; i < n; i++) {
      arr1.push(Array.from(data1.slice(i * n, (i + 1) * n)));
      arr2.push(Array.from(data2.slice(i * n, (i + 1) * n)));
    }

    results.matmul_mathjs = benchmark('mathjs.multiply', () => {
      math.multiply(arr1, arr2);
    }, iterations);
    log(`mathjs matmul: ${results.matmul_mathjs.mean.toFixed(3)}ms ± ${results.matmul_mathjs.std.toFixed(3)}ms`);

    results.transpose_mathjs = benchmark('mathjs.transpose', () => {
      math.transpose(arr1);
    }, iterations);
    log(`mathjs transpose: ${results.transpose_mathjs.mean.toFixed(3)}ms ± ${results.transpose_mathjs.std.toFixed(3)}ms`);
  } catch (e) {
    log(`mathjs error: ${e.message}`);
  }

  return results;
}

async function benchmarkReductions(size, iterations) {
  const data = randomArray(size);
  const results = {};

  log(`\n--- Reductions (${size} elements, ${iterations} iterations) ---`);

  // TensorFlow.js
  try {
    const tfData = tf.tensor1d(data);

    results.sum_tfjs = benchmark('tf.sum', () => {
      const r = tf.sum(tfData);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js sum: ${results.sum_tfjs.mean.toFixed(3)}ms ± ${results.sum_tfjs.std.toFixed(3)}ms`);

    results.mean_tfjs = benchmark('tf.mean', () => {
      const r = tf.mean(tfData);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js mean: ${results.mean_tfjs.mean.toFixed(3)}ms ± ${results.mean_tfjs.std.toFixed(3)}ms`);

    results.max_tfjs = benchmark('tf.max', () => {
      const r = tf.max(tfData);
      r.dispose();
    }, iterations);
    log(`TensorFlow.js max: ${results.max_tfjs.mean.toFixed(3)}ms ± ${results.max_tfjs.std.toFixed(3)}ms`);

    tfData.dispose();
  } catch (e) {
    log(`TensorFlow.js error: ${e.message}`);
  }

  // mathjs
  try {
    const mjData = Array.from(data);

    results.sum_mathjs = benchmark('mathjs.sum', () => {
      math.sum(mjData);
    }, iterations);
    log(`mathjs sum: ${results.sum_mathjs.mean.toFixed(3)}ms ± ${results.sum_mathjs.std.toFixed(3)}ms`);

    results.mean_mathjs = benchmark('mathjs.mean', () => {
      math.mean(mjData);
    }, iterations);
    log(`mathjs mean: ${results.mean_mathjs.mean.toFixed(3)}ms ± ${results.mean_mathjs.std.toFixed(3)}ms`);

    results.max_mathjs = benchmark('mathjs.max', () => {
      math.max(mjData);
    }, iterations);
    log(`mathjs max: ${results.max_mathjs.mean.toFixed(3)}ms ± ${results.max_mathjs.std.toFixed(3)}ms`);
  } catch (e) {
    log(`mathjs error: ${e.message}`);
  }

  // Native JS
  results.sum_native = benchmark('native sum', () => {
    data.reduce((a, b) => a + b, 0);
  }, iterations);
  log(`Native JS sum: ${results.sum_native.mean.toFixed(3)}ms ± ${results.sum_native.std.toFixed(3)}ms`);

  return results;
}

async function benchmarkLinearRegression(samples, features, iterations) {
  log(`\n--- Linear Regression Pipeline (${samples} samples, ${features} features) ---`);

  const X = randomMatrix(samples, features);
  const y = randomArray(samples);
  const results = {};

  // TensorFlow.js - using normal equation: w = (X'X)^-1 X'y
  try {
    const tfX = tf.tensor2d(X, [samples, features]);
    const tfY = tf.tensor2d(Array.from(y), [samples, 1]);

    results.lr_tfjs = benchmark('tf linear regression', () => {
      const Xt = tf.transpose(tfX);
      const XtX = tf.matMul(Xt, tfX);
      const XtX_inv = tf.linalg.tensorDiag(tf.ones([features])); // Simplified - real inv is expensive
      const Xty = tf.matMul(Xt, tfY);
      const w = tf.matMul(XtX_inv, Xty);

      Xt.dispose();
      XtX.dispose();
      XtX_inv.dispose();
      Xty.dispose();
      w.dispose();
    }, iterations);
    log(`TensorFlow.js linear regression: ${results.lr_tfjs.mean.toFixed(3)}ms ± ${results.lr_tfjs.std.toFixed(3)}ms`);

    tfX.dispose();
    tfY.dispose();
  } catch (e) {
    log(`TensorFlow.js error: ${e.message}`);
  }

  // ml-matrix - using built-in least squares
  try {
    const mlX = new Matrix(samples, features);
    for (let i = 0; i < samples; i++) {
      for (let j = 0; j < features; j++) {
        mlX.set(i, j, X[i * features + j]);
      }
    }
    const mlY = Matrix.columnVector(Array.from(y));

    results.lr_mlMatrix = benchmark('ml-matrix linear regression', () => {
      // Normal equation: (X'X)^-1 X'y
      const Xt = mlX.transpose();
      const XtX = Xt.mmul(mlX);
      const XtX_inv = XtX.pseudoInverse();
      const Xty = Xt.mmul(mlY);
      const w = XtX_inv.mmul(Xty);
    }, Math.min(iterations, 20)); // Slower, fewer iterations
    log(`ml-matrix linear regression: ${results.lr_mlMatrix.mean.toFixed(3)}ms ± ${results.lr_mlMatrix.std.toFixed(3)}ms`);
  } catch (e) {
    log(`ml-matrix error: ${e.message}`);
  }

  return results;
}

// =====================
// Chart Updates
// =====================

function createChart(canvasId, title, labels, datasets) {
  const ctx = document.getElementById(canvasId).getContext('2d');

  if (charts[canvasId]) {
    charts[canvasId].destroy();
  }

  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: false
        },
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: '#888'
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#888' },
          grid: { color: '#2a2a2a' }
        },
        y: {
          title: {
            display: true,
            text: 'Time (ms)',
            color: '#888'
          },
          ticks: { color: '#888' },
          grid: { color: '#2a2a2a' }
        }
      }
    }
  });
}

function updateCharts(allResults) {
  // Element-wise chart
  if (allResults.elementwise) {
    const r = allResults.elementwise;
    createChart('elementwiseChart', 'Element-wise Operations', ['Add', 'Multiply', 'Exp'], [
      {
        label: 'TensorFlow.js',
        data: [r.add_tfjs?.mean || 0, r.mul_tfjs?.mean || 0, r.exp_tfjs?.mean || 0],
        backgroundColor: COLORS.tfjs
      },
      {
        label: 'ml-matrix',
        data: [r.add_mlMatrix?.mean || 0, r.mul_mlMatrix?.mean || 0, 0],
        backgroundColor: COLORS.mlMatrix
      },
      {
        label: 'mathjs',
        data: [r.add_mathjs?.mean || 0, r.mul_mathjs?.mean || 0, 0],
        backgroundColor: COLORS.mathjs
      },
      {
        label: 'Native JS',
        data: [r.add_native?.mean || 0, 0, 0],
        backgroundColor: '#555'
      }
    ]);
  }

  // Matrix chart
  if (allResults.matrix) {
    const r = allResults.matrix;
    createChart('matrixChart', 'Matrix Operations', ['MatMul', 'Transpose'], [
      {
        label: 'TensorFlow.js',
        data: [r.matmul_tfjs?.mean || 0, r.transpose_tfjs?.mean || 0],
        backgroundColor: COLORS.tfjs
      },
      {
        label: 'ml-matrix',
        data: [r.matmul_mlMatrix?.mean || 0, r.transpose_mlMatrix?.mean || 0],
        backgroundColor: COLORS.mlMatrix
      },
      {
        label: 'mathjs',
        data: [r.matmul_mathjs?.mean || 0, r.transpose_mathjs?.mean || 0],
        backgroundColor: COLORS.mathjs
      }
    ]);
  }

  // Reductions chart
  if (allResults.reductions) {
    const r = allResults.reductions;
    createChart('reductionsChart', 'Reductions', ['Sum', 'Mean', 'Max'], [
      {
        label: 'TensorFlow.js',
        data: [r.sum_tfjs?.mean || 0, r.mean_tfjs?.mean || 0, r.max_tfjs?.mean || 0],
        backgroundColor: COLORS.tfjs
      },
      {
        label: 'mathjs',
        data: [r.sum_mathjs?.mean || 0, r.mean_mathjs?.mean || 0, r.max_mathjs?.mean || 0],
        backgroundColor: COLORS.mathjs
      },
      {
        label: 'Native JS',
        data: [r.sum_native?.mean || 0, 0, 0],
        backgroundColor: '#555'
      }
    ]);
  }

  // Pipeline chart
  if (allResults.pipeline) {
    const r = allResults.pipeline;
    createChart('pipelineChart', 'Linear Regression', ['Full Pipeline'], [
      {
        label: 'TensorFlow.js',
        data: [r.lr_tfjs?.mean || 0],
        backgroundColor: COLORS.tfjs
      },
      {
        label: 'ml-matrix',
        data: [r.lr_mlMatrix?.mean || 0],
        backgroundColor: COLORS.mlMatrix
      }
    ]);
  }
}

function updateTable(allResults) {
  const tbody = document.querySelector('#resultsTable tbody');
  const rows = [];

  // Flatten results
  const ops = [
    { name: 'Add', keys: ['add_tfjs', 'add_mlMatrix', 'add_mathjs'], cat: 'elementwise' },
    { name: 'Multiply', keys: ['mul_tfjs', 'mul_mlMatrix', 'mul_mathjs'], cat: 'elementwise' },
    { name: 'MatMul', keys: ['matmul_tfjs', 'matmul_mlMatrix', 'matmul_mathjs'], cat: 'matrix' },
    { name: 'Transpose', keys: ['transpose_tfjs', 'transpose_mlMatrix', 'transpose_mathjs'], cat: 'matrix' },
    { name: 'Sum', keys: ['sum_tfjs', null, 'sum_mathjs'], cat: 'reductions' },
    { name: 'Mean', keys: ['mean_tfjs', null, 'mean_mathjs'], cat: 'reductions' },
    { name: 'Linear Regression', keys: ['lr_tfjs', 'lr_mlMatrix', null], cat: 'pipeline' }
  ];

  for (const op of ops) {
    const data = allResults[op.cat] || {};
    const values = [
      '—', // rumpy-ts placeholder
      data[op.keys[0]]?.mean?.toFixed(3) || '—',
      data[op.keys[1]]?.mean?.toFixed(3) || '—',
      data[op.keys[2]]?.mean?.toFixed(3) || '—',
      '—'  // ndarray placeholder
    ];

    // Find fastest
    const numericValues = values.slice(1, 4).map(v => v === '—' ? Infinity : parseFloat(v));
    const minIdx = numericValues.indexOf(Math.min(...numericValues));

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${op.name}</td>
      <td>${values[0]}</td>
      <td class="${minIdx === 0 ? 'fastest' : ''}">${values[1]}${minIdx === 0 ? ' ✓' : ''}</td>
      <td class="${minIdx === 1 ? 'fastest' : ''}">${values[2]}${minIdx === 1 ? ' ✓' : ''}</td>
      <td class="${minIdx === 2 ? 'fastest' : ''}">${values[3]}${minIdx === 2 ? ' ✓' : ''}</td>
      <td>${values[4]}</td>
    `;
    rows.push(row);
  }

  tbody.innerHTML = '';
  rows.forEach(r => tbody.appendChild(r));
}

// =====================
// Main Entry Points
// =====================

async function runBenchmarks() {
  const runBtn = document.getElementById('runBtn');
  const runQuickBtn = document.getElementById('runQuickBtn');
  runBtn.disabled = true;
  runQuickBtn.disabled = true;
  clearLog();

  const size = parseInt(document.getElementById('sizeSelect').value);
  const iterations = parseInt(document.getElementById('iterations').value);
  const matrixSize = Math.floor(Math.sqrt(size));

  log(`Starting full benchmark suite...`);
  log(`Array size: ${size}, Matrix size: ${matrixSize}x${matrixSize}, Iterations: ${iterations}`);

  // Initialize TensorFlow.js
  log('\nInitializing TensorFlow.js...');
  await tf.ready();
  log(`TensorFlow.js backend: ${tf.getBackend()}`);

  const allResults = {};

  try {
    allResults.elementwise = await benchmarkElementwise(size, iterations);
    allResults.matrix = await benchmarkMatrix(Math.min(matrixSize, 200), Math.floor(iterations / 2));
    allResults.reductions = await benchmarkReductions(size, iterations);
    allResults.pipeline = await benchmarkLinearRegression(500, 10, Math.floor(iterations / 5));

    log('\n✅ Benchmarks complete!');

    updateCharts(allResults);
    updateTable(allResults);
  } catch (e) {
    log(`\n❌ Error: ${e.message}`);
    console.error(e);
  }

  runBtn.disabled = false;
  runQuickBtn.disabled = false;
}

async function runQuickBenchmarks() {
  document.getElementById('sizeSelect').value = '1000';
  document.getElementById('iterations').value = '20';
  await runBenchmarks();
}

// Expose to global scope for button onclick
window.runBenchmarks = runBenchmarks;
window.runQuickBenchmarks = runQuickBenchmarks;

// Auto-initialize TF.js on load
tf.ready().then(() => {
  log(`TensorFlow.js ready (backend: ${tf.getBackend()})`);
});
