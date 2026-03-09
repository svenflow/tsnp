/**
 * Performance benchmarks: rumpy-ts vs TensorFlow.js
 *
 * Fair comparisons using f32:
 * - WASM vs WASM: rumpy-wasm (f32 optimized) vs tfjs-wasm
 * - WebGPU vs WebGPU: rumpy-webgpu vs tfjs-webgpu
 *
 * Runs in Playwright browser environment for accurate GPU/WASM timings.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmBackend } from './wasm-backend';
import { initWebGPUBackend, createWebGPUBackend, WebGPUBackend } from './webgpu-backend';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgpu';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

// Import the optimized f32 matmul directly
let wasmModule: any = null;

// Benchmark configuration
const WARMUP_RUNS = 3;
const BENCHMARK_RUNS = 10;
const MATRIX_SIZES = [512, 1024, 2048, 4096];

interface BenchmarkResult {
  backend: string;
  size: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  gflops: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

describe('benchmarks', () => {
  let rumpyWebgpuBackend: WebGPUBackend;

  beforeAll(async () => {
    // Initialize rumpy WASM (to get the optimized f32 matmul)
    await initWasmBackend();
    const module = await import('./wasm-pkg/rumpy_wasm.js');
    wasmModule = module;

    await initWebGPUBackend();
    rumpyWebgpuBackend = createWebGPUBackend() as WebGPUBackend;

    // Set WASM paths for tfjs
    setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/');
  });

  it('verifies numerical correctness between backends', async () => {
    const size = 4;
    const aData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const bData = new Float32Array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    const expected = [80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386];

    // rumpy-wasm (f32 optimized)
    const wasmResult = wasmModule.matmulF32OptimizedParallel(aData, bData, size, size, size);
    console.log('rumpy-wasm-f32 result:', Array.from(wasmResult));
    for (let i = 0; i < expected.length; i++) {
      expect(Math.abs(wasmResult[i] - expected[i])).toBeLessThan(1e-4);
    }

    // rumpy-webgpu
    const gpuA = rumpyWebgpuBackend.array(Array.from(aData), [size, size]);
    const gpuB = rumpyWebgpuBackend.array(Array.from(bData), [size, size]);
    const gpuC = await rumpyWebgpuBackend.matmulAsync(gpuA, gpuB);
    const gpuResult = gpuC.toArray();
    console.log('rumpy-webgpu result:', gpuResult);
    for (let i = 0; i < expected.length; i++) {
      expect(Math.abs(gpuResult[i] - expected[i])).toBeLessThan(1e-4);
    }

    console.log('✅ Numerical correctness verified!');
  });

  it('benchmarks matmul at various sizes', async () => {
    const results: BenchmarkResult[] = [];

    console.log('\n' + '='.repeat(70));
    console.log('MATMUL BENCHMARK: rumpy-ts vs TensorFlow.js (f32)');
    console.log('='.repeat(70) + '\n');

    for (const size of MATRIX_SIZES) {
      console.log(`\n=== ${size}x${size} matrices ===\n`);

      // Generate random f32 matrices
      const aData = new Float32Array(size * size);
      const bData = new Float32Array(size * size);
      for (let i = 0; i < size * size; i++) {
        aData[i] = Math.random();
        bData[i] = Math.random();
      }

      const flops = 2 * size * size * size;

      // ============ rumpy-ts WASM (f32 optimized with SIMD + parallel) ============
      {
        // Warmup
        for (let i = 0; i < WARMUP_RUNS; i++) {
          wasmModule.matmulF32OptimizedParallel(aData, bData, size, size, size);
        }
        await sleep(10);

        // Benchmark
        const times: number[] = [];
        for (let i = 0; i < BENCHMARK_RUNS; i++) {
          const start = performance.now();
          wasmModule.matmulF32OptimizedParallel(aData, bData, size, size, size);
          const end = performance.now();
          times.push(end - start);
        }

        const avg = times.reduce((a, b) => a + b) / times.length;
        const gflops = (flops / (avg / 1000)) / 1e9;
        results.push({ backend: 'rumpy-wasm', size, avgMs: avg, minMs: Math.min(...times), maxMs: Math.max(...times), gflops });
        console.log(`rumpy-wasm:     ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`);
      }

      // ============ tfjs-wasm ============
      try {
        await tf.setBackend('wasm');
        await tf.ready();

        const tfA = tf.tensor2d(aData, [size, size], 'float32');
        const tfB = tf.tensor2d(bData, [size, size], 'float32');

        for (let i = 0; i < WARMUP_RUNS; i++) {
          const c = tf.matMul(tfA, tfB);
          await c.data();
          c.dispose();
        }
        await sleep(10);

        const times: number[] = [];
        for (let i = 0; i < BENCHMARK_RUNS; i++) {
          const start = performance.now();
          const c = tf.matMul(tfA, tfB);
          await c.data();
          const end = performance.now();
          times.push(end - start);
          c.dispose();
        }

        tfA.dispose();
        tfB.dispose();

        const avg = times.reduce((a, b) => a + b) / times.length;
        const gflops = (flops / (avg / 1000)) / 1e9;
        results.push({ backend: 'tfjs-wasm', size, avgMs: avg, minMs: Math.min(...times), maxMs: Math.max(...times), gflops });
        console.log(`tfjs-wasm:      ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`);
      } catch (e) {
        console.log(`tfjs-wasm: unavailable (${e})`);
      }

      // ============ rumpy-ts WebGPU ============
      {
        const aDataArray = Array.from(aData);
        const bDataArray = Array.from(bData);
        const gpuA = rumpyWebgpuBackend.array(aDataArray, [size, size]);
        const gpuB = rumpyWebgpuBackend.array(bDataArray, [size, size]);

        for (let i = 0; i < WARMUP_RUNS; i++) await rumpyWebgpuBackend.matmulAsync(gpuA, gpuB);
        await sleep(10);

        const times: number[] = [];
        for (let i = 0; i < BENCHMARK_RUNS; i++) {
          const start = performance.now();
          await rumpyWebgpuBackend.matmulAsync(gpuA, gpuB);
          const end = performance.now();
          times.push(end - start);
        }

        const avg = times.reduce((a, b) => a + b) / times.length;
        const gflops = (flops / (avg / 1000)) / 1e9;
        results.push({ backend: 'rumpy-webgpu', size, avgMs: avg, minMs: Math.min(...times), maxMs: Math.max(...times), gflops });
        console.log(`rumpy-webgpu:   ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`);
      }

      // ============ tfjs-webgpu ============
      try {
        await tf.setBackend('webgpu');
        await tf.ready();

        const tfA = tf.tensor2d(aData, [size, size], 'float32');
        const tfB = tf.tensor2d(bData, [size, size], 'float32');

        for (let i = 0; i < WARMUP_RUNS; i++) {
          const c = tf.matMul(tfA, tfB);
          await c.data();
          c.dispose();
        }
        await sleep(10);

        const times: number[] = [];
        for (let i = 0; i < BENCHMARK_RUNS; i++) {
          const start = performance.now();
          const c = tf.matMul(tfA, tfB);
          await c.data();
          const end = performance.now();
          times.push(end - start);
          c.dispose();
        }

        tfA.dispose();
        tfB.dispose();

        const avg = times.reduce((a, b) => a + b) / times.length;
        const gflops = (flops / (avg / 1000)) / 1e9;
        results.push({ backend: 'tfjs-webgpu', size, avgMs: avg, minMs: Math.min(...times), maxMs: Math.max(...times), gflops });
        console.log(`tfjs-webgpu:    ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`);
      } catch (e) {
        console.log(`tfjs-webgpu: unavailable (${e})`);
      }
    }

    // Print summary
    console.log('\n\n' + '='.repeat(70));
    console.log('SUMMARY - FAIR COMPARISONS (f32)');
    console.log('='.repeat(70) + '\n');

    for (const size of MATRIX_SIZES) {
      console.log(`${size}x${size}:`);

      // WASM comparison
      const rumpyWasm = results.find(r => r.size === size && r.backend === 'rumpy-wasm');
      const tfjsWasm = results.find(r => r.size === size && r.backend === 'tfjs-wasm');
      if (rumpyWasm && tfjsWasm) {
        const wasmWinner = rumpyWasm.avgMs < tfjsWasm.avgMs ? 'rumpy' : 'tfjs';
        const wasmSpeedup = wasmWinner === 'rumpy'
          ? (tfjsWasm.avgMs / rumpyWasm.avgMs).toFixed(2)
          : (rumpyWasm.avgMs / tfjsWasm.avgMs).toFixed(2);
        console.log(`  WASM:   rumpy ${rumpyWasm.avgMs.toFixed(2)}ms vs tfjs ${tfjsWasm.avgMs.toFixed(2)}ms → ${wasmWinner} ${wasmSpeedup}x faster`);
      }

      // WebGPU comparison
      const rumpyGpu = results.find(r => r.size === size && r.backend === 'rumpy-webgpu');
      const tfjsGpu = results.find(r => r.size === size && r.backend === 'tfjs-webgpu');
      if (rumpyGpu && tfjsGpu) {
        const gpuWinner = rumpyGpu.avgMs < tfjsGpu.avgMs ? 'rumpy' : 'tfjs';
        const gpuSpeedup = gpuWinner === 'rumpy'
          ? (tfjsGpu.avgMs / rumpyGpu.avgMs).toFixed(2)
          : (rumpyGpu.avgMs / tfjsGpu.avgMs).toFixed(2);
        console.log(`  WebGPU: rumpy ${rumpyGpu.avgMs.toFixed(2)}ms vs tfjs ${tfjsGpu.avgMs.toFixed(2)}ms → ${gpuWinner} ${gpuSpeedup}x faster`);
      }

      console.log('');
    }

    (window as any).benchmarkResults = results;
    expect(results.length).toBeGreaterThan(0);
  }, 600000);
});
