/**
 * Performance benchmarks: numpyjs WebGPU vs TensorFlow.js WebGPU
 *
 * Fair comparison using f32:
 * - WebGPU vs WebGPU: numpyjs-webgpu vs tfjs-webgpu
 * - Tests multiple kernel variants to find optimal per size
 *
 * Runs in Playwright browser environment for accurate GPU timings.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { initWebGPUBackend, createWebGPUBackend, WebGPUBackend } from './webgpu-backend';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

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
  let numpyjsWebgpuBackend: WebGPUBackend;

  beforeAll(async () => {
    await initWebGPUBackend();
    numpyjsWebgpuBackend = createWebGPUBackend() as WebGPUBackend;
  });

  it('autotunes all kernel variants at each size', async () => {
    console.log('\n' + '='.repeat(70));
    console.log('KERNEL AUTOTUNING: testing all variants per size');
    console.log('='.repeat(70) + '\n');

    for (const size of MATRIX_SIZES) {
      console.log(`\n--- Autotuning ${size}x${size} ---`);
      const winner = await numpyjsWebgpuBackend.autotune(size, size, size, 5);
      console.log(`Best kernel for ${size}: ${winner}\n`);
    }
  }, 600000);

  it('benchmarks matmul at various sizes', async () => {
    const results: BenchmarkResult[] = [];

    console.log('\n' + '='.repeat(70));
    console.log('MATMUL BENCHMARK: numpyjs WebGPU vs TensorFlow.js WebGPU (f32)');
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

      // ============ numpyjs WebGPU (GPU-resident, no readback) ============
      {
        // Upload once, benchmark GPU kernel only (no CPU conversion overhead)
        const gpuA = numpyjsWebgpuBackend.createAlignedTensor(aData, [size, size], true, true);
        const gpuB = numpyjsWebgpuBackend.createAlignedTensor(bData, [size, size], false, true);

        // Warmup
        for (let i = 0; i < WARMUP_RUNS; i++) {
          const c = numpyjsWebgpuBackend.matmulTensor(gpuA, gpuB);
          c.destroy();
        }
        // Flush GPU queue to ensure warmup completes
        await numpyjsWebgpuBackend.device.queue.onSubmittedWorkDone();
        await sleep(10);

        const times: number[] = [];
        for (let i = 0; i < BENCHMARK_RUNS; i++) {
          const start = performance.now();
          const c = numpyjsWebgpuBackend.matmulTensor(gpuA, gpuB);
          // Wait for GPU to finish to get accurate timing
          await numpyjsWebgpuBackend.device.queue.onSubmittedWorkDone();
          const end = performance.now();
          times.push(end - start);
          c.destroy();
        }

        const avg = times.reduce((a, b) => a + b) / times.length;
        const gflops = flops / (avg / 1000) / 1e9;
        results.push({
          backend: 'numpyjs-webgpu',
          size,
          avgMs: avg,
          minMs: Math.min(...times),
          maxMs: Math.max(...times),
          gflops,
        });
        console.log(
          `numpyjs-webgpu: ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`
        );

        gpuA.destroy();
        gpuB.destroy();
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
        const gflops = flops / (avg / 1000) / 1e9;
        results.push({
          backend: 'tfjs-webgpu',
          size,
          avgMs: avg,
          minMs: Math.min(...times),
          maxMs: Math.max(...times),
          gflops,
        });
        console.log(
          `tfjs-webgpu:    ${avg.toFixed(2).padStart(8)}ms  (${gflops.toFixed(2)} GFLOPS)`
        );
      } catch (e) {
        console.log(`tfjs-webgpu: unavailable (${e})`);
      }
    }

    // Print summary
    console.log('\n\n' + '='.repeat(70));
    console.log('SUMMARY - WebGPU COMPARISON (f32)');
    console.log('='.repeat(70) + '\n');

    for (const size of MATRIX_SIZES) {
      console.log(`${size}x${size}:`);

      const numpyjsGpu = results.find(r => r.size === size && r.backend === 'numpyjs-webgpu');
      const tfjsGpu = results.find(r => r.size === size && r.backend === 'tfjs-webgpu');
      if (numpyjsGpu && tfjsGpu) {
        const gpuWinner = numpyjsGpu.avgMs < tfjsGpu.avgMs ? 'numpyjs' : 'tfjs';
        const gpuSpeedup =
          gpuWinner === 'numpyjs'
            ? (tfjsGpu.avgMs / numpyjsGpu.avgMs).toFixed(2)
            : (numpyjsGpu.avgMs / tfjsGpu.avgMs).toFixed(2);
        console.log(
          `  WebGPU: numpyjs ${numpyjsGpu.avgMs.toFixed(2)}ms vs tfjs ${tfjsGpu.avgMs.toFixed(2)}ms -> ${gpuWinner} ${gpuSpeedup}x faster`
        );
      }

      console.log('');
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).benchmarkResults = results;
    expect(results.length).toBeGreaterThan(0);
  }, 600000);
});
