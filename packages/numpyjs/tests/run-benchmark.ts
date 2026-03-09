#!/usr/bin/env -S npx tsx

/**
 * Run benchmarks in headless Chrome via Playwright
 *
 * Usage: npx tsx run-benchmark.ts
 */

import { chromium } from 'playwright';
import { createServer } from 'vite';

async function main() {
  console.log('Starting benchmark server...');

  // Start Vite dev server
  const server = await createServer({
    root: '.',
    server: { port: 5174 },
    optimizeDeps: { include: [] },
  });
  await server.listen();

  console.log('Starting browser...');

  // Launch browser with WebGPU support
  const browser = await chromium.launch({
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-gl=angle',
      '--use-angle=gl',
    ],
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  // Listen to console
  page.on('console', msg => console.log('Browser:', msg.text()));

  console.log('Loading benchmark page...');
  await page.goto('http://localhost:5174/benchmark.html');

  console.log('Running benchmarks (this may take several minutes)...\n');

  // Run benchmarks
  await page.evaluate(() => {
    return (window as any).runBenchmarks();
  });

  // Wait for completion
  await page.waitForFunction(
    () => !document.getElementById('runBtn')?.hasAttribute('disabled') ||
          document.getElementById('status')?.textContent?.includes('complete'),
    { timeout: 600000 } // 10 minute timeout
  );

  // Get results
  const results = await page.evaluate(() => {
    const allResults = (window as any).allResults || [];
    return allResults;
  });

  // Format and print results
  console.log('\n' + '='.repeat(70));
  console.log('MATMUL BENCHMARK RESULTS');
  console.log('='.repeat(70) + '\n');

  interface Result {
    backend: string;
    size: number;
    avgMs: number;
    minMs: number;
    maxMs: number;
    gflops: number;
  }

  const sizes = [...new Set(results.map((r: Result) => r.size))].sort((a, b) => Number(a) - Number(b));

  for (const size of sizes) {
    console.log(`${size}x${size} matrices:`);
    console.log('-'.repeat(65));
    console.log(`${'Backend'.padEnd(18)} ${'Avg (ms)'.padStart(10)} ${'Min'.padStart(10)} ${'Max'.padStart(10)} ${'GFLOPS'.padStart(10)}`);
    console.log('-'.repeat(65));

    const sizeResults = results
      .filter((r: Result) => r.size === size)
      .sort((a: Result, b: Result) => a.avgMs - b.avgMs);

    const fastest = sizeResults[0]?.backend;

    for (const r of sizeResults) {
      const marker = r.backend === fastest ? ' ⚡' : '';
      console.log(
        `${(r.backend + marker).padEnd(18)} ${r.avgMs.toFixed(2).padStart(10)} ${r.minMs.toFixed(2).padStart(10)} ${r.maxMs.toFixed(2).padStart(10)} ${r.gflops.toFixed(2).padStart(10)}`
      );
    }
    console.log('');
  }

  await browser.close();
  await server.close();
}

main().catch(console.error);
