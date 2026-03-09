#!/usr/bin/env node
/**
 * Headless f32 vs f64 benchmark runner using Playwright
 */

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';

const BENCHMARK_URL = 'http://localhost:5175/f32-benchmark.html';
const OUTPUT_FILE = 'f32-benchmark-results.json';

async function runBenchmark() {
  console.log('Starting headless f32 vs f64 benchmark...\n');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Collect console logs
  page.on('console', msg => {
    console.log(`[browser] ${msg.text()}`);
  });

  page.on('pageerror', err => {
    console.error(`[browser error] ${err.message}`);
  });

  console.log(`Navigating to ${BENCHMARK_URL}...`);
  await page.goto(BENCHMARK_URL, { waitUntil: 'networkidle' });
  console.log('Page loaded, waiting for initialization...');

  // Wait for init
  await page.waitForTimeout(3000);

  // Check if rumpy is ready
  const rumpyReady = await page.evaluate(() => !!window.rumpy);
  console.log(`rumpy ready: ${rumpyReady}`);

  if (!rumpyReady) {
    console.log('Waiting longer for WASM...');
    await page.waitForTimeout(5000);
  }

  // Run benchmarks
  console.log('Running benchmarks...');
  await page.click('#runBtn');

  // Wait for completion (check for window.benchmarkComplete)
  let completed = false;
  for (let i = 0; i < 120; i++) {
    await page.waitForTimeout(1000);
    completed = await page.evaluate(() => window.benchmarkComplete === true);
    if (completed) {
      console.log(`Completed after ${i}s`);
      break;
    }
    if (i % 10 === 0) console.log(`Waiting... (${i}s elapsed)`);
  }

  if (!completed) {
    console.error('Benchmark did not complete in time');
  }

  // Get results
  const results = await page.evaluate(() => window.benchmarkResults);

  // Write JSON
  writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\nResults written to ${OUTPUT_FILE}`);

  // Print summary
  console.log('\n=== BENCHMARK RESULTS ===\n');
  console.log('Size | rumpy-f64 | rumpy-f32 | tfjs-wasm | f32 speedup | vs tfjs');
  console.log('-'.repeat(70));

  for (const [size, times] of Object.entries(results)) {
    const f64 = times['rumpy-f64'];
    const f32 = times['rumpy-f32'];
    const tfjs = times['tfjs-wasm'];
    const speedup = f64 && f32 ? (f64 / f32).toFixed(2) : '—';
    const vsTfjs = f32 && tfjs ? (f32 / tfjs).toFixed(2) : '—';
    console.log(`${size.padStart(3)}  | ${(f64?.toFixed(3) || '—').padStart(9)} | ${(f32?.toFixed(3) || '—').padStart(9)} | ${(tfjs?.toFixed(3) || '—').padStart(9)} | ${speedup.padStart(11)} | ${vsTfjs.padStart(7)}`);
  }

  await browser.close();
  console.log('\nBenchmark complete!');
  return results;
}

runBenchmark().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
