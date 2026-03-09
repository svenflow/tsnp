#!/usr/bin/env node
/**
 * Headless benchmark runner using Playwright
 * Runs the benchmark suite and outputs results to JSON
 */

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';

const BENCHMARK_URL = 'http://localhost:5175';
const OUTPUT_FILE = 'benchmark-results.json';

async function runBenchmark() {
  console.log('Starting headless benchmark...\n');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Collect console logs
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    console.log(`[browser] ${text}`);
  });

  page.on('pageerror', err => {
    console.error(`[browser error] ${err.message}`);
  });

  console.log(`Navigating to ${BENCHMARK_URL}...`);
  await page.goto(BENCHMARK_URL, { waitUntil: 'networkidle' });
  console.log('Page loaded, waiting for JS to initialize...');

  // Wait longer for vite to load modules
  await page.waitForTimeout(5000);

  // Take a screenshot for debugging
  await page.screenshot({ path: '/tmp/benchmark-debug.png' });
  console.log('Screenshot saved to /tmp/benchmark-debug.png');

  // Check if runQuickBenchmarks exists
  const hasFunc = await page.evaluate(() => typeof window.runQuickBenchmarks === 'function');
  console.log(`runQuickBenchmarks exists: ${hasFunc}`);

  if (!hasFunc) {
    console.log('Function not found. Module may not have loaded. Waiting more...');
    await page.waitForTimeout(10000);

    const hasFunc2 = await page.evaluate(() => typeof window.runQuickBenchmarks === 'function');
    console.log(`runQuickBenchmarks exists (after wait): ${hasFunc2}`);
  }

  // Check status
  const statusText = await page.$eval('#status', el => el.textContent);
  console.log(`Status: ${statusText}`);

  // Run benchmark via evaluate instead of clicking
  console.log('Running quick benchmark via page.evaluate...');

  try {
    await page.evaluate(async () => {
      if (typeof window.runQuickBenchmarks === 'function') {
        await window.runQuickBenchmarks();
      } else {
        throw new Error('runQuickBenchmarks not defined');
      }
    });
  } catch (e) {
    console.log(`Evaluate failed: ${e.message}, trying click...`);
    await page.click('#runQuickBtn');
  }

  // Wait for benchmarks to complete
  let completed = false;
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(2000);
    const status = await page.$eval('#status', el => el.textContent);
    if (status.includes('complete') || status.includes('Error')) {
      completed = true;
      console.log(`Completed after ${i * 2}s`);
      break;
    }
    if (i % 10 === 0) console.log(`Waiting... (${i * 2}s elapsed)`);
  }

  if (!completed) {
    console.error('Benchmark did not complete in time');
  }

  // Get final status
  const finalStatus = await page.$eval('#status', el => el.textContent);

  // Get table results
  const tableResults = await page.$$eval('#resultsTable tbody tr', rows => {
    return rows.map(row => {
      const cells = row.querySelectorAll('td');
      return Array.from(cells).map(cell => cell.textContent.trim());
    });
  });

  // Parse results into structured JSON
  const results = {
    timestamp: new Date().toISOString(),
    status: finalStatus.includes('complete') ? 'success' : 'partial',
    operations: {}
  };

  const headers = ['operation', 'rumpy-ts', 'tensorflow.js', 'ml-matrix', 'mathjs', 'ndarray'];
  for (const row of tableResults) {
    if (row.length >= 5) {
      const op = row[0];
      results.operations[op] = {
        'rumpy-ts': parseFloat(row[1]) || null,
        'tensorflow.js': parseFloat(row[2]) || null,
        'ml-matrix': parseFloat(row[3]) || null,
        'mathjs': parseFloat(row[4]) || null,
        'ndarray': row.length > 5 ? (parseFloat(row[5]) || null) : null
      };
    }
  }

  // Extract detailed timings from log
  const timingRegex = /(\w+(?:-\w+)?)\s+(\w+):\s+([\d.]+)ms/g;
  let match;
  const detailedTimings = [];
  for (const log of logs) {
    while ((match = timingRegex.exec(log)) !== null) {
      detailedTimings.push({
        library: match[1],
        operation: match[2],
        time_ms: parseFloat(match[3])
      });
    }
  }
  results.detailed_timings = detailedTimings;
  results.raw_log = finalStatus;

  // Write JSON
  writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log(`\nResults written to ${OUTPUT_FILE}`);

  // Print summary
  console.log('\n=== BENCHMARK RESULTS ===\n');
  console.log('Operation | rumpy-ts | TensorFlow.js | ml-matrix | mathjs');
  console.log('-'.repeat(60));
  for (const [op, times] of Object.entries(results.operations)) {
    const vals = [
      times['rumpy-ts'] ? `${times['rumpy-ts'].toFixed(3)}ms` : '—',
      times['tensorflow.js'] ? `${times['tensorflow.js'].toFixed(3)}ms` : '—',
      times['ml-matrix'] ? `${times['ml-matrix'].toFixed(3)}ms` : '—',
      times['mathjs'] ? `${times['mathjs'].toFixed(3)}ms` : '—'
    ];
    console.log(`${op.padEnd(15)} | ${vals.join(' | ')}`);
  }

  await browser.close();
  console.log('\nBenchmark complete!');
  return results;
}

runBenchmark().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
