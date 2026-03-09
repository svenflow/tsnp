#!/usr/bin/env node
/**
 * Run WASM benchmark in headless browser with Playwright
 * Outputs JSON results to stdout
 */

import { chromium } from 'playwright';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { writeFileSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  // Start Vite
  console.error('Starting Vite dev server...');
  const vite = spawn('npx', ['vite', '--port', '5198'], {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: true
  });

  // Wait for Vite to start
  await new Promise((resolve) => {
    vite.stdout.on('data', (data) => {
      if (data.toString().includes('Local:')) {
        resolve();
      }
    });
    setTimeout(resolve, 5000);
  });

  console.error('Vite server ready');

  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      args: ['--enable-features=SharedArrayBuffer', '--enable-webgl']
    });

    const page = await browser.newPage();

    // Collect logs
    const logs = [];
    page.on('console', msg => {
      const text = msg.text();
      logs.push(text);
      console.error('Browser:', text);
    });

    page.on('pageerror', err => {
      console.error('Page error:', err.message);
    });

    console.error('Loading benchmark page...');
    await page.goto('http://localhost:5198/wasm-benchmark.html', { waitUntil: 'networkidle' });

    // Wait for libraries to initialize
    console.error('Waiting for libraries to load...');
    await page.waitForFunction(() => window.benchmarkReady === true, { timeout: 60000 });

    console.error('Running benchmarks...');
    await page.click('#runBtn');

    // Wait for completion
    await page.waitForFunction(() => window.benchmarkComplete === true, { timeout: 300000 });

    // Get results
    const results = await page.evaluate(() => window.benchmarkResults);

    // Output JSON
    const output = JSON.stringify(results, null, 2);
    console.log(output);

    // Save to file
    writeFileSync(join(__dirname, 'wasm-benchmark-results.json'), output);
    console.error('\nResults saved to wasm-benchmark-results.json');

  } finally {
    if (browser) await browser.close();
    vite.kill();
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
