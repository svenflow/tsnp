#!/usr/bin/env node
/**
 * Headless f32 matmul benchmark
 * Compares rumpy-ts (SIMD, parallel) vs TensorFlow.js XNNPACK
 * Outputs JSON results
 */

import { chromium } from 'playwright';
import { spawn } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const benchmarksDir = join(__dirname, '..');

async function startViteServer(port = 5201) {
  return new Promise((resolve, reject) => {
    const vite = spawn('npx', ['vite', '--port', String(port)], {
      cwd: benchmarksDir,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true
    });

    vite.stdout.on('data', (data) => {
      const output = data.toString();
      if (output.includes('Local:')) {
        resolve({ process: vite, port });
      }
    });

    setTimeout(() => reject(new Error('Vite startup timeout')), 30000);
  });
}

async function runBenchmarks() {
  console.error('Starting Vite server...');
  const { process: viteProcess, port } = await startViteServer();
  console.error(`Vite running on port ${port}`);

  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      args: [
        '--enable-features=SharedArrayBuffer',
        '--disable-web-security',
        '--enable-experimental-web-platform-features',
      ]
    });

    const context = await browser.newContext();

    // Intercept responses to add COOP/COEP headers for SharedArrayBuffer
    await context.route('**/*', async (route) => {
      const response = await route.fetch();
      const headers = response.headers();
      headers['cross-origin-embedder-policy'] = 'require-corp';
      headers['cross-origin-opener-policy'] = 'same-origin';
      await route.fulfill({
        response,
        headers
      });
    });

    const page = await context.newPage();

    // Collect console logs
    let resultJson = null;
    page.on('console', msg => {
      const text = msg.text();
      if (text.startsWith('BENCHMARK_RESULT:')) {
        resultJson = text.replace('BENCHMARK_RESULT:', '');
      } else {
        console.error('Browser:', text);
      }
    });
    page.on('pageerror', err => console.error('Page error:', err.message));

    console.error('Running benchmark (may take 1-2 minutes)...');
    await page.goto(`http://localhost:${port}/headless-bench.html`);

    // Wait for benchmarks to complete
    await page.waitForFunction(() => window.benchmarkComplete === true, { timeout: 300000 });

    if (resultJson) {
      console.log(resultJson);
    } else {
      const results = await page.evaluate(() => window.benchmarkResults);
      console.log(JSON.stringify(results, null, 2));
    }

  } finally {
    if (browser) await browser.close();
    viteProcess.kill();
  }
}

runBenchmarks().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
