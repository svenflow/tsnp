#!/usr/bin/env node
/**
 * Headless browser benchmark runner using Playwright
 * Runs benchmarks in a real browser with WebGL/WASM support and outputs JSON
 */

import { chromium } from 'playwright';
import { writeFileSync, mkdirSync } from 'fs';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function startViteServer() {
  return new Promise((resolve, reject) => {
    const vite = spawn('npx', ['vite', '--port', '5199'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true
    });

    vite.stdout.on('data', (data) => {
      const output = data.toString();
      if (output.includes('Local:')) {
        resolve(vite);
      }
    });

    vite.stderr.on('data', (data) => {
      console.error('Vite stderr:', data.toString());
    });

    setTimeout(() => reject(new Error('Vite startup timeout')), 30000);
  });
}

async function runBenchmarks() {
  console.log('Starting Vite dev server...');
  const viteProcess = await startViteServer();
  console.log('Vite server running on http://localhost:5199');

  let browser;
  try {
    console.log('Launching headless browser...');
    browser = await chromium.launch({
      headless: true,
      args: [
        '--enable-features=SharedArrayBuffer',
        '--enable-webgl',
        '--use-gl=angle',
      ]
    });

    const context = await browser.newContext({
      permissions: [],
    });

    // Set headers for SharedArrayBuffer support
    await context.route('**/*', (route) => {
      route.continue({
        headers: {
          ...route.request().headers(),
          'Cross-Origin-Embedder-Policy': 'require-corp',
          'Cross-Origin-Opener-Policy': 'same-origin',
        }
      });
    });

    const page = await context.newPage();

    // Capture console logs
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('BENCHMARK_RESULT:')) {
        // Will be parsed below
      } else {
        console.log('Browser:', text);
      }
    });

    page.on('pageerror', err => {
      console.error('Page error:', err.message);
    });

    console.log('Navigating to benchmark page...');
    await page.goto('http://localhost:5199/', { waitUntil: 'networkidle' });

    // Wait for libraries to load
    console.log('Waiting for libraries to load...');
    await page.waitForFunction(() => {
      return window.benchmarkReady === true;
    }, { timeout: 60000 }).catch(() => {
      console.log('Libraries may not have loaded, proceeding anyway...');
    });

    // Run the benchmark
    console.log('Running benchmarks (this may take a minute)...');

    const results = await page.evaluate(async () => {
      // Run the benchmark function and return results
      if (typeof window.runBenchmarksHeadless === 'function') {
        return await window.runBenchmarksHeadless();
      } else if (typeof window.runBenchmarks === 'function') {
        return await window.runBenchmarks();
      }
      return { error: 'No benchmark function found' };
    });

    if (results.error) {
      // Try clicking the button instead
      console.log('Trying to click run button...');
      await page.click('#runBtn');

      // Wait for completion
      await page.waitForFunction(() => {
        const status = document.getElementById('status');
        return status && status.textContent.includes('Complete');
      }, { timeout: 300000 });

      // Extract results from the page
      const extractedResults = await page.evaluate(() => {
        const rows = document.querySelectorAll('.results-table tbody tr');
        const results = {};
        rows.forEach(row => {
          const cells = row.querySelectorAll('td');
          if (cells.length > 1) {
            const op = cells[0].textContent.trim();
            results[op] = {};
            const libs = ['rumpy-ts', 'tensorflow.js', 'ml-matrix', 'mathjs', 'ndarray'];
            libs.forEach((lib, i) => {
              const val = cells[i + 1]?.textContent.trim();
              if (val && val !== '—') {
                results[op][lib] = parseFloat(val.replace('ms', ''));
              }
            });
          }
        });
        return results;
      });

      console.log('\n=== BENCHMARK RESULTS ===\n');
      console.log(JSON.stringify(extractedResults, null, 2));

      writeFileSync(
        join(__dirname, 'benchmark-results-browser.json'),
        JSON.stringify({ timestamp: new Date().toISOString(), operations: extractedResults }, null, 2)
      );
      console.log('\nResults written to benchmark-results-browser.json');
    } else {
      console.log('\n=== BENCHMARK RESULTS ===\n');
      console.log(JSON.stringify(results, null, 2));

      writeFileSync(
        join(__dirname, 'benchmark-results-browser.json'),
        JSON.stringify(results, null, 2)
      );
      console.log('\nResults written to benchmark-results-browser.json');
    }

  } catch (error) {
    console.error('Benchmark error:', error);
    throw error;
  } finally {
    if (browser) await browser.close();
    viteProcess.kill();
  }
}

runBenchmarks().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
