#!/usr/bin/env node
/**
 * Benchmark script comparing rumpy-ts matmulF32Optimized vs TensorFlow.js XNNPACK
 *
 * Run: node bench-optimized.js
 */

const { chromium } = require('playwright');

async function runBenchmark() {
  console.log('Launching Chrome for benchmark...\n');

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--enable-features=SharedArrayBuffer',
      '--enable-features=WebAssembly',
      '--enable-features=WebAssemblySimd',
      '--enable-features=WebAssemblyRelaxedSimd',
    ]
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  // Serve the benchmark page
  const { createServer } = require('http');
  const { readFileSync } = require('fs');
  const { join } = require('path');
  const { extname } = require('path');

  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.wasm': 'application/wasm',
  };

  const server = createServer((req, res) => {
    let filePath = join(__dirname, req.url === '/' ? '/bench-optimized.html' : req.url);
    try {
      const content = readFileSync(filePath);
      const ext = extname(filePath);
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
      res.end(content);
    } catch (e) {
      res.statusCode = 404;
      res.end('Not found');
    }
  });

  await new Promise(resolve => server.listen(8888, resolve));
  console.log('Server running on http://localhost:8888');

  // Navigate and wait for results
  page.on('console', msg => console.log(msg.text()));

  await page.goto('http://localhost:8888');

  // Wait for benchmark to complete (indicated by results being available)
  const results = await page.evaluate(async () => {
    // Wait for init
    await new Promise(r => setTimeout(r, 2000));

    // Run the benchmark and return results
    return await window.runBenchmark();
  });

  console.log('\n=== BENCHMARK RESULTS ===\n');
  console.log(JSON.stringify(results, null, 2));

  await browser.close();
  server.close();
}

runBenchmark().catch(e => {
  console.error(e);
  process.exit(1);
});
