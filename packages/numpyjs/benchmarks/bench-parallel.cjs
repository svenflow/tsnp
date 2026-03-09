#!/usr/bin/env node
/**
 * Benchmark script comparing rumpy-ts parallel vs TensorFlow.js XNNPACK multi-threaded
 *
 * Run: node bench-parallel.cjs
 */

const { chromium } = require('playwright');

async function runBenchmark() {
  console.log('Launching Chrome for parallel benchmark...\n');

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--enable-features=SharedArrayBuffer',
      '--enable-features=WebAssembly',
      '--enable-features=WebAssemblySimd',
      '--enable-features=WebAssemblyRelaxedSimd',
      '--enable-features=WebAssemblyThreads',
    ]
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  const { createServer } = require('http');
  const { readFileSync } = require('fs');
  const { join, extname } = require('path');

  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.wasm': 'application/wasm',
  };

  const server = createServer((req, res) => {
    let url = req.url;

    // Handle package directory imports (workers do `import('../../..')` which resolves to /pkg/)
    // This needs to serve the main entry point from package.json
    if (url === '/pkg/' || url === '/pkg') {
      url = '/pkg/rumpy_wasm.js';
    }

    let filePath = join(__dirname, url === '/' ? '/bench-parallel.html' : url);
    try {
      const content = readFileSync(filePath);
      const ext = extname(filePath);
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
      res.end(content);
    } catch (e) {
      console.error(`404: ${req.url} -> ${filePath}`);
      res.statusCode = 404;
      res.end('Not found: ' + req.url);
    }
  });

  await new Promise(resolve => server.listen(8890, resolve));
  console.log('Server running on http://localhost:8890');

  // Log all browser console messages including worker errors
  page.on('console', msg => {
    const type = msg.type();
    if (type === 'error') {
      console.error(`[BROWSER ERROR]: ${msg.text()}`);
    } else {
      console.log(msg.text());
    }
  });

  // Catch unhandled page errors
  page.on('pageerror', err => {
    console.error(`[PAGE ERROR]: ${err.message}`);
  });

  // Monitor failed network requests
  page.on('requestfailed', request => {
    console.error(`[NETWORK FAIL]: ${request.url()} - ${request.failure()?.errorText || 'unknown'}`);
  });

  await page.goto('http://localhost:8890');

  const results = await page.evaluate(async () => {
    // Wait for init to complete (with timeout)
    if (window.initPromise) {
      await window.initPromise;
    } else {
      await new Promise(r => setTimeout(r, 5000));
    }
    return await window.runBenchmark();
  });

  console.log('\n=== BENCHMARK RESULTS (JSON) ===\n');
  console.log(JSON.stringify(results, null, 2));

  await browser.close();
  server.close();
}

runBenchmark().catch(e => {
  console.error(e);
  process.exit(1);
});
