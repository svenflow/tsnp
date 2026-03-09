#!/usr/bin/env node
/**
 * Run NumPy compatibility tests in headless browser with Playwright
 * Outputs JSON results to stdout
 */

import { chromium } from 'playwright';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  // Start Vite
  console.error('Starting Vite dev server...');
  const vite = spawn('npx', ['vite', '--port', '5199'], {
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
      args: ['--enable-features=SharedArrayBuffer']
    });

    const page = await browser.newPage();

    // Collect test results
    let passed = 0;
    let failed = 0;
    let testResults = [];

    page.on('console', msg => {
      const text = msg.text();
      console.error('Browser:', text);

      // Parse test results from console
      if (text.includes('✓')) {
        passed++;
        testResults.push({ status: 'pass', message: text });
      } else if (text.includes('✗')) {
        failed++;
        testResults.push({ status: 'fail', message: text });
      }
    });

    page.on('pageerror', err => {
      console.error('Page error:', err.message);
    });

    console.error('Loading test page...');
    await page.goto('http://localhost:5199/test-numpy-compat.html', { waitUntil: 'networkidle' });

    // Wait for tests to complete (wait for summary section to appear)
    console.error('Waiting for tests to complete...');
    await page.waitForSelector('div.section h2:has-text("Summary")', { timeout: 120000 });

    // Wait a bit more for final rendering
    await new Promise(r => setTimeout(r, 1000));

    // Get summary text
    const summary = await page.textContent('#summary');
    console.error('\n' + '='.repeat(50));
    console.error('SUMMARY:', summary);
    console.error('='.repeat(50));

    // Get all test results
    const allTests = await page.$$eval('.test-row', rows =>
      rows.map(row => ({
        passed: row.classList.contains('pass') || row.textContent.includes('✓'),
        text: row.textContent
      }))
    );

    const passCount = allTests.filter(t => t.passed).length;
    const failCount = allTests.filter(t => !t.passed).length;

    // Output JSON
    const output = {
      passed: passCount,
      failed: failCount,
      total: allTests.length,
      summary: summary,
      results: allTests
    };

    console.log(JSON.stringify(output, null, 2));

    // Return exit code based on results
    if (failCount > 0) {
      console.error(`\n${failCount} tests failed!`);
      process.exitCode = 1;
    } else {
      console.error(`\nAll ${passCount} tests passed!`);
    }

  } finally {
    if (browser) await browser.close();
    vite.kill();
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
