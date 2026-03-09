import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait(),
  ],
  server: {
    headers: {
      // Required for SharedArrayBuffer (needed by wasm-bindgen-rayon)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  test: {
    globals: true,
    include: ['index.test.ts', 'benchmark.test.ts'],
    fileParallelism: false,  // Sequential test files to avoid WASM/WebGPU conflicts
    testTimeout: 120000,  // 2 minute timeout for slow GPU tests
    // Use playwright browser for WASM tests (full SIMD + SharedArrayBuffer support)
    browser: {
      enabled: true,
      provider: playwright({
        launch: {
          headless: true,
          args: [
            // Enable WebGPU in headless mode
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--enable-unsafe-webgpu',
            '--enable-features=SharedArrayBuffer',
          ],
        },
      }),
      instances: [
        { browser: 'chromium' },
      ],
    },
  },
});
